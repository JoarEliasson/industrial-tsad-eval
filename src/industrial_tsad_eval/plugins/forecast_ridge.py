"""Forecasting-residual detector plugin based on Ridge regression."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from industrial_tsad_eval.ports.detectors import Detector, DetectorRunConfig
from industrial_tsad_eval.ports.repositories import PreparedDatasetRepository


@dataclass(frozen=True)
class ForecastRidgeConfig:
    """Hyperparameters for the ForecastRidge detector."""

    window: int = 128
    stride: int = 16
    alpha: float = 1.0
    lags: int = 1
    standardize: bool = True
    seed: int = 1337
    score_workers: int = 4


class ForecastRidgeDetector:
    """Next-step forecasting baseline using ridge regression residuals."""

    def __init__(self, config: ForecastRidgeConfig):
        self.config = config
        self.model = Ridge(alpha=config.alpha, random_state=config.seed)
        self.scaler = StandardScaler() if config.standardize else None
        self.features: list[str] = []
        self.is_fitted = False
        self._score_batch_telemetry: dict[str, Any] = {}

    def train(self, repository: PreparedDatasetRepository, protocol: str) -> None:
        """Fit the ridge forecaster on normal train and validation runs."""
        self.features = _feature_columns(repository)
        split = _protocol_split(repository, protocol)
        train_runs = list(split.get("train_runs", [])) + list(split.get("val_runs", []))
        x_all: list[np.ndarray] = []
        y_all: list[np.ndarray] = []

        for run_id in train_runs:
            frame = repository.read_run(str(run_id))
            x_run, y_run = self._build_supervised_samples(frame)
            if len(x_run) > 0:
                x_all.append(x_run)
                y_all.append(y_run)

        if not x_all:
            raise ValueError(f"No valid training windows found for protocol {protocol!r}.")

        x_train = np.concatenate(x_all, axis=0)
        y_train = np.concatenate(y_all, axis=0)
        if self.scaler is not None:
            x_train = self.scaler.fit_transform(x_train)
        self.model.fit(x_train, y_train)
        self.is_fitted = True

    def score_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Score one prepared run and return Score Contract v1 rows."""
        if not self.is_fitted:
            raise RuntimeError("ForecastRidgeDetector must be trained before scoring.")

        frame = repository.read_run(run_id)
        ts_ns = frame["ts_ns"].to_numpy(dtype=np.int64)
        x_test, y_test = self._build_supervised_samples(frame)
        if len(x_test) == 0:
            return pd.DataFrame({"ts_ns": ts_ns, "score": np.zeros(len(ts_ns), dtype=np.float64)})

        if self.scaler is not None:
            x_test = self.scaler.transform(x_test)
        y_pred = self.model.predict(x_test)
        residual_scores = np.mean((y_pred - y_test) ** 2, axis=1)
        pad = np.full(self.config.lags, residual_scores[0], dtype=np.float64)
        point_scores = np.concatenate([pad, residual_scores.astype(np.float64)])

        window_count = (len(ts_ns) - self.config.window) // self.config.stride + 1
        if window_count <= 0:
            return pd.DataFrame(
                {"ts_ns": [int(ts_ns[-1])], "score": [float(np.mean(point_scores))]}
            )

        window_ends: list[int] = []
        scores: list[float] = []
        for index in range(window_count):
            start = index * self.config.stride
            end = start + self.config.window
            window_ends.append(int(ts_ns[end - 1]))
            scores.append(float(np.mean(point_scores[start:end])))
        return pd.DataFrame({"ts_ns": window_ends, "score": scores})

    def score_runs(
        self,
        repository: PreparedDatasetRepository,
        run_ids: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Score runs with bounded CPU parallelism."""
        workers = _resolved_score_workers(self.config.score_workers, len(run_ids))
        if workers <= 1 or len(run_ids) <= 1:
            output = {run_id: self.score_run(repository, run_id) for run_id in run_ids}
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                frames = executor.map(lambda run_id: self.score_run(repository, run_id), run_ids)
                output = dict(zip(run_ids, frames, strict=True))
        self._score_batch_telemetry = {
            "mode": "cpu-parallel",
            "workers": workers,
            "run_count": len(run_ids),
            "run_read_count": len(run_ids),
            "total_windows": int(sum(len(frame) for frame in output.values())),
        }
        return output

    def score_batch_telemetry(self) -> dict[str, Any]:
        """Return telemetry from the most recent batch scoring call."""
        return dict(self._score_batch_telemetry)

    def metadata(self) -> dict[str, Any]:
        """Return detector metadata for score artifact provenance."""
        return {
            "detector": "forecast-ridge",
            "window": self.config.window,
            "stride": self.config.stride,
            "alpha": self.config.alpha,
            "lags": self.config.lags,
            "standardize": self.config.standardize,
            "seed": self.config.seed,
            "score_workers": self.config.score_workers,
            "feature_columns": list(self.features),
        }

    def _build_supervised_samples(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        data = frame.reindex(columns=self.features, fill_value=0.0).to_numpy(dtype=np.float64)
        row_count, feature_count = data.shape
        if row_count <= self.config.lags:
            return np.empty((0, feature_count * self.config.lags)), np.empty((0, feature_count))
        x_parts = [
            data[offset : row_count - self.config.lags + offset]
            for offset in range(self.config.lags)
        ]
        x = np.concatenate(x_parts, axis=1)
        y = data[self.config.lags :]
        return x, y


class ForecastRidgePlugin:
    """Factory for the ForecastRidge detector."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "forecast-ridge"

    @property
    def requires_torch(self) -> bool:
        """Return whether this plugin requires torch."""
        return False

    def create(self, config: DetectorRunConfig) -> Detector:
        """Create an unfitted ForecastRidge detector."""
        params = dict(config.parameters)
        detector_config = ForecastRidgeConfig(
            window=int(params.get("window", ForecastRidgeConfig.window)),
            stride=int(params.get("stride", ForecastRidgeConfig.stride)),
            alpha=float(params.get("alpha", ForecastRidgeConfig.alpha)),
            lags=int(params.get("lags", ForecastRidgeConfig.lags)),
            standardize=bool(params.get("standardize", ForecastRidgeConfig.standardize)),
            seed=int(params.get("seed", ForecastRidgeConfig.seed)),
            score_workers=int(params.get("score_workers", ForecastRidgeConfig.score_workers)),
        )
        return ForecastRidgeDetector(detector_config)


def _feature_columns(repository: PreparedDatasetRepository) -> list[str]:
    schema = repository.schema()
    tags = schema.get("tags", [])
    columns = [
        str(tag["browse_path"]) for tag in tags if isinstance(tag, dict) and tag.get("browse_path")
    ]
    if columns:
        return columns
    first_run = repository.run_ids()[0]
    frame = repository.read_run(first_run)
    return [str(column) for column in frame.columns if column != "ts_ns"]


def _protocol_split(repository: PreparedDatasetRepository, protocol: str) -> dict[str, list[str]]:
    splits = repository.splits()
    selected = splits.get(protocol, splits.get("naive", splits))
    if not isinstance(selected, dict):
        raise ValueError(f"Split protocol {protocol!r} is not an object.")
    return {
        "train_runs": [str(run_id) for run_id in selected.get("train_runs", [])],
        "val_runs": [str(run_id) for run_id in selected.get("val_runs", [])],
        "test_runs": [str(run_id) for run_id in selected.get("test_runs", [])],
    }


def _resolved_score_workers(configured: int, run_count: int) -> int:
    env_value = os.environ.get("INDUSTRIAL_TSAD_CPU_SCORE_WORKERS")
    if env_value:
        configured = int(env_value)
    cpu_count = os.cpu_count() or 1
    return max(1, min(configured, run_count, cpu_count))
