"""Optional torch-backed detector plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.plugins.torch_common import (
    FeatureStandardizer,
    TorchTrainingConfig,
    batch_slices,
    cap_aligned_arrays,
    empty_score_frame,
    feature_columns,
    forecast_window_batches,
    forecast_window_batches_at,
    forecast_windows,
    read_feature_array,
    require_torch,
    resolve_torch_device,
    score_frame,
    set_torch_seed,
    torch_device_name,
    training_arrays,
    window_end_batches,
    window_end_batches_at,
    window_end_windows,
)
from industrial_tsad_eval.plugins.torch_models import (
    build_drcad,
    build_hvae,
    build_lstm_forecaster,
    build_tcn_forecaster,
)
from industrial_tsad_eval.ports.detectors import Detector, DetectorRunConfig
from industrial_tsad_eval.ports.repositories import PreparedDatasetRepository


@dataclass(frozen=True)
class ForecastLSTMConfig:
    """Configuration for the ForecastLSTM plugin."""

    common: TorchTrainingConfig
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> ForecastLSTMConfig:
        """Build config from detector parameters."""
        config = cls(
            common=TorchTrainingConfig.from_parameters(parameters),
            hidden_size=int(parameters.get("hidden_size", cls.hidden_size)),
            num_layers=int(parameters.get("num_layers", cls.num_layers)),
            dropout=float(parameters.get("dropout", cls.dropout)),
        )
        if config.hidden_size <= 0:
            raise ValueError("hidden_size must be greater than 0.")
        if config.num_layers <= 0:
            raise ValueError("num_layers must be greater than 0.")
        if config.dropout < 0:
            raise ValueError("dropout must be non-negative.")
        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible metadata."""
        return {
            **self.common.to_dict(),
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class DRAConfig:
    """Configuration for DRA-style TCN detection and saliency."""

    common: TorchTrainingConfig
    d: int = 16

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> DRAConfig:
        """Build config from detector parameters."""
        config = cls(
            common=TorchTrainingConfig.from_parameters(parameters),
            d=int(parameters.get("d", cls.d)),
        )
        if config.d <= 0:
            raise ValueError("d must be greater than 0.")
        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible metadata."""
        return {**self.common.to_dict(), "d": self.d}


@dataclass(frozen=True)
class InterFusionConfig:
    """Configuration for InterFusion-style HVAE detection and attribution."""

    common: TorchTrainingConfig
    latent_dim: int = 3
    kl_warmup: int = 2
    mc_samples: int = 8

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> InterFusionConfig:
        """Build config from detector parameters."""
        config = cls(
            common=TorchTrainingConfig.from_parameters(parameters),
            latent_dim=int(parameters.get("latent_dim", cls.latent_dim)),
            kl_warmup=int(parameters.get("kl_warmup", cls.kl_warmup)),
            mc_samples=int(parameters.get("mc_samples", cls.mc_samples)),
        )
        if config.latent_dim <= 0:
            raise ValueError("latent_dim must be greater than 0.")
        if config.kl_warmup < 0:
            raise ValueError("kl_warmup must be non-negative.")
        if config.mc_samples <= 0:
            raise ValueError("mc_samples must be greater than 0.")
        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible metadata."""
        return {
            **self.common.to_dict(),
            "latent_dim": self.latent_dim,
            "kl_warmup": self.kl_warmup,
            "mc_samples": self.mc_samples,
        }


@dataclass(frozen=True)
class DRCADConfig:
    """Configuration for DRCAD-style detection and counterfactual attribution."""

    common: TorchTrainingConfig
    patch_size: int = 4
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    mlp_dim: int = 64
    dropout: float = 0.1

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> DRCADConfig:
        """Build config from detector parameters."""
        config = cls(
            common=TorchTrainingConfig.from_parameters(parameters),
            patch_size=int(parameters.get("patch_size", cls.patch_size)),
            d_model=int(parameters.get("d_model", cls.d_model)),
            n_heads=int(parameters.get("n_heads", cls.n_heads)),
            n_layers=int(parameters.get("n_layers", cls.n_layers)),
            mlp_dim=int(parameters.get("mlp_dim", cls.mlp_dim)),
            dropout=float(parameters.get("dropout", cls.dropout)),
        )
        if config.patch_size <= 0:
            raise ValueError("patch_size must be greater than 0.")
        if config.common.window % config.patch_size != 0:
            raise ValueError("window must be divisible by patch_size.")
        if config.d_model <= 0:
            raise ValueError("d_model must be greater than 0.")
        if config.n_heads <= 0:
            raise ValueError("n_heads must be greater than 0.")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if config.n_layers <= 0:
            raise ValueError("n_layers must be greater than 0.")
        if config.mlp_dim <= 0:
            raise ValueError("mlp_dim must be greater than 0.")
        if config.dropout < 0:
            raise ValueError("dropout must be non-negative.")
        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible metadata."""
        return {
            **self.common.to_dict(),
            "patch_size": self.patch_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "mlp_dim": self.mlp_dim,
            "dropout": self.dropout,
        }


@dataclass(frozen=True)
class _ScoreContext:
    """Per-run score rows retained for sparse native explanation selection."""

    run_id: str
    row_indices: np.ndarray
    scores: np.ndarray


class ForecastLSTMDetector:
    """Many-to-one LSTM next-step forecasting detector."""

    def __init__(self, config: ForecastLSTMConfig):
        self.config = config
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device = "cpu"
        self.device_name: str | None = None
        self.features: list[str] = []
        self.standardizer = FeatureStandardizer(enabled=False)
        self.train_window_count = 0
        self.training_losses: list[float] = []
        self._score_batch_telemetry: dict[str, Any] = {}

    def train(self, repository: PreparedDatasetRepository, protocol: str) -> None:
        """Fit the LSTM forecaster on normal train and validation runs."""
        torch = _initialize_torch(self, self.config.common)
        self.features = feature_columns(repository)
        _run_ids, arrays = training_arrays(repository, protocol, self.features)
        self.standardizer = FeatureStandardizer.fit(arrays, self.config.common.standardize)
        x_train, y_train = _forecast_training_matrices(
            self.config.common, arrays, self.standardizer
        )
        self.train_window_count = int(x_train.shape[0])
        self.model = build_lstm_forecaster(
            torch,
            feature_count=len(self.features),
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)
        self.training_losses = _train_forecast_model(
            torch,
            self.model,
            x_train,
            y_train,
            self.config.common,
            input_layout="time_feature",
        )

    def score_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Score one prepared run using next-step forecast residuals."""
        torch, model = _fitted(self)
        data, ts_ns = read_feature_array(repository, run_id, self.features)
        data = self.standardizer.transform(data)
        scores, target_indices = _score_forecast_run(
            torch,
            model,
            data,
            self.config.common,
            self.device,
            input_layout="time_feature",
        )
        if len(target_indices) == 0:
            return empty_score_frame()
        return score_frame(ts_ns, target_indices, scores)

    def score_runs(
        self,
        repository: PreparedDatasetRepository,
        run_ids: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Score multiple runs with cross-run GPU batches."""
        frames, telemetry = _score_forecast_runs_batched(
            detector=self,
            repository=repository,
            run_ids=run_ids,
            input_layout="time_feature",
            absolute=False,
        )
        self._score_batch_telemetry = telemetry
        return frames

    def score_batch_telemetry(self) -> dict[str, Any]:
        """Return telemetry from the most recent batch scoring call."""
        return dict(self._score_batch_telemetry)

    def metadata(self) -> dict[str, Any]:
        """Return detector metadata for score artifact provenance."""
        return _metadata(
            "forecast-lstm",
            self.config.to_dict(),
            self.device,
            self.device_name,
            self.features,
            self.train_window_count,
            self.training_losses,
        )


class DRADetector:
    """DRA-style TCN detector with residual-gradient saliency explanations."""

    def __init__(self, config: DRAConfig):
        self.config = config
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device = "cpu"
        self.device_name: str | None = None
        self.features: list[str] = []
        self.standardizer = FeatureStandardizer(enabled=False)
        self.train_window_count = 0
        self.training_losses: list[float] = []
        self._last_score_context: _ScoreContext | None = None
        self._score_batch_telemetry: dict[str, Any] = {}

    def train(self, repository: PreparedDatasetRepository, protocol: str) -> None:
        """Fit the TCN forecaster on normal train and validation runs."""
        torch = _initialize_torch(self, self.config.common)
        self.features = feature_columns(repository)
        _run_ids, arrays = training_arrays(repository, protocol, self.features)
        self.standardizer = FeatureStandardizer.fit(arrays, self.config.common.standardize)
        x_train, y_train = _forecast_training_matrices(
            self.config.common, arrays, self.standardizer
        )
        self.train_window_count = int(x_train.shape[0])
        self.model = build_tcn_forecaster(
            torch,
            feature_count=len(self.features),
            d=self.config.d,
        ).to(self.device)
        self.training_losses = _train_forecast_model(
            torch,
            self.model,
            x_train,
            y_train,
            self.config.common,
            input_layout="feature_time",
        )

    def score_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Score one prepared run using mean absolute forecast residual."""
        torch, model = _fitted(self)
        data, ts_ns = read_feature_array(repository, run_id, self.features)
        data = self.standardizer.transform(data)
        scores, target_indices = _score_forecast_run(
            torch,
            model,
            data,
            self.config.common,
            self.device,
            input_layout="feature_time",
            absolute=True,
        )
        if len(target_indices) == 0:
            self._last_score_context = None
            return empty_score_frame()
        self._last_score_context = _ScoreContext(run_id, target_indices, scores)
        return score_frame(ts_ns, target_indices, scores)

    def score_runs(
        self,
        repository: PreparedDatasetRepository,
        run_ids: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Score multiple runs with cross-run GPU batches."""
        frames, telemetry = _score_forecast_runs_batched(
            detector=self,
            repository=repository,
            run_ids=run_ids,
            input_layout="feature_time",
            absolute=True,
        )
        self._score_batch_telemetry = telemetry
        return frames

    def score_batch_telemetry(self) -> dict[str, Any]:
        """Return telemetry from the most recent batch scoring call."""
        return dict(self._score_batch_telemetry)

    def explain_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Explain one prepared run using residual-weighted input gradients."""
        torch, model = _fitted(self)
        data, ts_ns = read_feature_array(repository, run_id, self.features)
        data = self.standardizer.transform(data)
        context = (
            self._last_score_context
            if self._last_score_context is not None and self._last_score_context.run_id == run_id
            else None
        )
        target_pool = (
            _forecast_target_pool(
                len(ts_ns),
                self.config.common.window,
                self.config.common.score_stride,
            )
            if context is None
            else context.row_indices
        )
        target_indices = _selected_explanation_indices(
            repository,
            run_id,
            ts_ns,
            target_pool,
            np.zeros(len(target_pool), dtype=np.float64) if context is None else context.scores,
            self.config.common,
        )
        if len(target_indices) == 0:
            return _empty_explanation_frame()
        frames: list[pd.DataFrame] = []
        for x, y, batch_targets in forecast_window_batches_at(
            data,
            target_indices,
            window=self.config.common.window,
            batch_size=self.config.common.score_batch_size,
        ):
            importances = _forecast_saliency(
                torch,
                model,
                x,
                y,
                self.config.common,
                self.device,
                input_layout="feature_time",
                absolute=True,
            )
            frames.append(
                _ranked_explanation_frame(
                    ts_ns=ts_ns,
                    target_indices=batch_targets,
                    features=self.features,
                    importances=importances,
                    method="dra-residual-gradient-saliency",
                    window=self.config.common.window,
                    top_k=self.config.common.explanation_top_k,
                )
            )
        return _concat_explanation_frames(frames)

    def metadata(self) -> dict[str, Any]:
        """Return detector metadata for score artifact provenance."""
        return _metadata(
            "dra",
            self.config.to_dict(),
            self.device,
            self.device_name,
            self.features,
            self.train_window_count,
            self.training_losses,
        )


class InterFusionDetector:
    """InterFusion-style HVAE detector with stochastic reconstruction attribution."""

    def __init__(self, config: InterFusionConfig):
        self.config = config
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device = "cpu"
        self.device_name: str | None = None
        self.features: list[str] = []
        self.standardizer = FeatureStandardizer(enabled=False)
        self.train_window_count = 0
        self.training_losses: list[float] = []
        self._last_score_context: _ScoreContext | None = None
        self._score_batch_telemetry: dict[str, Any] = {}

    def train(self, repository: PreparedDatasetRepository, protocol: str) -> None:
        """Fit the HVAE on normal train and validation windows."""
        torch = _initialize_torch(self, self.config.common)
        self.features = feature_columns(repository)
        _run_ids, arrays = training_arrays(repository, protocol, self.features)
        self.standardizer = FeatureStandardizer.fit(arrays, self.config.common.standardize)
        x_train = _window_training_matrix(self.config.common, arrays, self.standardizer)
        self.train_window_count = int(x_train.shape[0])
        self.model = build_hvae(
            torch,
            feature_count=len(self.features),
            window=self.config.common.window,
            latent_dim=self.config.latent_dim,
        ).to(self.device)
        self.training_losses = _train_hvae_model(torch, self.model, x_train, self.config)

    def score_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Score one prepared run using HVAE reconstruction error."""
        torch, model = _fitted(self)
        data, ts_ns = read_feature_array(repository, run_id, self.features)
        data = self.standardizer.transform(data)
        scores, end_indices = _score_window_run(
            torch,
            model,
            data,
            self.config.common,
            self.device,
        )
        if len(end_indices) == 0:
            self._last_score_context = None
            return empty_score_frame()
        self._last_score_context = _ScoreContext(run_id, end_indices, scores)
        return score_frame(ts_ns, end_indices, scores)

    def score_runs(
        self,
        repository: PreparedDatasetRepository,
        run_ids: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Score multiple runs with cross-run GPU batches."""
        frames, telemetry = _score_window_runs_batched(
            detector=self,
            repository=repository,
            run_ids=run_ids,
        )
        self._score_batch_telemetry = telemetry
        return frames

    def score_batch_telemetry(self) -> dict[str, Any]:
        """Return telemetry from the most recent batch scoring call."""
        return dict(self._score_batch_telemetry)

    def explain_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Explain one prepared run using Monte Carlo reconstruction/imputation error."""
        torch, model = _fitted(self)
        data, ts_ns = read_feature_array(repository, run_id, self.features)
        data = self.standardizer.transform(data)
        context = (
            self._last_score_context
            if self._last_score_context is not None and self._last_score_context.run_id == run_id
            else None
        )
        end_pool = (
            _window_end_pool(len(ts_ns), self.config.common.window, self.config.common.score_stride)
            if context is None
            else context.row_indices
        )
        end_indices = _selected_explanation_indices(
            repository,
            run_id,
            ts_ns,
            end_pool,
            np.zeros(len(end_pool), dtype=np.float64) if context is None else context.scores,
            self.config.common,
        )
        if len(end_indices) == 0:
            return _empty_explanation_frame()
        frames: list[pd.DataFrame] = []
        for x, batch_ends in window_end_batches_at(
            data,
            end_indices,
            window=self.config.common.window,
            batch_size=self.config.common.score_batch_size,
        ):
            importances = _hvae_reconstruction_importance(
                torch,
                model,
                x,
                self.config,
                self.device,
            )
            frames.append(
                _ranked_explanation_frame(
                    ts_ns=ts_ns,
                    target_indices=batch_ends,
                    features=self.features,
                    importances=importances,
                    method="interfusion-mc-reconstruction-imputation",
                    window=self.config.common.window,
                    top_k=self.config.common.explanation_top_k,
                )
            )
        return _concat_explanation_frames(frames)

    def metadata(self) -> dict[str, Any]:
        """Return detector metadata for score artifact provenance."""
        return _metadata(
            "interfusion",
            self.config.to_dict(),
            self.device,
            self.device_name,
            self.features,
            self.train_window_count,
            self.training_losses,
        )


class DRCADDetector:
    """DRCAD-style dual-view detector with counterfactual reconstruction attribution."""

    def __init__(self, config: DRCADConfig):
        self.config = config
        self.model: Any | None = None
        self.torch: Any | None = None
        self.device = "cpu"
        self.device_name: str | None = None
        self.features: list[str] = []
        self.standardizer = FeatureStandardizer(enabled=False)
        self.train_window_count = 0
        self.training_losses: list[float] = []
        self._last_score_context: _ScoreContext | None = None
        self._score_batch_telemetry: dict[str, Any] = {}

    def train(self, repository: PreparedDatasetRepository, protocol: str) -> None:
        """Fit the dual-view contrastive model on normal windows."""
        torch = _initialize_torch(self, self.config.common)
        self.features = feature_columns(repository)
        _run_ids, arrays = training_arrays(repository, protocol, self.features)
        self.standardizer = FeatureStandardizer.fit(arrays, self.config.common.standardize)
        x_train = _window_training_matrix(self.config.common, arrays, self.standardizer)
        self.train_window_count = int(x_train.shape[0])
        self.model = build_drcad(
            torch,
            feature_count=len(self.features),
            window=self.config.common.window,
            patch_size=self.config.patch_size,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            mlp_dim=self.config.mlp_dim,
            dropout=self.config.dropout,
        ).to(self.device)
        self.training_losses = _train_drcad_model(torch, self.model, x_train, self.config.common)

    def score_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Score one prepared run using symmetric dual-view KL divergence."""
        torch, model = _fitted(self)
        data, ts_ns = read_feature_array(repository, run_id, self.features)
        data = self.standardizer.transform(data)
        scores, end_indices = _score_window_run(
            torch,
            model,
            data,
            self.config.common,
            self.device,
        )
        if len(end_indices) == 0:
            self._last_score_context = None
            return empty_score_frame()
        self._last_score_context = _ScoreContext(run_id, end_indices, scores)
        return score_frame(ts_ns, end_indices, scores)

    def score_runs(
        self,
        repository: PreparedDatasetRepository,
        run_ids: list[str],
    ) -> dict[str, pd.DataFrame]:
        """Score multiple runs with cross-run GPU batches."""
        frames, telemetry = _score_window_runs_batched(
            detector=self,
            repository=repository,
            run_ids=run_ids,
        )
        self._score_batch_telemetry = telemetry
        return frames

    def score_batch_telemetry(self) -> dict[str, Any]:
        """Return telemetry from the most recent batch scoring call."""
        return dict(self._score_batch_telemetry)

    def explain_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Explain one prepared run using counterfactual reconstruction deltas."""
        torch, model = _fitted(self)
        data, ts_ns = read_feature_array(repository, run_id, self.features)
        data = self.standardizer.transform(data)
        context = (
            self._last_score_context
            if self._last_score_context is not None and self._last_score_context.run_id == run_id
            else None
        )
        end_pool = (
            _window_end_pool(len(ts_ns), self.config.common.window, self.config.common.score_stride)
            if context is None
            else context.row_indices
        )
        end_indices = _selected_explanation_indices(
            repository,
            run_id,
            ts_ns,
            end_pool,
            np.zeros(len(end_pool), dtype=np.float64) if context is None else context.scores,
            self.config.common,
        )
        if len(end_indices) == 0:
            return _empty_explanation_frame()
        frames: list[pd.DataFrame] = []
        for x, batch_ends in window_end_batches_at(
            data,
            end_indices,
            window=self.config.common.window,
            batch_size=self.config.common.score_batch_size,
        ):
            importances = _drcad_counterfactual_importance(
                torch,
                model,
                x,
                self.config.common,
                self.device,
            )
            frames.append(
                _ranked_explanation_frame(
                    ts_ns=ts_ns,
                    target_indices=batch_ends,
                    features=self.features,
                    importances=importances,
                    method="drcad-cvae-counterfactual-delta",
                    window=self.config.common.window,
                    top_k=self.config.common.explanation_top_k,
                )
            )
        return _concat_explanation_frames(frames)

    def metadata(self) -> dict[str, Any]:
        """Return detector metadata for score artifact provenance."""
        return _metadata(
            "drcad",
            self.config.to_dict(),
            self.device,
            self.device_name,
            self.features,
            self.train_window_count,
            self.training_losses,
        )


class ForecastLSTMPlugin:
    """Factory for the ForecastLSTM detector."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "forecast-lstm"

    @property
    def requires_torch(self) -> bool:
        """Return whether this plugin requires torch."""
        return True

    def create(self, config: DetectorRunConfig) -> Detector:
        """Create an unfitted ForecastLSTM detector."""
        return ForecastLSTMDetector(ForecastLSTMConfig.from_parameters(config.parameters))


class DRAPlugin:
    """Factory for the DRA detector."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "dra"

    @property
    def requires_torch(self) -> bool:
        """Return whether this plugin requires torch."""
        return True

    def create(self, config: DetectorRunConfig) -> Detector:
        """Create an unfitted DRA detector."""
        return DRADetector(DRAConfig.from_parameters(config.parameters))


class InterFusionPlugin:
    """Factory for the InterFusion detector."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "interfusion"

    @property
    def requires_torch(self) -> bool:
        """Return whether this plugin requires torch."""
        return True

    def create(self, config: DetectorRunConfig) -> Detector:
        """Create an unfitted InterFusion detector."""
        return InterFusionDetector(InterFusionConfig.from_parameters(config.parameters))


class DRCADPlugin:
    """Factory for the DRCAD detector."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "drcad"

    @property
    def requires_torch(self) -> bool:
        """Return whether this plugin requires torch."""
        return True

    def create(self, config: DetectorRunConfig) -> Detector:
        """Create an unfitted DRCAD detector."""
        return DRCADDetector(DRCADConfig.from_parameters(config.parameters))


def _initialize_torch(detector: Any, config: TorchTrainingConfig) -> Any:
    torch = require_torch()
    set_torch_seed(torch, config.seed)
    detector.torch = torch
    detector.device = resolve_torch_device(torch, config.device)
    detector.device_name = torch_device_name(torch, detector.device)
    return torch


def _fitted(detector: Any) -> tuple[Any, Any]:
    if detector.torch is None or detector.model is None:
        raise RuntimeError("Detector must be trained before scoring.")
    return detector.torch, detector.model


def _score_forecast_run(
    torch: Any,
    model: Any,
    data: np.ndarray,
    config: TorchTrainingConfig,
    device: str,
    *,
    input_layout: str,
    absolute: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    score_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for x, y, target_indices in forecast_window_batches(
        data,
        window=config.window,
        stride=config.score_stride,
        batch_size=config.score_batch_size,
    ):
        score_parts.append(
            _score_forecast_model(
                torch,
                model,
                x,
                y,
                config,
                device,
                input_layout=input_layout,
                absolute=absolute,
            )
        )
        index_parts.append(target_indices)
    if not index_parts:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    return np.concatenate(score_parts), np.concatenate(index_parts)


def _score_window_run(
    torch: Any,
    model: Any,
    data: np.ndarray,
    config: TorchTrainingConfig,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    score_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for x, end_indices in window_end_batches(
        data,
        window=config.window,
        stride=config.score_stride,
        batch_size=config.score_batch_size,
    ):
        score_parts.append(_score_window_model(torch, model, x, config, device))
        index_parts.append(end_indices)
    if not index_parts:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.int64)
    return np.concatenate(score_parts), np.concatenate(index_parts)


def _score_forecast_runs_batched(
    *,
    detector: Any,
    repository: PreparedDatasetRepository,
    run_ids: list[str],
    input_layout: str,
    absolute: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    torch, model = _fitted(detector)
    frames: dict[str, pd.DataFrame] = {}
    parts: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    pending_windows = 0
    batch_count = 0
    max_batch_windows = 0
    total_windows = 0

    def flush() -> None:
        nonlocal batch_count, max_batch_windows, pending_windows, total_windows
        if not parts:
            return
        x = np.concatenate([part[2] for part in parts], axis=0)
        y = np.concatenate([part[3] for part in parts], axis=0)
        scores = _score_forecast_model(
            torch,
            model,
            x,
            y,
            detector.config.common,
            detector.device,
            input_layout=input_layout,
            absolute=absolute,
        )
        batch_count += 1
        max_batch_windows = max(max_batch_windows, int(x.shape[0]))
        total_windows += int(x.shape[0])
        offset = 0
        for run_id, ts_ns, part_x, _part_y, target_indices in parts:
            count = int(part_x.shape[0])
            frame = score_frame(ts_ns, target_indices, scores[offset : offset + count])
            frames[run_id] = (
                pd.concat([frames[run_id], frame], ignore_index=True) if run_id in frames else frame
            )
            offset += count
        parts.clear()
        pending_windows = 0

    for run_id in run_ids:
        data, ts_ns = read_feature_array(repository, run_id, detector.features)
        data = detector.standardizer.transform(data)
        has_windows = False
        for x, y, target_indices in forecast_window_batches(
            data,
            window=detector.config.common.window,
            stride=detector.config.common.score_stride,
            batch_size=detector.config.common.score_batch_size,
        ):
            has_windows = True
            parts.append((run_id, ts_ns, x, y, target_indices))
            pending_windows += int(x.shape[0])
            if pending_windows >= detector.config.common.score_batch_size:
                flush()
        if not has_windows:
            frames[run_id] = empty_score_frame()
    flush()
    return _ordered_frames(frames, run_ids), {
        "mode": "cross-run-gpu-batch",
        "run_count": len(run_ids),
        "run_read_count": len(run_ids),
        "batch_count": batch_count,
        "max_batch_windows": max_batch_windows,
        "total_windows": total_windows,
        "score_batch_size": detector.config.common.score_batch_size,
    }


def _score_window_runs_batched(
    *,
    detector: Any,
    repository: PreparedDatasetRepository,
    run_ids: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    torch, model = _fitted(detector)
    frames: dict[str, pd.DataFrame] = {}
    parts: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    pending_windows = 0
    batch_count = 0
    max_batch_windows = 0
    total_windows = 0

    def flush() -> None:
        nonlocal batch_count, max_batch_windows, pending_windows, total_windows
        if not parts:
            return
        x = np.concatenate([part[2] for part in parts], axis=0)
        scores = _score_window_model(
            torch,
            model,
            x,
            detector.config.common,
            detector.device,
        )
        batch_count += 1
        max_batch_windows = max(max_batch_windows, int(x.shape[0]))
        total_windows += int(x.shape[0])
        offset = 0
        for run_id, ts_ns, part_x, end_indices in parts:
            count = int(part_x.shape[0])
            frame = score_frame(ts_ns, end_indices, scores[offset : offset + count])
            frames[run_id] = (
                pd.concat([frames[run_id], frame], ignore_index=True) if run_id in frames else frame
            )
            offset += count
        parts.clear()
        pending_windows = 0

    for run_id in run_ids:
        data, ts_ns = read_feature_array(repository, run_id, detector.features)
        data = detector.standardizer.transform(data)
        has_windows = False
        for x, end_indices in window_end_batches(
            data,
            window=detector.config.common.window,
            stride=detector.config.common.score_stride,
            batch_size=detector.config.common.score_batch_size,
        ):
            has_windows = True
            parts.append((run_id, ts_ns, x, end_indices))
            pending_windows += int(x.shape[0])
            if pending_windows >= detector.config.common.score_batch_size:
                flush()
        if not has_windows:
            frames[run_id] = empty_score_frame()
    flush()
    return _ordered_frames(frames, run_ids), {
        "mode": "cross-run-gpu-batch",
        "run_count": len(run_ids),
        "run_read_count": len(run_ids),
        "batch_count": batch_count,
        "max_batch_windows": max_batch_windows,
        "total_windows": total_windows,
        "score_batch_size": detector.config.common.score_batch_size,
    }


def _ordered_frames(
    frames: dict[str, pd.DataFrame],
    run_ids: list[str],
) -> dict[str, pd.DataFrame]:
    return {run_id: frames.get(run_id, empty_score_frame()) for run_id in run_ids}


def _selected_explanation_indices(
    repository: PreparedDatasetRepository,
    run_id: str,
    ts_ns: np.ndarray,
    row_indices: np.ndarray,
    scores: np.ndarray,
    config: TorchTrainingConfig,
) -> np.ndarray:
    if len(row_indices) == 0:
        return row_indices.astype(np.int64)
    selected = np.zeros(len(row_indices), dtype=bool)
    timestamps = ts_ns[row_indices]
    read_events = getattr(repository, "read_events", None)
    events = read_events() if callable(read_events) else []
    for event in events:
        if getattr(event, "run_id", None) != run_id:
            continue
        selected |= (timestamps >= int(event.start_ts_ns)) & (timestamps < int(event.end_ts_ns))

    if len(scores) == len(row_indices) and len(scores) > 0:
        finite_scores = np.asarray(scores, dtype=np.float64)
        if np.isfinite(finite_scores).any():
            threshold = float(np.nanquantile(finite_scores, config.explanation_score_quantile))
            background = np.flatnonzero(finite_scores >= threshold)
            if len(background) > config.explanation_max_background_windows:
                ranked = np.argsort(-finite_scores[background], kind="stable")
                background = background[ranked[: config.explanation_max_background_windows]]
            selected[background] = True

            if int(selected.sum()) < min(config.explanation_min_windows, len(row_indices)):
                ranked = np.argsort(-finite_scores, kind="stable")
                selected[ranked[: min(config.explanation_min_windows, len(row_indices))]] = True

    if not selected.any():
        selected[: min(config.explanation_min_windows, len(row_indices))] = True
    return row_indices[np.flatnonzero(selected)].astype(np.int64)


def _forecast_target_pool(row_count: int, window: int, stride: int) -> np.ndarray:
    if row_count <= window:
        return np.empty(0, dtype=np.int64)
    return np.arange(window, row_count, stride, dtype=np.int64)


def _window_end_pool(row_count: int, window: int, stride: int) -> np.ndarray:
    if row_count < window:
        return np.empty(0, dtype=np.int64)
    return np.arange(window - 1, row_count, stride, dtype=np.int64)


def _concat_explanation_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_explanation_frame()
    return pd.concat(frames, ignore_index=True)


def _forecast_training_matrices(
    config: TorchTrainingConfig,
    arrays: list[np.ndarray],
    standardizer: FeatureStandardizer,
) -> tuple[np.ndarray, np.ndarray]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for array in arrays:
        x_run, y_run, _indices = forecast_windows(
            standardizer.transform(array),
            window=config.window,
            stride=config.train_stride,
        )
        if len(x_run) > 0:
            x_parts.append(x_run)
            y_parts.append(y_run)
    if not x_parts:
        raise ValueError("No valid training windows produced.")
    x_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    x_train, y_train = cap_aligned_arrays(
        (x_train, y_train),
        max_count=config.max_train_windows,
        seed=config.seed,
    )
    return x_train, y_train


def _window_training_matrix(
    config: TorchTrainingConfig,
    arrays: list[np.ndarray],
    standardizer: FeatureStandardizer,
) -> np.ndarray:
    x_parts: list[np.ndarray] = []
    for array in arrays:
        x_run, _indices = window_end_windows(
            standardizer.transform(array),
            window=config.window,
            stride=config.train_stride,
        )
        if len(x_run) > 0:
            x_parts.append(x_run)
    if not x_parts:
        raise ValueError("No valid training windows produced.")
    x_train = np.concatenate(x_parts, axis=0)
    (x_train,) = cap_aligned_arrays(
        (x_train,),
        max_count=config.max_train_windows,
        seed=config.seed,
    )
    return x_train


def _train_forecast_model(
    torch: Any,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TorchTrainingConfig,
    *,
    input_layout: str,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    x_tensor = torch.tensor(_layout(x_train, input_layout), dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    losses: list[float] = []
    model.train()
    for _epoch in range(config.epochs):
        total_loss = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(config.device if config.device != "auto" else "cpu")
            batch_y = batch_y.to(config.device if config.device != "auto" else "cpu")
            batch_x = batch_x.to(next(model.parameters()).device)
            batch_y = batch_y.to(next(model.parameters()).device)
            prediction = model(batch_x)
            loss = torch.nn.functional.mse_loss(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch_x.shape[0])
            seen += int(batch_x.shape[0])
        losses.append(total_loss / max(seen, 1))
    return losses


def _score_forecast_model(
    torch: Any,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    config: TorchTrainingConfig,
    device: str,
    *,
    input_layout: str,
    absolute: bool = False,
) -> np.ndarray:
    scores = np.empty(x.shape[0], dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for window_slice in batch_slices(x.shape[0], config.score_batch_size):
            batch_x = torch.tensor(_layout(x[window_slice], input_layout), dtype=torch.float32).to(
                device
            )
            batch_y = torch.tensor(y[window_slice], dtype=torch.float32).to(device)
            prediction = model(batch_x)
            residual = prediction - batch_y
            batch_scores = residual.abs().mean(dim=-1) if absolute else (residual**2).mean(dim=-1)
            scores[window_slice] = batch_scores.cpu().numpy()
    return scores


def _train_hvae_model(
    torch: Any,
    model: Any,
    x_train: np.ndarray,
    config: InterFusionConfig,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.common.lr)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.common.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.common.seed),
    )
    losses: list[float] = []
    model.train()
    for epoch in range(config.common.epochs):
        kl_weight = min(1.0, (epoch + 1) / max(config.kl_warmup, 1))
        losses.append(
            _train_window_epoch(torch, model, optimizer, loader, config.common.device, kl_weight)
        )
    return losses


def _train_drcad_model(
    torch: Any,
    model: Any,
    x_train: np.ndarray,
    config: TorchTrainingConfig,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    losses: list[float] = []
    model.train()
    for _epoch in range(config.epochs):
        losses.append(_train_window_epoch(torch, model, optimizer, loader, config.device, 1.0))
    return losses


def _train_window_epoch(
    torch: Any,
    model: Any,
    optimizer: Any,
    loader: Any,
    requested_device: str,
    kl_weight: float,
) -> float:
    total_loss = 0.0
    seen = 0
    for (batch_x,) in loader:
        batch_x = batch_x.to(requested_device if requested_device != "auto" else "cpu")
        batch_x = batch_x.to(next(model.parameters()).device)
        if hasattr(model, "negative_elbo"):
            loss = model.negative_elbo(batch_x, kl_weight)
        else:
            loss = model.contrastive_loss(batch_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(batch_x.shape[0])
        seen += int(batch_x.shape[0])
    return total_loss / max(seen, 1)


def _score_window_model(
    torch: Any,
    model: Any,
    x: np.ndarray,
    config: TorchTrainingConfig,
    device: str,
) -> np.ndarray:
    scores = np.empty(x.shape[0], dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for window_slice in batch_slices(x.shape[0], config.score_batch_size):
            batch_x = torch.tensor(x[window_slice], dtype=torch.float32).to(device)
            scores[window_slice] = model.deterministic_score(batch_x).cpu().numpy()
    return scores


def _forecast_saliency(
    torch: Any,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    config: TorchTrainingConfig,
    device: str,
    *,
    input_layout: str,
    absolute: bool,
) -> np.ndarray:
    importances = np.empty((x.shape[0], x.shape[2]), dtype=np.float64)
    model.eval()
    for window_slice in batch_slices(x.shape[0], config.score_batch_size):
        batch_x = torch.tensor(
            _layout(x[window_slice], input_layout),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        batch_y = torch.tensor(y[window_slice], dtype=torch.float32, device=device)
        prediction = model(batch_x)
        residual = prediction - batch_y
        batch_scores = residual.abs().mean(dim=-1) if absolute else (residual**2).mean(dim=-1)
        model.zero_grad(set_to_none=True)
        batch_scores.sum().backward()
        gradient = batch_x.grad.detach().abs().cpu().numpy()
        if input_layout == "feature_time":
            gradient = np.transpose(gradient, (0, 2, 1))
        gradient_importance = gradient.mean(axis=1)
        residual_importance = residual.detach().abs().cpu().numpy()
        importances[window_slice] = gradient_importance + residual_importance
    return importances


def _hvae_reconstruction_importance(
    torch: Any,
    model: Any,
    x: np.ndarray,
    config: InterFusionConfig,
    device: str,
) -> np.ndarray:
    importances = np.empty((x.shape[0], x.shape[2]), dtype=np.float64)
    was_training = bool(model.training)
    try:
        model.train()
        with torch.no_grad():
            for window_slice in batch_slices(x.shape[0], config.common.score_batch_size):
                batch_x = torch.tensor(x[window_slice], dtype=torch.float32, device=device)
                accumulated = torch.zeros(
                    (batch_x.shape[0], batch_x.shape[2]),
                    dtype=torch.float32,
                    device=device,
                )
                for _sample in range(config.mc_samples):
                    reconstruction, _mu, _logvar = model(batch_x)
                    accumulated += (reconstruction - batch_x).abs().mean(dim=1)
                importances[window_slice] = (accumulated / float(config.mc_samples)).cpu().numpy()
    finally:
        model.train(was_training)
    return importances


def _drcad_counterfactual_importance(
    torch: Any,
    model: Any,
    x: np.ndarray,
    config: TorchTrainingConfig,
    device: str,
) -> np.ndarray:
    importances = np.empty((x.shape[0], x.shape[2]), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for window_slice in batch_slices(x.shape[0], config.score_batch_size):
            batch_x = torch.tensor(x[window_slice], dtype=torch.float32, device=device)
            reconstruction = model.reconstruct(batch_x)
            recon_delta = (reconstruction - batch_x).abs().mean(dim=1)
            scores = model.deterministic_score(batch_x).reshape(-1, 1)
            importances[window_slice] = (recon_delta * (1.0 + scores)).cpu().numpy()
    return importances


def _ranked_explanation_frame(
    *,
    ts_ns: np.ndarray,
    target_indices: np.ndarray,
    features: list[str],
    importances: np.ndarray,
    method: str,
    window: int,
    top_k: int,
) -> pd.DataFrame:
    top_count = min(top_k, len(features))
    if top_count <= 0 or len(target_indices) == 0:
        return _empty_explanation_frame()

    positive_importances = np.maximum(importances, 0.0)
    ranked = np.argsort(-positive_importances, axis=1)[:, :top_count]
    row_positions = np.repeat(np.arange(len(target_indices), dtype=np.int64), top_count)
    feature_indices = ranked.reshape(-1)
    targets = np.repeat(target_indices.astype(np.int64), top_count)
    window_starts = np.maximum(targets - window + 1, 0)
    ranks = np.tile(np.arange(1, top_count + 1, dtype=np.int64), len(target_indices))
    feature_names = np.asarray(features, dtype=object)[feature_indices]
    importance_values = positive_importances[row_positions, feature_indices].astype(np.float64)
    return pd.DataFrame(
        {
            "ts_ns": ts_ns[targets].astype(np.int64),
            "variable": feature_names,
            "importance": importance_values,
            "rank": ranks,
            "method": np.full(len(targets), method, dtype=object),
            "window_start_ts_ns": ts_ns[window_starts].astype(np.int64),
            "window_end_ts_ns": ts_ns[targets].astype(np.int64),
        }
    )


def _empty_explanation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_ns": np.empty(0, dtype=np.int64),
            "variable": np.empty(0, dtype=object),
            "importance": np.empty(0, dtype=np.float64),
            "rank": np.empty(0, dtype=np.int64),
            "method": np.empty(0, dtype=object),
            "window_start_ts_ns": np.empty(0, dtype=np.int64),
            "window_end_ts_ns": np.empty(0, dtype=np.int64),
        }
    )


def _layout(array: np.ndarray, layout: str) -> np.ndarray:
    if layout == "feature_time":
        return np.transpose(array, (0, 2, 1)).copy()
    return array


def _metadata(
    detector: str,
    parameters: dict[str, Any],
    device: str,
    device_name: str | None,
    features: list[str],
    train_window_count: int,
    training_losses: list[float],
) -> dict[str, Any]:
    return {
        "detector": detector,
        "parameters": parameters,
        "resolved_device": device,
        "device_name": device_name,
        "feature_columns": list(features),
        "train_window_count": train_window_count,
        "training_losses": list(training_losses),
    }
