"""Shared optional torch utilities for detector plugins."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.errors import OptionalDependencyError
from industrial_tsad_eval.ports.repositories import PreparedDatasetRepository


@dataclass(frozen=True)
class TorchTrainingConfig:
    """Common hyperparameters for torch detector plugins."""

    window: int = 32
    train_stride: int = 4
    score_stride: int = 4
    max_train_windows: int | None = 256
    epochs: int = 2
    batch_size: int = 32
    lr: float = 1e-3
    seed: int = 1337
    device: str = "auto"
    standardize: bool = True
    score_batch_size: int = 512
    explanation_top_k: int = 10
    explanation_score_quantile: float = 0.995
    explanation_min_windows: int = 256
    explanation_max_background_windows: int = 50000

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> TorchTrainingConfig:
        """Build and validate common torch detector parameters."""
        config = cls(
            window=_int_parameter(parameters, "window", cls.window),
            train_stride=_int_parameter(parameters, "train_stride", cls.train_stride),
            score_stride=_int_parameter(parameters, "score_stride", cls.score_stride),
            max_train_windows=_optional_int_parameter(
                parameters, "max_train_windows", cls.max_train_windows
            ),
            epochs=_int_parameter(parameters, "epochs", cls.epochs),
            batch_size=_int_parameter(parameters, "batch_size", cls.batch_size),
            lr=_float_parameter(parameters, "lr", cls.lr),
            seed=_int_parameter(parameters, "seed", cls.seed),
            device=str(parameters.get("device", cls.device)),
            standardize=_bool_parameter(parameters, "standardize", cls.standardize),
            score_batch_size=_int_parameter(parameters, "score_batch_size", cls.score_batch_size),
            explanation_top_k=_int_parameter(
                parameters, "explanation_top_k", cls.explanation_top_k
            ),
            explanation_score_quantile=_float_parameter(
                parameters,
                "explanation_score_quantile",
                cls.explanation_score_quantile,
            ),
            explanation_min_windows=_int_parameter(
                parameters,
                "explanation_min_windows",
                cls.explanation_min_windows,
            ),
            explanation_max_background_windows=_int_parameter(
                parameters,
                "explanation_max_background_windows",
                cls.explanation_max_background_windows,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Raise a value error for invalid common parameters."""
        if self.window <= 1:
            raise ValueError("window must be greater than 1.")
        if self.train_stride <= 0:
            raise ValueError("train_stride must be greater than 0.")
        if self.score_stride <= 0:
            raise ValueError("score_stride must be greater than 0.")
        if self.max_train_windows is not None and self.max_train_windows <= 0:
            raise ValueError("max_train_windows must be greater than 0 when provided.")
        if self.epochs <= 0:
            raise ValueError("epochs must be greater than 0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if self.lr <= 0:
            raise ValueError("lr must be greater than 0.")
        if self.score_batch_size <= 0:
            raise ValueError("score_batch_size must be greater than 0.")
        if self.explanation_top_k <= 0:
            raise ValueError("explanation_top_k must be greater than 0.")
        if not 0.0 < self.explanation_score_quantile < 1.0:
            raise ValueError("explanation_score_quantile must be in (0, 1).")
        if self.explanation_min_windows <= 0:
            raise ValueError("explanation_min_windows must be greater than 0.")
        if self.explanation_max_background_windows <= 0:
            raise ValueError("explanation_max_background_windows must be greater than 0.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible metadata."""
        return {
            "window": self.window,
            "train_stride": self.train_stride,
            "score_stride": self.score_stride,
            "max_train_windows": self.max_train_windows,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "seed": self.seed,
            "device": self.device,
            "standardize": self.standardize,
            "score_batch_size": self.score_batch_size,
            "explanation_top_k": self.explanation_top_k,
            "explanation_score_quantile": self.explanation_score_quantile,
            "explanation_min_windows": self.explanation_min_windows,
            "explanation_max_background_windows": self.explanation_max_background_windows,
        }


@dataclass(frozen=True)
class FeatureStandardizer:
    """Feature-wise standardizer fitted on training runs."""

    enabled: bool
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    @classmethod
    def fit(cls, arrays: list[np.ndarray], enabled: bool) -> FeatureStandardizer:
        """Fit a standardizer from one or more feature arrays."""
        if not arrays or sum(int(array.shape[0]) for array in arrays) == 0:
            raise ValueError("No training feature rows available.")
        if not enabled:
            return cls(enabled=False)
        values = np.concatenate(arrays, axis=0).astype(np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return cls(enabled=True, mean=mean.astype(np.float32), std=std)

    def transform(self, array: np.ndarray) -> np.ndarray:
        """Apply feature standardization when enabled."""
        if not self.enabled or self.mean is None or self.std is None:
            return array.astype(np.float32)
        return ((array.astype(np.float32) - self.mean) / self.std).astype(np.float32)


def torch_available() -> bool:
    """Return true when the optional torch dependency is importable."""
    return importlib.util.find_spec("torch") is not None


def require_torch() -> Any:
    """Import torch lazily or raise a clear optional dependency error."""
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Torch-backed detectors require the optional torch extra. "
            'Install with `python -m pip install -e ".[torch]"`, or install a '
            "PyTorch wheel matching your CPU/GPU runtime."
        ) from exc


def resolve_torch_device(torch: Any, requested_device: str) -> str:
    """Resolve `auto`, `cpu`, `cuda`, or `xpu` to a runnable torch device."""
    requested = requested_device.strip().lower()
    if requested not in {"auto", "cpu", "cuda", "xpu"}:
        raise ValueError("device must be one of: auto, cpu, cuda, xpu.")

    cuda_available = bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False
    xpu_available = _xpu_available(torch)
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if xpu_available:
            return "xpu"
        return "cpu"
    if requested == "cuda" and not cuda_available:
        raise ValueError("Requested torch device 'cuda' is not available.")
    if requested == "xpu" and not xpu_available:
        raise ValueError("Requested torch device 'xpu' is not available.")
    return requested


def torch_device_name(torch: Any, device: str) -> str | None:
    """Return a best-effort display name for a resolved device."""
    if device == "cuda":
        try:
            return str(torch.cuda.get_device_name(0))
        except Exception:
            return None
    if device == "xpu" and hasattr(torch, "xpu"):
        try:
            return str(torch.xpu.get_device_name(0))
        except Exception:
            return None
    return "cpu"


def set_torch_seed(torch: Any, seed: int) -> None:
    """Set deterministic seeds for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        with suppress(Exception):
            torch.cuda.manual_seed_all(seed)


def feature_columns(repository: PreparedDatasetRepository) -> list[str]:
    """Return canonical feature columns from schema or the first run."""
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


def protocol_split(
    repository: PreparedDatasetRepository,
    protocol: str,
) -> dict[str, list[str]]:
    """Resolve a split protocol into train, validation, and test runs."""
    splits = repository.splits()
    selected = splits.get(protocol, splits.get("naive", splits))
    if not isinstance(selected, dict):
        raise ValueError(f"Split protocol {protocol!r} is not an object.")
    return {
        "train_runs": [str(run_id) for run_id in selected.get("train_runs", [])],
        "val_runs": [str(run_id) for run_id in selected.get("val_runs", [])],
        "test_runs": [str(run_id) for run_id in selected.get("test_runs", [])],
    }


def read_feature_array(
    repository: PreparedDatasetRepository,
    run_id: str,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Read one prepared run as feature matrix and timestamp vector."""
    frame = repository.read_run(run_id)
    ts_ns = frame["ts_ns"].to_numpy(dtype=np.int64)
    features = frame.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=np.float32)
    return features, ts_ns


def training_arrays(
    repository: PreparedDatasetRepository,
    protocol: str,
    columns: list[str],
    *,
    include_validation: bool = True,
) -> tuple[list[str], list[np.ndarray]]:
    """Load normal training arrays from a prepared split."""
    split = protocol_split(repository, protocol)
    run_ids = list(split["train_runs"])
    if include_validation:
        run_ids.extend(split["val_runs"])
    arrays = [read_feature_array(repository, run_id, columns)[0] for run_id in run_ids]
    return run_ids, arrays


def forecast_windows(
    data: np.ndarray,
    *,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create next-step forecast windows and target row indices."""
    row_count = data.shape[0]
    starts = np.arange(0, max(row_count - window, 0), stride, dtype=np.int64)
    target_indices = starts + window
    valid = target_indices < row_count
    starts = starts[valid]
    target_indices = target_indices[valid]
    if len(starts) == 0:
        feature_count = data.shape[1]
        return (
            np.empty((0, window, feature_count), dtype=np.float32),
            np.empty((0, feature_count), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    x = np.stack([data[start : start + window] for start in starts]).astype(np.float32)
    y = data[target_indices].astype(np.float32)
    return x, y, target_indices.astype(np.int64)


def forecast_window_batches(
    data: np.ndarray,
    *,
    window: int,
    stride: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield next-step forecast windows without materializing the full run."""
    count = _forecast_window_count(data.shape[0], window, stride)
    for window_slice in batch_slices(count, batch_size):
        starts = np.arange(window_slice.start, window_slice.stop, dtype=np.int64) * stride
        target_indices = starts + window
        x = np.stack([data[start : start + window] for start in starts]).astype(np.float32)
        y = data[target_indices].astype(np.float32)
        yield x, y, target_indices.astype(np.int64)


def forecast_window_batches_at(
    data: np.ndarray,
    target_indices: np.ndarray,
    *,
    window: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield forecast windows for selected target row indices."""
    selected = _valid_forecast_targets(target_indices, data.shape[0], window)
    for window_slice in batch_slices(len(selected), batch_size):
        batch_targets = selected[window_slice]
        starts = batch_targets - window
        x = np.stack([data[start : start + window] for start in starts]).astype(np.float32)
        y = data[batch_targets].astype(np.float32)
        yield x, y, batch_targets.astype(np.int64)


def window_end_windows(
    data: np.ndarray,
    *,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create reconstruction/contrastive windows and window-end row indices."""
    row_count = data.shape[0]
    starts = np.arange(0, max(row_count - window + 1, 0), stride, dtype=np.int64)
    if len(starts) == 0:
        return (
            np.empty((0, window, data.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    x = np.stack([data[start : start + window] for start in starts]).astype(np.float32)
    end_indices = (starts + window - 1).astype(np.int64)
    return x, end_indices


def window_end_batches(
    data: np.ndarray,
    *,
    window: int,
    stride: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield reconstruction windows without materializing the full run."""
    count = _window_end_count(data.shape[0], window, stride)
    for window_slice in batch_slices(count, batch_size):
        starts = np.arange(window_slice.start, window_slice.stop, dtype=np.int64) * stride
        x = np.stack([data[start : start + window] for start in starts]).astype(np.float32)
        end_indices = starts + window - 1
        yield x, end_indices.astype(np.int64)


def window_end_batches_at(
    data: np.ndarray,
    end_indices: np.ndarray,
    *,
    window: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield reconstruction windows for selected window-end row indices."""
    selected = _valid_window_ends(end_indices, data.shape[0], window)
    for window_slice in batch_slices(len(selected), batch_size):
        batch_ends = selected[window_slice]
        starts = batch_ends - window + 1
        x = np.stack([data[start : start + window] for start in starts]).astype(np.float32)
        yield x, batch_ends.astype(np.int64)


def cap_aligned_arrays(
    arrays: tuple[np.ndarray, ...],
    *,
    max_count: int | None,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Deterministically cap aligned arrays along axis 0."""
    if not arrays:
        return arrays
    count = int(arrays[0].shape[0])
    if max_count is None or count <= max_count:
        return arrays
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(count, max_count, replace=False))
    return tuple(array[indices] for array in arrays)


def score_frame(ts_ns: np.ndarray, row_indices: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Build a Score Contract v1 dataframe from aligned score rows."""
    return pd.DataFrame(
        {
            "ts_ns": ts_ns[row_indices].astype(np.int64),
            "score": scores.astype(np.float64),
        }
    )


def empty_score_frame() -> pd.DataFrame:
    """Return an empty valid Score Contract v1 dataframe."""
    return pd.DataFrame(
        {
            "ts_ns": np.empty(0, dtype=np.int64),
            "score": np.empty(0, dtype=np.float64),
        }
    )


def batch_slices(count: int, batch_size: int) -> Iterator[slice]:
    """Yield contiguous inference/training batch slices."""
    for start in range(0, count, batch_size):
        yield slice(start, min(start + batch_size, count))


def _forecast_window_count(row_count: int, window: int, stride: int) -> int:
    if row_count <= window:
        return 0
    return ((row_count - window - 1) // stride) + 1


def _window_end_count(row_count: int, window: int, stride: int) -> int:
    if row_count < window:
        return 0
    return ((row_count - window) // stride) + 1


def _valid_forecast_targets(
    target_indices: np.ndarray,
    row_count: int,
    window: int,
) -> np.ndarray:
    selected = np.asarray(target_indices, dtype=np.int64)
    selected = np.unique(selected[(selected >= window) & (selected < row_count)])
    return selected.astype(np.int64)


def _valid_window_ends(
    end_indices: np.ndarray,
    row_count: int,
    window: int,
) -> np.ndarray:
    selected = np.asarray(end_indices, dtype=np.int64)
    selected = np.unique(selected[(selected >= window - 1) & (selected < row_count)])
    return selected.astype(np.int64)


def _xpu_available(torch: Any) -> bool:
    if not hasattr(torch, "xpu"):
        return False
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


def _int_parameter(parameters: dict[str, Any], key: str, default: int) -> int:
    return int(parameters.get(key, default))


def _optional_int_parameter(
    parameters: dict[str, Any],
    key: str,
    default: int | None,
) -> int | None:
    value = parameters.get(key, default)
    if value is None:
        return None
    return int(value)


def _float_parameter(parameters: dict[str, Any], key: str, default: float) -> float:
    return float(parameters.get(key, default))


def _bool_parameter(parameters: dict[str, Any], key: str, default: bool) -> bool:
    value = parameters.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)
