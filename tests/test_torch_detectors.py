from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from industrial_tsad_eval.application.benchmark import RunBenchmark
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidateScores
from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkDetectorConfig,
)
from industrial_tsad_eval.domain.errors import OptionalDependencyError
from industrial_tsad_eval.infrastructure.explanation_repository import LocalExplanationRepository
from industrial_tsad_eval.plugins.registry import default_detector_registry
from industrial_tsad_eval.plugins.torch_common import (
    TorchTrainingConfig,
    cap_aligned_arrays,
    forecast_window_batches,
    forecast_window_batches_at,
    forecast_windows,
    require_torch,
    resolve_torch_device,
    window_end_batches,
    window_end_batches_at,
    window_end_windows,
)


def test_torch_missing_error_is_lazy_and_clear(monkeypatch: pytest.MonkeyPatch):
    registry = default_detector_registry()
    detector = registry.get("forecast-lstm").create(
        _run_config({"window": 16, "epochs": 1, "device": "cpu"})
    )

    real_import_module = importlib.import_module

    def blocked_import(name: str, package: str | None = None):
        if name == "torch":
            raise ImportError("blocked")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked_import)

    assert "forecast-lstm" in registry.names()
    with pytest.raises(OptionalDependencyError, match="Torch-backed detectors require"):
        require_torch()
    with pytest.raises(OptionalDependencyError, match="Torch-backed detectors require"):
        detector.train(_NoopRepository(), "naive")


def test_window_helpers_align_timestamps_and_subsample_deterministically():
    data = np.arange(60, dtype=np.float32).reshape(20, 3)

    x_forecast, y_forecast, target_indices = forecast_windows(data, window=4, stride=3)
    x_window, end_indices = window_end_windows(data, window=4, stride=3)
    capped_a, capped_b = cap_aligned_arrays(
        (x_forecast, y_forecast),
        max_count=2,
        seed=123,
    )
    capped_a_again, capped_b_again = cap_aligned_arrays(
        (x_forecast, y_forecast),
        max_count=2,
        seed=123,
    )

    assert target_indices.tolist() == [4, 7, 10, 13, 16, 19]
    assert end_indices.tolist() == [3, 6, 9, 12, 15, 18]
    np.testing.assert_array_equal(capped_a, capped_a_again)
    np.testing.assert_array_equal(capped_b, capped_b_again)


def test_streaming_window_helpers_match_materialized_windows():
    data = np.arange(120, dtype=np.float32).reshape(40, 3)
    full_x, full_y, full_targets = forecast_windows(data, window=5, stride=4)
    stream = list(forecast_window_batches(data, window=5, stride=4, batch_size=3))
    stream_x = np.concatenate([part[0] for part in stream], axis=0)
    stream_y = np.concatenate([part[1] for part in stream], axis=0)
    stream_targets = np.concatenate([part[2] for part in stream], axis=0)

    np.testing.assert_array_equal(stream_x, full_x)
    np.testing.assert_array_equal(stream_y, full_y)
    np.testing.assert_array_equal(stream_targets, full_targets)
    assert all(part[0].shape[0] <= 3 for part in stream)

    selected = np.array([5, 13, 37], dtype=np.int64)
    selected_stream = list(forecast_window_batches_at(data, selected, window=5, batch_size=2))
    selected_targets = np.concatenate([part[2] for part in selected_stream], axis=0)
    np.testing.assert_array_equal(selected_targets, selected)

    full_window_x, full_ends = window_end_windows(data, window=5, stride=4)
    window_stream = list(window_end_batches(data, window=5, stride=4, batch_size=3))
    np.testing.assert_array_equal(
        np.concatenate([part[0] for part in window_stream], axis=0),
        full_window_x,
    )
    np.testing.assert_array_equal(
        np.concatenate([part[1] for part in window_stream], axis=0),
        full_ends,
    )

    selected_ends = np.array([4, 12, 36], dtype=np.int64)
    selected_end_stream = list(window_end_batches_at(data, selected_ends, window=5, batch_size=2))
    np.testing.assert_array_equal(
        np.concatenate([part[1] for part in selected_end_stream], axis=0),
        selected_ends,
    )


def test_invalid_torch_config_fails_early():
    with pytest.raises(ValueError, match="window"):
        TorchTrainingConfig.from_parameters({"window": 1})
    with pytest.raises(ValueError, match="explanation_score_quantile"):
        TorchTrainingConfig.from_parameters({"explanation_score_quantile": 1.0})
    with pytest.raises(ValueError, match="device"):
        resolve_torch_device(object(), "mps")
    with pytest.raises(ValueError, match="divisible"):
        default_detector_registry().get("drcad").create(
            _run_config({"window": 15, "patch_size": 4})
        )


@pytest.mark.parametrize(
    ("detector", "parameters"),
    [
        ("forecast-lstm", {"hidden_size": 8, "num_layers": 1, "dropout": 0.0}),
        ("dra", {"d": 8}),
        ("interfusion", {"latent_dim": 2, "kl_warmup": 1}),
        ("drcad", {"patch_size": 4, "d_model": 16, "n_heads": 4, "n_layers": 1, "mlp_dim": 32}),
    ],
)
def test_torch_detector_scores_validate_and_write_metadata(
    opcua_prepared: Path,
    tmp_path: Path,
    detector: str,
    parameters: dict[str, object],
):
    pytest.importorskip("torch")
    scores = tmp_path / "scores" / detector
    result = ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=opcua_prepared,
        scores=scores,
        detector_name=detector,
        detector_parameters={**_tiny_torch_parameters(), **parameters},
    ).run()

    report = ValidateScores(opcua_prepared, scores).run()
    metadata = json.loads((scores / "model_meta.json").read_text(encoding="utf-8"))

    assert report.ok
    assert result.runs_scored == [
        "opcua/train/normal_001",
        "opcua/val/normal_001",
        "opcua/test/fault_001",
    ]
    assert metadata["detector"] == detector
    assert metadata["resolved_device"] == "cpu"
    assert metadata["feature_columns"]
    assert metadata["train_window_count"] > 0
    if detector in {"dra", "interfusion", "drcad"}:
        explanations = LocalExplanationRepository(scores / "explanations")
        discovered = explanations.discover()
        assert set(discovered) == set(result.runs_scored)
        frame = explanations.read_run_explanations(result.runs_scored[-1])
        assert {"ts_ns", "variable", "importance", "rank", "method"}.issubset(frame.columns)
        assert frame["importance"].ge(0.0).all()
        assert frame["rank"].min() == 1
    else:
        assert not (scores / "explanations" / "manifest.json").exists()


def test_benchmark_can_run_torch_detector(opcua_prepared: Path, tmp_path: Path):
    pytest.importorskip("torch")
    config = BenchmarkConfig(
        name="torch-smoke",
        protocols=["naive"],
        datasets=[BenchmarkDatasetConfig(id="opcua", prepared=str(opcua_prepared))],
        detectors=[
            BenchmarkDetectorConfig(
                id="forecast-lstm-tiny",
                name="forecast-lstm",
                parameters={**_tiny_torch_parameters(), "hidden_size": 8},
            )
        ],
    )

    result = RunBenchmark(
        config=config,
        detector_registry=default_detector_registry(),
        out=tmp_path / "bench",
        run_id="torch-run",
    ).run()

    assert result.ok
    assert result.results[0].status == "completed"


def _tiny_torch_parameters() -> dict[str, object]:
    return {
        "window": 16,
        "train_stride": 8,
        "score_stride": 8,
        "max_train_windows": 24,
        "epochs": 1,
        "batch_size": 8,
        "lr": 0.002,
        "seed": 123,
        "device": "cpu",
        "standardize": True,
        "explanation_top_k": 3,
    }


def _run_config(parameters: dict[str, object]):
    from industrial_tsad_eval.ports.detectors import DetectorRunConfig

    return DetectorRunConfig(parameters=dict(parameters))


class _NoopRepository:
    @property
    def root(self) -> Path:
        return Path(".")

    @property
    def dataset_name(self) -> str:
        return "noop"

    def manifest(self) -> dict[str, object]:
        return {}

    def schema(self) -> dict[str, object]:
        return {"tags": [{"browse_path": "x"}]}

    def splits(self) -> dict[str, object]:
        return {"naive": {"train_runs": ["run"], "val_runs": [], "test_runs": []}}

    def run_ids(self) -> list[str]:
        return ["run"]

    def read_run(self, run_id: str, columns: list[str] | None = None):
        raise AssertionError(f"read_run should not be called for {run_id}")
