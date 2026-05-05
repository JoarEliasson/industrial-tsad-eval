from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from industrial_tsad_eval.application.benchmark import RunBenchmark, summary_row
from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkDetectorConfig,
    BenchmarkExperimentResult,
)
from industrial_tsad_eval.domain.errors import BenchmarkConfigError
from industrial_tsad_eval.infrastructure.benchmark_config import load_benchmark_config
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_benchmark_toml_parsing_resolves_relative_prepared_paths(tmp_path):
    config_path = tmp_path / "benchmark.toml"
    config_path.write_text(
        """
[benchmark]
name = "parse-smoke"
protocols = ["naive"]

[benchmark.evaluation]
threshold_quantile = 0.99

[[datasets]]
id = "opcua"
prepared = "prepared/OPCUA_SYNTH"

[[detectors]]
id = "forecast-ridge-default"
name = "forecast-ridge"
parameters = { window = 32, stride = 4, lags = 1 }
""",
        encoding="utf-8",
    )

    config = load_benchmark_config(config_path)

    assert config.name == "parse-smoke"
    assert config.evaluation.threshold_quantile == 0.99
    assert config.datasets[0].prepared == str((tmp_path / "prepared" / "OPCUA_SYNTH").resolve())
    assert config.detectors[0].parameters["window"] == 32


def test_benchmark_config_rejects_unsafe_ids():
    with pytest.raises(BenchmarkConfigError, match="may only contain"):
        BenchmarkConfig.from_mapping(
            {
                "benchmark": {"name": "bad", "protocols": ["naive"]},
                "datasets": [{"id": "../bad", "prepared": "prepared/OPCUA_SYNTH"}],
                "detectors": [{"id": "forecast", "name": "forecast-ridge"}],
            }
        )


def test_benchmark_matrix_expansion_has_stable_ids():
    config = _benchmark_config(
        "matrix",
        Path("prepared/OPCUA_SYNTH"),
        protocols=["naive", "all_in_one"],
    )

    experiment_ids = [experiment.experiment_id for experiment in config.experiments()]

    assert experiment_ids == [
        "opcua__forecast-ridge-default__naive",
        "opcua__forecast-ridge-default__all_in_one",
    ]


def test_summary_row_extracts_public_metric_columns():
    row = summary_row(
        BenchmarkExperimentResult(
            experiment_id="exp",
            dataset="dataset",
            detector="detector",
            protocol="naive",
            status="completed",
            scores_dir="scores",
            eval_dir="eval",
            threshold=1.25,
            metrics={
                "event": {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "n_gt": 2.0},
                "delay": {"mean": 10.0},
                "far": {"false_events_per_hour": 0.25},
            },
        )
    )

    assert row["event_precision"] == 0.5
    assert row["event_recall"] == 1.0
    assert row["delay_mean_ns"] == 10.0
    assert row["far_false_events_per_hour"] == 0.25


def test_run_benchmark_writes_artifacts_for_opcua_fixture(tmp_path):
    prepared = make_opcua_fixture(tmp_path / "examples")
    config = _benchmark_config("opcua-smoke", prepared)

    result = RunBenchmark(
        config=config,
        detector_registry=default_detector_registry(),
        out=tmp_path / "runs",
        run_id="fixed-run",
    ).run()
    run_dir = Path(result.run_dir)

    assert result.ok
    assert (run_dir / "config" / "benchmark.toml").exists()
    assert (run_dir / "resolved_config.json").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.csv").exists()
    status_path = run_dir / "experiments" / "opcua__forecast-ridge-default__naive" / "status.json"
    assert status_path.exists()
    assert (
        run_dir / "experiments" / "opcua__forecast-ridge-default__naive" / "eval" / "metrics.json"
    ).exists()

    rows = list(csv.DictReader((run_dir / "summary.csv").read_text(encoding="utf-8").splitlines()))
    assert rows[0]["experiment_id"] == "opcua__forecast-ridge-default__naive"
    assert rows[0]["status"] == "completed"


def test_run_benchmark_supports_multiple_protocols_with_existing_splits(tmp_path):
    prepared = make_opcua_fixture(tmp_path / "examples")
    config = _benchmark_config("opcua-multi", prepared, protocols=["naive", "all_in_one"])

    result = RunBenchmark(
        config=config,
        detector_registry=default_detector_registry(),
        out=tmp_path / "runs",
        run_id="multi-run",
    ).run()

    assert result.ok
    assert [item.experiment_id for item in result.results] == [
        "opcua__forecast-ridge-default__naive",
        "opcua__forecast-ridge-default__all_in_one",
    ]


def test_run_benchmark_continues_after_unknown_detector_failure(tmp_path):
    prepared = make_opcua_fixture(tmp_path / "examples")
    config = BenchmarkConfig(
        name="mixed",
        protocols=["naive"],
        datasets=[BenchmarkDatasetConfig(id="opcua", prepared=str(prepared))],
        detectors=[
            BenchmarkDetectorConfig(
                id="forecast-ridge-default",
                name="forecast-ridge",
                parameters={"window": 32, "stride": 4, "lags": 1},
            ),
            BenchmarkDetectorConfig(id="missing", name="missing-detector"),
        ],
    )

    result = RunBenchmark(
        config=config,
        detector_registry=default_detector_registry(),
        out=tmp_path / "runs",
        run_id="mixed-run",
    ).run()

    statuses = {item.experiment_id: item.status for item in result.results}
    assert result.ok is False
    assert statuses["opcua__forecast-ridge-default__naive"] == "completed"
    assert statuses["opcua__missing__naive"] == "failed"
    summary = json.loads((Path(result.run_dir) / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] is False


def test_run_benchmark_marks_invalid_prepared_dataset_failed(tmp_path):
    config = _benchmark_config("invalid", tmp_path / "missing")

    result = RunBenchmark(
        config=config,
        detector_registry=default_detector_registry(),
        out=tmp_path / "runs",
        run_id="invalid-run",
    ).run()

    assert result.ok is False
    assert result.results[0].status == "failed"
    assert "Prepared dataset validation failed" in str(result.results[0].error)


def _benchmark_config(
    name: str,
    prepared: Path,
    *,
    protocols: list[str] | None = None,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        name=name,
        protocols=protocols or ["naive"],
        datasets=[BenchmarkDatasetConfig(id="opcua", prepared=str(prepared))],
        detectors=[
            BenchmarkDetectorConfig(
                id="forecast-ridge-default",
                name="forecast-ridge",
                parameters={"window": 32, "stride": 4, "lags": 1},
            )
        ],
    )
