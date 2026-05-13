from __future__ import annotations

import json
from pathlib import Path

from industrial_tsad_eval.application.benchmark import RunBenchmark
from industrial_tsad_eval.application.reproduction import DiagnoseThesisReproduction
from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkDetectorConfig,
)
from industrial_tsad_eval.domain.progress import ProgressEvent
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture
from industrial_tsad_eval.infrastructure.progress import LocalProgressSink, read_run_progress
from industrial_tsad_eval.infrastructure.reproduction_config import (
    load_reproduction_config,
    write_default_reproduction_config,
)
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_progress_event_serializes_core_fields():
    event = ProgressEvent(
        run_id="run",
        stage="benchmark",
        item_id="exp",
        status="completed",
        ordinal=1,
        total=2,
        metrics={"event_f1": 1.0},
    )

    payload = event.to_dict()

    assert payload["format_version"] == "progress-event-v1"
    assert event.key == "benchmark:exp"
    assert payload["metrics"] == {"event_f1": 1.0}


def test_local_progress_sink_writes_jsonl_and_snapshot(tmp_path: Path):
    sink = LocalProgressSink(tmp_path, "run")

    sink.emit(ProgressEvent(run_id="run", stage="stage", item_id="a", status="running"))
    sink.emit(ProgressEvent(run_id="run", stage="stage", item_id="a", status="completed"))

    lines = (tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    snapshot = json.loads((tmp_path / "progress_snapshot.json").read_text(encoding="utf-8"))

    assert len(lines) == 2
    assert snapshot["counts"] == {"completed": 1}
    assert read_run_progress(tmp_path)["latest_event"]["status"] == "completed"


def test_benchmark_progress_artifacts_are_ordered(tmp_path: Path):
    prepared = make_opcua_fixture(tmp_path / "examples")
    config = BenchmarkConfig(
        name="progress-smoke",
        protocols=["naive", "all_in_one"],
        datasets=[BenchmarkDatasetConfig(id="opcua", prepared=str(prepared))],
        detectors=[
            BenchmarkDetectorConfig(
                id="forecast-ridge",
                name="forecast-ridge",
                parameters={"window": 24, "stride": 4, "lags": 1},
            )
        ],
    )

    result = RunBenchmark(
        config=config,
        detector_registry=default_detector_registry(),
        out=tmp_path / "runs",
        run_id="progress-run",
    ).run()
    snapshot = read_run_progress(Path(result.run_dir))

    assert result.ok
    assert (Path(result.run_dir) / "progress.jsonl").exists()
    assert snapshot["counts"] == {"completed": 2}
    assert snapshot["items"]["benchmark:opcua__forecast-ridge__naive"]["ordinal"] == 1


def test_thesis_verification_profile_contains_expected_detector_mix(tmp_path: Path):
    config_path = tmp_path / "verification.toml"
    write_default_reproduction_config(config_path, profile="thesis-verification")

    config = load_reproduction_config(config_path)
    detector_names = [detector.name for detector in config.benchmark.detectors]

    assert config.name == "thesis-verification"
    assert detector_names == ["forecast-ridge", "forecast-lstm", "dra", "interfusion", "drcad"]
    assert config.benchmark.protocols == ["naive", "all_in_one", "zero_shot"]
    assert [experiment.experiment_id for experiment in config.benchmark.experiments()] == [
        "TEP__forecast-ridge__naive",
        "TEP__forecast-ridge__all_in_one",
        "TEP__forecast-ridge__zero_shot",
        "SWaT__forecast-ridge__naive",
        "SWaT__forecast-ridge__all_in_one",
        "SWaT__forecast-ridge__zero_shot",
        "SWaT__forecast-lstm-tiny__naive",
        "SWaT__dra-tiny__naive",
        "SWaT__interfusion-tiny__naive",
        "SWaT__drcad-tiny__naive",
        "HAI__forecast-ridge__naive",
        "HAI__forecast-ridge__all_in_one",
        "HAI__forecast-ridge__zero_shot",
        "HAI__forecast-lstm-tiny__naive",
        "HAI-CPPS__forecast-ridge__naive",
        "HAI-CPPS__forecast-ridge__all_in_one",
        "HAI-CPPS__forecast-ridge__zero_shot",
        "HAI-CPPS__forecast-lstm-tiny__naive",
    ]


def test_default_config_written_under_config_points_to_project_paths(tmp_path: Path):
    config_path = tmp_path / "config" / "thesis_smoke.docker.toml"
    write_default_reproduction_config(config_path, profile="thesis-smoke")

    text = config_path.read_text(encoding="utf-8")
    config = load_reproduction_config(config_path)

    assert 'prepared = "../examples/generated/OPCUA_SYNTH"' in text
    assert "../../examples/generated/OPCUA_SYNTH" not in text
    assert (
        Path(config.benchmark.datasets[0].prepared)
        == (tmp_path / "examples" / "generated" / "OPCUA_SYNTH").resolve()
    )
    assert (
        Path(config.assistant.prepared)
        == (tmp_path / "examples" / "generated" / "OPCUA_SYNTH").resolve()
    )


def test_thesis_full_profile_lists_current_draft_detector_plugins(tmp_path: Path):
    config_path = tmp_path / "full.toml"
    write_default_reproduction_config(config_path, profile="thesis-full")

    config = load_reproduction_config(config_path)
    detector_names = [detector.name for detector in config.benchmark.detectors]

    assert detector_names == ["forecast-ridge", "dra", "interfusion", "drcad"]
    assert config.profile_experiment_limit == 0
    assert config.resources.profile_mode == "inline"
    assert config.resources.benchmark_workers == "auto"
    assert config.resources.memory_limit_gb == 16


def test_reproduction_diagnostics_group_native_evidence_failures(tmp_path: Path):
    run = tmp_path / "run"
    (run / "benchmark").mkdir(parents=True)
    (run / "benchmark" / "summary.json").write_text(
        json.dumps(
            {
                "ok": True,
                "experiments": [
                    {"experiment_id": "a", "status": "completed"},
                    {"experiment_id": "b", "status": "completed"},
                ],
            }
        ),
        encoding="utf-8",
    )
    events = [
        {
            "run_id": "run",
            "stage": "evidence",
            "item_id": "a:operational",
            "status": "failed",
            "error": "EvidenceError: Native explanation artifacts contain no rows inside event.",
        }
    ]
    (run / "progress.jsonl").write_text(
        "\n".join(json.dumps(item) for item in events),
        encoding="utf-8",
    )

    report = DiagnoseThesisReproduction(run).run_diagnostics()

    assert report["classification"] == "downstream_operational_native_evidence_coverage"
    assert report["failures_by_error"] == {"native_explanation_missing_event_overlap": 1}
    assert (run / "diagnostics" / "failure_report.md").exists()
