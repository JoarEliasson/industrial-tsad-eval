from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import pytest

from industrial_tsad_eval.application.preflight import PreflightInput, RunPreflight
from industrial_tsad_eval.application.profiling import (
    ProfileScoreEvaluate,
    ProfileScoreEvaluateConfig,
)
from industrial_tsad_eval.domain.errors import PreflightError
from industrial_tsad_eval.domain.system import PreflightCheck, SystemGpu
from industrial_tsad_eval.infrastructure.profiling import (
    StageMonitor,
    percentile,
    summarize_samples,
)
from industrial_tsad_eval.infrastructure.system import (
    classify_gpu_name,
    classify_machine_profile,
    normalize_device_request,
    probe_torch_runtime,
    recommend_backend_for_gpus,
)
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_gpu_classification_and_backend_recommendation():
    gpus = [
        SystemGpu("Intel(R) Arc(TM) A770", "intel", "test"),
        SystemGpu("NVIDIA RTX 4090", "nvidia", "test"),
    ]

    assert classify_gpu_name("NVIDIA RTX 4090") == "nvidia"
    assert classify_gpu_name("Intel Arc Graphics") == "intel"
    assert classify_gpu_name("AMD Radeon") == "amd"
    assert recommend_backend_for_gpus(gpus) == "cuda"


def test_device_request_and_torch_missing_runtime(monkeypatch: pytest.MonkeyPatch):
    real_import_module = importlib.import_module

    def blocked_import(name: str, package: str | None = None):
        if name == "torch":
            raise ImportError("blocked")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked_import)

    assert normalize_device_request("AUTO") == "auto"
    with pytest.raises(ValueError):
        normalize_device_request("mps")
    status = probe_torch_runtime("auto")
    assert not status.ready
    assert not status.torch_available
    assert status.resolved_device == "cpu"


def test_machine_profile_classification():
    assert classify_machine_profile(8.0) == "small"
    assert classify_machine_profile(32.0) == "medium"
    assert classify_machine_profile(128.0) == "large"
    assert classify_machine_profile(None) == "medium"


def test_stage_monitor_and_profile_summary():
    with StageMonitor("sample", sample_interval_ms=1) as monitor:
        time.sleep(0.002)

    assert monitor.sample is not None
    assert monitor.sample.duration_ms > 0
    summary = summarize_samples([monitor.sample])
    assert summary["stage_count"] == 1
    assert summary["stages"]["sample"]["duration_ms"]["max"] > 0
    assert percentile([1.0, 3.0], 0.5) == 2.0


def test_preflight_pass_warn_and_fail_statuses(opcua_prepared: Path, tmp_path: Path):
    registry = default_detector_registry()

    pass_report = RunPreflight(
        detector_registry=registry,
        config=PreflightInput(
            prepared=opcua_prepared,
            detector_name="forecast-ridge",
            out=tmp_path / "pass",
        ),
    ).run()
    warn_report = RunPreflight(detector_registry=registry, config=PreflightInput()).run()
    fail_report = RunPreflight(
        detector_registry=registry,
        config=PreflightInput(prepared=tmp_path / "missing", detector_name="missing"),
    ).run()

    assert pass_report.status in {"pass", "warn"}
    assert warn_report.status == "warn"
    assert fail_report.status == "fail"
    assert (tmp_path / "pass" / "preflight.json").exists()


def test_strict_preflight_raises_on_failure(tmp_path: Path):
    with pytest.raises(PreflightError):
        RunPreflight(
            detector_registry=default_detector_registry(),
            config=PreflightInput(prepared=tmp_path / "missing", strict=True),
        ).run()


def test_preflight_torch_detector_reports_missing_torch(
    monkeypatch: pytest.MonkeyPatch,
    opcua_prepared: Path,
):
    real_import_module = importlib.import_module

    def blocked_import(name: str, package: str | None = None):
        if name == "torch":
            raise ImportError("blocked")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", blocked_import)

    report = RunPreflight(
        detector_registry=default_detector_registry(),
        config=PreflightInput(
            prepared=opcua_prepared,
            detector_name="forecast-lstm",
            detector_parameters={"window": 16},
        ),
    ).run()

    assert report.status == "fail"
    assert any(check.name == "torch-runtime" for check in report.checks)


def test_profile_score_evaluate_writes_artifact_layout(opcua_prepared: Path, tmp_path: Path):
    result = ProfileScoreEvaluate(
        detector_registry=default_detector_registry(),
        config=ProfileScoreEvaluateConfig(
            prepared=opcua_prepared,
            detector_name="forecast-ridge",
            out=tmp_path / "profiles",
            detector_parameters={"window": 24, "stride": 4, "lags": 1},
            profile_id="ridge-profile",
        ),
    ).run()
    profile_dir = Path(result.profile_dir)

    assert result.ok
    assert (profile_dir / "machine_env.json").exists()
    assert (profile_dir / "preflight.json").exists()
    assert (profile_dir / "stages.csv").exists()
    assert (profile_dir / "summary.json").exists()
    assert (profile_dir / "budget_check.md").exists()
    assert (profile_dir / "artifacts" / "scores" / "manifest.json").exists()
    assert (profile_dir / "artifacts" / "eval" / "metrics.json").exists()


def test_profile_score_evaluate_torch_detector_when_available(
    opcua_prepared: Path,
    tmp_path: Path,
):
    pytest.importorskip("torch")
    result = ProfileScoreEvaluate(
        detector_registry=default_detector_registry(),
        config=ProfileScoreEvaluateConfig(
            prepared=opcua_prepared,
            detector_name="forecast-lstm",
            out=tmp_path / "profiles",
            detector_parameters={
                "window": 16,
                "train_stride": 8,
                "score_stride": 8,
                "max_train_windows": 24,
                "epochs": 1,
                "batch_size": 8,
                "device": "cpu",
                "hidden_size": 8,
            },
            device="cpu",
            profile_id="lstm-profile",
        ),
    ).run()

    summary = json.loads((Path(result.profile_dir) / "summary.json").read_text(encoding="utf-8"))
    assert summary["detector"] == "forecast-lstm"


def test_preflight_report_ok_property():
    machine_report = RunPreflight(
        detector_registry=default_detector_registry(),
        config=PreflightInput(),
    ).run()
    check = PreflightCheck("x", "pass", "ok")

    assert check.to_dict()["status"] == "pass"
    assert machine_report.ok
