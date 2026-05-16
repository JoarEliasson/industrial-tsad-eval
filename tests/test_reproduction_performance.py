from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_tsad_eval.application.benchmark import RunBenchmark
from industrial_tsad_eval.application.reproduction import (
    AssembleReproductionSlices,
    StopThesisReproduction,
    filter_reproduction_config,
)
from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkDetectorConfig,
)
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture
from industrial_tsad_eval.infrastructure.progress import read_run_progress
from industrial_tsad_eval.infrastructure.reproduction_config import (
    load_reproduction_config,
    write_default_reproduction_config,
)
from industrial_tsad_eval.plugins.registry import DetectorRegistry
from industrial_tsad_eval.ports.detectors import DetectorRunConfig


class _SleepDetector:
    def __init__(self, delay_s: float):
        self.delay_s = delay_s

    def train(self, repository: Any, protocol: str) -> None:
        time.sleep(self.delay_s)

    def score_run(self, repository: Any, run_id: str) -> pd.DataFrame:
        frame = repository.read_run(run_id, columns=["ts_ns"])
        return pd.DataFrame({"ts_ns": frame["ts_ns"], "score": 0.0})

    def metadata(self) -> dict[str, Any]:
        return {"name": "sleep-detector", "delay_s": self.delay_s}


class _SleepPlugin:
    def __init__(self, name: str, *, requires_torch: bool, delay_s: float):
        self._name = name
        self._requires_torch = requires_torch
        self._delay_s = delay_s

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires_torch(self) -> bool:
        return self._requires_torch

    def create(self, config: DetectorRunConfig) -> _SleepDetector:
        return _SleepDetector(self._delay_s)


def test_benchmark_scheduler_keeps_cpu_queue_from_waiting_on_gpu_slots(tmp_path: Path):
    prepared = make_opcua_fixture(tmp_path / "examples")
    registry = DetectorRegistry()
    registry.register(_SleepPlugin("gpu-a", requires_torch=True, delay_s=0.2))
    registry.register(_SleepPlugin("gpu-b", requires_torch=True, delay_s=0.2))
    registry.register(_SleepPlugin("cpu-fast", requires_torch=False, delay_s=0.0))
    config = BenchmarkConfig(
        name="queue",
        protocols=["naive"],
        datasets=[BenchmarkDatasetConfig(id="opcua", prepared=str(prepared))],
        detectors=[
            BenchmarkDetectorConfig(id="gpu-a", name="gpu-a"),
            BenchmarkDetectorConfig(id="gpu-b", name="gpu-b"),
            BenchmarkDetectorConfig(id="cpu-fast", name="cpu-fast"),
        ],
    )

    result = RunBenchmark(
        config=config,
        detector_registry=registry,
        out=tmp_path / "runs",
        run_id="queue-run",
        worker_count=2,
        gpu_slots=1,
    ).run()

    assert result.ok
    progress_lines = (
        (Path(result.run_dir) / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    )
    events = [json.loads(line) for line in progress_lines]
    completed = [
        event
        for event in events
        if event.get("stage") == "benchmark" and event.get("status") == "completed"
    ]
    assert completed[0]["item_id"] == "opcua__cpu-fast__naive"
    snapshot = read_run_progress(Path(result.run_dir))
    queue_metrics = snapshot["items"]["benchmark_queue:state"]["metrics"]
    assert queue_metrics["completed"] == 3
    assert queue_metrics["failed"] == 0


def test_slice_filter_produces_single_detector_dataset_protocol(tmp_path: Path):
    config_path = tmp_path / "verification.toml"
    write_default_reproduction_config(config_path, profile="thesis-verification")
    config = load_reproduction_config(config_path)

    sliced = filter_reproduction_config(
        config,
        datasets=["SWaT"],
        detectors=["dra-tiny"],
        protocols=["naive"],
        stages=["benchmark"],
    )

    assert [experiment.experiment_id for experiment in sliced.benchmark.experiments()] == [
        "SWaT__dra-tiny__naive"
    ]
    assert sliced.run_evidence is False
    assert sliced.run_xai is False
    assert sliced.run_assistant is False


def test_assemble_reproduction_slices_writes_provenance(tmp_path: Path):
    config_path = tmp_path / "smoke.toml"
    write_default_reproduction_config(config_path, profile="thesis-smoke")
    config = load_reproduction_config(config_path)
    run_a = _write_slice_run(tmp_path / "slice-a", config.to_dict(), "a")
    run_b = _write_slice_run(tmp_path / "slice-b", config.to_dict(), "b")

    payload = AssembleReproductionSlices(
        runs=[run_a, run_b],
        out=tmp_path / "assembled",
        run_id="assembled-run",
    ).run()

    run_dir = Path(payload["run_dir"])
    assert payload["assembled"] is True
    assert (run_dir / "assembly_manifest.json").exists()
    detection_summary = run_dir / "summaries" / "detection_summary.csv"
    assert "experiment_id" in detection_summary.read_text(encoding="utf-8")


def test_stop_reproduction_writes_marker_and_safe_command(tmp_path: Path):
    payload = StopThesisReproduction(
        run=tmp_path / "run",
        container="itse-thesis-full",
    ).run_stop()

    assert payload["commands"] == ["docker stop itse-thesis-full"]
    assert (tmp_path / "run" / "run_control" / "cancel_requested.json").exists()


def _write_slice_run(root: Path, resolved: dict[str, Any], suffix: str) -> Path:
    (root / "summaries").mkdir(parents=True)
    (root / "resolved_config.json").write_text(json.dumps(resolved), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (root / "summaries" / "detection_summary.csv").write_text(
        f"experiment_id,status\n{suffix},completed\n",
        encoding="utf-8",
    )
    (root / "summaries" / "xai_summary.csv").write_text(
        f"experiment_id,status\n{suffix},completed\n",
        encoding="utf-8",
    )
    (root / "summaries" / "assistant_summary.csv").write_text(
        f"experiment_id,status\n{suffix},completed\n",
        encoding="utf-8",
    )
    return root
