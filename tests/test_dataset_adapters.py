from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from industrial_tsad_eval.application.preparation import PrepareDataset
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.domain.errors import PreparationError
from industrial_tsad_eval.plugins.registry import (
    default_dataset_adapter_registry,
    default_detector_registry,
)


def test_prepare_dataset_refuses_existing_output_without_overwrite(tmp_path):
    raw = _make_swat_raw(tmp_path / "raw")
    out = tmp_path / "prepared"
    registry = default_dataset_adapter_registry()

    PrepareDataset(
        adapter_registry=registry,
        dataset="swat",
        raw=raw,
        out=out,
        config=DatasetAdapterConfig(extra={"remove_startup": False}),
    ).run()

    with pytest.raises(PreparationError, match="already exists"):
        PrepareDataset(
            adapter_registry=registry,
            dataset="swat",
            raw=raw,
            out=out,
            config=DatasetAdapterConfig(extra={"remove_startup": False}),
        ).run()


def test_tep_adapter_prepares_synthetic_csv_and_scores(tmp_path):
    raw = _make_tep_raw(tmp_path / "raw")
    result = _prepare("tep", raw, tmp_path / "prepared")
    prepared = Path(result.prepared_path)

    assert result.dataset_name == "TEP"
    assert result.run_count == 2
    assert result.event_count == 1
    _assert_valid_and_score(prepared, tmp_path)

    schema = json.loads((prepared / "meta" / "schema.json").read_text(encoding="utf-8"))
    browse_paths = {tag["browse_path"] for tag in schema["tags"]}
    assert "Plant/TEP/XMEAS_01" in browse_paths
    assert "Plant/TEP/XMV_11" in browse_paths


def test_swat_adapter_prepares_synthetic_csv_and_scores(tmp_path):
    raw = _make_swat_raw(tmp_path / "raw")
    result = _prepare(
        "swat",
        raw,
        tmp_path / "prepared",
        DatasetAdapterConfig(extra={"remove_startup": False}),
    )
    prepared = Path(result.prepared_path)

    assert result.dataset_name == "SWaT"
    assert result.run_count == 2
    assert result.event_count == 1
    _assert_valid_and_score(prepared, tmp_path)

    events = _read_events(prepared)
    assert events[0]["event_type"] == "attack"
    assert events[0]["run_id"] == "swat/test/attack_run_001"


def test_hai_adapter_prepares_synthetic_csv_and_scores(tmp_path):
    raw = _make_hai_raw(tmp_path / "raw")
    result = _prepare("hai", raw, tmp_path / "prepared")
    prepared = Path(result.prepared_path)

    assert result.dataset_name == "HAI"
    assert result.run_count == 2
    assert result.event_count == 1
    _assert_valid_and_score(prepared, tmp_path)

    splits = json.loads((prepared / "meta" / "splits.json").read_text(encoding="utf-8"))
    assert splits["naive"]["train_runs"] == ["train1"]
    assert splits["naive"]["test_runs"] == ["test1"]


def test_hai_cpps_adapter_prepares_synthetic_scenarios_and_scores(tmp_path):
    raw = _make_hai_cpps_raw(tmp_path / "raw")
    result = _prepare(
        "hai-cpps",
        raw,
        tmp_path / "prepared",
        DatasetAdapterConfig(extra={"mode": "continuous", "injection_step": 20}),
    )
    prepared = Path(result.prepared_path)

    assert result.dataset_name == "HAI-CPPS"
    assert result.run_count == 2
    assert result.event_count == 1
    _assert_valid_and_score(prepared, tmp_path)

    event = _read_events(prepared)[0]
    assert event["metadata"]["effective_injection_step"] == 20
    assert event["metadata"]["target_module"] == "mixer"


def test_hai_cpps_adapter_clamps_out_of_range_onset(tmp_path):
    raw = tmp_path / "raw"
    run = raw / "ds1_mixer0_anom_test"
    run.mkdir(parents=True)
    (run / "sim_setup.json").write_text(json.dumps({"period_ms": 100}), encoding="utf-8")
    pd.DataFrame({"step": [0, 1, 2], "mixer0_temp": [1.0, 2.0, 3.0]}).to_csv(
        run / "anom_continuous.csv",
        index=False,
    )

    result = _prepare(
        "hai-cpps",
        raw,
        tmp_path / "prepared",
        DatasetAdapterConfig(extra={"mode": "continuous", "injection_step": 999}),
    )

    event = _read_events(Path(result.prepared_path))[0]
    assert event["metadata"]["injection_step_clamped"] is True
    assert event["metadata"]["effective_injection_step"] == 2


def _prepare(
    dataset: str,
    raw: Path,
    out: Path,
    config: DatasetAdapterConfig | None = None,
) -> DatasetAdapterResult:
    return PrepareDataset(
        adapter_registry=default_dataset_adapter_registry(),
        dataset=dataset,
        raw=raw,
        out=out,
        config=config,
    ).run()


def _assert_valid_and_score(prepared: Path, tmp_path: Path) -> None:
    report = ValidatePreparedDataset(prepared).run()
    assert report.ok, report.errors

    scores = tmp_path / f"scores_{prepared.name.lower().replace('-', '_')}"
    ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=prepared,
        scores=scores,
        detector_name="forecast-ridge",
        detector_parameters={"window": 4, "stride": 2, "lags": 1},
    ).run()
    score_report = ValidateScores(prepared, scores).run()
    assert score_report.ok, score_report.errors


def _read_events(prepared: Path) -> list[dict[str, object]]:
    events_path = prepared / "events" / "events.jsonl"
    return [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]


def _make_tep_raw(root: Path) -> Path:
    root.mkdir(parents=True)
    _tep_frame(fault=0, rows=50).to_csv(root / "TEP_FaultFree_Training.csv", index=False)
    _tep_frame(fault=1, rows=170).to_csv(root / "TEP_Faulty_Testing_01.csv", index=False)
    return root


def _tep_frame(*, fault: int, rows: int) -> pd.DataFrame:
    samples = np.arange(1, rows + 1)
    payload: dict[str, object] = {
        "faultNumber": np.full(rows, fault, dtype=np.int64),
        "simulationRun": np.ones(rows, dtype=np.int64),
        "sample": samples,
    }
    for index in range(1, 42):
        payload[f"xmeas_{index}"] = np.sin(samples / (index + 2)) + fault
    for index in range(1, 12):
        payload[f"xmv_{index}"] = np.cos(samples / (index + 3)) + fault
    return pd.DataFrame(payload)


def _make_swat_raw(root: Path) -> Path:
    root.mkdir(parents=True)
    timestamps = pd.date_range("2020-01-01", periods=24, freq="1s")
    pd.DataFrame(
        {
            "Timestamp": timestamps,
            "FIT101": np.linspace(1.0, 2.0, len(timestamps)),
            "MV101": np.zeros(len(timestamps), dtype=np.int64),
            "P101": np.ones(len(timestamps), dtype=np.int64),
            "Normal/Attack": ["Normal"] * len(timestamps),
        }
    ).to_csv(root / "SWaT_Dataset_Normal_v1.csv", index=False)

    labels = ["Normal"] * len(timestamps)
    labels[10:14] = ["Attack"] * 4
    pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-02", periods=24, freq="1s"),
            "FIT101": np.linspace(1.5, 2.5, len(timestamps)),
            "MV101": np.ones(len(timestamps), dtype=np.int64),
            "P101": np.ones(len(timestamps), dtype=np.int64),
            "Normal/Attack": labels,
        }
    ).to_csv(root / "SWaT_Dataset_Attack_v0.csv", index=False)
    return root


def _make_hai_raw(root: Path) -> Path:
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=24, freq="1s"),
            "P1_B2004": np.linspace(1.0, 3.0, 24),
            "P2_FLOW": np.linspace(0.0, 1.0, 24),
            "attack": [0] * 24,
        }
    ).to_csv(root / "train1.csv", index=False)

    attack = [0] * 24
    attack[8:12] = [1] * 4
    pd.DataFrame(
        {
            "time": pd.date_range("2020-01-02", periods=24, freq="1s"),
            "P1_B2004": np.linspace(2.0, 4.0, 24),
            "P2_FLOW": np.linspace(1.0, 2.0, 24),
            "attack": attack,
        }
    ).to_csv(root / "test1.csv", index=False)
    return root


def _make_hai_cpps_raw(root: Path) -> Path:
    normal = root / "ds1"
    normal.mkdir(parents=True)
    (normal / "sim_setup.json").write_text(json.dumps({"period_ms": 100}), encoding="utf-8")
    pd.DataFrame(
        {
            "step": list(range(32)),
            "mixer0_temp": np.linspace(10.0, 20.0, 32),
            "pump0_flow": np.linspace(1.0, 2.0, 32),
        }
    ).to_csv(normal / "ds1_continuous.csv", index=False)

    anomalous = root / "ds1_mixer0_anom_pump50"
    anomalous.mkdir(parents=True)
    (anomalous / "sim_setup.json").write_text(
        json.dumps({"period_ms": 100, "has_reset": True}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "step": list(range(40)),
            "mixer0_temp": np.linspace(12.0, 22.0, 40),
            "pump0_flow": np.linspace(1.5, 2.5, 40),
        }
    ).to_csv(anomalous / "ds1_mixer0_anom_pump50_continuous.csv", index=False)
    return root
