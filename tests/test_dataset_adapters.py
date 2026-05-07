from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from industrial_tsad_eval.application.preparation import PrepareDataset
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.plugins.registry import (
    default_dataset_adapter_registry,
    default_detector_registry,
)
from tests.conftest import write_hai_cpps_raw, write_hai_raw, write_swat_raw, write_tep_raw


@pytest.mark.parametrize(
    ("adapter", "writer", "dataset_name"),
    [
        ("tep", write_tep_raw, "TEP"),
        ("swat", write_swat_raw, "SWaT"),
        ("hai", write_hai_raw, "HAI"),
        ("hai-cpps", write_hai_cpps_raw, "HAI_CPPS"),
    ],
)
def test_dataset_adapters_prepare_valid_prepared_datasets(
    tmp_path: Path,
    adapter: str,
    writer,
    dataset_name: str,
):
    raw = writer(tmp_path / "raw")

    result = PrepareDataset(
        adapter_registry=default_dataset_adapter_registry(),
        dataset=adapter,
        raw=raw,
        out=tmp_path / "prepared",
    ).run()

    assert result.dataset_name == dataset_name
    assert result.run_count >= 2
    assert result.event_count >= 1
    assert ValidatePreparedDataset(result.prepared_path).run().ok


def test_prepared_adapter_output_can_be_scored_with_forecast_ridge(tmp_path: Path):
    raw = write_swat_raw(tmp_path / "raw")
    result = PrepareDataset(
        adapter_registry=default_dataset_adapter_registry(),
        dataset="swat",
        raw=raw,
        out=tmp_path / "prepared",
    ).run()

    scoring = ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=result.prepared_path,
        scores=tmp_path / "scores",
        detector_name="forecast-ridge",
        detector_parameters={"window": 24, "stride": 4, "lags": 1},
    ).run()

    assert len(scoring.runs_scored) == result.run_count


def test_swat_real_slash_timestamp_format_prepares(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    normal = pd.DataFrame(
        {
            "Timestamp": [
                "28/12/2015 10:00:01 AM",
                "28/12/2015 10:00:00 AM",
                "28/12/2015 10:00:00 AM",
            ],
            "FIT101": [1.1, 1.0, 1.05],
            "Normal/Attack": ["Normal", "Normal", "Normal"],
        }
    )
    attack = pd.DataFrame(
        {
            "Timestamp": [
                "28/12/2015 10:00:02 AM",
                "28/12/2015 10:00:03 AM",
                "28/12/2015 10:00:04 AM",
            ],
            "FIT101": [1.0, 4.0, 4.5],
            "Normal/Attack": ["Normal", "Attack", "Attack"],
        }
    )
    normal.to_csv(raw / "SWaT_Dataset_Normal.csv", index=False)
    attack.to_csv(raw / "SWaT_Dataset_Attack.csv", index=False)

    result = PrepareDataset(
        adapter_registry=default_dataset_adapter_registry(),
        dataset="swat",
        raw=raw,
        out=tmp_path / "prepared",
    ).run()

    report = ValidatePreparedDataset(result.prepared_path).run()
    assert report.ok
    assert not any("duplicate timestamps" in warning for warning in report.warnings)
    assert result.event_count == 1


def test_hai_label_files_are_paired_not_written_as_runs(tmp_path: Path):
    raw = tmp_path / "raw" / "hai-23.05"
    raw.mkdir(parents=True)
    timestamps = pd.date_range("2026-01-01", periods=8, freq="s")
    train = pd.DataFrame({"timestamp": timestamps, "P1_FIT101": np.arange(8)})
    test = pd.DataFrame(
        {
            "timestamp": timestamps,
            "P1_FIT101": np.arange(8) + 10,
            "P2_MV201": np.arange(8) + 20,
        }
    )
    labels = pd.DataFrame({"timestamp": timestamps, "label": [0, 0, 1, 1, 0, 0, 0, 0]})
    train.to_csv(raw / "hai-train1.csv", index=False)
    test.to_csv(raw / "hai-test1.csv", index=False)
    labels.to_csv(raw / "label-test1.csv", index=False)

    result = PrepareDataset(
        adapter_registry=default_dataset_adapter_registry(),
        dataset="hai",
        raw=raw.parent,
        out=tmp_path / "prepared",
    ).run()

    assert result.run_count == 2
    assert result.event_count == 1
    report = ValidatePreparedDataset(result.prepared_path).run()
    assert report.ok
    assert not any("schema tags missing from run" in warning for warning in report.warnings)


def test_hai_cpps_aligns_scenario_feature_schema(tmp_path: Path):
    raw = tmp_path / "raw"
    normal = raw / "normal_scenario"
    attack = raw / "fault_scenario"
    normal.mkdir(parents=True)
    attack.mkdir(parents=True)
    pd.DataFrame(
        {
            "Timestamp": pd.date_range("2026-01-01", periods=8, freq="s"),
            "SensorA": np.arange(8),
        }
    ).to_csv(normal / "continuous.csv", index=False)
    pd.DataFrame(
        {
            "Timestamp": pd.date_range("2026-01-01", periods=8, freq="s"),
            "SensorB": np.arange(8),
        }
    ).to_csv(attack / "continuous.csv", index=False)
    (attack / "sim_setup.json").write_text('{"attack_start": 2, "attack_duration": 2}')

    result = PrepareDataset(
        adapter_registry=default_dataset_adapter_registry(),
        dataset="hai-cpps",
        raw=raw,
        out=tmp_path / "prepared",
    ).run()
    report = ValidatePreparedDataset(result.prepared_path).run()

    assert report.ok
    assert not any("schema tags missing from run" in warning for warning in report.warnings)


def test_tep_rdata_support_when_pyreadr_available(tmp_path: Path):
    pyreadr = pytest.importorskip("pyreadr")
    raw = tmp_path / "raw"
    raw.mkdir()
    frame = pd.DataFrame(
        {
            "simulationRun": [1, 1, 1, 1],
            "sample": [1, 161, 162, 163],
            "faultNumber": [1, 1, 1, 1],
            "xmeas_1": [1.0, 2.0, 3.0, 4.0],
            "xmv_1": [2.0, 2.1, 2.2, 2.3],
        }
    )
    pyreadr.write_rdata(str(raw / "tep_test.RData"), frame, df_name="fault1")

    result = PrepareDataset(
        adapter_registry=default_dataset_adapter_registry(),
        dataset="tep",
        raw=raw,
        out=tmp_path / "prepared",
    ).run()

    assert result.run_count == 1
    assert result.event_count == 1
    assert ValidatePreparedDataset(result.prepared_path).run().ok
