from __future__ import annotations

from pathlib import Path

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
