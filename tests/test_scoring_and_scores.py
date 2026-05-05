from __future__ import annotations

from pathlib import Path

from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidateScores
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_forecast_ridge_scores_and_validates_synthetic_fixture(
    opcua_prepared: Path,
    tmp_path: Path,
):
    scores = tmp_path / "scores"

    result = ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=opcua_prepared,
        scores=scores,
        detector_name="forecast-ridge",
        detector_parameters={"window": 24, "stride": 4, "lags": 1},
    ).run()

    assert result.runs_scored == [
        "opcua/train/normal_001",
        "opcua/val/normal_001",
        "opcua/test/fault_001",
    ]
    report = ValidateScores(opcua_prepared, scores).run()
    assert report.ok
