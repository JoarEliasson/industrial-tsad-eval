from __future__ import annotations

from pathlib import Path

import pandas as pd

from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidateScores
from industrial_tsad_eval.infrastructure.score_repository import LocalScoreRepository
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
    assert result.telemetry["combined_scores_run_count"] == 3
    assert (scores / "combined_scores.parquet").exists()
    report = ValidateScores(opcua_prepared, scores).run()
    assert report.ok


def test_score_repository_combined_sidecar_round_trips_and_is_not_a_run(tmp_path: Path):
    repository = LocalScoreRepository(tmp_path / "scores")
    repository.write_run_scores(
        "plant/run_001",
        pd.DataFrame({"ts_ns": [1, 2], "score": [0.1, 0.2]}),
    )
    telemetry = repository.write_combined_scores(
        {"plant/run_001": pd.DataFrame({"ts_ns": [1, 2], "score": [0.1, 0.2]})}
    )

    assert telemetry["combined_scores_rows"] == 2
    combined = repository.read_combined_scores()
    assert combined["run_id"].tolist() == ["plant/run_001", "plant/run_001"]
    assert repository.discover() == {
        "plant/run_001": tmp_path / "scores" / "plant__run_001.parquet"
    }


def test_score_validation_uses_combined_sidecar_when_present(
    monkeypatch,
    opcua_prepared: Path,
    tmp_path: Path,
):
    scores = tmp_path / "scores_sidecar_validate"
    ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=opcua_prepared,
        scores=scores,
        detector_name="forecast-ridge",
        detector_parameters={"window": 24, "stride": 4, "lags": 1},
    ).run()

    def fail_per_run_read(*_args, **_kwargs):
        raise AssertionError("per-run score validation reads should not be used with sidecar")

    monkeypatch.setattr(LocalScoreRepository, "read_run_scores", fail_per_run_read)
    report = ValidateScores(opcua_prepared, scores).run()

    assert report.ok
