from __future__ import annotations

from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidateScores
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture
from industrial_tsad_eval.infrastructure.score_repository import LocalScoreRepository
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_forecast_ridge_scores_fixture_and_writes_score_contract(tmp_path):
    prepared = make_opcua_fixture(tmp_path / "examples")
    scores = tmp_path / "scores"

    result = ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=prepared,
        scores=scores,
        detector_name="forecast-ridge",
        detector_parameters={"window": 32, "stride": 4, "lags": 2},
    ).run()

    assert result.runs_scored == [
        "opcua/train/normal_001",
        "opcua/val/normal_001",
        "opcua/test/fault_001",
    ]
    assert (scores / "manifest.json").exists()
    assert (scores / "model_meta.json").exists()
    assert ValidateScores(prepared, scores).run().ok

    repository = LocalScoreRepository(scores)
    scored_test = repository.read_run_scores("opcua/test/fault_001")
    assert set(scored_test.columns) == {"ts_ns", "score"}
    assert scored_test["score"].max() > scored_test["score"].median()
