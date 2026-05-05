from __future__ import annotations

from pathlib import Path

import numpy as np

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.domain.evaluation import (
    cluster_anomalies_to_pred_events,
    compute_delays,
    compute_event_prf,
    compute_false_alarm_rates,
    match_pred_to_gt_events,
)
from industrial_tsad_eval.domain.events import GTEvent
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_event_matching_delay_and_far_metrics():
    gt = [GTEvent("run", "gt-1", 10, 30, "attack", {})]
    pred = cluster_anomalies_to_pred_events("run", np.array([12, 13, 100]), 5, 1, 1)

    gt_matches, _pred_matches = match_pred_to_gt_events(gt, pred, grace_ns=0)
    event = compute_event_prf(gt, pred, gt_matches)
    delay = compute_delays(gt, gt_matches, pred)
    far = compute_false_alarm_rates(
        ["normal"], {"normal": [pred[-1]]}, {"normal": 3_600_000_000_000}, {"normal": 100}
    )

    assert event["precision"] == 0.5
    assert event["recall"] == 1.0
    assert delay["mean"] == 2.0
    assert far["false_events_per_hour"] == 1.0


def test_evaluate_scores_writes_report_artifacts(opcua_prepared: Path, tmp_path: Path):
    scores = tmp_path / "scores"
    out = tmp_path / "eval"
    ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=opcua_prepared,
        scores=scores,
        detector_name="forecast-ridge",
        detector_parameters={"window": 24, "stride": 4, "lags": 1},
    ).run()

    result = EvaluateScores(prepared=opcua_prepared, scores=scores, out=out).run()

    assert result.metrics["event"]["n_gt"] == 1.0
    assert (out / "metrics.json").exists()
    assert (out / "threshold.json").exists()
    assert (out / "event_matches.json").exists()
