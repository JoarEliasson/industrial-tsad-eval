from __future__ import annotations

from pathlib import Path

import numpy as np

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.domain.evaluation import (
    cluster_anomalies_to_pred_events,
    compute_affiliation_prf,
    compute_delays,
    compute_event_prf,
    compute_false_alarm_rates,
    compute_point_prf_for_scores,
    match_pred_to_gt_events,
)
from industrial_tsad_eval.domain.events import GTEvent
from industrial_tsad_eval.domain.policy import EvalPolicy
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


def test_point_and_affiliation_metrics_cover_edge_cases():
    gt = [GTEvent("run", "gt-1", 10, 30, "attack", {})]
    timestamps = {"run": np.array([0, 10, 20, 30, 40], dtype=np.int64)}
    values = {"run": np.array([0.1, 0.1, 0.9, 0.1, 0.9], dtype=np.float64)}

    point = compute_point_prf_for_scores(timestamps, values, gt, threshold=0.5)
    adjusted = compute_point_prf_for_scores(
        timestamps,
        values,
        gt,
        threshold=0.5,
        point_adjusted=True,
    )
    affiliation = compute_affiliation_prf(
        gt,
        cluster_anomalies_to_pred_events("run", np.array([20, 40]), 0, 10, 10),
    )

    assert point["precision"] == 0.5
    assert point["recall"] == 0.5
    assert adjusted["recall"] == 1.0
    assert affiliation["precision"] < 1.0
    assert affiliation["recall"] > 0.0


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


def test_evaluate_scores_honors_metric_compute_groups(opcua_prepared: Path, tmp_path: Path):
    scores = tmp_path / "scores_compute"
    out = tmp_path / "eval_compute"
    ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=opcua_prepared,
        scores=scores,
        detector_name="forecast-ridge",
        detector_parameters={"window": 24, "stride": 4, "lags": 1},
    ).run()

    result = EvaluateScores(
        prepared=opcua_prepared,
        scores=scores,
        out=out,
        policy=EvalPolicy(compute=["point", "point_adjusted", "affiliation"]),
    ).run()

    assert "event" not in result.metrics
    assert "point" in result.metrics
    assert "point_adjusted" in result.metrics
    assert "affiliation" in result.metrics
