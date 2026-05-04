from __future__ import annotations

import json

import numpy as np

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.domain.evaluation import (
    cluster_anomalies_to_pred_events,
    compute_event_prf,
    match_pred_to_gt_events,
)
from industrial_tsad_eval.domain.events import GTEvent
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_event_matching_and_prf_are_deterministic():
    gt_events = [
        GTEvent(
            run_id="r1",
            event_id="e1",
            start_ts_ns=100,
            end_ts_ns=200,
            event_type="fault",
        )
    ]
    pred_events = cluster_anomalies_to_pred_events(
        "r1",
        np.array([120, 130, 500], dtype=np.int64),
        merge_gap_ns=15,
        stride_ns=10,
        period_ns_fallback=10,
    )

    gt_matches, _pred_matches = match_pred_to_gt_events(gt_events, pred_events, grace_ns=0)
    metrics = compute_event_prf(gt_events, pred_events, gt_matches)

    assert [event.pred_event_id for event in pred_events] == ["pred_r1_0000", "pred_r1_0001"]
    assert gt_matches == {"e1": "pred_r1_0000"}
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0


def test_evaluation_vertical_slice_writes_artifacts(tmp_path):
    prepared = make_opcua_fixture(tmp_path / "examples")
    scores = tmp_path / "scores"
    out = tmp_path / "eval"
    ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=prepared,
        scores=scores,
        detector_name="forecast-ridge",
        detector_parameters={"window": 32, "stride": 4, "lags": 2},
    ).run()

    result = EvaluateScores(prepared=prepared, scores=scores, out=out, threshold=0.1).run()

    assert (out / "metrics.json").exists()
    assert (out / "event_matches.json").exists()
    assert (out / "threshold.json").exists()
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert result.dataset == "OPCUA_SYNTH"
    assert metrics["event"]["n_gt"] == 1.0
    assert metrics["event"]["n_pred"] >= 1.0
