from __future__ import annotations

import json
from pathlib import Path

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.evidence import (
    BuildGroundTruthTagMap,
    GenerateEvidence,
    ValidateEvidence,
    ValidateGroundTruthTagMap,
)
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.xai import EvaluateEvidence, EvaluateEvidenceConfig
from industrial_tsad_eval.domain.evidence import EvidenceBundle, EvidenceVariable
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_evidence_bundle_serialization_round_trip():
    bundle = EvidenceBundle(
        dataset="D",
        run_id="run/1",
        event_id="event-1",
        event_source="oracle",
        event_start_ts_ns=10,
        event_end_ts_ns=20,
        top_variables=[
            EvidenceVariable(variable="Plant/TagA", rank=1, importance=2.0, mean_abs_z=2.0)
        ],
        top_time_windows=[],
    )

    parsed = EvidenceBundle.from_dict(bundle.to_dict())

    assert parsed.run_id == "run/1"
    assert parsed.top_variables[0].variable == "Plant/TagA"


def test_oracle_evidence_generation_validation_and_xai_eval(
    opcua_prepared: Path,
    tmp_path: Path,
):
    scores, _eval_dir = _score_and_eval(opcua_prepared, tmp_path)
    evidence_dir = tmp_path / "evidence"
    gt_map = tmp_path / "gt_map.json"
    xai_out = tmp_path / "xai"

    generated = GenerateEvidence(
        prepared=opcua_prepared,
        scores=scores,
        out=evidence_dir,
        top_k=5,
        max_events=5,
    ).run()
    evidence_report = ValidateEvidence(opcua_prepared, evidence_dir).run()
    gt_result = BuildGroundTruthTagMap(prepared=opcua_prepared, out=gt_map).run()
    gt_report = ValidateGroundTruthTagMap(gt_map).run()
    xai_result = EvaluateEvidence(
        EvaluateEvidenceConfig(
            prepared=opcua_prepared,
            evidence=evidence_dir,
            gt_map=gt_map,
            out=xai_out,
            ks=[1, 3, 5],
        )
    ).run()

    assert generated.bundle_count == 1
    assert evidence_report.ok
    assert gt_result.mapped_count == 1
    assert gt_report.ok
    assert (evidence_dir / "manifest.json").exists()
    assert (evidence_dir / "index.jsonl").exists()
    assert (xai_out / "metrics.json").exists()
    assert (xai_out / "bundle_metrics.csv").exists()
    assert xai_result.metrics["valid_bundle_count"] == 1
    assert xai_result.metrics["hitrate_at_k"]["5"] == 1.0
    assert xai_result.metrics["masking"]["status"] == "computed"


def test_operational_evidence_handles_matched_and_unmatched_predictions(
    opcua_prepared: Path,
    tmp_path: Path,
):
    scores, eval_dir = _score_and_eval(opcua_prepared, tmp_path)
    _write_operational_matches(opcua_prepared, eval_dir)
    evidence_dir = tmp_path / "operational_evidence"
    gt_map = tmp_path / "gt_map.json"
    xai_out = tmp_path / "operational_xai"
    BuildGroundTruthTagMap(prepared=opcua_prepared, out=gt_map).run()

    generated = GenerateEvidence(
        prepared=opcua_prepared,
        scores=scores,
        eval_dir=eval_dir,
        out=evidence_dir,
        event_source="operational",
        top_k=5,
        max_events=10,
    ).run()
    result = EvaluateEvidence(
        EvaluateEvidenceConfig(
            prepared=opcua_prepared,
            evidence=evidence_dir,
            gt_map=gt_map,
            out=xai_out,
            ks=[3],
        )
    ).run()

    assert generated.bundle_count == 2
    assert result.metrics["bundle_count"] == 2
    assert result.metrics["valid_bundle_count"] == 1
    assert result.skipped["missing_gt_entry"] == 1


def _score_and_eval(prepared: Path, tmp_path: Path) -> tuple[Path, Path]:
    scores = tmp_path / "scores"
    eval_dir = tmp_path / "eval"
    ScoreRuns(
        detector_registry=default_detector_registry(),
        prepared=prepared,
        scores=scores,
        detector_name="forecast-ridge",
        detector_parameters={"window": 24, "stride": 4, "lags": 1},
    ).run()
    EvaluateScores(prepared=prepared, scores=scores, out=eval_dir).run()
    return scores, eval_dir


def _write_operational_matches(prepared: Path, eval_dir: Path) -> None:
    event = json.loads(
        (prepared / "events" / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    payload = {
        "gt_matches": {event["event_id"]: "pred_match"},
        "pred_matches": {"pred_match": event["event_id"], "pred_unmatched": None},
        "pred_events": [
            {
                "run_id": event["run_id"],
                "pred_event_id": "pred_match",
                "start_ts_ns": event["start_ts_ns"],
                "end_ts_ns": event["end_ts_ns"],
                "first_detect_ts_ns": event["start_ts_ns"],
            },
            {
                "run_id": event["run_id"],
                "pred_event_id": "pred_unmatched",
                "start_ts_ns": event["start_ts_ns"],
                "end_ts_ns": event["end_ts_ns"],
                "first_detect_ts_ns": event["start_ts_ns"],
            },
        ],
    }
    (eval_dir / "event_matches.json").write_text(json.dumps(payload), encoding="utf-8")
