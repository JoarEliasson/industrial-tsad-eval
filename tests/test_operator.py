from __future__ import annotations

import json
from pathlib import Path

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.evidence import GenerateEvidence
from industrial_tsad_eval.application.operator import (
    GenerateOperatorCards,
    RetrieveOperatorEvidence,
    ValidateOperatorCards,
)
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.domain.operator import OperatorCard
from industrial_tsad_eval.infrastructure.json_utils import read_json
from industrial_tsad_eval.infrastructure.operator_repository import render_operator_card_markdown
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_operator_card_serialization_and_markdown_citations():
    card = OperatorCard(
        dataset="D",
        run_id="run/1",
        event_id="event-1",
        query="What should I check?",
        status="answered",
        situation_summary="Tag A is elevated [C1].",
        evidence_highlights=["Tag A is ranked first [C1]."],
        checks=["Inspect Tag A [C1]."],
        recommended_actions=["Preserve artifacts [C1]."],
        escalation_criteria=["Escalate if repeated [C1]."],
        citations=[
            {
                "citation_id": "C1",
                "source_id": "evidence::event-1::top_variables",
                "source_type": "evidence_bundle",
                "title": "Evidence Bundle: event-1",
                "role": "top_variables",
                "event_id": "event-1",
            }
        ],
    )

    parsed = OperatorCard.from_dict(card.to_dict())
    markdown = render_operator_card_markdown(parsed)

    assert parsed.status == "answered"
    assert "[C1]" in markdown
    assert "Operator Card: event-1" in markdown


def test_operator_retrieval_filters_ranks_and_uses_playbooks(opcua_prepared: Path, tmp_path: Path):
    evidence_dir = _evidence(opcua_prepared, tmp_path)
    playbooks = tmp_path / "playbooks"
    playbooks.mkdir()
    (playbooks / "response.md").write_text(
        "# Response Playbook\n\nPreserve artifacts and compare ranked tags before reset.",
        encoding="utf-8",
    )

    result = RetrieveOperatorEvidence(
        prepared=opcua_prepared,
        evidence=evidence_dir,
        query="preserve artifacts and inspect ranked tags",
        playbooks=playbooks,
        top_k=8,
    ).run()

    assert result.query.detected_intent in {"checks", "recommended_actions"}
    assert result.hits
    assert [hit.rank for hit in result.hits] == list(range(1, len(result.hits) + 1))
    assert any(hit.source_type == "playbook" for hit in result.hits)


def test_operator_retrieval_abstains_when_filter_has_no_hits(
    opcua_prepared: Path,
    tmp_path: Path,
):
    evidence_dir = _evidence(opcua_prepared, tmp_path)

    result = RetrieveOperatorEvidence(
        prepared=opcua_prepared,
        evidence=evidence_dir,
        query="inspect event",
        event_id="missing-event",
    ).run()

    assert result.hits == []
    assert result.diagnostics["returned_count"] == 0


def test_generate_validate_operator_cards_from_oracle_evidence(
    opcua_prepared: Path,
    tmp_path: Path,
):
    evidence_dir = _evidence(opcua_prepared, tmp_path)
    out = tmp_path / "operator"

    generated = GenerateOperatorCards(
        prepared=opcua_prepared,
        evidence=evidence_dir,
        out=out,
        query="What should the operator check and preserve?",
    ).run()
    report = ValidateOperatorCards(
        prepared=opcua_prepared,
        evidence=evidence_dir,
        cards=out,
    ).run()
    card_path = next(out.glob("cards/*/operator_card.json"))
    card = read_json(card_path)

    assert generated.card_count == 1
    assert generated.statuses["answered"] == 1
    assert report.ok
    assert (out / "manifest.json").exists()
    assert (out / "index.jsonl").exists()
    assert (out / "retrieval" / "retrieval_result.json").exists()
    assert card["format_version"] == "operator-card-v1"
    assert card["citations"]
    assert "[C" in (card_path.parent / "operator_card.md").read_text(encoding="utf-8")


def test_generate_operator_card_abstains_for_missing_event(
    opcua_prepared: Path,
    tmp_path: Path,
):
    evidence_dir = _evidence(opcua_prepared, tmp_path)
    out = tmp_path / "operator"

    generated = GenerateOperatorCards(
        prepared=opcua_prepared,
        evidence=evidence_dir,
        out=out,
        event_id="missing-event",
    ).run()
    report = ValidateOperatorCards(
        prepared=opcua_prepared,
        evidence=evidence_dir,
        cards=out,
    ).run()
    card = read_json(next(out.glob("cards/*/operator_card.json")))

    assert generated.statuses["abstained"] == 1
    assert card["status"] == "abstained"
    assert card["abstain_reason"]
    assert report.ok


def test_operational_evidence_operator_cards_include_prediction_identity(
    opcua_prepared: Path,
    tmp_path: Path,
):
    scores, eval_dir = _score_and_eval(opcua_prepared, tmp_path)
    _write_operational_matches(opcua_prepared, eval_dir)
    evidence_dir = tmp_path / "operational_evidence"
    GenerateEvidence(
        prepared=opcua_prepared,
        scores=scores,
        eval_dir=eval_dir,
        out=evidence_dir,
        event_source="operational",
        max_events=10,
    ).run()
    out = tmp_path / "operator"

    generated = GenerateOperatorCards(
        prepared=opcua_prepared,
        evidence=evidence_dir,
        out=out,
    ).run()
    report = ValidateOperatorCards(
        prepared=opcua_prepared,
        evidence=evidence_dir,
        cards=out,
    ).run()

    assert generated.card_count == 2
    assert generated.statuses["answered"] == 2
    assert report.ok


def _evidence(prepared: Path, tmp_path: Path) -> Path:
    scores, _eval_dir = _score_and_eval(prepared, tmp_path)
    evidence_dir = tmp_path / "evidence"
    GenerateEvidence(prepared=prepared, scores=scores, out=evidence_dir).run()
    return evidence_dir


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
