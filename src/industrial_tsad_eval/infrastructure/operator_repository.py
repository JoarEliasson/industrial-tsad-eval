"""Filesystem repository for deterministic operator cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from industrial_tsad_eval.domain.operator import (
    OPERATOR_CARD_MANIFEST_VERSION,
    OperatorCard,
    OperatorCardIndexRow,
)
from industrial_tsad_eval.infrastructure.json_utils import read_json, write_json, write_jsonl


class LocalOperatorCardRepository:
    """Read and write Operator Card v1 artifacts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_card_set(
        self,
        *,
        dataset: str,
        cards: list[OperatorCard],
        retrieval_payload: dict[str, Any],
    ) -> list[OperatorCardIndexRow]:
        """Write cards, Markdown views, retrieval payload, index rows, and manifest."""
        rows: list[OperatorCardIndexRow] = []
        for card in cards:
            json_relative = _card_json_relative_path(card.event_id)
            markdown_relative = _card_markdown_relative_path(card.event_id)
            write_json(self.root / json_relative, card.to_dict())
            (self.root / markdown_relative).parent.mkdir(parents=True, exist_ok=True)
            (self.root / markdown_relative).write_text(
                render_operator_card_markdown(card),
                encoding="utf-8",
            )
            rows.append(
                OperatorCardIndexRow(
                    dataset=card.dataset,
                    run_id=card.run_id,
                    event_id=card.event_id,
                    status=card.status,
                    relative_json_path=json_relative,
                    relative_markdown_path=markdown_relative,
                    citation_count=len(card.citations),
                )
            )

        write_json(self.root / "retrieval" / "retrieval_result.json", retrieval_payload)
        write_jsonl(self.root / "index.jsonl", [row.to_dict() for row in rows])
        write_json(
            self.root / "manifest.json",
            {
                "format_version": OPERATOR_CARD_MANIFEST_VERSION,
                "dataset": dataset,
                "card_count": len(rows),
                "index_path": "index.jsonl",
                "card_root": "cards",
                "retrieval_path": "retrieval/retrieval_result.json",
            },
        )
        return rows

    def manifest(self) -> dict[str, Any]:
        """Read the operator-card manifest."""
        return read_json(self.root / "manifest.json")

    def index_rows(self) -> list[OperatorCardIndexRow]:
        """Read operator-card index rows."""
        index_path = self.root / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Operator card index not found: {index_path}")
        rows: list[OperatorCardIndexRow] = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError(f"Operator card index row must be an object: {index_path}")
                    rows.append(OperatorCardIndexRow.from_dict(payload))
        return rows

    def read_card(self, row: OperatorCardIndexRow) -> OperatorCard:
        """Read one operator card from an index row."""
        payload = read_json(self.root / row.relative_json_path)
        return OperatorCard.from_dict(payload)


def render_operator_card_markdown(card: OperatorCard) -> str:
    """Render an operator card as Markdown."""
    lines = [
        f"# Operator Card: {card.event_id}",
        "",
        f"- Dataset: {card.dataset}",
        f"- Run: {card.run_id or 'n/a'}",
        f"- Status: {card.status}",
    ]
    if card.abstain_reason:
        lines.append(f"- Abstention: {card.abstain_reason}")
    lines.extend(
        [
            "",
            "## Situation",
            card.situation_summary or "No supported situation summary is available.",
            "",
            "## Evidence Highlights",
            *_list_section(card.evidence_highlights),
            "",
            "## Checks",
            *_list_section(card.checks),
            "",
            "## Recommended Actions",
            *_list_section(card.recommended_actions),
            "",
            "## Escalation Criteria",
            *_list_section(card.escalation_criteria),
            "",
            "## Citations",
            *_citation_section(card.citations),
            "",
        ]
    )
    return "\n".join(lines)


def _card_json_relative_path(event_id: str) -> str:
    return f"cards/{_safe_id(event_id)}/operator_card.json"


def _card_markdown_relative_path(event_id: str) -> str:
    return f"cards/{_safe_id(event_id)}/operator_card.md"


def _safe_id(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )
    return cleaned.strip("._-") or "item"


def _list_section(items: list[str]) -> list[str]:
    if not items:
        return ["- None."]
    return [f"- {item}" for item in items]


def _citation_section(citations: list[dict[str, Any]]) -> list[str]:
    if not citations:
        return ["- None."]
    return [
        (
            f"- [{citation.get('citation_id', 'C?')}] "
            f"{citation.get('title', 'Untitled source')} "
            f"({citation.get('source_type', 'unknown')}, role={citation.get('role', 'unknown')})"
        )
        for citation in citations
    ]
