"""Deterministic operator-assistant contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

OPERATOR_CARD_VERSION = "operator-card-v1"
OPERATOR_CARD_MANIFEST_VERSION = "operator-card-manifest-v1"
OPERATOR_RETRIEVAL_VERSION = "operator-retrieval-v1"

OperatorIntent = Literal[
    "general",
    "checks",
    "recommended_actions",
    "likely_causes",
    "escalation_criteria",
]
OperatorSourceType = Literal["evidence_bundle", "playbook"]
OperatorCardStatus = Literal["answered", "abstained"]


@dataclass(frozen=True)
class OperatorQuery:
    """Resolved operator query and deterministic retrieval intent."""

    query: str
    top_k: int = 8
    dataset: str | None = None
    event_id: str | None = None
    detected_intent: OperatorIntent = "general"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class OperatorEvidenceHit:
    """One ranked evidence or playbook hit used by operator cards."""

    source_id: str
    source_type: OperatorSourceType
    title: str
    role: str
    rank: int
    score: float
    text: str
    citation_id: str
    dataset: str | None = None
    event_id: str | None = None
    run_id: str | None = None
    relative_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OperatorEvidenceHit:
        """Parse an operator evidence hit."""
        return cls(
            source_id=str(payload["source_id"]),
            source_type=_source_type(str(payload["source_type"])),
            title=str(payload["title"]),
            role=str(payload["role"]),
            rank=int(payload["rank"]),
            score=float(payload["score"]),
            text=str(payload["text"]),
            citation_id=str(payload["citation_id"]),
            dataset=_optional_str(payload.get("dataset")),
            event_id=_optional_str(payload.get("event_id")),
            run_id=_optional_str(payload.get("run_id")),
            relative_path=_optional_str(payload.get("relative_path")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class OperatorRetrievalResult:
    """Ranked retrieval result for one operator query."""

    query: OperatorQuery
    hits: list[OperatorEvidenceHit]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    format_version: str = OPERATOR_RETRIEVAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "format_version": self.format_version,
            "query": self.query.to_dict(),
            "hits": [hit.to_dict() for hit in self.hits],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class OperatorCard:
    """Deterministic operator-facing card grounded in cited evidence."""

    dataset: str
    event_id: str
    query: str
    status: OperatorCardStatus
    situation_summary: str
    evidence_highlights: list[str]
    checks: list[str]
    recommended_actions: list[str]
    escalation_criteria: list[str]
    citations: list[dict[str, Any]]
    run_id: str | None = None
    event_source: str | None = None
    matched_gt_event_id: str | None = None
    abstain_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    format_version: str = OPERATOR_CARD_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "format_version": self.format_version,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "event_source": self.event_source,
            "matched_gt_event_id": self.matched_gt_event_id,
            "query": self.query,
            "status": self.status,
            "abstain_reason": self.abstain_reason,
            "situation_summary": self.situation_summary,
            "evidence_highlights": list(self.evidence_highlights),
            "checks": list(self.checks),
            "recommended_actions": list(self.recommended_actions),
            "escalation_criteria": list(self.escalation_criteria),
            "citations": [dict(citation) for citation in self.citations],
            "diagnostics": dict(self.diagnostics),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OperatorCard:
        """Parse an Operator Card v1 payload."""
        return cls(
            dataset=str(payload["dataset"]),
            run_id=_optional_str(payload.get("run_id")),
            event_id=str(payload["event_id"]),
            event_source=_optional_str(payload.get("event_source")),
            matched_gt_event_id=_optional_str(payload.get("matched_gt_event_id")),
            query=str(payload.get("query", "")),
            status=_card_status(str(payload["status"])),
            abstain_reason=_optional_str(payload.get("abstain_reason")),
            situation_summary=str(payload.get("situation_summary", "")),
            evidence_highlights=[str(item) for item in payload.get("evidence_highlights", [])],
            checks=[str(item) for item in payload.get("checks", [])],
            recommended_actions=[str(item) for item in payload.get("recommended_actions", [])],
            escalation_criteria=[str(item) for item in payload.get("escalation_criteria", [])],
            citations=[dict(item) for item in _list_of_dicts(payload.get("citations", []))],
            diagnostics=dict(payload.get("diagnostics", {})),
            provenance=dict(payload.get("provenance", {})),
            format_version=str(payload.get("format_version", OPERATOR_CARD_VERSION)),
        )


@dataclass(frozen=True)
class OperatorCardIndexRow:
    """Discovery row for one operator card."""

    dataset: str
    event_id: str
    status: OperatorCardStatus
    relative_json_path: str
    relative_markdown_path: str
    citation_count: int
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OperatorCardIndexRow:
        """Parse an operator-card index row."""
        return cls(
            dataset=str(payload["dataset"]),
            run_id=_optional_str(payload.get("run_id")),
            event_id=str(payload["event_id"]),
            status=_card_status(str(payload["status"])),
            relative_json_path=str(payload["relative_json_path"]),
            relative_markdown_path=str(payload["relative_markdown_path"]),
            citation_count=int(payload["citation_count"]),
        )


def _card_status(value: str) -> OperatorCardStatus:
    normalized = value.strip().lower()
    if normalized not in {"answered", "abstained"}:
        raise ValueError("operator card status must be 'answered' or 'abstained'.")
    return cast(OperatorCardStatus, normalized)


def _source_type(value: str) -> OperatorSourceType:
    normalized = value.strip().lower()
    if normalized not in {"evidence_bundle", "playbook"}:
        raise ValueError("operator evidence source_type must be 'evidence_bundle' or 'playbook'.")
    return cast(OperatorSourceType, normalized)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]
