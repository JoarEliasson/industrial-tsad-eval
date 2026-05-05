"""Evidence and explanation-evaluation contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

EvidenceSource = Literal["oracle", "operational"]

EVIDENCE_FORMAT_VERSION = "evidence-bundle-v1"
EVIDENCE_MANIFEST_VERSION = "evidence-manifest-v1"
GT_TAG_MAP_VERSION = "gt-tag-map-v1"


@dataclass(frozen=True)
class EvidenceVariable:
    """One ranked explanatory variable."""

    variable: str
    rank: int
    importance: float
    mean_abs_z: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class EvidenceTimeWindow:
    """One ranked explanatory time window."""

    start_ts_ns: int
    end_ts_ns: int
    rank: int
    importance: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class EvidenceBundle:
    """Evidence Bundle v1 for one oracle or operational event."""

    dataset: str
    run_id: str
    event_id: str
    event_source: EvidenceSource
    event_start_ts_ns: int
    event_end_ts_ns: int
    top_variables: list[EvidenceVariable]
    top_time_windows: list[EvidenceTimeWindow]
    source_event_id: str | None = None
    matched_gt_event_id: str | None = None
    is_matched_pred_event: bool = False
    score_context: dict[str, Any] = field(default_factory=dict)
    local_rankings: list[dict[str, Any]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    format_version: str = EVIDENCE_FORMAT_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "format_version": self.format_version,
            "dataset": self.dataset,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "event_source": self.event_source,
            "source_event_id": self.source_event_id,
            "matched_gt_event_id": self.matched_gt_event_id,
            "is_matched_pred_event": self.is_matched_pred_event,
            "event_start_ts_ns": self.event_start_ts_ns,
            "event_end_ts_ns": self.event_end_ts_ns,
            "top_variables": [variable.to_dict() for variable in self.top_variables],
            "top_time_windows": [window.to_dict() for window in self.top_time_windows],
            "score_context": dict(self.score_context),
            "local_rankings": [dict(row) for row in self.local_rankings],
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidenceBundle:
        """Parse an Evidence Bundle v1 payload."""
        return cls(
            dataset=str(payload["dataset"]),
            run_id=str(payload["run_id"]),
            event_id=str(payload["event_id"]),
            event_source=_evidence_source(str(payload["event_source"])),
            source_event_id=_optional_str(payload.get("source_event_id")),
            matched_gt_event_id=_optional_str(payload.get("matched_gt_event_id")),
            is_matched_pred_event=bool(payload.get("is_matched_pred_event", False)),
            event_start_ts_ns=int(payload["event_start_ts_ns"]),
            event_end_ts_ns=int(payload["event_end_ts_ns"]),
            top_variables=[
                EvidenceVariable(
                    variable=str(item["variable"]),
                    rank=int(item["rank"]),
                    importance=float(item["importance"]),
                    mean_abs_z=float(item.get("mean_abs_z", item["importance"])),
                )
                for item in _list_of_dicts(payload.get("top_variables", []))
            ],
            top_time_windows=[
                EvidenceTimeWindow(
                    start_ts_ns=int(item["start_ts_ns"]),
                    end_ts_ns=int(item["end_ts_ns"]),
                    rank=int(item["rank"]),
                    importance=float(item["importance"]),
                )
                for item in _list_of_dicts(payload.get("top_time_windows", []))
            ],
            score_context=dict(payload.get("score_context", {})),
            local_rankings=[
                dict(item) for item in _list_of_dicts(payload.get("local_rankings", []))
            ],
            provenance=dict(payload.get("provenance", {})),
            format_version=str(payload.get("format_version", EVIDENCE_FORMAT_VERSION)),
        )


@dataclass(frozen=True)
class EvidenceIndexRow:
    """Discovery row for one evidence bundle."""

    dataset: str
    run_id: str
    event_id: str
    event_source: EvidenceSource
    relative_path: str
    matched_gt_event_id: str | None
    is_matched_pred_event: bool
    top_variables: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidenceIndexRow:
        """Parse an evidence index row."""
        return cls(
            dataset=str(payload["dataset"]),
            run_id=str(payload["run_id"]),
            event_id=str(payload["event_id"]),
            event_source=_evidence_source(str(payload["event_source"])),
            relative_path=str(payload["relative_path"]),
            matched_gt_event_id=_optional_str(payload.get("matched_gt_event_id")),
            is_matched_pred_event=bool(payload.get("is_matched_pred_event", False)),
            top_variables=[str(item) for item in payload.get("top_variables", [])],
        )


@dataclass(frozen=True)
class GroundTruthTagMap:
    """Event-keyed ground-truth tag map for XAI metrics."""

    dataset: str
    key_mode: str
    entries: dict[str, list[str]]
    format_version: str = GT_TAG_MAP_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "format_version": self.format_version,
            "dataset": self.dataset,
            "key_mode": self.key_mode,
            "entries": {
                str(key): sorted({str(tag) for tag in tags})
                for key, tags in sorted(self.entries.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GroundTruthTagMap:
        """Parse a ground-truth tag map."""
        entries_payload = payload.get("entries", payload.get("faults", {}))
        entries: dict[str, list[str]] = {}
        if isinstance(entries_payload, dict):
            for key, value in entries_payload.items():
                tags = value.get("tags", []) if isinstance(value, dict) else value
                entries[str(key)] = sorted({str(tag) for tag in _tag_values(tags)})
        return cls(
            dataset=str(payload.get("dataset", "")),
            key_mode=str(payload.get("key_mode", "event_id")),
            entries=entries,
            format_version=str(payload.get("format_version", GT_TAG_MAP_VERSION)),
        )


@dataclass(frozen=True)
class XAIEvaluationResult:
    """Application-level result for an XAI evaluation run."""

    dataset: str
    evidence_dir: str
    out_dir: str
    metrics: dict[str, Any]
    bundle_metrics: list[dict[str, Any]]
    skipped: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "dataset": self.dataset,
            "evidence_dir": self.evidence_dir,
            "out_dir": self.out_dir,
            "metrics": dict(self.metrics),
            "bundle_metrics": [dict(row) for row in self.bundle_metrics],
            "skipped": dict(self.skipped),
        }


def _evidence_source(value: str) -> EvidenceSource:
    normalized = value.strip().lower()
    if normalized not in {"oracle", "operational"}:
        raise ValueError("event_source must be either 'oracle' or 'operational'.")
    return cast(EvidenceSource, normalized)


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _tag_values(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]
