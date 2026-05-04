"""Versioned evaluation policy models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_COMPUTE_GROUPS = ["event", "delay", "far"]
DEFAULT_EVENT_TYPES = ["attack", "fault", "anomaly"]


@dataclass(frozen=True)
class EvalConfig:
    """Resolved evaluation configuration."""

    grace_ns: int
    merge_gap_ns: int
    threshold: float
    merge_gap_mode: str = "fixed_seconds"
    merge_gap_skipped_samples: int = 0
    merge_gap_jitter_ratio: float = 0.0
    compute: list[str] = field(default_factory=lambda: list(DEFAULT_COMPUTE_GROUPS))
    event_types: list[str] = field(default_factory=lambda: list(DEFAULT_EVENT_TYPES))


@dataclass(frozen=True)
class EvalPolicy:
    """Serializable policy describing score binarization and event evaluation."""

    policy_version: str = "industrial_v1"
    dataset: str | None = None
    protocol: str = "naive"
    threshold_source: str = "quantile"
    threshold_quantile: float = 0.995
    binarization_rule: str = "score_gte_threshold"
    min_pred_event_ns: int = 0
    merge_gap_s: float = 10.0
    grace_s: float = 5.0
    far_counting_unit: str = "per_hour"
    delay_definition: str = "first_detect_minus_gt_start_floored_zero"
    merge_gap_mode: str = "fixed_seconds"
    merge_gap_skipped_samples: int = 0
    merge_gap_jitter_ratio: float = 0.0
    compute: list[str] = field(default_factory=lambda: list(DEFAULT_COMPUTE_GROUPS))
    event_types: list[str] = field(default_factory=lambda: list(DEFAULT_EVENT_TYPES))

    def to_config(self, threshold: float) -> EvalConfig:
        """Create a resolved runtime configuration with a concrete threshold."""
        return EvalConfig(
            grace_ns=int(self.grace_s * 1_000_000_000),
            merge_gap_ns=int(self.merge_gap_s * 1_000_000_000),
            threshold=threshold,
            merge_gap_mode=self.merge_gap_mode,
            merge_gap_skipped_samples=self.merge_gap_skipped_samples,
            merge_gap_jitter_ratio=self.merge_gap_jitter_ratio,
            compute=list(self.compute),
            event_types=list(self.event_types),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy to JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalPolicy:
        """Build a policy from JSON-compatible data."""
        known_fields = set(cls.__dataclass_fields__)
        filtered = {key: value for key, value in payload.items() if key in known_fields}
        return cls(**filtered)
