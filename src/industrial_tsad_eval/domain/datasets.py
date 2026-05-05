"""Dataset adapter contracts shared across preparation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetAdapterConfig:
    """Configuration supplied to dataset adapter plugins."""

    base_epoch_iso: str = "2020-01-01T00:00:00Z"
    default_period_ms: int = 100
    strict: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetAdapterResult:
    """Summary returned by dataset preparation use cases."""

    dataset_name: str
    prepared_path: str
    run_count: int
    event_count: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to JSON-compatible data."""
        return {
            "dataset_name": self.dataset_name,
            "prepared_path": self.prepared_path,
            "run_count": self.run_count,
            "event_count": self.event_count,
            "warnings": list(self.warnings),
        }
