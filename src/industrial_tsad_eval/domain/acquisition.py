"""Raw dataset acquisition contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetSourceConfig:
    """Configuration supplied to raw dataset source plugins."""

    method: str = "manual"
    manual_path: str | None = None
    ref: str | None = None
    overwrite: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSourceResult:
    """Summary returned by raw dataset acquisition use cases."""

    source_name: str
    dataset_name: str
    method: str
    raw_path: str
    file_count: int
    provenance_path: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result to JSON-compatible data."""
        return {
            "source_name": self.source_name,
            "dataset_name": self.dataset_name,
            "method": self.method,
            "raw_path": self.raw_path,
            "file_count": self.file_count,
            "provenance_path": self.provenance_path,
            "warnings": list(self.warnings),
        }
