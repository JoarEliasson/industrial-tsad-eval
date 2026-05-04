"""Validation report model shared by contract checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationReport:
    """Structured result returned by validation use cases."""

    subject: str
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return true when no validation errors were collected."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to JSON-compatible data."""
        return {
            "subject": self.subject,
            "path": self.path,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }
