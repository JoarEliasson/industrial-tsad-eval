"""Reproducibility audit contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

AuditStatus = Literal["pass", "warn", "fail", "skipped"]


@dataclass(frozen=True)
class AuditCheck:
    """One audit check result."""

    name: str
    status: AuditStatus
    required: bool
    message: str
    duration_s: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    log_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class AuditRunResult:
    """Summary of a reproducibility audit run."""

    audit_id: str
    audit_dir: str
    ok: bool
    checks: list[AuditCheck]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "audit_id": self.audit_id,
            "audit_dir": self.audit_dir,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "counts": {
                "pass": sum(1 for check in self.checks if check.status == "pass"),
                "warn": sum(1 for check in self.checks if check.status == "warn"),
                "fail": sum(1 for check in self.checks if check.status == "fail"),
                "skipped": sum(1 for check in self.checks if check.status == "skipped"),
            },
        }
