"""Runtime profiling contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageSample:
    """Timing and resource sample for one profiled stage."""

    stage: str
    start_ts_ns: int
    end_ts_ns: int
    duration_ms: float
    rss_before_bytes: int | None
    rss_after_bytes: int | None
    rss_peak_bytes: int | None
    python_current_bytes: int | None
    python_peak_bytes: int | None
    torch_before_bytes: int | None
    torch_after_bytes: int | None
    torch_peak_bytes: int | None
    vram_before_bytes: int | None
    vram_after_bytes: int | None
    vram_peak_bytes: int | None
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class ProfileRunResult:
    """Application-level result for a profiling run."""

    profile_id: str
    profile_dir: str
    ok: bool
    stages: list[StageSample]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "profile_id": self.profile_id,
            "profile_dir": self.profile_dir,
            "ok": self.ok,
            "stages": [stage.to_dict() for stage in self.stages],
            "summary": self.summary,
        }
