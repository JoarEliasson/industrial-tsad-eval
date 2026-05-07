"""Progress reporting contracts for long-running workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

ProgressStatus = Literal["planned", "running", "completed", "failed", "skipped", "warn"]


@dataclass(frozen=True)
class ProgressEvent:
    """One machine-readable progress event."""

    run_id: str
    stage: str
    item_id: str
    status: ProgressStatus
    timestamp_utc: str = field(default_factory=lambda: _utc_now())
    ordinal: int | None = None
    total: int | None = None
    path: str | None = None
    duration_s: float | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None
    message: str | None = None
    format_version: str = "progress-event-v1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)

    @property
    def key(self) -> str:
        """Return the stable item key used in progress snapshots."""
        return f"{self.stage}:{self.item_id}"


class ProgressSink(Protocol):
    """Receiver for progress events."""

    def emit(self, event: ProgressEvent) -> None:
        """Handle one progress event."""


class NullProgressSink:
    """Progress sink that discards all events."""

    def emit(self, event: ProgressEvent) -> None:
        """Discard one progress event."""


class CompositeProgressSink:
    """Fan progress events out to several sinks."""

    def __init__(self, sinks: list[ProgressSink | None]):
        self.sinks = [sink for sink in sinks if sink is not None]

    def emit(self, event: ProgressEvent) -> None:
        """Emit one event to every configured sink."""
        for sink in self.sinks:
            sink.emit(event)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
