"""Event models used by the evaluation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GTEvent:
    """Ground-truth anomaly event."""

    run_id: str
    event_id: str
    start_ts_ns: int
    end_ts_ns: int
    event_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredEvent:
    """Predicted anomaly event derived from detector scores."""

    run_id: str
    pred_event_id: str
    start_ts_ns: int
    end_ts_ns: int
    first_detect_ts_ns: int
