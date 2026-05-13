"""Filesystem progress recording and status summaries."""

from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from industrial_tsad_eval.domain.progress import ProgressEvent
from industrial_tsad_eval.infrastructure.json_utils import read_json, write_json


class LocalProgressSink:
    """Persist progress events as JSONL and a latest-state snapshot."""

    def __init__(self, root: str | Path, run_id: str):
        self.root = Path(root)
        self.run_id = run_id
        self.events_path = self.root / "progress.jsonl"
        self.snapshot_path = self.root / "progress_snapshot.json"
        self.items: dict[str, dict[str, Any]] = {}
        self.latest_event: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def emit(self, event: ProgressEvent) -> None:
        """Append one event and update the run snapshot."""
        payload = event.to_dict()
        with self._lock:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            self.items[event.key] = payload
            self.latest_event = payload
            write_json(self.snapshot_path, progress_snapshot(self.run_id, self.items, payload))


def progress_snapshot(
    run_id: str,
    items: dict[str, dict[str, Any]],
    latest_event: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a snapshot from latest item events."""
    counts = Counter(str(item.get("status", "unknown")) for item in items.values())
    latest_failure = _latest_event(items, {"failed"})
    return {
        "format_version": "progress-snapshot-v1",
        "run_id": run_id,
        "counts": dict(sorted(counts.items())),
        "total_items": len(items),
        "latest_event": latest_event,
        "latest_failure": latest_failure,
        "items": dict(sorted(items.items())),
    }


def read_run_progress(run: str | Path) -> dict[str, Any]:
    """Read a run progress snapshot or derive a fallback status."""
    run_path = Path(run)
    snapshot_path = run_path / "progress_snapshot.json"
    if snapshot_path.exists():
        return read_json(snapshot_path)
    manifest_path = run_path / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    summary_path = run_path / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    status = str(manifest.get("status") or ("completed" if summary else "unknown"))
    return {
        "format_version": "progress-snapshot-v1",
        "run_id": str(manifest.get("run_id") or run_path.name),
        "counts": {status: 1},
        "total_items": 1 if status != "unknown" else 0,
        "latest_event": None,
        "latest_failure": None,
        "items": {},
        "manifest": manifest,
        "summary": summary,
    }


def _latest_event(
    items: dict[str, dict[str, Any]],
    statuses: set[str],
) -> dict[str, Any] | None:
    matches = [item for item in items.values() if str(item.get("status")) in statuses]
    if not matches:
        return None
    return sorted(matches, key=lambda item: str(item.get("timestamp_utc", "")))[-1]
