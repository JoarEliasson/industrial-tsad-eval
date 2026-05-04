"""Filesystem repository for Prepared Format v1 datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_tsad_eval.domain.events import GTEvent
from industrial_tsad_eval.infrastructure.json_utils import read_json


class LocalPreparedDatasetRepository:
    """Read a Prepared Format v1 dataset from the local filesystem."""

    def __init__(self, root: str | Path):
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """Return the dataset root path."""
        return self._root

    @property
    def dataset_name(self) -> str:
        """Return the manifest dataset name or directory name."""
        manifest_path = self._root / "meta" / "manifest.json"
        if manifest_path.exists():
            return str(read_json(manifest_path).get("dataset", self._root.name))
        return self._root.name

    def manifest(self) -> dict[str, Any]:
        """Load `meta/manifest.json`."""
        return read_json(self._root / "meta" / "manifest.json")

    def schema(self) -> dict[str, Any]:
        """Load `meta/schema.json`."""
        return read_json(self._root / "meta" / "schema.json")

    def splits(self) -> dict[str, Any]:
        """Load `meta/splits.json`."""
        return read_json(self._root / "meta" / "splits.json")

    def run_ids(self) -> list[str]:
        """Return run IDs discovered from run directories."""
        runs_root = self._root / "runs"
        run_ids: list[str] = []
        for timeseries_path in sorted(runs_root.rglob("timeseries.parquet")):
            run_ids.append("/".join(timeseries_path.parent.relative_to(runs_root).parts))
        return run_ids

    def read_run(self, run_id: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Read a run timeseries parquet file."""
        return pd.read_parquet(self._root / "runs" / run_id / "timeseries.parquet", columns=columns)

    def read_events(self, event_types: list[str] | None = None) -> list[GTEvent]:
        """Read ground-truth events from `events/events.jsonl`."""
        events_path = self._root / "events" / "events.jsonl"
        if not events_path.exists():
            return []

        allowed = set(event_types or [])
        events: list[GTEvent] = []
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                event_type = str(payload.get("event_type", "anomaly"))
                if allowed and event_type not in allowed:
                    continue
                events.append(
                    GTEvent(
                        run_id=str(payload["run_id"]),
                        event_id=str(payload["event_id"]),
                        start_ts_ns=int(payload["start_ts_ns"]),
                        end_ts_ns=int(payload["end_ts_ns"]),
                        event_type=event_type,
                        metadata=dict(payload.get("metadata", {})),
                    )
                )
        return events
