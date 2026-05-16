"""Filesystem repository for detector-native explanation artifacts."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_tsad_eval.infrastructure.json_utils import write_json

REQUIRED_EXPLANATION_COLUMNS = {"ts_ns", "variable", "importance", "rank", "method"}


class LocalExplanationRepository:
    """Read and write per-run native explanation parquet files."""

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._manifest: dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def root(self) -> Path:
        """Return the explanation artifact root path."""
        return self._root

    def discover(self) -> dict[str, Path]:
        """Discover explanation parquet files by manifest or encoded filename."""
        if not self._root.exists():
            return {}

        manifest_path = self._root / "manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Explanation manifest must be an object: {manifest_path}")
            return {
                str(run_id): self._root / str(relative_path)
                for run_id, relative_path in payload.items()
            }

        return {path.stem.replace("__", "/"): path for path in sorted(self._root.glob("*.parquet"))}

    def read_run_explanations(self, run_id: str) -> pd.DataFrame:
        """Read one run's native explanation parquet file."""
        mapping = self.discover()
        if run_id not in mapping:
            raise FileNotFoundError(f"No explanation file found for run {run_id!r}.")
        return pd.read_parquet(mapping[run_id])

    def write_run_explanations(self, run_id: str, explanations: pd.DataFrame) -> None:
        """Write one run's explanation rows and remember it for manifest emission."""
        missing = REQUIRED_EXPLANATION_COLUMNS - set(explanations.columns)
        if missing:
            raise ValueError(f"Explanation rows missing required columns: {sorted(missing)}")
        self._root.mkdir(parents=True, exist_ok=True)
        filename = f"{run_id.replace('/', '__')}.parquet"
        explanations.to_parquet(self._root / filename, index=False)
        with self._lock:
            self._manifest[run_id] = filename

    def write_manifest(self) -> None:
        """Write the run-to-file explanation manifest."""
        with self._lock:
            manifest = dict(self._manifest)
        if manifest:
            write_json(self._root / "manifest.json", manifest)

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        """Write explanation metadata beside explanation files."""
        with self._lock:
            has_manifest = bool(self._manifest)
        if has_manifest:
            write_json(self._root / "explanation_meta.json", metadata)
