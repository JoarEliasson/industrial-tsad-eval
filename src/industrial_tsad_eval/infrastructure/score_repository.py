"""Filesystem repository for Score Contract v1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from industrial_tsad_eval.infrastructure.json_utils import write_json


class LocalScoreRepository:
    """Read and write Score Contract v1 score parquet files."""

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._manifest: dict[str, str] = {}

    @property
    def root(self) -> Path:
        """Return the score artifact root path."""
        return self._root

    def discover(self) -> dict[str, Path]:
        """Discover score parquet files by manifest or encoded filename."""
        if not self._root.exists():
            raise FileNotFoundError(f"Scores directory not found: {self._root}")

        manifest_path = self._root / "manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Score manifest must be an object: {manifest_path}")
            return {
                str(run_id): self._root / str(relative_path)
                for run_id, relative_path in payload.items()
            }

        return {
            path.stem.replace("__", "/"): path
            for path in sorted(self._root.glob("*.parquet"))
            if path.name != "manifest.parquet"
        }

    def read_run_scores(self, run_id: str) -> pd.DataFrame:
        """Read one run's score parquet."""
        mapping = self.discover()
        if run_id not in mapping:
            raise FileNotFoundError(f"No score file found for run {run_id!r}.")
        return pd.read_parquet(mapping[run_id])

    def write_run_scores(self, run_id: str, scores: pd.DataFrame) -> None:
        """Write one run's scores and remember it for manifest emission."""
        self._root.mkdir(parents=True, exist_ok=True)
        filename = f"{run_id.replace('/', '__')}.parquet"
        scores.to_parquet(self._root / filename, index=False)
        self._manifest[run_id] = filename

    def write_manifest(self) -> None:
        """Write the run-to-file score manifest."""
        write_json(self._root / "manifest.json", self._manifest)

    def write_model_metadata(self, metadata: dict[str, Any]) -> None:
        """Write model metadata beside score files."""
        write_json(self._root / "model_meta.json", metadata)
