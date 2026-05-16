"""Filesystem repository for Score Contract v1 artifacts."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from industrial_tsad_eval.infrastructure.json_utils import write_json

COMBINED_SCORES_FILENAME = "combined_scores.parquet"
RESERVED_SCORE_PARQUETS = {COMBINED_SCORES_FILENAME, "manifest.parquet"}


class LocalScoreRepository:
    """Read and write Score Contract v1 score parquet files."""

    def __init__(self, root: str | Path):
        self._root = Path(root)
        self._manifest: dict[str, str] = {}
        self._lock = threading.Lock()

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
            if path.name not in RESERVED_SCORE_PARQUETS
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
        with self._lock:
            self._manifest[run_id] = filename

    def has_combined_scores(self) -> bool:
        """Return whether the optional combined score sidecar exists."""
        return (self._root / COMBINED_SCORES_FILENAME).exists()

    def read_combined_scores(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Read the optional combined score sidecar."""
        path = self._root / COMBINED_SCORES_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"Combined score sidecar not found: {path}")
        return pd.read_parquet(path, columns=columns)

    def write_combined_scores(self, scores_by_run: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Write an additive combined score sidecar for faster bulk readers."""
        self._root.mkdir(parents=True, exist_ok=True)
        sidecar_path = self._root / COMBINED_SCORES_FILENAME
        schema = pa.schema(
            [
                pa.field("run_id", pa.string()),
                pa.field("ts_ns", pa.int64()),
                pa.field("score", pa.float64()),
            ]
        )
        writer: pq.ParquetWriter | None = None
        row_count = 0
        for run_id, frame in scores_by_run.items():
            if frame.empty:
                continue
            combined = frame.loc[:, ["ts_ns", "score"]].copy()
            combined["ts_ns"] = combined["ts_ns"].astype("int64", copy=False)
            combined["score"] = combined["score"].astype("float64", copy=False)
            combined.insert(0, "run_id", str(run_id))
            table = pa.Table.from_pandas(
                combined,
                schema=schema,
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(sidecar_path, schema)
            writer.write_table(table)
            row_count += int(table.num_rows)
        if writer is None:
            pq.write_table(pa.Table.from_arrays([[], [], []], schema=schema), sidecar_path)
        else:
            writer.close()
        return {
            "combined_scores_path": COMBINED_SCORES_FILENAME,
            "combined_scores_rows": row_count,
            "combined_scores_run_count": int(len(scores_by_run)),
        }

    def write_manifest(self) -> None:
        """Write the run-to-file score manifest."""
        with self._lock:
            manifest = dict(self._manifest)
        write_json(self._root / "manifest.json", manifest)

    def write_model_metadata(self, metadata: dict[str, Any]) -> None:
        """Write model metadata beside score files."""
        write_json(self._root / "model_meta.json", metadata)
