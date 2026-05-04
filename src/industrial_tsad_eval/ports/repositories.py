"""Repository ports for filesystem-backed TSAD artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from industrial_tsad_eval.domain.events import GTEvent


class PreparedDatasetRepository(Protocol):
    """Read-only access to a Prepared Format v1 dataset."""

    @property
    def root(self) -> Path:
        """Return the dataset root path."""

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""

    def manifest(self) -> dict[str, Any]:
        """Load `meta/manifest.json`."""

    def schema(self) -> dict[str, Any]:
        """Load `meta/schema.json`."""

    def splits(self) -> dict[str, Any]:
        """Load `meta/splits.json`."""

    def run_ids(self) -> list[str]:
        """Return run IDs discovered from `runs/**/timeseries.parquet`."""

    def read_run(self, run_id: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Read a run timeseries."""

    def read_events(self, event_types: list[str] | None = None) -> list[GTEvent]:
        """Read ground-truth events."""


class ScoreRepository(Protocol):
    """Access to Score Contract v1 artifacts."""

    @property
    def root(self) -> Path:
        """Return the score artifact root path."""

    def discover(self) -> dict[str, Path]:
        """Return score parquet files keyed by run ID."""

    def read_run_scores(self, run_id: str) -> pd.DataFrame:
        """Read score rows for one run."""

    def write_run_scores(self, run_id: str, scores: pd.DataFrame) -> None:
        """Write score rows for one run."""

    def write_manifest(self) -> None:
        """Write a run-to-file manifest."""

    def write_model_metadata(self, metadata: dict[str, Any]) -> None:
        """Write detector metadata."""
