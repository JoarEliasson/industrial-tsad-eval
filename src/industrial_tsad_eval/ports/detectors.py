"""Detector plugin ports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd

from industrial_tsad_eval.ports.repositories import PreparedDatasetRepository


@dataclass(frozen=True)
class DetectorRunConfig:
    """Detector configuration supplied by application services."""

    parameters: dict[str, Any] = field(default_factory=dict)


class Detector(Protocol):
    """Fitted detector interface used by application services."""

    def train(self, repository: PreparedDatasetRepository, protocol: str) -> None:
        """Fit the detector against the prepared dataset."""

    def score_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Return Score Contract v1 rows for a single run."""

    def metadata(self) -> dict[str, Any]:
        """Return JSON-compatible model metadata."""


class DetectorExplainer(Protocol):
    """Optional fitted detector interface for native explanation artifacts."""

    def explain_run(self, repository: PreparedDatasetRepository, run_id: str) -> pd.DataFrame:
        """Return ranked explanation rows for a single run."""


class DetectorPlugin(Protocol):
    """Factory and metadata interface for a detector plugin."""

    @property
    def name(self) -> str:
        """Return the stable plugin name."""

    def create(self, config: DetectorRunConfig) -> Detector:
        """Create an unfitted detector instance."""
