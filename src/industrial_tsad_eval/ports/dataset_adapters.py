"""Dataset adapter plugin ports."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult


class DatasetAdapterPlugin(Protocol):
    """Raw-to-prepared dataset adapter plugin interface."""

    @property
    def name(self) -> str:
        """Return the stable adapter registry name."""

    @property
    def dataset_name(self) -> str:
        """Return the prepared dataset directory name."""

    def describe_expected_raw_layout(self) -> str:
        """Describe the local raw data layout expected by the adapter."""

    def prepare(
        self,
        *,
        raw: Path,
        prepared: Path,
        config: DatasetAdapterConfig,
    ) -> DatasetAdapterResult:
        """Write a Prepared Format v1 dataset and return a summary."""
