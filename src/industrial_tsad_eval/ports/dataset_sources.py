"""Raw dataset source plugin ports."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig, DatasetSourceResult


class DatasetSourcePlugin(Protocol):
    """Raw dataset acquisition plugin interface."""

    @property
    def name(self) -> str:
        """Return the stable source registry name."""

    @property
    def dataset_name(self) -> str:
        """Return the raw dataset directory name."""

    def supported_methods(self) -> list[str]:
        """Return supported acquisition method names."""

    def describe(self) -> str:
        """Describe supported raw acquisition paths for this source."""

    def acquire(
        self,
        *,
        target: Path,
        config: DatasetSourceConfig,
    ) -> DatasetSourceResult:
        """Materialize raw files into a target directory and return a summary."""
