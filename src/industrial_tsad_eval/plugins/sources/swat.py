"""SWaT raw dataset source plugin."""

from __future__ import annotations

from pathlib import Path

from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig, DatasetSourceResult
from industrial_tsad_eval.plugins.sources.common import acquire_kaggle, acquire_manual


class SWaTDatasetSourcePlugin:
    """Acquire local or optional Kaggle SWaT raw files."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "swat"

    @property
    def dataset_name(self) -> str:
        """Return the raw dataset directory name."""
        return "SWaT"

    def supported_methods(self) -> list[str]:
        """Return supported acquisition methods."""
        return ["manual", "kaggle"]

    def describe(self) -> str:
        """Describe accepted raw acquisition inputs."""
        return (
            "Use manual for local SWaT files obtained through approved channels. Kaggle is "
            "optional for user-provided dataset refs; official iTrust access remains manual."
        )

    def acquire(
        self,
        *,
        target: Path,
        config: DatasetSourceConfig,
    ) -> DatasetSourceResult:
        """Acquire SWaT raw files into a target directory."""
        if config.method == "manual":
            return acquire_manual(
                source_name=self.name,
                dataset_name=self.dataset_name,
                target=target,
                config=config,
            )
        return acquire_kaggle(
            source_name=self.name,
            dataset_name=self.dataset_name,
            target=target,
            config=config,
        )
