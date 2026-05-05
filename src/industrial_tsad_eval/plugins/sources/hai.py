"""HAI raw dataset source plugin."""

from __future__ import annotations

from pathlib import Path

from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig, DatasetSourceResult
from industrial_tsad_eval.plugins.sources.common import acquire_git, acquire_kaggle, acquire_manual


class HAIDatasetSourcePlugin:
    """Acquire local, Kaggle, or git-hosted HAI raw files."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "hai"

    @property
    def dataset_name(self) -> str:
        """Return the raw dataset directory name."""
        return "HAI"

    def supported_methods(self) -> list[str]:
        """Return supported acquisition methods."""
        return ["manual", "kaggle", "git"]

    def describe(self) -> str:
        """Describe accepted raw acquisition inputs."""
        return (
            "Use manual for local HAI CSV/parquet layouts, kaggle for optional Kaggle refs, "
            "or git for public repository mirrors that contain raw tables."
        )

    def acquire(
        self,
        *,
        target: Path,
        config: DatasetSourceConfig,
    ) -> DatasetSourceResult:
        """Acquire HAI raw files into a target directory."""
        if config.method == "manual":
            return acquire_manual(
                source_name=self.name,
                dataset_name=self.dataset_name,
                target=target,
                config=config,
            )
        if config.method == "git":
            return acquire_git(
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
