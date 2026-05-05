"""Tennessee Eastman Process raw dataset source plugin."""

from __future__ import annotations

from pathlib import Path

from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig, DatasetSourceResult
from industrial_tsad_eval.plugins.sources.common import acquire_http, acquire_kaggle, acquire_manual


class TEPDatasetSourcePlugin:
    """Acquire local or optional online TEP raw files."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "tep"

    @property
    def dataset_name(self) -> str:
        """Return the raw dataset directory name."""
        return "TEP"

    def supported_methods(self) -> list[str]:
        """Return supported acquisition methods."""
        return ["manual", "mathworks-http", "kaggle"]

    def describe(self) -> str:
        """Describe accepted raw acquisition inputs."""
        return (
            "Use manual for local TEP CSV folders or archives. mathworks-http downloads a "
            "user-provided HTTP(S) ref and unpacks archives. kaggle requires the optional "
            "acquisition extra and a Kaggle dataset ref."
        )

    def acquire(
        self,
        *,
        target: Path,
        config: DatasetSourceConfig,
    ) -> DatasetSourceResult:
        """Acquire TEP raw files into a target directory."""
        if config.method == "manual":
            return acquire_manual(
                source_name=self.name,
                dataset_name=self.dataset_name,
                target=target,
                config=config,
            )
        if config.method == "mathworks-http":
            return acquire_http(
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
