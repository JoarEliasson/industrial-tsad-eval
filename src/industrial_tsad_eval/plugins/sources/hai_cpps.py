"""HAI-CPPS raw dataset source plugin."""

from __future__ import annotations

from pathlib import Path

from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig, DatasetSourceResult
from industrial_tsad_eval.plugins.sources.common import acquire_manual


class HAICPPSDatasetSourcePlugin:
    """Acquire user-provided HAI-CPPS scenario files."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "hai-cpps"

    @property
    def dataset_name(self) -> str:
        """Return the raw dataset directory name."""
        return "HAI_CPPS"

    def supported_methods(self) -> list[str]:
        """Return supported acquisition methods."""
        return ["manual"]

    def describe(self) -> str:
        """Describe accepted raw acquisition inputs."""
        return (
            "Use manual for local HAI-CPPS scenario directories or archives. IEEE/request-gated "
            "downloads and simulator execution are intentionally outside this acquisition helper."
        )

    def acquire(
        self,
        *,
        target: Path,
        config: DatasetSourceConfig,
    ) -> DatasetSourceResult:
        """Acquire HAI-CPPS raw files into a target directory."""
        return acquire_manual(
            source_name=self.name,
            dataset_name=self.dataset_name,
            target=target,
            config=config,
        )
