"""Plugin registry implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from industrial_tsad_eval.domain.errors import PluginNotFoundError
from industrial_tsad_eval.ports.dataset_adapters import DatasetAdapterPlugin
from industrial_tsad_eval.ports.detectors import DetectorPlugin


@dataclass
class DetectorRegistry:
    """In-memory registry for detector plugins."""

    _plugins: dict[str, DetectorPlugin] = field(default_factory=dict)

    def register(self, plugin: DetectorPlugin) -> None:
        """Register or replace a detector plugin by stable name."""
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> DetectorPlugin:
        """Return a registered detector plugin."""
        try:
            return self._plugins[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._plugins)) or "<none>"
            raise PluginNotFoundError(
                f"Unknown detector plugin {name!r}. Available plugins: {available}."
            ) from exc

    def names(self) -> list[str]:
        """Return registered detector names."""
        return sorted(self._plugins)


@dataclass
class DatasetAdapterRegistry:
    """In-memory registry for dataset adapter plugins."""

    _plugins: dict[str, DatasetAdapterPlugin] = field(default_factory=dict)

    def register(self, plugin: DatasetAdapterPlugin) -> None:
        """Register or replace a dataset adapter by stable name."""
        self._plugins[plugin.name] = plugin

    def get_dataset_adapter(self, name: str) -> DatasetAdapterPlugin:
        """Return a registered dataset adapter plugin."""
        try:
            return self._plugins[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._plugins)) or "<none>"
            raise PluginNotFoundError(
                f"Unknown dataset adapter {name!r}. Available adapters: {available}."
            ) from exc

    def names(self) -> list[str]:
        """Return registered dataset adapter names."""
        return sorted(self._plugins)


def default_detector_registry() -> DetectorRegistry:
    """Create the default detector registry."""
    from industrial_tsad_eval.plugins.forecast_ridge import ForecastRidgePlugin
    from industrial_tsad_eval.plugins.torch_detectors import (
        DRAPlugin,
        DRCADPlugin,
        ForecastLSTMPlugin,
        InterFusionPlugin,
    )

    registry = DetectorRegistry()
    registry.register(ForecastRidgePlugin())
    registry.register(ForecastLSTMPlugin())
    registry.register(DRAPlugin())
    registry.register(InterFusionPlugin())
    registry.register(DRCADPlugin())
    return registry


def default_dataset_adapter_registry() -> DatasetAdapterRegistry:
    """Create the default dataset adapter registry."""
    from industrial_tsad_eval.plugins.datasets.hai import HAIDatasetAdapterPlugin
    from industrial_tsad_eval.plugins.datasets.hai_cpps import HAICPPSDatasetAdapterPlugin
    from industrial_tsad_eval.plugins.datasets.swat import SWaTDatasetAdapterPlugin
    from industrial_tsad_eval.plugins.datasets.tep import TEPDatasetAdapterPlugin

    registry = DatasetAdapterRegistry()
    registry.register(TEPDatasetAdapterPlugin())
    registry.register(SWaTDatasetAdapterPlugin())
    registry.register(HAIDatasetAdapterPlugin())
    registry.register(HAICPPSDatasetAdapterPlugin())
    return registry
