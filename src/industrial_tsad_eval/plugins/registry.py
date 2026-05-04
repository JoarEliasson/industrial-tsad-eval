"""Plugin registry implementation."""

from __future__ import annotations

from dataclasses import dataclass, field

from industrial_tsad_eval.domain.errors import PluginNotFoundError
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


def default_detector_registry() -> DetectorRegistry:
    """Create the default detector registry."""
    from industrial_tsad_eval.plugins.forecast_ridge import ForecastRidgePlugin

    registry = DetectorRegistry()
    registry.register(ForecastRidgePlugin())
    return registry
