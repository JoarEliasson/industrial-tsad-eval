from __future__ import annotations

import pytest

from industrial_tsad_eval.domain.errors import PluginNotFoundError
from industrial_tsad_eval.plugins.registry import default_detector_registry


def test_default_detector_registry_contains_forecast_ridge():
    registry = default_detector_registry()

    assert registry.names() == ["forecast-ridge"]
    assert registry.get("forecast-ridge").name == "forecast-ridge"


def test_detector_registry_reports_unknown_plugin():
    registry = default_detector_registry()

    with pytest.raises(PluginNotFoundError, match="Unknown detector plugin"):
        registry.get("missing-detector")
