from __future__ import annotations

import pytest

from industrial_tsad_eval.domain.errors import PluginNotFoundError
from industrial_tsad_eval.plugins.registry import (
    default_dataset_adapter_registry,
    default_detector_registry,
)


def test_detector_registry_looks_up_forecast_ridge_and_rejects_unknown():
    registry = default_detector_registry()

    assert registry.get("forecast-ridge").name == "forecast-ridge"
    with pytest.raises(PluginNotFoundError):
        registry.get("missing")


def test_dataset_adapter_registry_exposes_phase_two_plugins():
    registry = default_dataset_adapter_registry()

    assert registry.names() == ["hai", "hai-cpps", "swat", "tep"]
    assert registry.get_dataset_adapter("swat").dataset_name == "SWaT"
    with pytest.raises(PluginNotFoundError):
        registry.get_dataset_adapter("missing")
