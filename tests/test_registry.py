from __future__ import annotations

import pytest

from industrial_tsad_eval.domain.errors import PluginNotFoundError
from industrial_tsad_eval.plugins.registry import (
    default_dataset_adapter_registry,
    default_detector_registry,
)


def test_default_detector_registry_contains_forecast_ridge():
    registry = default_detector_registry()

    assert registry.names() == ["forecast-ridge"]
    assert registry.get("forecast-ridge").name == "forecast-ridge"


def test_detector_registry_reports_unknown_plugin():
    registry = default_detector_registry()

    with pytest.raises(PluginNotFoundError, match="Unknown detector plugin"):
        registry.get("missing-detector")


def test_default_dataset_adapter_registry_contains_phase_two_adapters():
    registry = default_dataset_adapter_registry()

    assert registry.names() == ["hai", "hai-cpps", "swat", "tep"]
    assert registry.get_dataset_adapter("tep").dataset_name == "TEP"
    assert registry.get_dataset_adapter("hai-cpps").dataset_name == "HAI-CPPS"


def test_dataset_adapter_registry_reports_unknown_plugin():
    registry = default_dataset_adapter_registry()

    with pytest.raises(PluginNotFoundError, match="Unknown dataset adapter"):
        registry.get_dataset_adapter("missing-dataset")
