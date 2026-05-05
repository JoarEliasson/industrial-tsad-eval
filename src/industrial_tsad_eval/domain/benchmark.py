"""Benchmark orchestration contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from industrial_tsad_eval.domain.errors import BenchmarkConfigError

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class BenchmarkEvaluationConfig:
    """Evaluation defaults applied to benchmark experiments."""

    threshold_quantile: float = 0.995

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {"threshold_quantile": self.threshold_quantile}


@dataclass(frozen=True)
class BenchmarkDatasetConfig:
    """Prepared dataset entry in a benchmark config."""

    id: str
    prepared: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {"id": self.id, "prepared": self.prepared}


@dataclass(frozen=True)
class BenchmarkDetectorConfig:
    """Detector plugin entry in a benchmark config."""

    id: str
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {"id": self.id, "name": self.name, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class BenchmarkExperiment:
    """Resolved dataset-detector-protocol experiment."""

    experiment_id: str
    dataset: BenchmarkDatasetConfig
    detector: BenchmarkDetectorConfig
    protocol: str


@dataclass(frozen=True)
class BenchmarkExperimentResult:
    """Status and artifacts for one benchmark experiment."""

    experiment_id: str
    dataset: str
    detector: str
    protocol: str
    status: str
    scores_dir: str | None = None
    eval_dir: str | None = None
    threshold: float | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "detector": self.detector,
            "protocol": self.protocol,
            "status": self.status,
            "scores_dir": self.scores_dir,
            "eval_dir": self.eval_dir,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "error": self.error,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    """Resolved benchmark configuration."""

    name: str
    protocols: list[str]
    datasets: list[BenchmarkDatasetConfig]
    detectors: list[BenchmarkDetectorConfig]
    evaluation: BenchmarkEvaluationConfig = field(default_factory=BenchmarkEvaluationConfig)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> BenchmarkConfig:
        """Build a benchmark config from TOML-compatible data."""
        benchmark = _required_mapping(payload, "benchmark")
        name = _required_string(benchmark, "name", "benchmark.name")
        protocols = _required_string_list(benchmark, "protocols", "benchmark.protocols")
        evaluation = _evaluation_config(benchmark.get("evaluation", {}))
        datasets = _dataset_configs(payload.get("datasets"))
        detectors = _detector_configs(payload.get("detectors"))
        _ensure_unique("dataset", [item.id for item in datasets])
        _ensure_unique("detector", [item.id for item in detectors])
        for protocol in protocols:
            _validate_safe_id(protocol, "protocol")
        return cls(
            name=name,
            protocols=protocols,
            datasets=datasets,
            detectors=detectors,
            evaluation=evaluation,
        )

    def experiments(self) -> list[BenchmarkExperiment]:
        """Expand the benchmark matrix in deterministic order."""
        experiments: list[BenchmarkExperiment] = []
        for dataset_config in self.datasets:
            for detector_config in self.detectors:
                for protocol in self.protocols:
                    experiments.append(
                        BenchmarkExperiment(
                            experiment_id=experiment_id(
                                dataset_config.id,
                                detector_config.id,
                                protocol,
                            ),
                            dataset=dataset_config,
                            detector=detector_config,
                            protocol=protocol,
                        )
                    )
        return experiments

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "benchmark": {
                "name": self.name,
                "protocols": list(self.protocols),
                "evaluation": self.evaluation.to_dict(),
            },
            "datasets": [dataset_config.to_dict() for dataset_config in self.datasets],
            "detectors": [detector_config.to_dict() for detector_config in self.detectors],
        }


def experiment_id(dataset_id: str, detector_id: str, protocol: str) -> str:
    """Return the stable benchmark experiment id."""
    return f"{dataset_id}__{detector_id}__{protocol}"


def sanitize_run_id(value: str) -> str:
    """Convert a benchmark name into a filesystem-safe run id prefix."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return sanitized or "benchmark"


def _evaluation_config(payload: object) -> BenchmarkEvaluationConfig:
    if payload is None:
        return BenchmarkEvaluationConfig()
    if not isinstance(payload, dict):
        raise BenchmarkConfigError("benchmark.evaluation must be a table.")
    quantile = float(payload.get("threshold_quantile", 0.995))
    if not 0.0 < quantile < 1.0:
        raise BenchmarkConfigError("benchmark.evaluation.threshold_quantile must be in (0, 1).")
    return BenchmarkEvaluationConfig(threshold_quantile=quantile)


def _dataset_configs(payload: object) -> list[BenchmarkDatasetConfig]:
    if not isinstance(payload, list) or not payload:
        raise BenchmarkConfigError("datasets must be a non-empty array of tables.")
    datasets: list[BenchmarkDatasetConfig] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise BenchmarkConfigError(f"datasets[{index}] must be a table.")
        item_id = _required_string(item, "id", f"datasets[{index}].id")
        prepared = _required_string(item, "prepared", f"datasets[{index}].prepared")
        _validate_safe_id(item_id, f"datasets[{index}].id")
        datasets.append(BenchmarkDatasetConfig(id=item_id, prepared=prepared))
    return datasets


def _detector_configs(payload: object) -> list[BenchmarkDetectorConfig]:
    if not isinstance(payload, list) or not payload:
        raise BenchmarkConfigError("detectors must be a non-empty array of tables.")
    detectors: list[BenchmarkDetectorConfig] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise BenchmarkConfigError(f"detectors[{index}] must be a table.")
        item_id = _required_string(item, "id", f"detectors[{index}].id")
        name = _required_string(item, "name", f"detectors[{index}].name")
        parameters = item.get("parameters", {})
        if not isinstance(parameters, dict):
            raise BenchmarkConfigError(f"detectors[{index}].parameters must be an object.")
        _validate_safe_id(item_id, f"detectors[{index}].id")
        detectors.append(BenchmarkDetectorConfig(id=item_id, name=name, parameters=parameters))
    return detectors


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"{key} must be a table.")
    return value


def _required_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkConfigError(f"{label} must be a non-empty string.")
    return value.strip()


def _required_string_list(payload: dict[str, Any], key: str, label: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise BenchmarkConfigError(f"{label} must be a non-empty string list.")
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise BenchmarkConfigError(f"{label}[{index}] must be a non-empty string.")
        output.append(item.strip())
    return output


def _ensure_unique(label: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise BenchmarkConfigError(f"Duplicate {label} ids: {duplicates}.")


def _validate_safe_id(value: str, label: str) -> None:
    if not _SAFE_ID_PATTERN.match(value):
        raise BenchmarkConfigError(
            f"{label} {value!r} may only contain letters, numbers, '.', '_', and '-'."
        )
