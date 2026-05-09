"""Benchmark orchestration contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from industrial_tsad_eval.domain.errors import BenchmarkConfigError
from industrial_tsad_eval.domain.policy import EvalPolicy

_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class BenchmarkEvaluationConfig:
    """Evaluation defaults applied to benchmark experiments."""

    threshold_quantile: float = 0.995
    policy: EvalPolicy = field(default_factory=EvalPolicy)
    dataset_policies: dict[str, EvalPolicy] = field(default_factory=dict)

    def policy_for(self, dataset: str, protocol: str) -> EvalPolicy:
        """Return the effective evaluation policy for a dataset/protocol pair."""
        base = self.policy.to_dict()
        override = self.dataset_policies.get(dataset)
        if override is not None:
            base.update(override.to_dict())
        base["dataset"] = dataset
        base["protocol"] = protocol
        base["threshold_quantile"] = self.threshold_quantile
        return EvalPolicy.from_dict(base)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "threshold_quantile": self.threshold_quantile,
            "policy": self.policy.to_dict(),
            "dataset_policies": {
                dataset: policy.to_dict()
                for dataset, policy in sorted(self.dataset_policies.items())
            },
        }


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
    datasets: list[str] | None = None
    protocols: list[str] | None = None
    parameter_overrides: list[BenchmarkParameterOverride] = field(default_factory=list)

    def for_experiment(self, dataset: str, protocol: str) -> BenchmarkDetectorConfig:
        """Return a detector config with matching parameter overrides applied."""
        parameters = dict(self.parameters)
        for override in self.parameter_overrides:
            if override.matches(dataset, protocol):
                parameters.update(override.parameters)
        return BenchmarkDetectorConfig(
            id=self.id,
            name=self.name,
            parameters=parameters,
            datasets=list(self.datasets) if self.datasets is not None else None,
            protocols=list(self.protocols) if self.protocols is not None else None,
            parameter_overrides=list(self.parameter_overrides),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "parameters": dict(self.parameters),
        }
        if self.datasets is not None:
            payload["datasets"] = list(self.datasets)
        if self.protocols is not None:
            payload["protocols"] = list(self.protocols)
        if self.parameter_overrides:
            payload["parameter_overrides"] = [
                override.to_dict() for override in self.parameter_overrides
            ]
        return payload


@dataclass(frozen=True)
class BenchmarkParameterOverride:
    """Detector parameter override for a selected dataset/protocol subset."""

    parameters: dict[str, Any]
    dataset: str | None = None
    protocol: str | None = None

    def matches(self, dataset: str, protocol: str) -> bool:
        """Return true when this override applies to a benchmark experiment."""
        dataset_matches = self.dataset is None or self.dataset == dataset
        protocol_matches = self.protocol is None or self.protocol == protocol
        return dataset_matches and protocol_matches

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        payload: dict[str, Any] = {"parameters": dict(self.parameters)}
        if self.dataset is not None:
            payload["dataset"] = self.dataset
        if self.protocol is not None:
            payload["protocol"] = self.protocol
        return payload


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
        _validate_detector_filters(detectors, datasets, protocols)
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
                if detector_config.datasets is not None and dataset_config.id not in set(
                    detector_config.datasets
                ):
                    continue
                protocols = detector_config.protocols or self.protocols
                for protocol in protocols:
                    experiments.append(
                        BenchmarkExperiment(
                            experiment_id=experiment_id(
                                dataset_config.id,
                                detector_config.id,
                                protocol,
                            ),
                            dataset=dataset_config,
                            detector=detector_config.for_experiment(
                                dataset_config.id,
                                protocol,
                            ),
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
    policy = EvalPolicy.from_dict(
        _optional_mapping(payload, "policy", "benchmark.evaluation.policy")
    )
    dataset_policy_payload = payload.get("dataset_policies", {})
    if not isinstance(dataset_policy_payload, dict):
        raise BenchmarkConfigError("benchmark.evaluation.dataset_policies must be a table.")
    dataset_policies: dict[str, EvalPolicy] = {}
    for dataset, policy_payload in dataset_policy_payload.items():
        if not _is_mapping(policy_payload, f"benchmark.evaluation.dataset_policies.{dataset}"):
            continue
        merged = policy.to_dict()
        merged.update(dict(policy_payload))
        dataset_policies[str(dataset)] = EvalPolicy.from_dict(merged)
    return BenchmarkEvaluationConfig(
        threshold_quantile=quantile,
        policy=policy,
        dataset_policies=dataset_policies,
    )


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
        datasets = _optional_string_list(item, "datasets", f"detectors[{index}].datasets")
        protocols = _optional_string_list(item, "protocols", f"detectors[{index}].protocols")
        parameter_overrides = _parameter_overrides(
            item.get("parameter_overrides", []),
            f"detectors[{index}].parameter_overrides",
        )
        _validate_safe_id(item_id, f"detectors[{index}].id")
        detectors.append(
            BenchmarkDetectorConfig(
                id=item_id,
                name=name,
                parameters=parameters,
                datasets=datasets,
                protocols=protocols,
                parameter_overrides=parameter_overrides,
            )
        )
    return detectors


def _parameter_overrides(payload: object, label: str) -> list[BenchmarkParameterOverride]:
    if payload is None:
        return []
    if not isinstance(payload, list):
        raise BenchmarkConfigError(f"{label} must be an array of tables.")
    overrides: list[BenchmarkParameterOverride] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise BenchmarkConfigError(f"{label}[{index}] must be a table.")
        parameters = item.get("parameters", {})
        if not isinstance(parameters, dict):
            raise BenchmarkConfigError(f"{label}[{index}].parameters must be an object.")
        dataset = _optional_string(item.get("dataset"))
        protocol = _optional_string(item.get("protocol"))
        if dataset is not None:
            _validate_safe_id(dataset, f"{label}[{index}].dataset")
        if protocol is not None:
            _validate_safe_id(protocol, f"{label}[{index}].protocol")
        overrides.append(
            BenchmarkParameterOverride(
                dataset=dataset,
                protocol=protocol,
                parameters=dict(parameters),
            )
        )
    return overrides


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"{key} must be a table.")
    return value


def _optional_mapping(payload: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"{label} must be a table.")
    return dict(value)


def _is_mapping(value: object, label: str) -> bool:
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"{label} must be a table.")
    return True


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


def _optional_string_list(
    payload: dict[str, Any],
    key: str,
    label: str,
) -> list[str] | None:
    if key not in payload:
        return None
    return _required_string_list(payload, key, label)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _validate_detector_filters(
    detectors: list[BenchmarkDetectorConfig],
    datasets: list[BenchmarkDatasetConfig],
    protocols: list[str],
) -> None:
    dataset_ids = {dataset.id for dataset in datasets}
    protocol_ids = set(protocols)
    for detector in detectors:
        for dataset_id in detector.datasets or []:
            _validate_safe_id(dataset_id, f"detectors[{detector.id}].datasets")
            if dataset_id not in dataset_ids:
                raise BenchmarkConfigError(
                    f"Detector {detector.id!r} references unknown dataset {dataset_id!r}."
                )
        for protocol in detector.protocols or []:
            _validate_safe_id(protocol, f"detectors[{detector.id}].protocols")
            if protocol not in protocol_ids:
                raise BenchmarkConfigError(
                    f"Detector {detector.id!r} references unknown protocol {protocol!r}."
                )
        for override in detector.parameter_overrides:
            if override.dataset is not None and override.dataset not in dataset_ids:
                raise BenchmarkConfigError(
                    f"Detector {detector.id!r} override references unknown dataset "
                    f"{override.dataset!r}."
                )
            if override.protocol is not None and override.protocol not in protocol_ids:
                raise BenchmarkConfigError(
                    f"Detector {detector.id!r} override references unknown protocol "
                    f"{override.protocol!r}."
                )


def _ensure_unique(label: str, values: list[str]) -> None:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise BenchmarkConfigError(f"Duplicate {label} ids: {duplicates}.")


def _validate_safe_id(value: str, label: str) -> None:
    if not _SAFE_ID_PATTERN.match(value):
        raise BenchmarkConfigError(
            f"{label} {value!r} may only contain letters, numbers, '.', '_', and '-'."
        )
