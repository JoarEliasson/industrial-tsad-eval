"""TOML benchmark configuration I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkDetectorConfig,
    BenchmarkParameterOverride,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10.
    import tomli as tomllib


DEFAULT_BENCHMARK_CONFIG_TOML = """[benchmark]
name = "opcua-smoke"
protocols = ["naive"]

[benchmark.evaluation]
threshold_quantile = 0.995

[[datasets]]
id = "opcua"
prepared = "examples/generated/OPCUA_SYNTH"

[[detectors]]
id = "forecast-ridge-default"
name = "forecast-ridge"
parameters = { window = 32, stride = 4, lags = 1, alpha = 1.0, standardize = true, seed = 1337 }
"""


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load and validate a TOML benchmark config."""
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config = BenchmarkConfig.from_mapping(payload)
    return _resolve_prepared_paths(config, config_path.parent)


def write_default_benchmark_config(path: str | Path) -> Path:
    """Write the default benchmark config template."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(DEFAULT_BENCHMARK_CONFIG_TOML, encoding="utf-8")
    return output


def render_benchmark_config_toml(config: BenchmarkConfig) -> str:
    """Render a benchmark config for artifact capture."""
    lines = [
        "[benchmark]",
        f'name = "{_toml_string(config.name)}"',
        "protocols = [" + ", ".join(f'"{_toml_string(item)}"' for item in config.protocols) + "]",
        "",
        "[benchmark.evaluation]",
        f"threshold_quantile = {config.evaluation.threshold_quantile}",
        "",
        "[benchmark.evaluation.policy]",
        *_policy_lines(config.evaluation.policy.to_dict()),
        "",
    ]
    for dataset, policy in sorted(config.evaluation.dataset_policies.items()):
        lines.extend(
            [
                f'[benchmark.evaluation.dataset_policies."{_toml_string(dataset)}"]',
                *_policy_lines(policy.to_dict()),
                "",
            ]
        )
    for dataset_config in config.datasets:
        lines.extend(
            [
                "[[datasets]]",
                f'id = "{_toml_string(dataset_config.id)}"',
                f'prepared = "{_toml_string(dataset_config.prepared)}"',
                "",
            ]
        )
    for detector_config in config.detectors:
        lines.extend(
            [
                "[[detectors]]",
                f'id = "{_toml_string(detector_config.id)}"',
                f'name = "{_toml_string(detector_config.name)}"',
            ]
        )
        if detector_config.datasets is not None:
            lines.append(f"datasets = {_string_list(detector_config.datasets)}")
        if detector_config.protocols is not None:
            lines.append(f"protocols = {_string_list(detector_config.protocols)}")
        lines.extend(
            [
                f"parameters = {_inline_table(detector_config.parameters)}",
            ]
        )
        for override in detector_config.parameter_overrides:
            lines.extend(_override_lines(override))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _resolve_prepared_paths(config: BenchmarkConfig, base_dir: Path) -> BenchmarkConfig:
    datasets = [
        BenchmarkDatasetConfig(
            id=dataset_config.id,
            prepared=str(_resolve_path(dataset_config.prepared, base_dir)),
        )
        for dataset_config in config.datasets
    ]
    detectors = [
        BenchmarkDetectorConfig(
            id=detector_config.id,
            name=detector_config.name,
            parameters=dict(detector_config.parameters),
            datasets=list(detector_config.datasets)
            if detector_config.datasets is not None
            else None,
            protocols=list(detector_config.protocols)
            if detector_config.protocols is not None
            else None,
            parameter_overrides=list(detector_config.parameter_overrides),
        )
        for detector_config in config.detectors
    ]
    return BenchmarkConfig(
        name=config.name,
        protocols=list(config.protocols),
        datasets=datasets,
        detectors=detectors,
        evaluation=config.evaluation,
    )


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _inline_table(payload: dict[str, Any]) -> str:
    if not payload:
        return "{}"
    parts = [f"{key} = {_toml_value(value)}" for key, value in sorted(payload.items())]
    return "{ " + ", ".join(parts) + " }"


def _override_lines(override: BenchmarkParameterOverride) -> list[str]:
    lines = ["[[detectors.parameter_overrides]]"]
    if override.dataset is not None:
        lines.append(f'dataset = "{_toml_string(override.dataset)}"')
    if override.protocol is not None:
        lines.append(f'protocol = "{_toml_string(override.protocol)}"')
    lines.append(f"parameters = {_inline_table(override.parameters)}")
    return lines


def _policy_lines(payload: dict[str, Any]) -> list[str]:
    omitted = {"policy_version", "dataset", "protocol"}
    return [
        f"{key} = {_toml_value(value)}"
        for key, value in sorted(payload.items())
        if key not in omitted and value not in (None, [], {})
    ]


def _string_list(values: list[str]) -> str:
    return "[" + ", ".join(f'"{_toml_string(value)}"' for value in values) + "]"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return f'"{_toml_string(value)}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML parameter value: {value!r}")


def _toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
