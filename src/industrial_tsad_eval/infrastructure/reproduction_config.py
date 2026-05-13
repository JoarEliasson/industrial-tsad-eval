"""TOML configuration I/O for thesis-style reproduction and assistant replay."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from industrial_tsad_eval.domain.assistant_replay import (
    THESIS_ASSISTANT_QUERY_TEMPLATE,
    AssistantReplayConfig,
)
from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkDetectorConfig,
    BenchmarkParameterOverride,
)
from industrial_tsad_eval.domain.reproduction import ReproductionConfig, ReproductionReuseConfig

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
    import tomli as tomllib


THESIS_ASSISTANT_QUERY_TOML = THESIS_ASSISTANT_QUERY_TEMPLATE.replace("\\", "\\\\").replace(
    '"', '\\"'
)
DRA_THESIS_PARAMS = (
    "{ window = 100, train_stride = 10, score_stride = 1, "
    "max_train_windows = 5000, epochs = 30, batch_size = 64, "
    'seed = 1337, device = "auto", evidence_explanation_source = "native", '
    'evidence_native_missing_policy = "skip_bundle" }'
)
INTERFUSION_THESIS_PARAMS = (
    "{ window = 100, train_stride = 10, score_stride = 1, latent_dim = 3, "
    "kl_warmup = 10, max_train_windows = 5000, epochs = 30, batch_size = 64, "
    'seed = 1337, device = "auto", evidence_explanation_source = "native", '
    'evidence_native_missing_policy = "skip_bundle" }'
)
DRCAD_THESIS_PARAMS = (
    "{ window = 100, train_stride = 10, score_stride = 1, patch_size = 5, "
    "d_model = 64, n_heads = 4, n_layers = 2, mlp_dim = 128, lr = 0.0001, "
    'epochs = 10, batch_size = 64, seed = 1337, device = "auto", '
    'evidence_explanation_source = "native", evidence_native_missing_policy = "skip_bundle" }'
)

THESIS_SMOKE_CONFIG_TOML = """[reproduction]
name = "thesis-smoke"
run_evidence = true
run_xai = true
run_profiles = false
run_assistant = true
xai_ks = [1, 3, 5]
evidence_sources = ["oracle"]
assistant_evidence_source = "oracle"
profile_experiment_limit = 1

[reproduction.resources]
cpu_threads = 12
memory_limit_gb = 16
benchmark_workers = "auto"
evidence_workers = "auto"
xai_workers = "auto"
assistant_workers = "conservative"
gpu_slots = 1
profile_mode = "inline"
require_cuda_for_torch = true
require_llama_gpu = false

[benchmark]
name = "thesis-smoke"
protocols = ["naive"]

[benchmark.evaluation]
threshold_quantile = 0.995

[[datasets]]
id = "opcua"
prepared = "../examples/generated/OPCUA_SYNTH"

[[detectors]]
id = "forecast-ridge-smoke"
name = "forecast-ridge"
parameters = { window = 24, stride = 4, lags = 1, alpha = 1.0, standardize = true, seed = 1337 }

[assistant]
suite_id = "thesis-smoke-assistant"
prepared = "../examples/generated/OPCUA_SYNTH"
cases_per_dataset = 2
top_k = 6
minimum_supported_claims = 1
prompt_budget_chars = 8000
query_template = "__THESIS_assistant replay_QUERY__"

[assistant.provider]
name = "fake"
model = "fake-assistant"
""".replace("__THESIS_assistant replay_QUERY__", THESIS_ASSISTANT_QUERY_TOML)


THESIS_VERIFICATION_CONFIG_TOML = """[reproduction]
name = "thesis-verification"
run_evidence = true
run_xai = true
run_profiles = true
run_assistant = true
xai_ks = [1, 3, 5]
evidence_sources = ["oracle"]
assistant_evidence_source = "oracle"
profile_experiment_limit = 1

[reproduction.resources]
cpu_threads = 12
memory_limit_gb = 16
benchmark_workers = "auto"
evidence_workers = "auto"
xai_workers = "auto"
assistant_workers = "conservative"
gpu_slots = 1
profile_mode = "inline"
require_cuda_for_torch = true
require_llama_gpu = true

[benchmark]
name = "thesis-verification"
protocols = ["naive", "all_in_one", "zero_shot"]

[benchmark.evaluation]
threshold_quantile = 0.995

[[datasets]]
id = "TEP"
prepared = "../prepared/TEP"

[[datasets]]
id = "SWaT"
prepared = "../prepared/SWaT"

[[datasets]]
id = "HAI"
prepared = "../prepared/HAI"

[[datasets]]
id = "HAI-CPPS"
prepared = "../prepared/HAI_CPPS"

[[detectors]]
id = "forecast-ridge"
name = "forecast-ridge"
parameters = { window = 32, stride = 4, lags = 1, alpha = 1.0, standardize = true, seed = 1337 }

[[detectors]]
id = "forecast-lstm-tiny"
name = "forecast-lstm"
datasets = ["SWaT", "HAI", "HAI-CPPS"]
protocols = ["naive"]
parameters = { window = 32, max_train_windows = 256, epochs = 1, device = "auto" }

[[detectors]]
id = "dra-tiny"
name = "dra"
datasets = ["SWaT"]
protocols = ["naive"]
parameters = { window = 32, max_train_windows = 128, epochs = 1, batch_size = 16, device = "auto" }

[[detectors]]
id = "interfusion-tiny"
name = "interfusion"
datasets = ["SWaT"]
protocols = ["naive"]
parameters = { window = 32, max_train_windows = 128, epochs = 1, batch_size = 16, device = "auto" }

[[detectors]]
id = "drcad-tiny"
name = "drcad"
datasets = ["SWaT"]
protocols = ["naive"]
parameters = { window = 32, patch_size = 8, max_train_windows = 128, epochs = 1, device = "auto" }

[assistant]
suite_id = "thesis-verification-assistant"
prepared = "../prepared/TEP"
cases_per_dataset = 1
top_k = 8
minimum_supported_claims = 1
prompt_budget_chars = 12000
query_template = "__THESIS_assistant replay_QUERY__"

[assistant.provider]
name = "llama-cpp"
model = "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
base_url = "http://127.0.0.1:8080/v1"
timeout_s = 180.0
temperature = 0.0
top_p = 1.0
max_tokens = 700
seed = 1337

[assistant.provider.extra]
structured_output_mode = "json_object"
gpu_offload_required = true
gpu_layers = -1
""".replace("__THESIS_assistant replay_QUERY__", THESIS_ASSISTANT_QUERY_TOML)


THESIS_FULL_CONFIG_TOML = (
    """[reproduction]
name = "thesis-full"
run_evidence = true
run_xai = true
run_profiles = true
run_assistant = true
xai_ks = [1, 3, 5]
evidence_sources = ["oracle", "operational"]
assistant_evidence_source = "operational"
profile_experiment_limit = 0

[reproduction.resources]
cpu_threads = 12
memory_limit_gb = 16
benchmark_workers = "auto"
evidence_workers = "auto"
xai_workers = "auto"
assistant_workers = "conservative"
gpu_slots = 1
profile_mode = "inline"
require_cuda_for_torch = true
require_llama_gpu = true

[benchmark]
name = "thesis-full"
protocols = ["naive", "all_in_one", "zero_shot"]

[benchmark.evaluation]
threshold_quantile = 0.995

[benchmark.evaluation.policy]
merge_gap_s = 10.0
grace_s = 5.0
threshold_quantile = 0.995
compute = ["event", "delay", "far", "point", "point_adjusted", "affiliation"]
event_types = ["attack", "fault", "anomaly"]

[benchmark.evaluation.dataset_policies.TEP]
merge_gap_mode = "auto_period"
merge_gap_skipped_samples = 1
merge_gap_jitter_ratio = 0.1

[benchmark.evaluation.dataset_policies.HAI]
merge_gap_s = 30.0
grace_s = 120.0

[benchmark.evaluation.dataset_policies."HAI-CPPS"]
grace_s = 6.0

[[datasets]]
id = "TEP"
prepared = "../prepared/TEP"

[[datasets]]
id = "SWaT"
prepared = "../prepared/SWaT"

[[datasets]]
id = "HAI"
prepared = "../prepared/HAI"

[[datasets]]
id = "HAI-CPPS"
prepared = "../prepared/HAI_CPPS"

[[detectors]]
id = "forecast-ridge"
name = "forecast-ridge"
parameters = { window = 32, stride = 4, lags = 1, alpha = 1.0, standardize = true, seed = 1337 }

[[detectors]]
id = "dra"
name = "dra"
parameters = __DRA_THESIS_PARAMS__

[[detectors.parameter_overrides]]
dataset = "TEP"
protocol = "naive"
parameters = { train_stride = 5 }

[[detectors]]
id = "interfusion"
name = "interfusion"
parameters = __INTERFUSION_THESIS_PARAMS__

[[detectors.parameter_overrides]]
dataset = "TEP"
protocol = "naive"
parameters = { train_stride = 5 }

[[detectors]]
id = "drcad"
name = "drcad"
parameters = __DRCAD_THESIS_PARAMS__

[[detectors.parameter_overrides]]
dataset = "TEP"
protocol = "naive"
parameters = { train_stride = 5 }

[assistant]
suite_id = "thesis-assistant-all-datasets"
prepared = "../prepared/TEP"
cases_per_dataset = 4
top_k = 8
minimum_supported_claims = 1
prompt_budget_chars = 12000
query_template = "__THESIS_assistant replay_QUERY__"

[assistant.provider]
name = "llama-cpp"
model = "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
base_url = "http://127.0.0.1:8080/v1"
timeout_s = 180.0
temperature = 0.0
top_p = 1.0
max_tokens = 700
seed = 1337

[assistant.provider.extra]
structured_output_mode = "json_object"
gpu_offload_required = true
gpu_layers = -1
""".replace("__THESIS_assistant replay_QUERY__", THESIS_ASSISTANT_QUERY_TOML)
    .replace("__DRA_THESIS_PARAMS__", DRA_THESIS_PARAMS)
    .replace("__INTERFUSION_THESIS_PARAMS__", INTERFUSION_THESIS_PARAMS)
    .replace("__DRCAD_THESIS_PARAMS__", DRCAD_THESIS_PARAMS)
)


PROVIDER_CONFIG_TOML = """# Provider examples for assistant replay.

[assistant.provider]
# Recommended thesis-reproducibility path: run llama.cpp with an OpenAI-compatible server.
name = "llama-cpp"
model = "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
base_url = "http://127.0.0.1:8080/v1"
timeout_s = 180.0
temperature = 0.0
top_p = 1.0
max_tokens = 700
seed = 1337

[assistant.provider.extra]
structured_output_mode = "json_object"
gpu_offload_required = true
gpu_layers = -1

# Cloud examples use env vars only:
# name = "openai"
# model = "gpt-4.1-mini"
# api_key_env = "OPENAI_API_KEY"
#
# name = "anthropic"
# model = "claude-3-5-sonnet-latest"
# api_key_env = "ANTHROPIC_API_KEY"
#
# name = "google"
# model = "gemini-2.0-flash"
# api_key_env = "GOOGLE_API_KEY"
#
# name = "xai"
# model = "grok-3-mini"
# api_key_env = "XAI_API_KEY"
"""


def load_reproduction_config(path: str | Path) -> ReproductionConfig:
    """Load and validate a reproduction TOML config."""
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config = ReproductionConfig.from_mapping(payload)
    return _resolve_reproduction_paths(config, config_path.parent)


def load_assistant_config(path: str | Path) -> AssistantReplayConfig:
    """Load and validate an assistant replay TOML config."""
    config_path = Path(path)
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config = AssistantReplayConfig.from_mapping(payload)
    return _resolve_assistant_paths(config, config_path.parent)


def write_default_reproduction_config(path: str | Path, profile: str = "thesis-smoke") -> Path:
    """Write a starter thesis reproduction config."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if profile == "thesis-smoke":
        payload = THESIS_SMOKE_CONFIG_TOML
    elif profile == "thesis-verification":
        payload = THESIS_VERIFICATION_CONFIG_TOML
    elif profile == "thesis-full":
        payload = THESIS_FULL_CONFIG_TOML
    else:
        raise ValueError(
            "profile must be one of 'thesis-smoke', 'thesis-verification', or 'thesis-full'."
        )
    output.write_text(_default_paths_for_output(payload, output.parent), encoding="utf-8")
    return output


def write_provider_config_template(path: str | Path) -> Path:
    """Write provider example config snippets."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(PROVIDER_CONFIG_TOML, encoding="utf-8")
    return output


def render_reproduction_config_toml(config: ReproductionConfig) -> str:
    """Render a resolved reproduction config for artifact capture."""
    lines = [
        "[reproduction]",
        f'name = "{_toml_string(config.name)}"',
        f"run_evidence = {_bool(config.run_evidence)}",
        f"run_xai = {_bool(config.run_xai)}",
        f"run_profiles = {_bool(config.run_profiles)}",
        f"run_assistant = {_bool(config.run_assistant)}",
        "xai_ks = [" + ", ".join(str(item) for item in config.xai_ks) + "]",
        "evidence_sources = " + _string_list(config.evidence_sources),
        f'assistant_evidence_source = "{_toml_string(config.assistant_evidence_source)}"',
        "profile_experiment_limit = "
        + (
            "0" if config.profile_experiment_limit is None else str(config.profile_experiment_limit)
        ),
        "",
        "[reproduction.resources]",
        f"cpu_threads = {config.resources.cpu_threads}",
        f"memory_limit_gb = {config.resources.memory_limit_gb}",
        "benchmark_workers = " + _worker_toml(config.resources.benchmark_workers),
        "evidence_workers = " + _worker_toml(config.resources.evidence_workers),
        "xai_workers = " + _worker_toml(config.resources.xai_workers),
        "assistant_workers = " + _worker_toml(config.resources.assistant_workers),
        f"gpu_slots = {config.resources.gpu_slots}",
        f'profile_mode = "{config.resources.profile_mode}"',
        f"require_cuda_for_torch = {_bool(config.resources.require_cuda_for_torch)}",
        f"require_llama_gpu = {_bool(config.resources.require_llama_gpu)}",
        *(_reuse_lines(config) if config.reuse.benchmark_dir is not None else []),
        "",
        *_benchmark_lines(config.benchmark),
        "",
        *_assistant_lines(config.assistant),
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_assistant_config_toml(config: AssistantReplayConfig) -> str:
    """Render an assistant replay config for artifact capture."""
    return "\n".join(_assistant_lines(config)).rstrip() + "\n"


def _resolve_reproduction_paths(config: ReproductionConfig, base_dir: Path) -> ReproductionConfig:
    benchmark = _resolve_benchmark_paths(config.benchmark, base_dir)
    assistant = _resolve_assistant_paths(config.assistant, base_dir)
    return ReproductionConfig(
        name=config.name,
        benchmark=benchmark,
        assistant=assistant,
        run_evidence=config.run_evidence,
        run_xai=config.run_xai,
        run_profiles=config.run_profiles,
        run_assistant=config.run_assistant,
        xai_ks=list(config.xai_ks),
        evidence_sources=list(config.evidence_sources),
        assistant_evidence_source=config.assistant_evidence_source,
        profile_experiment_limit=config.profile_experiment_limit,
        resources=config.resources,
        reuse=ReproductionReuseConfig(
            benchmark_dir=str(_resolve_path(config.reuse.benchmark_dir, base_dir))
            if config.reuse.benchmark_dir is not None
            else None,
            mode=config.reuse.mode,
        ),
    )


def _resolve_benchmark_paths(config: BenchmarkConfig, base_dir: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        name=config.name,
        protocols=list(config.protocols),
        datasets=[
            BenchmarkDatasetConfig(
                id=dataset.id,
                prepared=str(_resolve_path(dataset.prepared, base_dir)),
            )
            for dataset in config.datasets
        ],
        detectors=[
            BenchmarkDetectorConfig(
                id=detector.id,
                name=detector.name,
                parameters=dict(detector.parameters),
                datasets=list(detector.datasets) if detector.datasets is not None else None,
                protocols=list(detector.protocols) if detector.protocols is not None else None,
                parameter_overrides=list(detector.parameter_overrides),
            )
            for detector in config.detectors
        ],
        evaluation=config.evaluation,
    )


def _resolve_assistant_paths(
    config: AssistantReplayConfig, base_dir: Path
) -> AssistantReplayConfig:
    playbooks = (
        str(_resolve_path(config.playbooks, base_dir)) if config.playbooks is not None else None
    )
    return AssistantReplayConfig(
        suite_id=config.suite_id,
        prepared=str(_resolve_path(config.prepared, base_dir)),
        provider=config.provider,
        query_template=config.query_template,
        cases_per_dataset=config.cases_per_dataset,
        top_k=config.top_k,
        minimum_supported_claims=config.minimum_supported_claims,
        prompt_budget_chars=config.prompt_budget_chars,
        playbooks=playbooks,
        include_operator_cards=config.include_operator_cards,
    )


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _default_paths_for_output(payload: str, output_dir: Path) -> str:
    replacements = {
        "../examples/generated/OPCUA_SYNTH": _default_relative_path(
            output_dir,
            Path("examples") / "generated" / "OPCUA_SYNTH",
        ),
        "../prepared/TEP": _default_relative_path(output_dir, Path("prepared") / "TEP"),
        "../prepared/SWaT": _default_relative_path(output_dir, Path("prepared") / "SWaT"),
        "../prepared/HAI": _default_relative_path(output_dir, Path("prepared") / "HAI"),
        "../prepared/HAI_CPPS": _default_relative_path(output_dir, Path("prepared") / "HAI_CPPS"),
    }
    for old, new in replacements.items():
        payload = payload.replace(f'prepared = "{old}"', f'prepared = "{new}"')
    return payload


def _default_relative_path(output_dir: Path, project_relative: Path) -> str:
    base = output_dir.parent if output_dir.name == "config" else output_dir
    target = base / project_relative
    return os.path.relpath(target, output_dir).replace("\\", "/")


def _reuse_lines(config: ReproductionConfig) -> list[str]:
    if config.reuse.benchmark_dir is None:
        return []
    return [
        "",
        "[reproduction.reuse]",
        f'benchmark_dir = "{_toml_string(config.reuse.benchmark_dir)}"',
        f'mode = "{config.reuse.mode or "diagnostic"}"',
    ]


def _worker_toml(value: int | str) -> str:
    return str(value) if isinstance(value, int) else f'"{_toml_string(value)}"'


def _benchmark_lines(config: BenchmarkConfig) -> list[str]:
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
    for dataset_id, policy in sorted(config.evaluation.dataset_policies.items()):
        lines.extend(
            [
                f'[benchmark.evaluation.dataset_policies."{_toml_string(dataset_id)}"]',
                *_policy_lines(policy.to_dict()),
                "",
            ]
        )
    for dataset in config.datasets:
        lines.extend(
            [
                "[[datasets]]",
                f'id = "{_toml_string(dataset.id)}"',
                f'prepared = "{_toml_string(dataset.prepared)}"',
                "",
            ]
        )
    for detector in config.detectors:
        lines.extend(
            [
                "[[detectors]]",
                f'id = "{_toml_string(detector.id)}"',
                f'name = "{_toml_string(detector.name)}"',
            ]
        )
        if detector.datasets is not None:
            lines.append(f"datasets = {_string_list(detector.datasets)}")
        if detector.protocols is not None:
            lines.append(f"protocols = {_string_list(detector.protocols)}")
        lines.extend(
            [
                f"parameters = {_inline_table(detector.parameters)}",
            ]
        )
        for override in detector.parameter_overrides:
            lines.extend(_override_lines(override))
        lines.append("")
    return lines


def _assistant_lines(config: AssistantReplayConfig) -> list[str]:
    lines = [
        "[assistant]",
        f'suite_id = "{_toml_string(config.suite_id)}"',
        f'prepared = "{_toml_string(config.prepared)}"',
        f"cases_per_dataset = {config.cases_per_dataset}",
        f"top_k = {config.top_k}",
        f"minimum_supported_claims = {config.minimum_supported_claims}",
        f"prompt_budget_chars = {config.prompt_budget_chars}",
        f'query_template = "{_toml_string(config.query_template)}"',
    ]
    if config.playbooks is not None:
        lines.append(f'playbooks = "{_toml_string(config.playbooks)}"')
    lines.extend(
        [
            "",
            "[assistant.provider]",
            f'name = "{_toml_string(config.provider.name)}"',
            f'model = "{_toml_string(config.provider.model)}"',
        ]
    )
    if config.provider.base_url is not None:
        lines.append(f'base_url = "{_toml_string(config.provider.base_url)}"')
    if config.provider.api_key_env is not None:
        lines.append(f'api_key_env = "{_toml_string(config.provider.api_key_env)}"')
    lines.extend(
        [
            f"timeout_s = {config.provider.timeout_s}",
            f"temperature = {config.provider.temperature}",
            f"top_p = {config.provider.top_p}",
            f"max_tokens = {config.provider.max_tokens}",
        ]
    )
    if config.provider.seed is not None:
        lines.append(f"seed = {config.provider.seed}")
    if config.provider.extra:
        lines.extend(["", "[assistant.provider.extra]"])
        lines.extend(
            f"{key} = {_toml_value(value)}" for key, value in sorted(config.provider.extra.items())
        )
    return lines


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
        return _bool(value)
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return f'"{_toml_string(value)}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value: {value!r}")


def _toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _bool(value: bool) -> str:
    return "true" if value else "false"
