"""Thesis-style reproduction contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

from industrial_tsad_eval.domain.assistant_replay import AssistantReplayConfig
from industrial_tsad_eval.domain.benchmark import BenchmarkConfig
from industrial_tsad_eval.domain.errors import BenchmarkConfigError

ReproductionStageStatus = Literal["pending", "completed", "failed", "skipped"]
ReproductionProfileMode = Literal["inline", "standalone"]
ReproductionReuseMode = Literal["diagnostic"]


@dataclass(frozen=True)
class ReproductionResourcePolicy:
    """Resource policy for thesis-style reproduction runs."""

    cpu_threads: int = 12
    memory_limit_gb: int = 16
    benchmark_workers: int | str = "auto"
    evidence_workers: int | str = "auto"
    xai_workers: int | str = "auto"
    assistant_workers: int | str = "conservative"
    gpu_slots: int = 1
    profile_mode: ReproductionProfileMode = "inline"
    require_cuda_for_torch: bool = True
    require_llama_gpu: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ReproductionResourcePolicy:
        """Build a resource policy from TOML-compatible data."""
        resources = payload.get("resources", {})
        if resources is not None and not isinstance(resources, dict):
            raise BenchmarkConfigError("reproduction.resources must be a table.")
        raw = dict(resources or {})
        profile_mode = str(raw.get("profile_mode", "inline")).strip().lower()
        if profile_mode not in {"inline", "standalone"}:
            raise BenchmarkConfigError(
                "reproduction.resources.profile_mode must be inline or standalone."
            )
        cpu_threads = int(raw.get("cpu_threads", 12))
        memory_limit_gb = int(raw.get("memory_limit_gb", 16))
        gpu_slots = int(raw.get("gpu_slots", 1))
        if cpu_threads <= 0:
            raise BenchmarkConfigError("reproduction.resources.cpu_threads must be positive.")
        if memory_limit_gb <= 0:
            raise BenchmarkConfigError("reproduction.resources.memory_limit_gb must be positive.")
        if gpu_slots < 0:
            raise BenchmarkConfigError("reproduction.resources.gpu_slots cannot be negative.")
        return cls(
            cpu_threads=cpu_threads,
            memory_limit_gb=memory_limit_gb,
            benchmark_workers=_worker_value(raw.get("benchmark_workers", "auto")),
            evidence_workers=_worker_value(raw.get("evidence_workers", "auto")),
            xai_workers=_worker_value(raw.get("xai_workers", "auto")),
            assistant_workers=_worker_value(raw.get("assistant_workers", "conservative")),
            gpu_slots=gpu_slots,
            profile_mode=cast(ReproductionProfileMode, profile_mode),
            require_cuda_for_torch=bool(raw.get("require_cuda_for_torch", True)),
            require_llama_gpu=bool(raw.get("require_llama_gpu", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class ReproductionReuseConfig:
    """Diagnostic reuse policy for completed upstream artifacts."""

    benchmark_dir: str | None = None
    mode: ReproductionReuseMode | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ReproductionReuseConfig:
        """Build a reuse config from TOML-compatible data."""
        reuse = payload.get("reuse", {})
        if reuse is None:
            return cls()
        if not isinstance(reuse, dict):
            raise BenchmarkConfigError("reproduction.reuse must be a table.")
        benchmark_dir = reuse.get("benchmark_dir")
        mode = reuse.get("mode")
        if benchmark_dir is None and mode is None:
            return cls()
        normalized_mode = str(mode or "diagnostic").strip().lower()
        if normalized_mode != "diagnostic":
            raise BenchmarkConfigError("reproduction.reuse.mode must be diagnostic.")
        if benchmark_dir is None or not str(benchmark_dir).strip():
            raise BenchmarkConfigError("reproduction.reuse.benchmark_dir is required.")
        return cls(benchmark_dir=str(benchmark_dir), mode="diagnostic")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class ReproductionConfig:
    """Resolved thesis-style reproduction configuration."""

    name: str
    benchmark: BenchmarkConfig
    assistant: AssistantReplayConfig
    run_evidence: bool = True
    run_xai: bool = True
    run_profiles: bool = False
    run_assistant: bool = True
    xai_ks: list[int] = field(default_factory=lambda: [1, 3, 5])
    evidence_sources: list[str] = field(default_factory=lambda: ["oracle"])
    assistant_evidence_source: str = "oracle"
    profile_experiment_limit: int | None = 1
    resources: ReproductionResourcePolicy = field(default_factory=ReproductionResourcePolicy)
    reuse: ReproductionReuseConfig = field(default_factory=ReproductionReuseConfig)

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> ReproductionConfig:
        """Build a reproduction config from TOML-compatible data."""
        reproduction = payload.get("reproduction", {})
        if reproduction is not None and not isinstance(reproduction, dict):
            raise BenchmarkConfigError("reproduction must be a table.")
        reproduction_payload = dict(reproduction or {})
        benchmark = BenchmarkConfig.from_mapping(payload)
        assistant = AssistantReplayConfig.from_mapping(payload)
        xai_ks = [int(item) for item in reproduction_payload.get("xai_ks", [1, 3, 5])]
        if not xai_ks or any(item <= 0 for item in xai_ks):
            raise BenchmarkConfigError("reproduction.xai_ks must contain positive integers.")
        evidence_sources = [
            str(item).strip().lower()
            for item in reproduction_payload.get("evidence_sources", ["oracle"])
        ]
        if not evidence_sources or any(
            item not in {"oracle", "operational"} for item in evidence_sources
        ):
            raise BenchmarkConfigError(
                "reproduction.evidence_sources must contain oracle and/or operational."
            )
        assistant_evidence_source = (
            str(reproduction_payload.get("assistant_evidence_source", evidence_sources[0]))
            .strip()
            .lower()
        )
        if assistant_evidence_source not in set(evidence_sources):
            raise BenchmarkConfigError(
                "reproduction.assistant_evidence_source must be listed in evidence_sources."
            )
        profile_limit_raw = reproduction_payload.get("profile_experiment_limit", 1)
        profile_experiment_limit = None if profile_limit_raw is None else int(profile_limit_raw)
        if profile_experiment_limit is not None and profile_experiment_limit < 0:
            raise BenchmarkConfigError(
                "reproduction.profile_experiment_limit must be non-negative or null."
            )
        resources = ReproductionResourcePolicy.from_mapping(reproduction_payload)
        reuse = ReproductionReuseConfig.from_mapping(reproduction_payload)
        return cls(
            name=str(reproduction_payload.get("name", benchmark.name)).strip() or benchmark.name,
            benchmark=benchmark,
            assistant=assistant,
            run_evidence=bool(reproduction_payload.get("run_evidence", True)),
            run_xai=bool(reproduction_payload.get("run_xai", True)),
            run_profiles=bool(reproduction_payload.get("run_profiles", False)),
            run_assistant=bool(reproduction_payload.get("run_assistant", True)),
            xai_ks=xai_ks,
            evidence_sources=evidence_sources,
            assistant_evidence_source=assistant_evidence_source,
            profile_experiment_limit=profile_experiment_limit,
            resources=resources,
            reuse=reuse,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "reproduction": {
                "name": self.name,
                "run_evidence": self.run_evidence,
                "run_xai": self.run_xai,
                "run_profiles": self.run_profiles,
                "run_assistant": self.run_assistant,
                "xai_ks": list(self.xai_ks),
                "evidence_sources": list(self.evidence_sources),
                "assistant_evidence_source": self.assistant_evidence_source,
                "profile_experiment_limit": self.profile_experiment_limit,
                "resources": self.resources.to_dict(),
                "reuse": self.reuse.to_dict(),
            },
            "benchmark": self.benchmark.to_dict(),
            "assistant": self.assistant.to_dict(),
        }


def _worker_value(value: object) -> int | str:
    if isinstance(value, int):
        if value <= 0:
            raise BenchmarkConfigError("worker counts must be positive, auto, or conservative.")
        return value
    normalized = str(value).strip().lower()
    if normalized in {"auto", "conservative"}:
        return normalized
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise BenchmarkConfigError(
            "worker counts must be positive integers, auto, or conservative."
        ) from exc
    if parsed <= 0:
        raise BenchmarkConfigError("worker counts must be positive, auto, or conservative.")
    return parsed


@dataclass(frozen=True)
class ReproductionStageResult:
    """Status for one reproduction stage or experiment substage."""

    stage: str
    status: ReproductionStageStatus
    path: str | None = None
    metrics: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class ReproductionRunResult:
    """Application-level result for a thesis-style reproduction run."""

    run_id: str
    run_dir: str
    ok: bool
    stages: list[ReproductionStageResult]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "ok": self.ok,
            "stages": [stage.to_dict() for stage in self.stages],
        }
