"""Thesis-style reproduction contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from industrial_tsad_eval.domain.assistant_replay import AssistantReplayConfig
from industrial_tsad_eval.domain.benchmark import BenchmarkConfig
from industrial_tsad_eval.domain.errors import BenchmarkConfigError

ReproductionStageStatus = Literal["pending", "completed", "failed", "skipped"]


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
        parsed_profile_limit = None if profile_limit_raw is None else int(profile_limit_raw)
        profile_experiment_limit = None if parsed_profile_limit == 0 else parsed_profile_limit
        if profile_experiment_limit is not None and profile_experiment_limit < 0:
            raise BenchmarkConfigError(
                "reproduction.profile_experiment_limit must be non-negative or null."
            )
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
            },
            "benchmark": self.benchmark.to_dict(),
            "assistant": self.assistant.to_dict(),
        }


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
