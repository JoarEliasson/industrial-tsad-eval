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
        return cls(
            name=str(reproduction_payload.get("name", benchmark.name)).strip() or benchmark.name,
            benchmark=benchmark,
            assistant=assistant,
            run_evidence=bool(reproduction_payload.get("run_evidence", True)),
            run_xai=bool(reproduction_payload.get("run_xai", True)),
            run_profiles=bool(reproduction_payload.get("run_profiles", False)),
            run_assistant=bool(reproduction_payload.get("run_assistant", True)),
            xai_ks=xai_ks,
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
