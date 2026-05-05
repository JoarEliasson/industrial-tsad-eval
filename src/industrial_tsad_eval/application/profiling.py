"""Score/evaluate profiling application service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.preflight import PreflightInput, RunPreflight
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.errors import ProfileRunError
from industrial_tsad_eval.domain.profiling import ProfileRunResult, StageSample
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository
from industrial_tsad_eval.infrastructure.profiling import (
    StageMonitor,
    render_budget_markdown,
    summarize_samples,
    write_stage_csv,
)
from industrial_tsad_eval.plugins.registry import DetectorRegistry


@dataclass(frozen=True)
class ProfileScoreEvaluateConfig:
    """Configuration for profiling the score/evaluate workflow."""

    prepared: str | Path
    detector_name: str
    out: str | Path
    protocol: str = "naive"
    detector_parameters: dict[str, Any] | None = None
    device: str = "auto"
    profile_id: str | None = None
    enable_vram: bool = False
    sample_interval_ms: int = 10


class ProfileScoreEvaluate:
    """Profile validate, score, validate scores, and evaluate stages."""

    def __init__(self, *, detector_registry: DetectorRegistry, config: ProfileScoreEvaluateConfig):
        self.detector_registry = detector_registry
        self.config = config

    def run(self) -> ProfileRunResult:
        """Run the profiled workflow and write profile artifacts."""
        profile_id = self.config.profile_id or self._default_profile_id()
        profile_root = Path(self.config.out) / profile_id
        if profile_root.exists():
            raise ProfileRunError(f"Profile output already exists: {profile_root}")

        artifacts_root = profile_root / "artifacts"
        scores_dir = artifacts_root / "scores"
        eval_dir = artifacts_root / "eval"
        writer = LocalArtifactWriter(profile_root)
        detector_parameters = _parameters_with_device(
            self.config.detector_parameters,
            self.config.device,
        )

        preflight = RunPreflight(
            detector_registry=self.detector_registry,
            config=PreflightInput(
                prepared=self.config.prepared,
                detector_name=self.config.detector_name,
                detector_parameters=detector_parameters,
                device=self.config.device,
                out=profile_root,
                strict=True,
            ),
        ).run()

        samples: list[StageSample] = []
        with StageMonitor(
            "end_to_end",
            enable_vram=self.config.enable_vram,
            sample_interval_ms=self.config.sample_interval_ms,
        ) as end_to_end:
            samples.append(self._validate_prepared())
            samples.append(self._score(scores_dir, detector_parameters))
            samples.append(self._validate_scores(scores_dir))
            samples.append(self._evaluate(scores_dir, eval_dir))
        samples.append(_sample(end_to_end))

        summary = summarize_samples(samples)
        summary.update(
            {
                "profile_id": profile_id,
                "prepared": str(self.config.prepared),
                "detector": self.config.detector_name,
                "protocol": self.config.protocol,
                "artifacts": {"scores": str(scores_dir), "eval": str(eval_dir)},
                "preflight_status": preflight.status,
            }
        )
        write_stage_csv(profile_root / "stages.csv", samples)
        writer.write_json("summary.json", summary)
        writer.write_text("budget_check.md", render_budget_markdown(summary))
        return ProfileRunResult(
            profile_id=profile_id,
            profile_dir=str(profile_root),
            ok=True,
            stages=samples,
            summary=summary,
        )

    def _validate_prepared(self) -> StageSample:
        with StageMonitor(
            "validate_prepared",
            enable_vram=self.config.enable_vram,
            sample_interval_ms=self.config.sample_interval_ms,
        ) as monitor:
            report = ValidatePreparedDataset(self.config.prepared).run()
            monitor.meta["ok"] = report.ok
            if not report.ok:
                raise ProfileRunError(f"Prepared dataset validation failed: {report.errors}")
        return _sample(monitor)

    def _score(self, scores_dir: Path, detector_parameters: dict[str, Any]) -> StageSample:
        with StageMonitor(
            "score",
            meta={"detector": self.config.detector_name},
            enable_vram=self.config.enable_vram,
            sample_interval_ms=self.config.sample_interval_ms,
        ) as monitor:
            result = ScoreRuns(
                detector_registry=self.detector_registry,
                prepared=self.config.prepared,
                scores=scores_dir,
                detector_name=self.config.detector_name,
                protocol=self.config.protocol,
                detector_parameters=detector_parameters,
            ).run()
            monitor.meta["runs_scored"] = result.runs_scored
        return _sample(monitor)

    def _validate_scores(self, scores_dir: Path) -> StageSample:
        with StageMonitor(
            "validate_scores",
            enable_vram=self.config.enable_vram,
            sample_interval_ms=self.config.sample_interval_ms,
        ) as monitor:
            report = ValidateScores(self.config.prepared, scores_dir).run()
            monitor.meta["ok"] = report.ok
            if not report.ok:
                raise ProfileRunError(f"Score validation failed: {report.errors}")
        return _sample(monitor)

    def _evaluate(self, scores_dir: Path, eval_dir: Path) -> StageSample:
        with StageMonitor(
            "evaluate",
            enable_vram=self.config.enable_vram,
            sample_interval_ms=self.config.sample_interval_ms,
        ) as monitor:
            result = EvaluateScores(
                prepared=self.config.prepared,
                scores=scores_dir,
                out=eval_dir,
                protocol=self.config.protocol,
            ).run()
            monitor.meta["threshold"] = result.threshold
        return _sample(monitor)

    def _default_profile_id(self) -> str:
        dataset = LocalPreparedDatasetRepository(self.config.prepared).dataset_name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{_safe_id(self.config.detector_name)}-{_safe_id(dataset)}-{timestamp}"


def _sample(monitor: StageMonitor) -> StageSample:
    if monitor.sample is None:
        raise ProfileRunError(f"Stage monitor did not produce a sample for {monitor.stage}.")
    return monitor.sample


def _parameters_with_device(parameters: dict[str, Any] | None, device: str) -> dict[str, Any]:
    merged = dict(parameters or {})
    merged.setdefault("device", device)
    return merged


def _safe_id(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "-" for character in value
    )
    return cleaned.strip("-._") or "profile"
