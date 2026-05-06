"""Thesis-style reproduction orchestration services."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.assistant_replay import (
    PreflightAssistantReplay,
    RunAssistantReplaySuite,
    summary_csv_from_runs,
)
from industrial_tsad_eval.application.benchmark import RunBenchmark
from industrial_tsad_eval.application.evidence import (
    BuildGroundTruthTagMap,
    GenerateEvidence,
)
from industrial_tsad_eval.application.profiling import (
    ProfileScoreEvaluate,
    ProfileScoreEvaluateConfig,
)
from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.application.xai import EvaluateEvidence, EvaluateEvidenceConfig
from industrial_tsad_eval.domain.benchmark import BenchmarkExperimentResult, sanitize_run_id
from industrial_tsad_eval.domain.errors import ReproductionError
from industrial_tsad_eval.domain.reproduction import (
    ReproductionConfig,
    ReproductionRunResult,
    ReproductionStageResult,
)
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.json_utils import read_json
from industrial_tsad_eval.infrastructure.reproduction_config import render_reproduction_config_toml
from industrial_tsad_eval.plugins.providers import LLMProviderRegistry
from industrial_tsad_eval.plugins.registry import DetectorRegistry


@dataclass(frozen=True)
class ReproductionPlan:
    """Resolved reproduction plan."""

    name: str
    stages: list[str]
    experiment_count: int
    assistant_provider: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "name": self.name,
            "stages": list(self.stages),
            "experiment_count": self.experiment_count,
            "assistant_provider": self.assistant_provider,
        }


class PlanThesisReproduction:
    """Create a deterministic reproduction execution plan."""

    def __init__(self, config: ReproductionConfig):
        self.config = config

    def run(self) -> ReproductionPlan:
        """Return the planned stages without executing them."""
        stages = ["validate_prepared", "benchmark"]
        if self.config.run_evidence:
            stages.append("evidence")
        if self.config.run_xai:
            stages.append("xai")
        if self.config.run_profiles:
            stages.append("profiles")
        if self.config.run_assistant:
            stages.append("assistant")
        stages.append("summaries")
        return ReproductionPlan(
            name=self.config.name,
            stages=stages,
            experiment_count=len(self.config.benchmark.experiments()),
            assistant_provider=self.config.assistant.provider.name,
        )


class PreflightThesisReproduction:
    """Run static reproduction preflight checks."""

    def __init__(self, *, config: ReproductionConfig, provider_registry: LLMProviderRegistry):
        self.config = config
        self.provider_registry = provider_registry

    def run(self) -> dict[str, Any]:
        """Return preflight checks for datasets and assistant replay provider config."""
        checks: list[dict[str, Any]] = []
        for dataset in self.config.benchmark.datasets:
            report = ValidatePreparedDataset(dataset.prepared).run()
            checks.append(
                {
                    "name": f"prepared:{dataset.id}",
                    "status": "pass" if report.ok else "fail",
                    "message": "Prepared dataset validation passed."
                    if report.ok
                    else "Prepared dataset validation failed.",
                    "details": report.to_dict(),
                }
            )
        assistant = PreflightAssistantReplay(
            config=self.config.assistant,
            provider_registry=self.provider_registry,
        ).run()
        checks.extend({"name": f"assistant:{check['name']}", **check} for check in assistant.checks)
        ok = not any(check["status"] == "fail" for check in checks)
        return {"ok": ok, "checks": checks}


class RunThesisReproduction:
    """Run benchmark, evidence, XAI, profiling, and assistant replay stages."""

    def __init__(
        self,
        *,
        config: ReproductionConfig,
        detector_registry: DetectorRegistry,
        provider_registry: LLMProviderRegistry,
        out: str | Path,
        run_id: str | None = None,
        source_config: str | Path | None = None,
    ):
        self.config = config
        self.detector_registry = detector_registry
        self.provider_registry = provider_registry
        self.out = Path(out)
        self.run_id = run_id or _default_run_id(config.name)
        self.source_config = Path(source_config) if source_config is not None else None

    def run(self) -> ReproductionRunResult:
        """Execute the reproduction workflow and write structured artifacts."""
        run_root = self.out / self.run_id
        if run_root.exists():
            raise ReproductionError(f"Reproduction run already exists: {run_root}")
        writer = LocalArtifactWriter(run_root)
        writer.write_text("config/reproduction.toml", render_reproduction_config_toml(self.config))
        if self.source_config is not None and self.source_config.exists():
            writer.write_text(
                "config/source_reproduction.toml",
                self.source_config.read_text(encoding="utf-8"),
            )
        writer.write_json("resolved_config.json", self.config.to_dict())
        writer.write_json(
            "run_manifest.json",
            {
                "run_id": self.run_id,
                "status": "running",
                "started_at_utc": _utc_now(),
                "plan": PlanThesisReproduction(self.config).run().to_dict(),
            },
        )

        stages: list[ReproductionStageResult] = []
        preflight = PreflightThesisReproduction(
            config=self.config,
            provider_registry=self.provider_registry,
        ).run()
        writer.write_json("preflight.json", preflight)
        stages.append(
            ReproductionStageResult(
                stage="preflight",
                status="completed" if preflight["ok"] else "failed",
                path=str(run_root / "preflight.json"),
            )
        )

        benchmark_result = RunBenchmark(
            config=self.config.benchmark,
            detector_registry=self.detector_registry,
            out=run_root,
            run_id="benchmark",
        ).run()
        stages.append(
            ReproductionStageResult(
                stage="benchmark",
                status="completed" if benchmark_result.ok else "failed",
                path=benchmark_result.run_dir,
            )
        )

        successful = [result for result in benchmark_result.results if result.status == "completed"]
        evidence_dirs = self._run_evidence_xai(run_root, successful, stages)
        self._run_profiles(run_root, successful, stages)
        self._run_assistant(run_root, successful, evidence_dirs, stages)
        self._write_summaries(run_root, benchmark_result.results, stages)

        ok = all(stage.status in {"completed", "skipped"} for stage in stages)
        writer.write_json(
            "run_manifest.json",
            {
                "run_id": self.run_id,
                "status": "completed" if ok else "failed",
                "finished_at_utc": _utc_now(),
                "stage_count": len(stages),
                "failed_count": sum(stage.status == "failed" for stage in stages),
            },
        )
        result = ReproductionRunResult(
            run_id=self.run_id,
            run_dir=str(run_root),
            ok=ok,
            stages=stages,
        )
        writer.write_json("summary.json", result.to_dict())
        return result

    def _run_evidence_xai(
        self,
        run_root: Path,
        experiments: list[BenchmarkExperimentResult],
        stages: list[ReproductionStageResult],
    ) -> dict[str, Path]:
        evidence_dirs: dict[str, Path] = {}
        if not self.config.run_evidence:
            stages.append(ReproductionStageResult("evidence", "skipped"))
            return evidence_dirs
        for experiment in experiments:
            evidence_dir = run_root / "evidence" / experiment.experiment_id
            dataset_config = _dataset_config(self.config, experiment.dataset)
            try:
                GenerateEvidence(
                    prepared=dataset_config.prepared,
                    scores=str(experiment.scores_dir),
                    eval_dir=str(experiment.eval_dir),
                    out=evidence_dir,
                    event_source="oracle",
                    protocol=experiment.protocol,
                ).run()
                evidence_dirs[experiment.experiment_id] = evidence_dir
                stages.append(
                    ReproductionStageResult(
                        f"evidence:{experiment.experiment_id}",
                        "completed",
                        path=str(evidence_dir),
                    )
                )
                if self.config.run_xai:
                    gt_map = run_root / "xai" / experiment.experiment_id / "gt_map.json"
                    xai_dir = run_root / "xai" / experiment.experiment_id
                    BuildGroundTruthTagMap(prepared=dataset_config.prepared, out=gt_map).run()
                    xai_result = EvaluateEvidence(
                        EvaluateEvidenceConfig(
                            prepared=dataset_config.prepared,
                            evidence=evidence_dir,
                            gt_map=gt_map,
                            out=xai_dir,
                            ks=list(self.config.xai_ks),
                            protocol=experiment.protocol,
                        )
                    ).run()
                    stages.append(
                        ReproductionStageResult(
                            f"xai:{experiment.experiment_id}",
                            "completed",
                            path=str(xai_dir),
                            metrics=xai_result.metrics,
                        )
                    )
            except Exception as exc:
                stages.append(
                    ReproductionStageResult(
                        f"evidence_xai:{experiment.experiment_id}",
                        "failed",
                        path=str(evidence_dir),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        if self.config.run_xai and not experiments:
            stages.append(
                ReproductionStageResult(
                    "xai",
                    "skipped",
                    error="No successful experiments.",
                )
            )
        return evidence_dirs

    def _run_profiles(
        self,
        run_root: Path,
        experiments: list[BenchmarkExperimentResult],
        stages: list[ReproductionStageResult],
    ) -> None:
        if not self.config.run_profiles:
            stages.append(ReproductionStageResult("profiles", "skipped"))
            return
        if not experiments:
            stages.append(ReproductionStageResult("profiles", "skipped", error="No experiments."))
            return
        experiment = experiments[0]
        dataset_config = _dataset_config(self.config, experiment.dataset)
        detector_config = _detector_config(self.config, experiment.detector)
        try:
            result = ProfileScoreEvaluate(
                detector_registry=self.detector_registry,
                config=ProfileScoreEvaluateConfig(
                    prepared=dataset_config.prepared,
                    detector_name=detector_config.name,
                    detector_parameters=detector_config.parameters,
                    protocol=experiment.protocol,
                    out=run_root / "profiles",
                    profile_id=experiment.experiment_id,
                ),
            ).run()
            stages.append(ReproductionStageResult("profiles", "completed", path=result.profile_dir))
        except Exception as exc:
            stages.append(
                ReproductionStageResult(
                    "profiles",
                    "failed",
                    path=str(run_root / "profiles"),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    def _run_assistant(
        self,
        run_root: Path,
        experiments: list[BenchmarkExperimentResult],
        evidence_dirs: dict[str, Path],
        stages: list[ReproductionStageResult],
    ) -> None:
        if not self.config.run_assistant:
            stages.append(ReproductionStageResult("assistant", "skipped"))
            return
        assistant_rows: list[dict[str, Any]] = []
        for experiment in experiments:
            evidence_dir = evidence_dirs.get(experiment.experiment_id)
            if evidence_dir is None:
                continue
            dataset_config = _dataset_config(self.config, experiment.dataset)
            assistant_config = self.config.assistant.with_prepared(dataset_config.prepared)
            assistant_out = run_root / "assistant" / "experiments" / experiment.experiment_id
            try:
                result = RunAssistantReplaySuite(
                    config=assistant_config,
                    evidence=evidence_dir,
                    out=assistant_out,
                    provider_registry=self.provider_registry,
                    benchmark=run_root / "benchmark",
                ).run()
                summary = result.metrics.to_dict()
                assistant_rows.append(
                    {
                        "experiment_id": experiment.experiment_id,
                        "dataset": experiment.dataset,
                        "detector": experiment.detector,
                        "protocol": experiment.protocol,
                        **_summary_row(summary),
                    }
                )
                stages.append(
                    ReproductionStageResult(
                        f"assistant:{experiment.experiment_id}",
                        "completed" if result.ok else "failed",
                        path=result.run_dir,
                        metrics=summary,
                    )
                )
            except Exception as exc:
                stages.append(
                    ReproductionStageResult(
                        f"assistant:{experiment.experiment_id}",
                        "failed",
                        path=str(assistant_out),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        writer = LocalArtifactWriter(run_root)
        writer.write_json(
            "assistant/assistant_summary.json",
            {"format_version": "assistant-reproduction-summary-v1", "rows": assistant_rows},
        )
        writer.write_text("assistant/assistant_summary.csv", summary_csv_from_runs(assistant_rows))
        if not assistant_rows:
            stages.append(
                ReproductionStageResult(
                    "assistant",
                    "failed",
                    error="No assistant replay rows produced.",
                )
            )

    def _write_summaries(
        self,
        run_root: Path,
        benchmark_results: list[BenchmarkExperimentResult],
        stages: list[ReproductionStageResult],
    ) -> None:
        writer = LocalArtifactWriter(run_root)
        benchmark_summary = run_root / "benchmark" / "summary.csv"
        if benchmark_summary.exists():
            writer.write_text(
                "summaries/detection_summary.csv",
                benchmark_summary.read_text(encoding="utf-8"),
            )
        xai_rows = _collect_xai_rows(run_root / "xai")
        writer.write_text("summaries/xai_summary.csv", _csv(xai_rows))
        assistant_summary = run_root / "assistant" / "assistant_summary.csv"
        writer.write_text(
            "summaries/assistant_summary.csv",
            assistant_summary.read_text(encoding="utf-8") if assistant_summary.exists() else "",
        )
        writer.write_json(
            "summaries/reproducibility_matrix.json",
            {
                "benchmark_experiments": [result.to_dict() for result in benchmark_results],
                "stages": [stage.to_dict() for stage in stages],
            },
        )
        writer.write_text("summaries/thesis_crosswalk.md", _crosswalk_markdown())
        stages.append(
            ReproductionStageResult(
                "summaries",
                "completed",
                path=str(run_root / "summaries"),
            )
        )


class SummarizeThesisReproduction:
    """Read a completed reproduction run summary."""

    def __init__(self, run: str | Path):
        self.run = Path(run)

    def run_summary(self) -> dict[str, Any]:
        """Return the reproduction summary payload."""
        return read_json(self.run / "summary.json")


def _dataset_config(config: ReproductionConfig, dataset_id: str) -> Any:
    for dataset in config.benchmark.datasets:
        if dataset.id == dataset_id:
            return dataset
    raise ReproductionError(f"Unknown dataset id in benchmark result: {dataset_id}")


def _detector_config(config: ReproductionConfig, detector_id: str) -> Any:
    for detector in config.benchmark.detectors:
        if detector.id == detector_id:
            return detector
    raise ReproductionError(f"Unknown detector id in benchmark result: {detector_id}")


def _collect_xai_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for summary_path in sorted(root.glob("*/summary.csv")):
        text = summary_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            row["experiment_id"] = summary_path.parent.name
            rows.append(dict(row))
    return rows


def _csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = sorted({key for row in rows for key in row})
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return handle.getvalue()


def _summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "runs_evaluated": summary.get("runs_evaluated"),
        "supported_claims": summary.get("supported_claims"),
        "citation_compliant_claims": summary.get("citation_compliant_claims"),
        "propositional_alignment_proxy": summary.get("propositional_alignment_proxy"),
        "citation_compliance_proxy": summary.get("citation_compliance_proxy"),
        "verified_response_safety_proxy": summary.get("verified_response_safety_proxy"),
        "abstain_rate": summary.get("abstain_rate"),
        "retrieval_expectation_hit_rate": summary.get("retrieval_expectation_hit_rate"),
        "document_grounding_coverage_proxy": summary.get("document_grounding_coverage_proxy"),
    }


def _default_run_id(name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{sanitize_run_id(name)}-{timestamp}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _crosswalk_markdown() -> str:
    return """# Thesis Result Crosswalk

| Thesis-era concern | Productized architecture |
| --- | --- |
| Prepared datasets | Prepared Format v1 repositories and adapter plugins |
| Detector scoring | Detector plugins and Score Contract v1 |
| Event metrics | Versioned evaluation policy and benchmark summaries |
| Evidence/XAI | Evidence Bundle v1 plus deterministic XAI metrics |
| System profiling | System preflight and profile reports |
| assistant replay evaluation | Provider-backed replay suites with claim/referee metrics |

Operator cards are optional rendering artifacts. They do not replace the assistant replay
metric source.
"""
