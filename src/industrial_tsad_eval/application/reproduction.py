"""Thesis-style reproduction orchestration services."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from industrial_tsad_eval.application.assistant_replay import (
    PreflightAssistantReplay,
    RunAssistantReplaySuite,
    summary_csv_from_runs,
)
from industrial_tsad_eval.application.benchmark import BenchmarkRunResult, RunBenchmark
from industrial_tsad_eval.application.evidence import (
    BuildGroundTruthTagMap,
    GenerateEvidence,
)
from industrial_tsad_eval.application.profiling import (
    ProfileScoreEvaluate,
    ProfileScoreEvaluateConfig,
)
from industrial_tsad_eval.application.thesis_exports import write_thesis_draft_exports
from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.application.xai import EvaluateEvidence, EvaluateEvidenceConfig
from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkDetectorConfig,
    BenchmarkExperimentResult,
    sanitize_run_id,
)
from industrial_tsad_eval.domain.errors import ReproductionError
from industrial_tsad_eval.domain.progress import CompositeProgressSink, ProgressEvent, ProgressSink
from industrial_tsad_eval.domain.reproduction import (
    ReproductionConfig,
    ReproductionRunResult,
    ReproductionStageResult,
)
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.json_utils import read_json
from industrial_tsad_eval.infrastructure.profiling import (
    StageMonitor,
    render_budget_markdown,
    summarize_samples,
    write_stage_csv,
)
from industrial_tsad_eval.infrastructure.progress import LocalProgressSink
from industrial_tsad_eval.infrastructure.reproduction_config import render_reproduction_config_toml
from industrial_tsad_eval.infrastructure.system import probe_torch_runtime
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

    def __init__(
        self,
        *,
        config: ReproductionConfig,
        provider_registry: LLMProviderRegistry,
        detector_registry: DetectorRegistry | None = None,
    ):
        self.config = config
        self.provider_registry = provider_registry
        self.detector_registry = detector_registry

    def run(self) -> dict[str, Any]:
        """Return preflight checks for datasets and assistant replay provider config."""
        checks: list[dict[str, Any]] = []
        resource_checks = self._resource_checks()
        checks.extend(resource_checks)
        if any(check["status"] == "fail" for check in resource_checks):
            return {
                "ok": False,
                "checks": checks,
                "short_circuited": True,
                "short_circuit_reason": "required resource preflight failed",
            }
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

    def _resource_checks(self) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        if self.detector_registry is not None and self.config.resources.require_cuda_for_torch:
            torch_experiments = [
                experiment
                for experiment in self.config.benchmark.experiments()
                if getattr(
                    self.detector_registry.get(experiment.detector.name),
                    "requires_torch",
                    False,
                )
            ]
            if torch_experiments:
                runtime = probe_torch_runtime("auto")
                checks.append(
                    {
                        "name": "resources:torch-cuda",
                        "status": "pass"
                        if runtime.ready and runtime.resolved_device == "cuda"
                        else "fail",
                        "message": "Torch CUDA is ready for torch-backed detectors."
                        if runtime.ready and runtime.resolved_device == "cuda"
                        else "Torch-backed detectors are configured but CUDA is not ready.",
                        "details": runtime.to_dict(),
                    }
                )
        if (
            self.config.run_assistant
            and self.config.resources.require_llama_gpu
            and self.config.assistant.provider.name == "llama-cpp"
        ):
            extra = dict(self.config.assistant.provider.extra)
            explicit = bool(extra.get("gpu_offload_required") or extra.get("gpu_offload"))
            offload = _llama_gpu_offload_details(
                self.config.assistant.provider.base_url,
                min_gpu_memory_mb=int(extra.get("min_gpu_memory_mb", 1000)),
            )
            ready = (
                explicit
                and bool(offload.get("endpoint_ready"))
                and (
                    bool(offload.get("active_llama_like_process"))
                    or bool(offload.get("gpu_memory_offload_evidence"))
                )
            )
            checks.append(
                {
                    "name": "resources:llama-gpu-offload",
                    "status": "pass" if ready else "fail",
                    "message": "llama.cpp GPU offload appears active."
                    if ready
                    else (
                        "llama.cpp GPU offload is required but was not verified. "
                        "Start the host server with --n_gpu_layers -1 and rerun preflight."
                    ),
                    "details": {
                        "base_url": self.config.assistant.provider.base_url,
                        "model": self.config.assistant.provider.model,
                        "gpu_layers": extra.get("gpu_layers"),
                        "expected_n_gpu_layers": -1,
                        "config_declares_offload": explicit,
                        **offload,
                    },
                }
            )
        return checks


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
        progress_sink: ProgressSink | None = None,
    ):
        self.config = config
        self.detector_registry = detector_registry
        self.provider_registry = provider_registry
        self.out = Path(out)
        self.run_id = run_id or _default_run_id(config.name)
        self.source_config = Path(source_config) if source_config is not None else None
        self.progress_sink = progress_sink

    def run(self) -> ReproductionRunResult:
        """Execute the reproduction workflow and write structured artifacts."""
        run_root = self.out / self.run_id
        if run_root.exists():
            raise ReproductionError(f"Reproduction run already exists: {run_root}")
        writer = LocalArtifactWriter(run_root)
        progress = CompositeProgressSink(
            [LocalProgressSink(run_root, self.run_id), self.progress_sink]
        )
        reproduction_toml = render_reproduction_config_toml(self.config)
        writer.write_text("config/reproduction.toml", reproduction_toml)
        if self.source_config is not None and self.source_config.exists():
            writer.write_text(
                "config/source_reproduction.toml",
                self.source_config.read_text(encoding="utf-8"),
            )
        writer.write_json("resolved_config.json", self.config.to_dict())
        writer.write_json("resource_budget.json", _resource_budget(self.config))
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
        profile_samples: list[Any] = []
        preflight_started = time.perf_counter()
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="reproduction",
                item_id="preflight",
                status="running",
                ordinal=1,
                total=5,
                path=str(run_root / "preflight.json"),
            )
        )
        with StageMonitor(
            "preflight",
            enable_vram=True,
            meta={"run_id": self.run_id},
        ) as monitor:
            preflight = PreflightThesisReproduction(
                config=self.config,
                provider_registry=self.provider_registry,
                detector_registry=self.detector_registry,
            ).run()
        _append_sample(profile_samples, monitor)
        writer.write_json("preflight.json", preflight)
        stages.append(
            ReproductionStageResult(
                stage="preflight",
                status="completed" if preflight["ok"] else "failed",
                path=str(run_root / "preflight.json"),
            )
        )
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="reproduction",
                item_id="preflight",
                status="completed" if preflight["ok"] else "failed",
                ordinal=1,
                total=5,
                path=str(run_root / "preflight.json"),
                duration_s=round(time.perf_counter() - preflight_started, 6),
                metrics={"ok": preflight["ok"]},
            )
        )

        benchmark_started = time.perf_counter()
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="reproduction",
                item_id="benchmark",
                status="running",
                ordinal=2,
                total=5,
                path=str(run_root / "benchmark"),
            )
        )
        benchmark_reuse = self._reused_benchmark(run_root, progress)
        if benchmark_reuse is None:
            with StageMonitor(
                "benchmark",
                enable_vram=True,
                meta={"run_id": self.run_id, "worker_count": _benchmark_workers(self.config)},
            ) as monitor:
                benchmark_result = RunBenchmark(
                    config=self.config.benchmark,
                    detector_registry=self.detector_registry,
                    out=run_root,
                    run_id="benchmark",
                    progress_sink=progress,
                    worker_count=_benchmark_workers(self.config),
                    gpu_slots=self.config.resources.gpu_slots,
                ).run()
            _append_sample(profile_samples, monitor)
        else:
            benchmark_result = benchmark_reuse
        stages.append(
            ReproductionStageResult(
                stage="benchmark",
                status="completed" if benchmark_result.ok else "failed",
                path=benchmark_result.run_dir,
            )
        )
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="reproduction",
                item_id="benchmark",
                status="completed" if benchmark_result.ok else "failed",
                ordinal=2,
                total=5,
                path=benchmark_result.run_dir,
                duration_s=round(time.perf_counter() - benchmark_started, 6),
                metrics={
                    "ok": benchmark_result.ok,
                    "completed": sum(
                        result.status == "completed" for result in benchmark_result.results
                    ),
                    "failed": sum(result.status == "failed" for result in benchmark_result.results),
                },
            )
        )

        successful = [result for result in benchmark_result.results if result.status == "completed"]
        with StageMonitor(
            "evidence_xai",
            enable_vram=True,
            meta={"run_id": self.run_id, "worker_count": _evidence_workers(self.config)},
        ) as monitor:
            evidence_dirs = self._run_evidence_xai(run_root, successful, stages, progress)
        _append_sample(profile_samples, monitor)
        self._run_profiles(run_root, successful, stages, progress, profile_samples)
        with StageMonitor(
            "assistant",
            enable_vram=True,
            meta={"run_id": self.run_id, "worker_count": _assistant_workers(self.config)},
        ) as monitor:
            self._run_assistant(run_root, successful, evidence_dirs, stages, progress)
        _append_sample(profile_samples, monitor)
        with StageMonitor("summaries", enable_vram=True, meta={"run_id": self.run_id}) as monitor:
            self._write_summaries(run_root, benchmark_result.results, stages, reproduction_toml)
        _append_sample(profile_samples, monitor)
        _write_inline_profile(run_root, profile_samples)
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="reproduction",
                item_id="summaries",
                status="completed",
                ordinal=5,
                total=5,
                path=str(run_root / "summaries"),
            )
        )

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

    def _reused_benchmark(
        self,
        run_root: Path,
        progress: ProgressSink,
    ) -> BenchmarkRunResult | None:
        if self.config.reuse.benchmark_dir is None:
            return None
        if self.config.reuse.mode != "diagnostic":
            raise ReproductionError("Only diagnostic benchmark reuse is supported.")
        source = Path(self.config.reuse.benchmark_dir)
        summary_path = source / "summary.json"
        if not summary_path.exists():
            raise ReproductionError(f"Benchmark reuse summary is missing: {summary_path}")
        payload = read_json(summary_path)
        rows = payload.get("experiments", [])
        if not isinstance(rows, list):
            raise ReproductionError("Benchmark reuse summary has no experiment rows.")
        expected = {experiment.experiment_id for experiment in self.config.benchmark.experiments()}
        observed = {str(row.get("experiment_id", "")) for row in rows if isinstance(row, dict)}
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ReproductionError(
                "Benchmark reuse does not match configured matrix: "
                f"missing={missing[:10]}, extra={extra[:10]}."
            )
        target = run_root / "benchmark"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "summary.csv", "run_manifest.json"):
            candidate = source / name
            if candidate.exists():
                shutil.copyfile(candidate, target / name)
        LocalArtifactWriter(target).write_json(
            "reuse_provenance.json",
            {
                "format_version": "benchmark-reuse-provenance-v1",
                "mode": self.config.reuse.mode,
                "source_benchmark_dir": str(source),
                "reportable_final_result": False,
            },
        )
        results = [
            BenchmarkExperimentResult(
                experiment_id=str(row.get("experiment_id")),
                dataset=str(row.get("dataset")),
                detector=str(row.get("detector")),
                protocol=str(row.get("protocol")),
                status=str(row.get("status")),
                scores_dir=str(row.get("scores_dir")) if row.get("scores_dir") else None,
                eval_dir=str(row.get("eval_dir")) if row.get("eval_dir") else None,
                threshold=float(row["threshold"]) if row.get("threshold") is not None else None,
                metrics=(
                    dict(row.get("metrics", {})) if isinstance(row.get("metrics"), dict) else None
                ),
                error=str(row.get("error")) if row.get("error") else None,
            )
            for row in rows
            if isinstance(row, dict)
        ]
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="benchmark",
                item_id="diagnostic-reuse",
                status="completed",
                ordinal=1,
                total=1,
                path=str(target),
                message=(
                    "Reused completed benchmark artifacts for diagnostic downstream validation."
                ),
                metrics={"source": str(source), "experiment_count": len(results)},
            )
        )
        return BenchmarkRunResult(
            run_id="benchmark",
            run_dir=str(target),
            ok=all(result.status == "completed" for result in results),
            results=results,
        )

    def _run_evidence_xai(
        self,
        run_root: Path,
        experiments: list[BenchmarkExperimentResult],
        stages: list[ReproductionStageResult],
        progress: ProgressSink,
    ) -> dict[tuple[str, str], Path]:
        evidence_dirs: dict[tuple[str, str], Path] = {}
        if not self.config.run_evidence:
            stages.append(ReproductionStageResult("evidence", "skipped"))
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="evidence",
                    item_id="all",
                    status="skipped",
                    ordinal=1,
                    total=1,
                    message="Evidence generation disabled.",
                )
            )
            return evidence_dirs
        total = len(experiments) * len(self.config.evidence_sources)
        tasks = [
            (experiment, evidence_source, ordinal)
            for experiment_ordinal, experiment in enumerate(experiments, start=1)
            for source_index, evidence_source in enumerate(self.config.evidence_sources)
            for ordinal in [
                (experiment_ordinal - 1) * len(self.config.evidence_sources) + source_index + 1
            ]
        ]
        worker_count = _evidence_workers(self.config)
        if worker_count == 1 or len(tasks) <= 1:
            completed = [
                self._run_one_evidence_xai(
                    run_root,
                    experiment,
                    evidence_source,
                    ordinal,
                    total,
                    progress,
                )
                for experiment, evidence_source, ordinal in tasks
            ]
        else:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(tasks))) as executor:
                futures = [
                    executor.submit(
                        self._run_one_evidence_xai,
                        run_root,
                        experiment,
                        evidence_source,
                        ordinal,
                        total,
                        progress,
                    )
                    for experiment, evidence_source, ordinal in tasks
                ]
                completed = [future.result() for future in as_completed(futures)]
        for experiment_id, evidence_source, evidence_dir, item_stages in completed:
            has_completed_evidence = any(
                stage.status == "completed" and stage.stage.startswith("evidence:")
                for stage in item_stages
            )
            if has_completed_evidence:
                evidence_dirs[(experiment_id, evidence_source)] = evidence_dir
            stages.extend(item_stages)
        if self.config.run_xai and not experiments:
            stages.append(
                ReproductionStageResult(
                    "xai",
                    "skipped",
                    error="No successful experiments.",
                )
            )
        return evidence_dirs

    def _run_one_evidence_xai(
        self,
        run_root: Path,
        experiment: BenchmarkExperimentResult,
        evidence_source: str,
        ordinal: int,
        total: int,
        progress: ProgressSink,
    ) -> tuple[str, str, Path, list[ReproductionStageResult]]:
        dataset_config = _dataset_config(self.config, experiment.dataset)
        evidence_dir = run_root / "evidence" / experiment.experiment_id / evidence_source
        item_id = f"{experiment.experiment_id}:{evidence_source}"
        detector_config = _detector_config(self.config, experiment.detector)
        started = time.perf_counter()
        item_stages: list[ReproductionStageResult] = []
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="evidence",
                item_id=item_id,
                status="running",
                ordinal=ordinal,
                total=total,
                path=str(evidence_dir),
            )
        )
        try:
            GenerateEvidence(
                prepared=dataset_config.prepared,
                scores=str(experiment.scores_dir),
                eval_dir=str(experiment.eval_dir),
                out=evidence_dir,
                event_source=evidence_source,
                protocol=experiment.protocol,
                explanation_source=str(
                    detector_config.parameters.get("evidence_explanation_source", "auto")
                ),
                native_missing_policy=str(
                    detector_config.parameters.get("evidence_native_missing_policy", "skip_bundle")
                ),
            ).run()
            item_stages.append(
                ReproductionStageResult(f"evidence:{item_id}", "completed", path=str(evidence_dir))
            )
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="evidence",
                    item_id=item_id,
                    status="completed",
                    ordinal=ordinal,
                    total=total,
                    path=str(evidence_dir),
                    duration_s=round(time.perf_counter() - started, 6),
                )
            )
            if self.config.run_xai:
                item_stages.extend(
                    self._run_one_xai(
                        run_root,
                        experiment,
                        evidence_source,
                        evidence_dir,
                        dataset_config.prepared,
                        item_id,
                        ordinal,
                        total,
                        progress,
                    )
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            item_stages.append(
                ReproductionStageResult(
                    f"evidence_xai:{item_id}",
                    "failed",
                    path=str(evidence_dir),
                    error=error,
                )
            )
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="evidence",
                    item_id=item_id,
                    status="failed",
                    ordinal=ordinal,
                    total=total,
                    path=str(evidence_dir),
                    duration_s=round(time.perf_counter() - started, 6),
                    error=error,
                )
            )
        return experiment.experiment_id, evidence_source, evidence_dir, item_stages

    def _run_one_xai(
        self,
        run_root: Path,
        experiment: BenchmarkExperimentResult,
        evidence_source: str,
        evidence_dir: Path,
        prepared: str,
        item_id: str,
        ordinal: int,
        total: int,
        progress: ProgressSink,
    ) -> list[ReproductionStageResult]:
        xai_dir = run_root / "xai" / experiment.experiment_id / evidence_source
        gt_map = xai_dir / "gt_map.json"
        xai_started = time.perf_counter()
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="xai",
                item_id=item_id,
                status="running",
                ordinal=ordinal,
                total=total,
                path=str(xai_dir),
            )
        )
        BuildGroundTruthTagMap(prepared=prepared, out=gt_map).run()
        xai_result = EvaluateEvidence(
            EvaluateEvidenceConfig(
                prepared=prepared,
                evidence=evidence_dir,
                gt_map=gt_map,
                out=xai_dir,
                ks=list(self.config.xai_ks),
                protocol=experiment.protocol,
            )
        ).run()
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="xai",
                item_id=item_id,
                status="completed",
                ordinal=ordinal,
                total=total,
                path=str(xai_dir),
                duration_s=round(time.perf_counter() - xai_started, 6),
                metrics=xai_result.metrics,
            )
        )
        return [
            ReproductionStageResult(
                f"xai:{item_id}",
                "completed",
                path=str(xai_dir),
                metrics=xai_result.metrics,
            )
        ]

    def _run_profiles(
        self,
        run_root: Path,
        experiments: list[BenchmarkExperimentResult],
        stages: list[ReproductionStageResult],
        progress: ProgressSink,
        profile_samples: list[Any],
    ) -> None:
        if not self.config.run_profiles:
            stages.append(ReproductionStageResult("profiles", "skipped"))
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="profiles",
                    item_id="all",
                    status="skipped",
                    ordinal=3,
                    total=5,
                    message="Profiling disabled.",
                )
            )
            return
        if self.config.resources.profile_mode == "inline":
            stages.append(
                ReproductionStageResult(
                    "profiles",
                    "completed",
                    path=str(run_root / "profiles"),
                    metrics={"mode": "inline", "sample_count": len(profile_samples)},
                )
            )
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="profiles",
                    item_id="inline",
                    status="completed",
                    ordinal=3,
                    total=5,
                    path=str(run_root / "profiles"),
                    metrics={"mode": "inline", "sample_count": len(profile_samples)},
                    message="Inline profiling records actual reproduction stages without reruns.",
                )
            )
            return
        if not experiments:
            stages.append(ReproductionStageResult("profiles", "skipped", error="No experiments."))
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="profiles",
                    item_id="all",
                    status="skipped",
                    ordinal=3,
                    total=5,
                    error="No experiments.",
                )
            )
            return
        if self.config.profile_experiment_limit == 0:
            stages.append(
                ReproductionStageResult(
                    "profiles",
                    "skipped",
                    error="Standalone profiling limit is 0.",
                )
            )
            return
        selected = experiments[: self.config.profile_experiment_limit]
        if self.config.profile_experiment_limit is None:
            selected = list(experiments)
        for ordinal, experiment in enumerate(selected, start=1):
            dataset_config = _dataset_config(self.config, experiment.dataset)
            detector_config = _detector_config(self.config, experiment.detector)
            started = time.perf_counter()
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="profiles",
                    item_id=experiment.experiment_id,
                    status="running",
                    ordinal=ordinal,
                    total=len(selected),
                    path=str(run_root / "profiles"),
                )
            )
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
                stages.append(
                    ReproductionStageResult(
                        f"profiles:{experiment.experiment_id}",
                        "completed",
                        path=result.profile_dir,
                    )
                )
                progress.emit(
                    ProgressEvent(
                        run_id=self.run_id,
                        stage="profiles",
                        item_id=experiment.experiment_id,
                        status="completed",
                        ordinal=ordinal,
                        total=len(selected),
                        path=result.profile_dir,
                        duration_s=round(time.perf_counter() - started, 6),
                    )
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                stages.append(
                    ReproductionStageResult(
                        f"profiles:{experiment.experiment_id}",
                        "failed",
                        path=str(run_root / "profiles"),
                        error=error,
                    )
                )
                progress.emit(
                    ProgressEvent(
                        run_id=self.run_id,
                        stage="profiles",
                        item_id=experiment.experiment_id,
                        status="failed",
                        ordinal=ordinal,
                        total=len(selected),
                        path=str(run_root / "profiles"),
                        duration_s=round(time.perf_counter() - started, 6),
                        error=error,
                    )
                )

    def _run_assistant(
        self,
        run_root: Path,
        experiments: list[BenchmarkExperimentResult],
        evidence_dirs: dict[tuple[str, str], Path],
        stages: list[ReproductionStageResult],
        progress: ProgressSink,
    ) -> None:
        if not self.config.run_assistant:
            stages.append(ReproductionStageResult("assistant", "skipped"))
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="assistant",
                    item_id="all",
                    status="skipped",
                    ordinal=4,
                    total=5,
                    message="Assistant replay disabled.",
                )
            )
            return
        assistant_rows: list[dict[str, Any]] = []
        tasks = [
            (ordinal, experiment)
            for ordinal, experiment in enumerate(experiments, start=1)
            if (experiment.experiment_id, self.config.assistant_evidence_source) in evidence_dirs
        ]
        worker_count = _assistant_workers(self.config)
        if worker_count == 1 or len(tasks) <= 1:
            completed = [
                self._run_one_assistant(
                    run_root,
                    experiment,
                    evidence_dirs[
                        (experiment.experiment_id, self.config.assistant_evidence_source)
                    ],
                    ordinal,
                    len(experiments),
                    progress,
                )
                for ordinal, experiment in tasks
            ]
        else:
            with ThreadPoolExecutor(max_workers=min(worker_count, len(tasks))) as executor:
                futures = [
                    executor.submit(
                        self._run_one_assistant,
                        run_root,
                        experiment,
                        evidence_dirs[
                            (experiment.experiment_id, self.config.assistant_evidence_source)
                        ],
                        ordinal,
                        len(experiments),
                        progress,
                    )
                    for ordinal, experiment in tasks
                ]
                completed = [future.result() for future in as_completed(futures)]
        for row, stage in completed:
            if row is not None:
                assistant_rows.append(row)
            stages.append(stage)
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

    def _run_one_assistant(
        self,
        run_root: Path,
        experiment: BenchmarkExperimentResult,
        evidence_dir: Path,
        ordinal: int,
        total: int,
        progress: ProgressSink,
    ) -> tuple[dict[str, Any] | None, ReproductionStageResult]:
        dataset_config = _dataset_config(self.config, experiment.dataset)
        assistant_config = self.config.assistant.with_prepared(dataset_config.prepared)
        assistant_out = run_root / "assistant" / "experiments" / experiment.experiment_id
        started = time.perf_counter()
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="assistant",
                item_id=experiment.experiment_id,
                status="running",
                ordinal=ordinal,
                total=total,
                path=str(assistant_out),
            )
        )
        try:
            result = RunAssistantReplaySuite(
                config=assistant_config,
                evidence=evidence_dir,
                out=assistant_out,
                provider_registry=self.provider_registry,
                benchmark=run_root / "benchmark",
                progress_sink=progress,
            ).run()
            summary = result.metrics.to_dict()
            row = {
                "experiment_id": experiment.experiment_id,
                "dataset": experiment.dataset,
                "detector": experiment.detector,
                "protocol": experiment.protocol,
                **_summary_row(summary),
            }
            stage = ReproductionStageResult(
                f"assistant:{experiment.experiment_id}",
                "completed" if result.ok else "failed",
                path=result.run_dir,
                metrics=summary,
            )
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="assistant",
                    item_id=experiment.experiment_id,
                    status="completed" if result.ok else "failed",
                    ordinal=ordinal,
                    total=total,
                    path=result.run_dir,
                    duration_s=round(time.perf_counter() - started, 6),
                    metrics=_summary_row(summary),
                )
            )
            return row, stage
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            stage = ReproductionStageResult(
                f"assistant:{experiment.experiment_id}",
                "failed",
                path=str(assistant_out),
                error=error,
            )
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="assistant",
                    item_id=experiment.experiment_id,
                    status="failed",
                    ordinal=ordinal,
                    total=total,
                    path=str(assistant_out),
                    duration_s=round(time.perf_counter() - started, 6),
                    error=error,
                )
            )
            return None, stage

    def _write_summaries(
        self,
        run_root: Path,
        benchmark_results: list[BenchmarkExperimentResult],
        stages: list[ReproductionStageResult],
        reproduction_toml: str,
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
        write_thesis_draft_exports(
            run_root=run_root,
            config=self.config,
            reproduction_toml=reproduction_toml,
        )
        stages.append(
            ReproductionStageResult(
                "summaries",
                "completed",
                path=str(run_root / "summaries"),
            )
        )


class RunReproductionSlice:
    """Run a filtered reproduction slice with regular reproduction artifacts."""

    def __init__(
        self,
        *,
        config: ReproductionConfig,
        detector_registry: DetectorRegistry,
        provider_registry: LLMProviderRegistry,
        out: str | Path,
        run_id: str,
        datasets: list[str] | None = None,
        detectors: list[str] | None = None,
        protocols: list[str] | None = None,
        stages: list[str] | None = None,
        source_config: str | Path | None = None,
        progress_sink: ProgressSink | None = None,
    ):
        self.config = filter_reproduction_config(
            config,
            datasets=datasets,
            detectors=detectors,
            protocols=protocols,
            stages=stages,
        )
        self.detector_registry = detector_registry
        self.provider_registry = provider_registry
        self.out = out
        self.run_id = run_id
        self.source_config = source_config
        self.progress_sink = progress_sink

    def run(self) -> ReproductionRunResult:
        """Run the filtered reproduction slice."""
        return RunThesisReproduction(
            config=self.config,
            detector_registry=self.detector_registry,
            provider_registry=self.provider_registry,
            out=self.out,
            run_id=self.run_id,
            source_config=self.source_config,
            progress_sink=self.progress_sink,
        ).run()


class AssembleReproductionSlices:
    """Assemble compatible slice summaries into one provenance-rich result pack."""

    def __init__(
        self,
        *,
        runs: list[str | Path],
        out: str | Path,
        run_id: str,
    ):
        self.runs = [Path(run) for run in runs]
        self.out = Path(out)
        self.run_id = run_id

    def run(self) -> dict[str, Any]:
        """Validate compatibility and write assembled summary artifacts."""
        if len(self.runs) < 2:
            raise ReproductionError("At least two slice run directories are required.")
        run_payloads = [_slice_payload(run) for run in self.runs]
        compatibility = _slice_compatibility(run_payloads)
        if not compatibility["ok"]:
            raise ReproductionError(
                "Slice runs are incompatible: "
                + "; ".join(str(item) for item in compatibility["errors"])
            )
        run_root = self.out / self.run_id
        if run_root.exists():
            raise ReproductionError(f"Assembled run already exists: {run_root}")
        writer = LocalArtifactWriter(run_root)
        detection_rows = _concat_csv_files(
            [run / "summaries" / "detection_summary.csv" for run in self.runs]
        )
        xai_rows = _concat_csv_files([run / "summaries" / "xai_summary.csv" for run in self.runs])
        assistant_rows = _concat_csv_files(
            [run / "summaries" / "assistant_summary.csv" for run in self.runs]
        )
        writer.write_text("summaries/detection_summary.csv", _csv(detection_rows))
        writer.write_text("summaries/xai_summary.csv", _csv(xai_rows))
        writer.write_text("summaries/assistant_summary.csv", _csv(assistant_rows))
        assembly = {
            "format_version": "reproduction-slice-assembly-v1",
            "run_id": self.run_id,
            "assembled": True,
            "reporting_note": "This result pack was assembled from compatible reproduction slices.",
            "source_runs": [payload["run"] for payload in run_payloads],
            "compatibility": compatibility,
            "created_at_utc": _utc_now(),
        }
        writer.write_json("assembly_manifest.json", assembly)
        writer.write_json(
            "summaries/reproducibility_matrix.json",
            {
                "assembled": True,
                "source_runs": [payload["run"] for payload in run_payloads],
                "detection_rows": detection_rows,
                "xai_rows": xai_rows,
                "assistant_rows": assistant_rows,
            },
        )
        ok = all(bool(payload.get("summary", {}).get("ok", False)) for payload in run_payloads)
        summary = {
            "run_id": self.run_id,
            "run_dir": str(run_root),
            "ok": ok,
            "assembled": True,
            "source_run_count": len(run_payloads),
            "artifacts": {
                "assembly_manifest": str(run_root / "assembly_manifest.json"),
                "detection_summary": str(run_root / "summaries" / "detection_summary.csv"),
                "xai_summary": str(run_root / "summaries" / "xai_summary.csv"),
                "assistant_summary": str(run_root / "summaries" / "assistant_summary.csv"),
            },
        }
        writer.write_json("summary.json", summary)
        writer.write_json(
            "run_manifest.json",
            {
                "run_id": self.run_id,
                "status": "completed" if ok else "failed",
                "assembled": True,
                "source_runs": [payload["run"] for payload in run_payloads],
                "finished_at_utc": _utc_now(),
            },
        )
        return summary


class StopThesisReproduction:
    """Write a safe cancellation marker for a long reproduction run."""

    def __init__(self, *, run: str | Path, container: str | None = None):
        self.run = Path(run)
        self.container = container

    def run_stop(self) -> dict[str, Any]:
        """Write the cancellation marker and return safe follow-up commands."""
        marker = self.run / "run_control" / "cancel_requested.json"
        payload = {
            "format_version": "reproduction-stop-request-v1",
            "run": str(self.run),
            "container": self.container,
            "requested_at_utc": _utc_now(),
            "commands": _stop_commands(self.container),
        }
        LocalArtifactWriter(marker.parent).write_json(marker.name, payload)
        return payload


class SummarizeThesisReproduction:
    """Read a completed reproduction run summary."""

    def __init__(self, run: str | Path):
        self.run = Path(run)

    def run_summary(self) -> dict[str, Any]:
        """Return the reproduction summary payload."""
        return read_json(self.run / "summary.json")


class DiagnoseThesisReproduction:
    """Build a post-run diagnostic report from progress and summary artifacts."""

    def __init__(self, run: str | Path):
        self.run = Path(run)

    def run_diagnostics(self) -> dict[str, Any]:
        """Write and return grouped reproduction diagnostics."""
        progress_path = self.run / "progress.jsonl"
        events = _read_progress_events(progress_path)
        summary_path = self.run / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        stages = list(summary.get("stages", [])) if isinstance(summary.get("stages"), list) else []
        failed_events = [event for event in events if str(event.get("status")) == "failed"]
        failed_stages = [
            stage for stage in stages if isinstance(stage, dict) and stage.get("status") == "failed"
        ]
        errors = [
            str(item.get("error") or "")
            for item in [*failed_events, *failed_stages]
            if str(item.get("error") or "")
        ]
        by_stage = Counter(str(event.get("stage", "unknown")) for event in failed_events)
        by_error = Counter(_error_signature(error) for error in errors)
        benchmark_summary = _benchmark_status(self.run / "benchmark")
        throughput = _throughput_summary(events, self.run)
        diagnosis = {
            "format_version": "reproduction-diagnostics-v1",
            "run": str(self.run),
            "summary_ok": summary.get("ok"),
            "benchmark": benchmark_summary,
            "throughput": throughput,
            "failed_event_count": len(failed_events),
            "failed_stage_count": len(failed_stages),
            "failures_by_stage": dict(sorted(by_stage.items())),
            "failures_by_error": dict(sorted(by_error.items())),
            "classification": _diagnostic_classification(errors, benchmark_summary),
            "diagnostic_only": _diagnostic_only(self.run, summary),
            "latest_failed_event": failed_events[-1] if failed_events else None,
            "next_files": _diagnostic_next_files(self.run),
        }
        writer = LocalArtifactWriter(self.run / "diagnostics")
        writer.write_json("failure_report.json", diagnosis)
        writer.write_text("failure_report.md", _diagnostic_markdown(diagnosis))
        return diagnosis


def filter_reproduction_config(
    config: ReproductionConfig,
    *,
    datasets: list[str] | None = None,
    detectors: list[str] | None = None,
    protocols: list[str] | None = None,
    stages: list[str] | None = None,
) -> ReproductionConfig:
    """Return a reproduction config narrowed to a deterministic execution slice."""
    requested_stages = {stage.strip().lower() for stage in stages or [] if stage.strip()}
    if (
        requested_stages
        and "benchmark" not in requested_stages
        and config.reuse.benchmark_dir is None
    ):
        raise ReproductionError(
            "Slice stages without benchmark require diagnostic benchmark reuse."
        )
    dataset_filter = set(datasets or [])
    detector_filter = set(detectors or [])
    protocol_filter = set(protocols or [])
    selected_datasets = [
        dataset
        for dataset in config.benchmark.datasets
        if not dataset_filter or dataset.id in dataset_filter
    ]
    selected_detectors = [
        detector
        for detector in config.benchmark.detectors
        if not detector_filter or detector.id in detector_filter or detector.name in detector_filter
    ]
    selected_protocols = [
        protocol
        for protocol in config.benchmark.protocols
        if not protocol_filter or protocol in protocol_filter
    ]
    if dataset_filter and len(selected_datasets) != len(dataset_filter):
        missing = sorted(dataset_filter - {dataset.id for dataset in selected_datasets})
        raise ReproductionError(f"Unknown dataset slice ids: {missing}.")
    if detector_filter and not selected_detectors:
        raise ReproductionError(f"No detectors matched slice ids/names: {sorted(detector_filter)}.")
    if protocol_filter and len(selected_protocols) != len(protocol_filter):
        missing = sorted(protocol_filter - set(selected_protocols))
        raise ReproductionError(f"Unknown protocol slice ids: {missing}.")
    benchmark = _filtered_benchmark(
        config.benchmark,
        datasets=selected_datasets,
        detectors=selected_detectors,
        protocols=selected_protocols,
    )
    if not benchmark.experiments():
        raise ReproductionError("Slice filters produced an empty benchmark matrix.")
    if not requested_stages:
        return replace(config, benchmark=benchmark)
    return replace(
        config,
        benchmark=benchmark,
        run_evidence="evidence" in requested_stages,
        run_xai="xai" in requested_stages,
        run_profiles="profiles" in requested_stages,
        run_assistant="assistant" in requested_stages,
    )


def _filtered_benchmark(
    config: BenchmarkConfig,
    *,
    datasets: list[BenchmarkDatasetConfig],
    detectors: list[BenchmarkDetectorConfig],
    protocols: list[str],
) -> BenchmarkConfig:
    protocol_set = set(protocols)
    adjusted_detectors: list[BenchmarkDetectorConfig] = []
    for detector in detectors:
        detector_protocols = detector.protocols
        if detector_protocols is not None:
            detector_protocols = [
                protocol for protocol in detector_protocols if protocol in protocol_set
            ]
        adjusted_detectors.append(replace(detector, protocols=detector_protocols))
    return replace(config, datasets=datasets, detectors=adjusted_detectors, protocols=protocols)


def _slice_payload(run: Path) -> dict[str, Any]:
    resolved_path = run / "resolved_config.json"
    if not resolved_path.exists():
        raise ReproductionError(f"Slice run is missing resolved_config.json: {run}")
    resolved = read_json(resolved_path)
    summary_path = run / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    return {
        "run": str(run),
        "resolved": resolved,
        "summary": summary,
        "config_hash": _hash_json(resolved),
        "fingerprints": _prepared_fingerprints(resolved),
    }


def _slice_compatibility(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    baseline = payloads[0]["resolved"]
    baseline_fingerprints = payloads[0]["fingerprints"]
    for payload in payloads[1:]:
        resolved = payload["resolved"]
        if resolved.get("assistant") != baseline.get("assistant"):
            errors.append(f"assistant config differs for {payload['run']}")
        if _benchmark_evaluation(resolved) != _benchmark_evaluation(baseline):
            errors.append(f"evaluation policy differs for {payload['run']}")
        for dataset_id, fingerprint in payload["fingerprints"].items():
            if (
                dataset_id in baseline_fingerprints
                and baseline_fingerprints[dataset_id] != fingerprint
            ):
                errors.append(f"prepared fingerprint differs for dataset {dataset_id}")
        errors.extend(
            f"{error} for {payload['run']}"
            for error in _compare_detector_configs(baseline, resolved)
        )
    return {
        "ok": not errors,
        "errors": errors,
        "config_hashes": [payload["config_hash"] for payload in payloads],
        "prepared_fingerprints": [payload["fingerprints"] for payload in payloads],
    }


def _benchmark_evaluation(resolved: dict[str, Any]) -> Any:
    benchmark = resolved.get("benchmark", {})
    if not isinstance(benchmark, dict):
        return None
    inner = benchmark.get("benchmark", {})
    return inner.get("evaluation") if isinstance(inner, dict) else None


def _compare_detector_configs(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    baseline_detectors = _detectors_by_id(baseline)
    for detector_id, detector_config in _detectors_by_id(candidate).items():
        if detector_id in baseline_detectors and baseline_detectors[detector_id] != detector_config:
            errors.append(f"detector config differs for {detector_id}")
    return errors


def _detectors_by_id(resolved: dict[str, Any]) -> dict[str, Any]:
    benchmark = resolved.get("benchmark", {})
    detectors = benchmark.get("detectors", []) if isinstance(benchmark, dict) else []
    if not isinstance(detectors, list):
        return {}
    return {
        str(detector.get("id")): detector
        for detector in detectors
        if isinstance(detector, dict) and detector.get("id")
    }


def _prepared_fingerprints(resolved: dict[str, Any]) -> dict[str, str]:
    benchmark = resolved.get("benchmark", {})
    datasets = benchmark.get("datasets", []) if isinstance(benchmark, dict) else []
    fingerprints: dict[str, str] = {}
    if not isinstance(datasets, list):
        return fingerprints
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        dataset_id = str(dataset.get("id", ""))
        prepared = dataset.get("prepared")
        if dataset_id and isinstance(prepared, str):
            fingerprints[dataset_id] = _prepared_fingerprint(Path(prepared))
    return fingerprints


def _prepared_fingerprint(prepared: Path) -> str:
    hasher = hashlib.sha256()
    for relative in ("meta/manifest.json", "meta/schema.json", "meta/splits.json"):
        path = prepared / relative
        hasher.update(relative.encode("utf-8"))
        if path.exists():
            hasher.update(path.read_bytes())
        else:
            hasher.update(f"missing:{path}".encode())
    return hasher.hexdigest()


def _hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _concat_csv_files(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        reader = csv.DictReader(io.StringIO(text))
        rows.extend(dict(row) for row in reader)
    return rows


def _stop_commands(container: str | None) -> list[str]:
    if not container:
        return ["Inspect the active run container, then stop it at a safe checkpoint."]
    return [f"docker stop {container}"]


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
    for summary_path in sorted(root.rglob("summary.csv")):
        text = summary_path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            relative = summary_path.relative_to(root)
            row["experiment_id"] = relative.parts[0] if relative.parts else ""
            if "evidence_source" not in row:
                row["evidence_source"] = relative.parts[1] if len(relative.parts) > 2 else "oracle"
            rows.append(dict(row))
    return rows


def _read_progress_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _benchmark_status(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        status_rows = _partial_benchmark_status_rows(root)
        if not status_rows:
            return {"status": "missing", "completed": 0, "failed": 0, "planned": 0}
        statuses = Counter(str(row.get("status", "unknown")) for row in status_rows)
        return {
            "status": "partial",
            "ok": False,
            "planned": len(status_rows),
            "completed": statuses.get("completed", 0),
            "failed": statuses.get("failed", 0),
            "by_status": dict(sorted(statuses.items())),
        }
    payload = read_json(summary_path)
    rows = payload.get("experiments", [])
    if not isinstance(rows, list):
        rows = []
    statuses = Counter(str(row.get("status", "unknown")) for row in rows if isinstance(row, dict))
    return {
        "status": "present",
        "ok": payload.get("ok"),
        "planned": len(rows),
        "completed": statuses.get("completed", 0),
        "failed": statuses.get("failed", 0),
        "by_status": dict(sorted(statuses.items())),
    }


def _partial_benchmark_status_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    experiments = root / "experiments"
    if not experiments.exists():
        return rows
    for status_path in sorted(experiments.glob("*/status.json")):
        try:
            payload = read_json(status_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _error_signature(error: str) -> str:
    text = error.strip()
    if "Native explanation artifacts contain no rows" in text:
        return "native_explanation_missing_event_overlap"
    if not text:
        return "unknown"
    return text.split(":", 1)[0]


def _diagnostic_classification(errors: list[str], benchmark: dict[str, Any]) -> str:
    if (
        errors
        and all(
            "Native explanation artifacts contain no rows" in error
            or "native_explanation_missing_event_overlap" in error
            for error in errors
        )
        and benchmark.get("failed") == 0
        and benchmark.get("completed")
    ):
        return "downstream_operational_native_evidence_coverage"
    if benchmark.get("failed", 0):
        return "benchmark_failures_present"
    if errors:
        return "downstream_failures_present"
    return "no_recorded_failures"


def _throughput_summary(events: list[dict[str, Any]], run: Path) -> dict[str, Any]:
    completed = [
        event
        for event in events
        if str(event.get("status")) == "completed" and event.get("duration_s") is not None
    ]
    slowest = sorted(
        completed,
        key=lambda event: float(event.get("duration_s") or 0.0),
        reverse=True,
    )[:10]
    benchmark_events = [event for event in completed if str(event.get("stage")) == "benchmark"]
    score_files = list((run / "benchmark" / "experiments").rglob("scores/*.parquet"))
    total_duration = sum(float(event.get("duration_s") or 0.0) for event in benchmark_events)
    files_per_hour = (len(score_files) / (total_duration / 3600.0)) if total_duration > 0 else None
    return {
        "completed_events_with_duration": len(completed),
        "benchmark_completed_events": len(benchmark_events),
        "score_parquet_files": len(score_files),
        "benchmark_files_per_hour": files_per_hour,
        "slowest_events": [
            {
                "stage": event.get("stage"),
                "item_id": event.get("item_id"),
                "duration_s": event.get("duration_s"),
                "path": event.get("path"),
            }
            for event in slowest
        ],
    }


def _diagnostic_only(run: Path, summary: dict[str, Any]) -> bool:
    if (run / "run_control" / "stopped_for_performance_triage.json").exists():
        return True
    if (run / "run_control" / "cancel_requested.json").exists():
        return True
    return bool(summary.get("assembled"))


def _diagnostic_next_files(run: Path) -> list[str]:
    candidates = [
        run / "progress_snapshot.json",
        run / "progress.jsonl",
        run / "benchmark" / "summary.json",
        run / "summaries" / "reproducibility_matrix.json",
        run / "summary.json",
    ]
    return [str(path) for path in candidates if path.exists()]


def _diagnostic_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reproduction Diagnostics",
        "",
        f"- Run: `{payload.get('run')}`",
        f"- Classification: `{payload.get('classification')}`",
        f"- Diagnostic only: `{payload.get('diagnostic_only')}`",
        f"- Summary ok: `{payload.get('summary_ok')}`",
        f"- Failed progress events: `{payload.get('failed_event_count')}`",
        f"- Failed stages: `{payload.get('failed_stage_count')}`",
        "",
        "## Benchmark",
        f"- Planned: `{dict(payload.get('benchmark', {})).get('planned')}`",
        f"- Completed: `{dict(payload.get('benchmark', {})).get('completed')}`",
        f"- Failed: `{dict(payload.get('benchmark', {})).get('failed')}`",
        "",
        "## Failures By Stage",
    ]
    for stage, count in dict(payload.get("failures_by_stage", {})).items():
        lines.append(f"- `{stage}`: {count}")
    lines.extend(["", "## Failures By Error"])
    for error, count in dict(payload.get("failures_by_error", {})).items():
        lines.append(f"- `{error}`: {count}")
    throughput = dict(payload.get("throughput", {}))
    lines.extend(
        [
            "",
            "## Throughput",
            f"- Score parquet files: `{throughput.get('score_parquet_files')}`",
            f"- Benchmark files/hour: `{throughput.get('benchmark_files_per_hour')}`",
            "- Slowest events:",
        ]
    )
    for event in throughput.get("slowest_events", []) or []:
        if isinstance(event, dict):
            lines.append(
                f"  - `{event.get('stage')}:{event.get('item_id')}` `{event.get('duration_s')}` s"
            )
    lines.extend(["", "## Next Files"])
    for path in payload.get("next_files", []):
        lines.append(f"- `{path}`")
    return "\n".join(lines).rstrip() + "\n"


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


def _append_sample(samples: list[Any], monitor: StageMonitor) -> None:
    if monitor.sample is not None:
        samples.append(monitor.sample)


def _write_inline_profile(run_root: Path, samples: list[Any]) -> None:
    if not samples:
        return
    profile_root = run_root / "profiles"
    summary = summarize_samples(samples)
    write_stage_csv(profile_root / "stages.csv", samples)
    LocalArtifactWriter(profile_root).write_json("summary.json", summary)
    LocalArtifactWriter(profile_root).write_text("budget_check.md", render_budget_markdown(summary))


def _resource_budget(config: ReproductionConfig) -> dict[str, Any]:
    env_keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "INDUSTRIAL_TSAD_CONTAINER_MEMORY_GB",
        "INDUSTRIAL_TSAD_CONTAINER_GPU",
        "INDUSTRIAL_TSAD_LLAMA_BASE_URL",
    ]
    return {
        "format_version": "reproduction-resource-budget-v1",
        "resources": config.resources.to_dict(),
        "host_logical_cpus": os.cpu_count(),
        "thread_env": {key: os.environ.get(key) for key in env_keys},
        "assistant_provider": {
            "name": config.assistant.provider.name,
            "base_url": config.assistant.provider.base_url,
            "model": config.assistant.provider.model,
            "extra": dict(config.assistant.provider.extra),
        },
    }


def _llama_gpu_offload_details(
    base_url: str | None = None,
    *,
    min_gpu_memory_mb: int = 1000,
) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    gpu_command = [
        "nvidia-smi",
        "--query-gpu=name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {
            "nvidia_smi_available": False,
            "active_llama_like_process": False,
            "probe_error": f"{type(exc).__name__}: {exc}",
        }
    try:
        gpu_completed = subprocess.run(
            gpu_command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        gpu_completed = None
        gpu_probe_error = f"{type(exc).__name__}: {exc}"
    else:
        gpu_probe_error = ""
    rows = _parse_nvidia_compute_rows(completed.stdout)
    llama_rows = [
        row
        for row in rows
        if "llama" in row.get("process_name", "").lower()
        or "python" in row.get("process_name", "").lower()
    ]
    gpu_summary = _parse_nvidia_gpu_row(gpu_completed.stdout if gpu_completed else "")
    gpu_memory_used_mb = gpu_summary.get("memory_used_mb")
    compute_memory_rows = [
        row
        for row in rows
        if (row.get("used_memory_mb") is not None and row["used_memory_mb"] >= min_gpu_memory_mb)
    ]
    gpu_memory_offload_evidence = bool(
        (gpu_memory_used_mb is not None and gpu_memory_used_mb >= min_gpu_memory_mb)
        or compute_memory_rows
    )
    return {
        "nvidia_smi_available": completed.returncode == 0,
        **_llama_endpoint_status(base_url),
        "active_llama_like_process": bool(llama_rows),
        "gpu_memory_offload_evidence": gpu_memory_offload_evidence,
        "min_gpu_memory_mb": min_gpu_memory_mb,
        "gpu": gpu_summary,
        "compute_processes": rows,
        "llama_like_processes": llama_rows,
        "compute_memory_rows": compute_memory_rows,
        "probe_stderr": completed.stderr.strip()[:1000],
        "gpu_probe_stderr": (gpu_completed.stderr.strip()[:1000] if gpu_completed else ""),
        "gpu_probe_error": gpu_probe_error,
    }


def _llama_endpoint_status(base_url: str | None) -> dict[str, Any]:
    if not base_url:
        return {"endpoint_ready": None, "endpoint_status_code": None}
    url = base_url.rstrip("/") + "/models"
    try:
        with urlopen(Request(url, method="GET"), timeout=3) as response:
            status_code = getattr(response, "status", None)
            return {
                "endpoint_ready": status_code is None or 200 <= int(status_code) < 500,
                "endpoint_status_code": status_code,
                "endpoint_url": url,
            }
    except HTTPError as exc:
        return {
            "endpoint_ready": 200 <= exc.code < 500,
            "endpoint_status_code": exc.code,
            "endpoint_url": url,
            "endpoint_error": f"{type(exc).__name__}: {exc}",
        }
    except (OSError, URLError, TimeoutError) as exc:
        return {
            "endpoint_ready": False,
            "endpoint_status_code": None,
            "endpoint_url": url,
            "endpoint_error": f"{type(exc).__name__}: {exc}",
        }


def _parse_nvidia_compute_rows(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            used_memory_mb: int | None = int(parts[2])
        except ValueError:
            used_memory_mb = None
        rows.append(
            {
                "pid": parts[0],
                "process_name": parts[1],
                "used_memory_mb": used_memory_mb,
            }
        )
    return rows


def _parse_nvidia_gpu_row(raw: str) -> dict[str, Any]:
    first = next((line for line in raw.splitlines() if line.strip()), "")
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 4:
        return {}
    return {
        "name": parts[0],
        "memory_used_mb": _int_or_none(parts[1]),
        "memory_total_mb": _int_or_none(parts[2]),
        "utilization_gpu_pct": _int_or_none(parts[3]),
    }


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _benchmark_workers(config: ReproductionConfig) -> int:
    return _resolve_workers(
        config.resources.benchmark_workers,
        config.resources.cpu_threads,
        max(1, config.resources.cpu_threads),
    )


def _evidence_workers(config: ReproductionConfig) -> int:
    return _resolve_workers(
        config.resources.evidence_workers,
        config.resources.cpu_threads,
        max(1, config.resources.cpu_threads),
    )


def _assistant_workers(config: ReproductionConfig) -> int:
    value = config.resources.assistant_workers
    if value == "conservative":
        return 1
    return _resolve_workers(value, config.resources.cpu_threads, 2)


def _resolve_workers(value: int | str, cpu_threads: int, cap: int) -> int:
    if isinstance(value, int):
        return max(1, value)
    if value == "auto":
        return max(1, min(cap, cpu_threads))
    if value == "conservative":
        return 1
    return max(1, int(value))


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
