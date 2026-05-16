"""Benchmark orchestration use case."""

from __future__ import annotations

import csv
import io
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.scoring import ScoreRuns, write_detector_explanations
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkExperiment,
    BenchmarkExperimentResult,
    sanitize_run_id,
)
from industrial_tsad_eval.domain.errors import BenchmarkRunError, IndustrialTSADError
from industrial_tsad_eval.domain.progress import CompositeProgressSink, ProgressEvent, ProgressSink
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.benchmark_config import render_benchmark_config_toml
from industrial_tsad_eval.infrastructure.explanation_repository import LocalExplanationRepository
from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository
from industrial_tsad_eval.infrastructure.progress import LocalProgressSink
from industrial_tsad_eval.plugins.registry import DetectorRegistry

SUMMARY_COLUMNS = [
    "experiment_id",
    "dataset",
    "detector",
    "protocol",
    "status",
    "threshold",
    "event_precision",
    "event_recall",
    "event_f1",
    "event_n_gt",
    "event_n_pred",
    "event_n_hits",
    "delay_mean_ns",
    "far_false_events_per_hour",
    "point_adjusted_f1",
    "affiliation_precision",
    "affiliation_recall",
    "affiliation_f1",
    "scores_dir",
    "eval_dir",
    "error",
]


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Summary returned by benchmark execution."""

    run_id: str
    run_dir: str
    ok: bool
    results: list[BenchmarkExperimentResult]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "ok": self.ok,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass
class _QueueState:
    """Thread-safe benchmark queue counters for progress snapshots."""

    run_id: str
    total: int
    cpu_queued: int = 0
    gpu_queued: int = 0
    cpu_running: int = 0
    gpu_running: int = 0
    completed: int = 0
    failed: int = 0

    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def emit(self, progress: ProgressSink) -> None:
        with self._lock:
            metrics = {
                "cpu_running": self.cpu_running,
                "gpu_running": self.gpu_running,
                "cpu_queued": self.cpu_queued,
                "gpu_queued": self.gpu_queued,
                "queued": self.cpu_queued + self.gpu_queued,
                "completed": self.completed,
                "failed": self.failed,
            }
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="benchmark_queue",
                item_id="state",
                status="running",
                ordinal=metrics["completed"] + metrics["failed"],
                total=self.total,
                metrics=metrics,
            )
        )

    def mark_started(self, queue: str, progress: ProgressSink) -> None:
        with self._lock:
            if queue == "gpu":
                self.gpu_queued = max(0, self.gpu_queued - 1)
                self.gpu_running += 1
            else:
                self.cpu_queued = max(0, self.cpu_queued - 1)
                self.cpu_running += 1
        self.emit(progress)

    def mark_finished(self, queue: str, status: str, progress: ProgressSink) -> None:
        with self._lock:
            if queue == "gpu":
                self.gpu_running = max(0, self.gpu_running - 1)
            else:
                self.cpu_running = max(0, self.cpu_running - 1)
            if status == "completed":
                self.completed += 1
            else:
                self.failed += 1
        self.emit(progress)


class RunBenchmark:
    """Run a detector/dataset/protocol benchmark matrix."""

    def __init__(
        self,
        *,
        config: BenchmarkConfig,
        detector_registry: DetectorRegistry,
        out: str | Path,
        run_id: str | None = None,
        source_config: str | Path | None = None,
        progress_sink: ProgressSink | None = None,
        worker_count: int = 1,
        gpu_slots: int = 1,
    ):
        self.config = config
        self.detector_registry = detector_registry
        self.out = Path(out)
        self.run_id = run_id or _default_run_id(config.name)
        self.source_config = Path(source_config) if source_config is not None else None
        self.progress_sink = progress_sink
        self.worker_count = max(int(worker_count), 1)
        self.gpu_slots = max(int(gpu_slots), 0)

    def run(self) -> BenchmarkRunResult:
        """Execute the benchmark matrix and write run artifacts."""
        run_root = self.out / self.run_id
        if run_root.exists():
            raise BenchmarkRunError(f"Benchmark run already exists: {run_root}")

        writer = LocalArtifactWriter(run_root)
        experiments = self.config.experiments()
        progress = CompositeProgressSink(
            [LocalProgressSink(run_root, self.run_id), self.progress_sink]
        )
        started_at = _utc_now()
        results: list[BenchmarkExperimentResult] = []

        _write_config_artifacts(writer, self.config, self.source_config)
        writer.write_json(
            "run_manifest.json",
            {
                "run_id": self.run_id,
                "benchmark": self.config.name,
                "status": "running",
                "started_at_utc": started_at,
                "experiment_count": len(experiments),
            },
        )

        validation_errors = _validate_datasets(self.config)
        for ordinal, experiment in enumerate(experiments, start=1):
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="benchmark",
                    item_id=experiment.experiment_id,
                    status="planned",
                    ordinal=ordinal,
                    total=len(experiments),
                    path=str(run_root / "experiments" / experiment.experiment_id),
                )
            )
        results = self._run_experiments(run_root, experiments, validation_errors, progress)

        ok = all(result.status == "completed" for result in results)
        rows = [summary_row(result) for result in results]
        writer.write_json(
            "summary.json",
            {
                "run_id": self.run_id,
                "benchmark": self.config.name,
                "ok": ok,
                "experiments": rows,
            },
        )
        writer.write_text("summary.csv", _summary_csv(rows))
        writer.write_json(
            "run_manifest.json",
            {
                "run_id": self.run_id,
                "benchmark": self.config.name,
                "status": "completed" if ok else "failed",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "experiment_count": len(experiments),
                "completed_count": sum(result.status == "completed" for result in results),
                "failed_count": sum(result.status == "failed" for result in results),
            },
        )
        return BenchmarkRunResult(
            run_id=self.run_id,
            run_dir=str(run_root),
            ok=ok,
            results=results,
        )

    def _run_experiments(
        self,
        run_root: Path,
        experiments: list[BenchmarkExperiment],
        validation_errors: dict[str, list[str]],
        progress: ProgressSink,
    ) -> list[BenchmarkExperimentResult]:
        if self.worker_count == 1 or len(experiments) <= 1:
            return [
                self._run_experiment_with_progress(
                    run_root,
                    experiment,
                    validation_errors,
                    progress,
                    ordinal,
                    len(experiments),
                )
                for ordinal, experiment in enumerate(experiments, start=1)
            ]

        ordered: list[BenchmarkExperimentResult | None] = [None] * len(experiments)
        gpu_slots = max(1, self.gpu_slots)
        gpu_tasks: list[tuple[int, BenchmarkExperiment]] = []
        cpu_tasks: list[tuple[int, BenchmarkExperiment]] = []
        for ordinal, experiment in enumerate(experiments, start=1):
            if self._requires_gpu(experiment):
                gpu_tasks.append((ordinal, experiment))
            else:
                cpu_tasks.append((ordinal, experiment))

        queue_state = _QueueState(
            run_id=self.run_id,
            total=len(experiments),
            cpu_queued=len(cpu_tasks),
            gpu_queued=len(gpu_tasks),
        )
        queue_state.emit(progress)
        cpu_workers = max(1, min(len(cpu_tasks), self.worker_count)) if cpu_tasks else 0
        if cpu_tasks and gpu_tasks:
            cpu_workers = max(1, min(len(cpu_tasks), max(1, self.worker_count - gpu_slots)))
        gpu_workers = min(gpu_slots, len(gpu_tasks))
        futures: dict[Any, int] = {}
        with (
            ThreadPoolExecutor(max_workers=cpu_workers or 1) as cpu_executor,
            ThreadPoolExecutor(max_workers=gpu_workers or 1) as gpu_executor,
        ):
            for ordinal, experiment in cpu_tasks:
                futures[
                    cpu_executor.submit(
                        self._run_queued_experiment,
                        "cpu",
                        run_root,
                        experiment,
                        validation_errors,
                        progress,
                        ordinal,
                        len(experiments),
                        queue_state,
                    )
                ] = ordinal - 1
            for ordinal, experiment in gpu_tasks:
                futures[
                    gpu_executor.submit(
                        self._run_queued_experiment,
                        "gpu",
                        run_root,
                        experiment,
                        validation_errors,
                        progress,
                        ordinal,
                        len(experiments),
                        queue_state,
                    )
                ] = ordinal - 1
            for future in as_completed(futures):
                ordered[futures[future]] = future.result()
        return [result for result in ordered if result is not None]

    def _requires_gpu(self, experiment: BenchmarkExperiment) -> bool:
        try:
            return bool(
                getattr(
                    self.detector_registry.get(experiment.detector.name),
                    "requires_torch",
                    False,
                )
            )
        except IndustrialTSADError:
            return False

    def _run_queued_experiment(
        self,
        queue: str,
        run_root: Path,
        experiment: BenchmarkExperiment,
        validation_errors: dict[str, list[str]],
        progress: ProgressSink,
        ordinal: int,
        total: int,
        queue_state: _QueueState,
    ) -> BenchmarkExperimentResult:
        queue_state.mark_started(queue, progress)
        result = self._run_experiment_with_progress(
            run_root, experiment, validation_errors, progress, ordinal, total
        )
        queue_state.mark_finished(queue, result.status, progress)
        return result

    def _run_experiment_with_progress(
        self,
        run_root: Path,
        experiment: BenchmarkExperiment,
        validation_errors: dict[str, list[str]],
        progress: ProgressSink,
        ordinal: int,
        total: int,
    ) -> BenchmarkExperimentResult:
        started = time.perf_counter()
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="benchmark",
                item_id=experiment.experiment_id,
                status="running",
                ordinal=ordinal,
                total=total,
                path=str(run_root / "experiments" / experiment.experiment_id),
                message=f"{experiment.dataset.id}/{experiment.detector.id}/{experiment.protocol}",
            )
        )
        result = self._run_experiment(run_root, experiment, validation_errors)
        progress.emit(
            ProgressEvent(
                run_id=self.run_id,
                stage="benchmark",
                item_id=experiment.experiment_id,
                status="completed" if result.status == "completed" else "failed",
                ordinal=ordinal,
                total=total,
                path=str(run_root / "experiments" / experiment.experiment_id),
                duration_s=round(time.perf_counter() - started, 6),
                metrics=_progress_metrics(result),
                error=result.error,
            )
        )
        return result

    def _run_experiment(
        self,
        run_root: Path,
        experiment: BenchmarkExperiment,
        validation_errors: dict[str, list[str]],
    ) -> BenchmarkExperimentResult:
        experiment_root = run_root / "experiments" / experiment.experiment_id
        writer = LocalArtifactWriter(experiment_root)
        scores_dir = experiment_root / "scores"
        eval_dir = experiment_root / "eval"

        writer.write_json(
            "status.json",
            {
                "experiment_id": experiment.experiment_id,
                "status": "running",
                "started_at_utc": _utc_now(),
            },
        )
        if experiment.dataset.id in validation_errors:
            result = _failed_result(
                experiment,
                f"Prepared dataset validation failed: {validation_errors[experiment.dataset.id]}",
                scores_dir,
                eval_dir,
            )
            writer.write_json("status.json", result.to_dict())
            return result

        try:
            score_result = ScoreRuns(
                detector_registry=self.detector_registry,
                prepared=experiment.dataset.prepared,
                scores=scores_dir,
                detector_name=experiment.detector.name,
                protocol=experiment.protocol,
                detector_parameters=experiment.detector.parameters,
                explanation_mode="none",
                write_workers=_score_write_workers(self.worker_count),
            ).run()
            score_report = ValidateScores(experiment.dataset.prepared, scores_dir).run()
            if not score_report.ok:
                raise BenchmarkRunError(f"Score validation failed: {score_report.errors}")
            eval_result = EvaluateScores(
                prepared=experiment.dataset.prepared,
                scores=scores_dir,
                out=eval_dir,
                protocol=experiment.protocol,
                policy=self.config.evaluation.policy_for(
                    experiment.dataset.id,
                    experiment.protocol,
                ),
            ).run()
            explained_runs = _write_selected_native_explanations(
                fitted_detector=score_result.fitted_detector,
                prepared=experiment.dataset.prepared,
                scores_dir=scores_dir,
                eval_dir=eval_dir,
                protocol=experiment.protocol,
                detector_id=experiment.detector.id,
                dataset=experiment.dataset.id,
                write_workers=_score_write_workers(self.worker_count),
            )
            result = BenchmarkExperimentResult(
                experiment_id=experiment.experiment_id,
                dataset=experiment.dataset.id,
                detector=experiment.detector.id,
                protocol=experiment.protocol,
                status="completed",
                scores_dir=str(scores_dir),
                eval_dir=str(eval_dir),
                threshold=eval_result.threshold,
                metrics=eval_result.metrics,
            )
            writer.write_json(
                "status.json",
                {
                    **result.to_dict(),
                    "finished_at_utc": _utc_now(),
                    "runs_scored": score_result.runs_scored,
                    "runs_explained": explained_runs,
                    "detector_parameters": dict(experiment.detector.parameters),
                    "evaluation_policy": eval_result.metrics.get("policy"),
                },
            )
            return result
        except (
            IndustrialTSADError,
            ValueError,
            RuntimeError,
            FileNotFoundError,
            MemoryError,
        ) as exc:
            result = _failed_result(
                experiment,
                f"{type(exc).__name__}: {exc}",
                scores_dir,
                eval_dir,
            )
            writer.write_json("status.json", {**result.to_dict(), "finished_at_utc": _utc_now()})
            return result


def summary_row(result: BenchmarkExperimentResult) -> dict[str, Any]:
    """Flatten an experiment result into the public summary row contract."""
    metrics = result.metrics or {}
    event = _mapping(metrics.get("event"))
    delay = _mapping(metrics.get("delay"))
    far = _mapping(metrics.get("far"))
    point_adjusted = _mapping(metrics.get("point_adjusted"))
    affiliation = _mapping(metrics.get("affiliation"))
    return {
        "experiment_id": result.experiment_id,
        "dataset": result.dataset,
        "detector": result.detector,
        "protocol": result.protocol,
        "status": result.status,
        "threshold": result.threshold,
        "event_precision": event.get("precision"),
        "event_recall": event.get("recall"),
        "event_f1": event.get("f1"),
        "event_n_gt": event.get("n_gt"),
        "event_n_pred": event.get("n_pred"),
        "event_n_hits": event.get("n_hits"),
        "delay_mean_ns": delay.get("mean"),
        "far_false_events_per_hour": far.get("false_events_per_hour"),
        "point_adjusted_f1": point_adjusted.get("f1"),
        "affiliation_precision": affiliation.get("precision"),
        "affiliation_recall": affiliation.get("recall"),
        "affiliation_f1": affiliation.get("f1"),
        "scores_dir": result.scores_dir,
        "eval_dir": result.eval_dir,
        "error": result.error,
    }


def _write_selected_native_explanations(
    *,
    fitted_detector: Any,
    prepared: str,
    scores_dir: Path,
    eval_dir: Path,
    protocol: str,
    detector_id: str,
    dataset: str,
    write_workers: int,
) -> list[str]:
    if fitted_detector is None or not callable(getattr(fitted_detector, "explain_run", None)):
        return []
    prepared_repository = LocalPreparedDatasetRepository(prepared)
    run_ids = _selected_explanation_run_ids(prepared_repository, eval_dir, protocol)
    if not run_ids:
        return []
    return write_detector_explanations(
        detector=fitted_detector,
        prepared_repository=prepared_repository,
        explanation_repository=LocalExplanationRepository(scores_dir / "explanations"),
        run_ids=run_ids,
        write_workers=write_workers,
        refresh_score_context=True,
        metadata={
            "detector": detector_id,
            "dataset": dataset,
            "protocol": protocol,
            "selection": "event-window-runs",
            "selection_event_limit": 100,
        },
    )


def _selected_explanation_run_ids(
    repository: LocalPreparedDatasetRepository,
    eval_dir: Path,
    protocol: str,
    *,
    event_limit: int = 100,
) -> list[str]:
    split = _protocol_split(repository.splits(), protocol)
    test_runs = set(split.get("test_runs", []))
    selected: list[str] = []
    seen: set[str] = set()
    for event in sorted(
        repository.read_events(),
        key=lambda item: (item.run_id, item.start_ts_ns, item.event_id),
    )[:event_limit]:
        if test_runs and event.run_id not in test_runs:
            continue
        if event.run_id not in seen:
            selected.append(event.run_id)
            seen.add(event.run_id)
    matches_path = eval_dir / "event_matches.json"
    if matches_path.exists():
        payload = json.loads(matches_path.read_text(encoding="utf-8"))
        pred_events = payload.get("pred_events", [])
        if isinstance(pred_events, list):
            for raw_event in pred_events[:event_limit]:
                if not isinstance(raw_event, dict):
                    continue
                run_id = str(raw_event.get("run_id", ""))
                if run_id and run_id not in seen:
                    selected.append(run_id)
                    seen.add(run_id)
    return selected


def _protocol_split(splits: dict[str, Any], protocol: str) -> dict[str, list[str]]:
    selected = splits.get(protocol, splits.get("naive", splits))
    if not isinstance(selected, dict):
        raise ValueError(f"Split protocol {protocol!r} is not an object.")
    return {
        "train_runs": [str(run_id) for run_id in selected.get("train_runs", [])],
        "val_runs": [str(run_id) for run_id in selected.get("val_runs", [])],
        "test_runs": [str(run_id) for run_id in selected.get("test_runs", [])],
    }


def _score_write_workers(worker_count: int) -> int:
    return max(1, min(4, worker_count))


def _progress_metrics(result: BenchmarkExperimentResult) -> dict[str, Any]:
    metrics = result.metrics or {}
    event = _mapping(metrics.get("event"))
    return {
        "status": result.status,
        "threshold": result.threshold,
        "event_f1": event.get("f1"),
        "event_n_gt": event.get("n_gt"),
        "event_n_pred": event.get("n_pred"),
    }


def _validate_datasets(config: BenchmarkConfig) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {}
    for dataset_config in config.datasets:
        report = ValidatePreparedDataset(dataset_config.prepared).run()
        if not report.ok:
            errors[dataset_config.id] = report.errors
    return errors


def _write_config_artifacts(
    writer: LocalArtifactWriter,
    config: BenchmarkConfig,
    source_config: Path | None,
) -> None:
    if source_config is not None and source_config.exists():
        writer.write_text(
            "config/benchmark.toml",
            source_config.read_text(encoding="utf-8"),
        )
    else:
        writer.write_text("config/benchmark.toml", render_benchmark_config_toml(config))
    writer.write_json("resolved_config.json", config.to_dict())


def _failed_result(
    experiment: BenchmarkExperiment,
    error: str,
    scores_dir: Path,
    eval_dir: Path,
) -> BenchmarkExperimentResult:
    return BenchmarkExperimentResult(
        experiment_id=experiment.experiment_id,
        dataset=experiment.dataset.id,
        detector=experiment.detector.id,
        protocol=experiment.protocol,
        status="failed",
        scores_dir=str(scores_dir),
        eval_dir=str(eval_dir),
        error=error,
    )


def _summary_csv(rows: list[dict[str, Any]]) -> str:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return handle.getvalue()


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _default_run_id(name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{sanitize_run_id(name)}-{timestamp}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
