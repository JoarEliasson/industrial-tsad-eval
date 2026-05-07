"""Benchmark orchestration use case."""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkExperiment,
    BenchmarkExperimentResult,
    sanitize_run_id,
)
from industrial_tsad_eval.domain.errors import BenchmarkRunError, IndustrialTSADError
from industrial_tsad_eval.domain.policy import EvalPolicy
from industrial_tsad_eval.domain.progress import CompositeProgressSink, ProgressEvent, ProgressSink
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.benchmark_config import render_benchmark_config_toml
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
    ):
        self.config = config
        self.detector_registry = detector_registry
        self.out = Path(out)
        self.run_id = run_id or _default_run_id(config.name)
        self.source_config = Path(source_config) if source_config is not None else None
        self.progress_sink = progress_sink

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
        for ordinal, experiment in enumerate(experiments, start=1):
            started = time.perf_counter()
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="benchmark",
                    item_id=experiment.experiment_id,
                    status="running",
                    ordinal=ordinal,
                    total=len(experiments),
                    path=str(run_root / "experiments" / experiment.experiment_id),
                    message=f"{experiment.dataset.id}/{experiment.detector.id}/{experiment.protocol}",
                )
            )
            result = self._run_experiment(run_root, experiment, validation_errors)
            results.append(result)
            progress.emit(
                ProgressEvent(
                    run_id=self.run_id,
                    stage="benchmark",
                    item_id=experiment.experiment_id,
                    status="completed" if result.status == "completed" else "failed",
                    ordinal=ordinal,
                    total=len(experiments),
                    path=str(run_root / "experiments" / experiment.experiment_id),
                    duration_s=round(time.perf_counter() - started, 6),
                    metrics=_progress_metrics(result),
                    error=result.error,
                )
            )

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
            ).run()
            score_report = ValidateScores(experiment.dataset.prepared, scores_dir).run()
            if not score_report.ok:
                raise BenchmarkRunError(f"Score validation failed: {score_report.errors}")
            eval_result = EvaluateScores(
                prepared=experiment.dataset.prepared,
                scores=scores_dir,
                out=eval_dir,
                protocol=experiment.protocol,
                policy=EvalPolicy(
                    protocol=experiment.protocol,
                    threshold_quantile=self.config.evaluation.threshold_quantile,
                ),
            ).run()
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
                },
            )
            return result
        except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
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
        "scores_dir": result.scores_dir,
        "eval_dir": result.eval_dir,
        "error": result.error,
    }


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
