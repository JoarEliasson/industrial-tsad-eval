"""Detector scoring use case."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from industrial_tsad_eval.infrastructure.explanation_repository import LocalExplanationRepository
from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository
from industrial_tsad_eval.infrastructure.score_repository import LocalScoreRepository
from industrial_tsad_eval.plugins.registry import DetectorRegistry
from industrial_tsad_eval.ports.detectors import Detector, DetectorRunConfig
from industrial_tsad_eval.ports.repositories import PreparedDatasetRepository


@dataclass(frozen=True)
class ScoreRunsResult:
    """Summary of a scoring use case execution."""

    dataset: str
    detector: str
    protocol: str
    runs_scored: list[str]
    scores_dir: str
    telemetry: dict[str, Any]
    fitted_detector: Detector | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize public scoring fields without the in-memory fitted detector."""
        return {
            "dataset": self.dataset,
            "detector": self.detector,
            "protocol": self.protocol,
            "runs_scored": list(self.runs_scored),
            "scores_dir": self.scores_dir,
            "telemetry": dict(self.telemetry),
        }


class ScoreRuns:
    """Train a detector plugin and produce Score Contract v1 artifacts."""

    def __init__(
        self,
        *,
        detector_registry: DetectorRegistry,
        prepared: str | Path,
        scores: str | Path,
        detector_name: str,
        protocol: str = "naive",
        detector_parameters: dict[str, Any] | None = None,
        explanation_mode: Literal["all", "none"] = "all",
        write_workers: int = 1,
    ):
        self.detector_registry = detector_registry
        self.prepared_repository = LocalPreparedDatasetRepository(prepared)
        self.score_repository = LocalScoreRepository(scores)
        self.explanation_repository = LocalExplanationRepository(Path(scores) / "explanations")
        self.detector_name = detector_name
        self.protocol = protocol
        self.detector_parameters = dict(detector_parameters or {})
        self.explanation_mode = explanation_mode
        self.write_workers = max(1, int(write_workers))
        if explanation_mode not in {"all", "none"}:
            raise ValueError("explanation_mode must be 'all' or 'none'.")

    def run(self) -> ScoreRunsResult:
        """Execute detector training and scoring."""
        plugin = self.detector_registry.get(self.detector_name)
        detector = plugin.create(DetectorRunConfig(parameters=self.detector_parameters))
        detector.train(self.prepared_repository, self.protocol)

        run_ids = _runs_for_protocol(self.prepared_repository.splits(), self.protocol)
        explanation_run_ids: list[str] = []
        write_futures: list[Any] = []
        score_started = time.perf_counter()
        scores_by_run = _score_runs(detector, self.prepared_repository, run_ids)
        score_duration_s = round(time.perf_counter() - score_started, 6)
        with ThreadPoolExecutor(max_workers=self.write_workers) as executor:
            for run_id, scores in scores_by_run.items():
                write_futures.append(
                    executor.submit(self.score_repository.write_run_scores, run_id, scores)
                )
                explain_run = getattr(detector, "explain_run", None)
                if self.explanation_mode == "all" and callable(explain_run):
                    explanations = explain_run(self.prepared_repository, run_id)
                    if isinstance(explanations, pd.DataFrame) and not explanations.empty:
                        write_futures.append(
                            executor.submit(
                                self.explanation_repository.write_run_explanations,
                                run_id,
                                explanations,
                            )
                        )
                        explanation_run_ids.append(run_id)
            for future in as_completed(write_futures):
                future.result()

        telemetry = self.score_repository.write_combined_scores(scores_by_run)
        detector_telemetry: dict[str, Any] = getattr(
            detector,
            "score_batch_telemetry",
            lambda: {},
        )()
        total_windows = _total_windows(detector_telemetry)
        telemetry.update(
            {
                "per_run_score_files": len(scores_by_run),
                "score_file_write_count": len(scores_by_run),
                "score_write_workers": self.write_workers,
                "batch_api_used": callable(getattr(detector, "score_runs", None)),
                "score_duration_s": score_duration_s,
                "score_windows_per_second": (
                    round(total_windows / score_duration_s, 6)
                    if total_windows is not None and score_duration_s > 0
                    else None
                ),
                "detector_batch_telemetry": detector_telemetry,
            }
        )
        self.score_repository.write_manifest()
        self.explanation_repository.write_manifest()
        self.explanation_repository.write_metadata(
            {
                "detector": self.detector_name,
                "dataset": self.prepared_repository.dataset_name,
                "protocol": self.protocol,
                "runs_explained": explanation_run_ids,
            }
        )
        self.score_repository.write_model_metadata(
            {
                **detector.metadata(),
                "dataset": self.prepared_repository.dataset_name,
                "protocol": self.protocol,
                "runs_scored": run_ids,
                "runs_explained": explanation_run_ids,
                "scoring_telemetry": telemetry,
            }
        )
        return ScoreRunsResult(
            dataset=self.prepared_repository.dataset_name,
            detector=self.detector_name,
            protocol=self.protocol,
            runs_scored=run_ids,
            scores_dir=str(self.score_repository.root),
            telemetry=telemetry,
            fitted_detector=detector,
        )


def _runs_for_protocol(splits: dict[str, Any], protocol: str) -> list[str]:
    selected = splits.get(protocol, splits.get("naive", splits))
    if not isinstance(selected, dict):
        raise ValueError(f"Split protocol {protocol!r} is not an object.")
    ordered: list[str] = []
    seen: set[str] = set()
    for split_name in ("train_runs", "val_runs", "test_runs"):
        for run_id in selected.get(split_name, []):
            run_text = str(run_id)
            if run_text not in seen:
                ordered.append(run_text)
                seen.add(run_text)
    return ordered


def _score_runs(
    detector: Detector,
    repository: PreparedDatasetRepository,
    run_ids: list[str],
) -> dict[str, pd.DataFrame]:
    score_runs = getattr(detector, "score_runs", None)
    if callable(score_runs):
        result = score_runs(repository, run_ids)
        if not isinstance(result, dict):
            raise TypeError("Detector batch score_runs must return a dict of run_id to DataFrame.")
        return {str(run_id): frame for run_id, frame in result.items()}
    return {run_id: detector.score_run(repository, run_id) for run_id in run_ids}


def _total_windows(telemetry: Any) -> int | None:
    if not isinstance(telemetry, dict):
        return None
    value = telemetry.get("total_windows")
    if value is None:
        return None
    return int(value)


def write_detector_explanations(
    *,
    detector: Detector,
    prepared_repository: PreparedDatasetRepository,
    explanation_repository: LocalExplanationRepository,
    run_ids: list[str],
    metadata: dict[str, Any],
    write_workers: int = 1,
    refresh_score_context: bool = False,
) -> list[str]:
    """Write native explanations for a selected set of runs when the detector supports it."""
    explain_run = getattr(detector, "explain_run", None)
    if not callable(explain_run):
        return []

    explained: list[str] = []
    futures: list[Any] = []
    with ThreadPoolExecutor(max_workers=max(1, int(write_workers))) as executor:
        for run_id in run_ids:
            if refresh_score_context:
                detector.score_run(prepared_repository, run_id)
            explanations = explain_run(prepared_repository, run_id)
            if isinstance(explanations, pd.DataFrame) and not explanations.empty:
                futures.append(
                    executor.submit(
                        explanation_repository.write_run_explanations,
                        run_id,
                        explanations,
                    )
                )
                explained.append(run_id)
        for future in as_completed(futures):
            future.result()
    explanation_repository.write_manifest()
    explanation_repository.write_metadata({**metadata, "runs_explained": explained})
    return explained
