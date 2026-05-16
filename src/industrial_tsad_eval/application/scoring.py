"""Detector scoring use case."""

from __future__ import annotations

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
    fitted_detector: Detector | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize public scoring fields without the in-memory fitted detector."""
        return {
            "dataset": self.dataset,
            "detector": self.detector,
            "protocol": self.protocol,
            "runs_scored": list(self.runs_scored),
            "scores_dir": self.scores_dir,
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
        with ThreadPoolExecutor(max_workers=self.write_workers) as executor:
            for run_id in run_ids:
                scores = detector.score_run(self.prepared_repository, run_id)
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
            }
        )
        return ScoreRunsResult(
            dataset=self.prepared_repository.dataset_name,
            detector=self.detector_name,
            protocol=self.protocol,
            runs_scored=run_ids,
            scores_dir=str(self.score_repository.root),
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
