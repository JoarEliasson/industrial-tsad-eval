"""Detector scoring use case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository
from industrial_tsad_eval.infrastructure.score_repository import LocalScoreRepository
from industrial_tsad_eval.plugins.registry import DetectorRegistry
from industrial_tsad_eval.ports.detectors import DetectorRunConfig


@dataclass(frozen=True)
class ScoreRunsResult:
    """Summary of a scoring use case execution."""

    dataset: str
    detector: str
    protocol: str
    runs_scored: list[str]
    scores_dir: str


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
    ):
        self.detector_registry = detector_registry
        self.prepared_repository = LocalPreparedDatasetRepository(prepared)
        self.score_repository = LocalScoreRepository(scores)
        self.detector_name = detector_name
        self.protocol = protocol
        self.detector_parameters = dict(detector_parameters or {})

    def run(self) -> ScoreRunsResult:
        """Execute detector training and scoring."""
        plugin = self.detector_registry.get(self.detector_name)
        detector = plugin.create(DetectorRunConfig(parameters=self.detector_parameters))
        detector.train(self.prepared_repository, self.protocol)

        run_ids = _runs_for_protocol(self.prepared_repository.splits(), self.protocol)
        for run_id in run_ids:
            scores = detector.score_run(self.prepared_repository, run_id)
            self.score_repository.write_run_scores(run_id, scores)

        self.score_repository.write_manifest()
        self.score_repository.write_model_metadata(
            {
                **detector.metadata(),
                "dataset": self.prepared_repository.dataset_name,
                "protocol": self.protocol,
                "runs_scored": run_ids,
            }
        )
        return ScoreRunsResult(
            dataset=self.prepared_repository.dataset_name,
            detector=self.detector_name,
            protocol=self.protocol,
            runs_scored=run_ids,
            scores_dir=str(self.score_repository.root),
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
