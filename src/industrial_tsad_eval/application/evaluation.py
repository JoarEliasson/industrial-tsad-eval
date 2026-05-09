"""Score evaluation use case."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from industrial_tsad_eval.domain.evaluation import (
    cluster_anomalies_to_pred_events,
    compute_affiliation_prf,
    compute_delays,
    compute_event_prf,
    compute_false_alarm_rates,
    compute_point_prf_for_scores,
    infer_period_ns,
    match_pred_to_gt_events,
    resolve_effective_merge_gap_ns,
)
from industrial_tsad_eval.domain.events import PredEvent
from industrial_tsad_eval.domain.policy import EvalPolicy
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository
from industrial_tsad_eval.infrastructure.score_repository import LocalScoreRepository


@dataclass(frozen=True)
class EvaluationResult:
    """Summary of score evaluation."""

    dataset: str
    protocol: str
    threshold: float
    metrics: dict[str, Any]
    out_dir: str


class EvaluateScores:
    """Evaluate Score Contract v1 artifacts against prepared events."""

    def __init__(
        self,
        *,
        prepared: str | Path,
        scores: str | Path,
        out: str | Path,
        protocol: str = "naive",
        policy: EvalPolicy | None = None,
        threshold: float | None = None,
    ):
        self.prepared_repository = LocalPreparedDatasetRepository(prepared)
        self.score_repository = LocalScoreRepository(scores)
        self.artifact_writer = LocalArtifactWriter(out)
        self.protocol = protocol
        self.policy = policy or EvalPolicy(protocol=protocol)
        self.threshold = threshold

    def run(self) -> EvaluationResult:
        """Run threshold calibration, event clustering, and metrics."""
        split = _protocol_split(self.prepared_repository.splits(), self.protocol)
        threshold = self.threshold
        if threshold is None:
            threshold = _calibrate_threshold(
                self.score_repository,
                split["train_runs"] + split["val_runs"],
                self.policy.threshold_quantile,
            )
        config = self.policy.to_config(threshold)
        score_files = self.score_repository.discover()
        pred_events_by_run: dict[str, list[PredEvent]] = {}
        run_durations_ns: dict[str, int] = {}
        window_count_by_run: dict[str, int] = {}
        score_timestamps_by_run: dict[str, np.ndarray] = {}
        score_values_by_run: dict[str, np.ndarray] = {}

        for run_id in sorted(score_files):
            score_frame = self.score_repository.read_run_scores(run_id)
            timestamps = score_frame["ts_ns"].to_numpy(dtype=np.int64)
            scores = score_frame["score"].to_numpy(dtype=np.float64)
            score_timestamps_by_run[run_id] = timestamps
            score_values_by_run[run_id] = scores
            period_ns = infer_period_ns(timestamps) or _prepared_period_ns(
                self.prepared_repository,
                run_id,
            )
            merge_gap_ns = resolve_effective_merge_gap_ns(
                config.merge_gap_ns,
                period_ns,
                config.merge_gap_mode,
                config.merge_gap_skipped_samples,
                config.merge_gap_jitter_ratio,
            )
            anomaly_ts_ns = timestamps[scores >= threshold]
            pred_events_by_run[run_id] = cluster_anomalies_to_pred_events(
                run_id,
                anomaly_ts_ns,
                merge_gap_ns,
                period_ns,
                period_ns,
            )
            run_durations_ns[run_id] = _run_duration_ns(self.prepared_repository, run_id)
            window_count_by_run[run_id] = int(len(score_frame))

        all_pred_events = [
            event for events_for_run in pred_events_by_run.values() for event in events_for_run
        ]
        gt_events = self.prepared_repository.read_events(config.event_types)
        gt_matches, pred_matches = match_pred_to_gt_events(
            gt_events,
            all_pred_events,
            config.grace_ns,
        )
        metrics = {
            "dataset": self.prepared_repository.dataset_name,
            "protocol": self.protocol,
            "threshold": threshold,
            "policy": self.policy.to_dict(),
        }
        compute_groups = set(config.compute)
        if "event" in compute_groups:
            metrics["event"] = compute_event_prf(gt_events, all_pred_events, gt_matches)
        if "delay" in compute_groups:
            metrics["delay"] = compute_delays(gt_events, gt_matches, all_pred_events)
        if "far" in compute_groups:
            metrics["far"] = compute_false_alarm_rates(
                split["train_runs"] + split["val_runs"],
                pred_events_by_run,
                run_durations_ns,
                window_count_by_run,
            )
        if "point" in compute_groups:
            metrics["point"] = compute_point_prf_for_scores(
                score_timestamps_by_run,
                score_values_by_run,
                gt_events,
                threshold,
            )
        if "point_adjusted" in compute_groups:
            metrics["point_adjusted"] = compute_point_prf_for_scores(
                score_timestamps_by_run,
                score_values_by_run,
                gt_events,
                threshold,
                point_adjusted=True,
            )
        if "affiliation" in compute_groups:
            metrics["affiliation"] = compute_affiliation_prf(gt_events, all_pred_events)
        self.artifact_writer.write_json("metrics.json", metrics)
        self.artifact_writer.write_json(
            "event_matches.json",
            {
                "gt_matches": gt_matches,
                "pred_matches": pred_matches,
                "pred_events": [_pred_event_to_dict(event) for event in all_pred_events],
            },
        )
        self.artifact_writer.write_json(
            "threshold.json",
            {
                "threshold": threshold,
                "source": self.policy.threshold_source if self.threshold is None else "manual",
                "q": self.policy.threshold_quantile,
            },
        )
        return EvaluationResult(
            dataset=self.prepared_repository.dataset_name,
            protocol=self.protocol,
            threshold=threshold,
            metrics=metrics,
            out_dir=str(self.artifact_writer.root),
        )


def _calibrate_threshold(
    score_repository: LocalScoreRepository,
    run_ids: list[str],
    q: float,
) -> float:
    score_arrays: list[np.ndarray] = []
    available = score_repository.discover()
    for run_id in run_ids:
        if run_id not in available:
            continue
        scores = score_repository.read_run_scores(run_id)["score"].to_numpy(dtype=np.float64)
        score_arrays.append(scores[~np.isnan(scores)])
    if not score_arrays:
        raise ValueError("No train/validation scores available for threshold calibration.")
    return float(np.quantile(np.concatenate(score_arrays), q))


def _protocol_split(splits: dict[str, Any], protocol: str) -> dict[str, list[str]]:
    selected = splits.get(protocol, splits.get("naive", splits))
    if not isinstance(selected, dict):
        raise ValueError(f"Split protocol {protocol!r} is not an object.")
    return {
        "train_runs": [str(run_id) for run_id in selected.get("train_runs", [])],
        "val_runs": [str(run_id) for run_id in selected.get("val_runs", [])],
        "test_runs": [str(run_id) for run_id in selected.get("test_runs", [])],
    }


def _prepared_period_ns(repository: LocalPreparedDatasetRepository, run_id: str) -> int:
    frame = repository.read_run(run_id, columns=["ts_ns"])
    inferred = infer_period_ns(frame["ts_ns"].to_numpy(dtype=np.int64))
    return inferred or 1_000_000_000


def _run_duration_ns(repository: LocalPreparedDatasetRepository, run_id: str) -> int:
    frame = repository.read_run(run_id, columns=["ts_ns"])
    if len(frame) == 0:
        return 0
    period_ns = infer_period_ns(frame["ts_ns"].to_numpy(dtype=np.int64)) or 1_000_000_000
    return int(frame["ts_ns"].iloc[-1] - frame["ts_ns"].iloc[0] + period_ns)


def _pred_event_to_dict(event: PredEvent) -> dict[str, Any]:
    return {
        "run_id": event.run_id,
        "pred_event_id": event.pred_event_id,
        "start_ts_ns": event.start_ts_ns,
        "end_ts_ns": event.end_ts_ns,
        "first_detect_ts_ns": event.first_detect_ts_ns,
    }
