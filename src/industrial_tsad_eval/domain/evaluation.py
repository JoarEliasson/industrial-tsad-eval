"""Pure evaluation functions for score-to-event metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from industrial_tsad_eval.domain.events import GTEvent, PredEvent


def infer_period_ns(ts_ns: np.ndarray) -> int | None:
    """Infer the median positive sampling period in nanoseconds."""
    if len(ts_ns) < 2:
        return None
    diffs = np.diff(np.asarray(ts_ns, dtype=np.int64))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return None
    return int(np.median(positive))


def resolve_effective_merge_gap_ns(
    fixed_merge_gap_ns: int,
    inferred_period_ns: int | None,
    merge_gap_mode: str,
    merge_gap_skipped_samples: int,
    merge_gap_jitter_ratio: float,
) -> int:
    """Resolve fixed or cadence-aware event merge gap."""
    if merge_gap_mode != "auto_period" or inferred_period_ns is None:
        return int(fixed_merge_gap_ns)
    multiplier = merge_gap_skipped_samples + 1 + merge_gap_jitter_ratio
    return max(int(fixed_merge_gap_ns), int(math.ceil(multiplier * inferred_period_ns)))


def cluster_anomalies_to_pred_events(
    run_id: str,
    anomaly_ts_ns: np.ndarray,
    merge_gap_ns: int,
    stride_ns: int | None,
    period_ns_fallback: int,
) -> list[PredEvent]:
    """Cluster anomalous score timestamps into deterministic predicted events."""
    if len(anomaly_ts_ns) == 0:
        return []

    timestamps = np.sort(np.asarray(anomaly_ts_ns, dtype=np.int64))
    end_offset = stride_ns if stride_ns is not None else period_ns_fallback
    events: list[PredEvent] = []
    current_start = int(timestamps[0])
    current_last = int(timestamps[0])

    for ts_value in timestamps[1:]:
        ts_int = int(ts_value)
        if ts_int - current_last <= merge_gap_ns:
            current_last = ts_int
            continue
        events.append(_pred_event(run_id, len(events), current_start, current_last, end_offset))
        current_start = ts_int
        current_last = ts_int

    events.append(_pred_event(run_id, len(events), current_start, current_last, end_offset))
    return events


def match_pred_to_gt_events(
    gt_events: list[GTEvent],
    pred_events: list[PredEvent],
    grace_ns: int,
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Match predicted events one-to-one to ground truth events."""
    gt_sorted = sorted(gt_events, key=lambda event: (event.start_ts_ns, event.event_id))
    pred_sorted = sorted(
        pred_events, key=lambda event: (event.first_detect_ts_ns, event.pred_event_id)
    )
    gt_matches: dict[str, str | None] = {event.event_id: None for event in gt_events}
    pred_matches: dict[str, str | None] = {event.pred_event_id: None for event in pred_events}
    used_pred_ids: set[str] = set()

    for gt_event in gt_sorted:
        for pred_event in pred_sorted:
            if pred_event.pred_event_id in used_pred_ids:
                continue
            if (
                gt_event.start_ts_ns
                <= pred_event.first_detect_ts_ns
                < gt_event.end_ts_ns + grace_ns
            ):
                gt_matches[gt_event.event_id] = pred_event.pred_event_id
                pred_matches[pred_event.pred_event_id] = gt_event.event_id
                used_pred_ids.add(pred_event.pred_event_id)
                break

    return gt_matches, pred_matches


def compute_event_prf(
    gt_events: list[GTEvent],
    pred_events: list[PredEvent],
    gt_matches: dict[str, str | None],
) -> dict[str, float]:
    """Compute event-level precision, recall, and F1."""
    n_gt = len(gt_events)
    n_pred = len(pred_events)
    n_hits = sum(1 for match in gt_matches.values() if match is not None)
    recall = 1.0 if n_gt == 0 else n_hits / n_gt
    precision = (0.0 if n_pred > 0 else 1.0) if n_gt == 0 else (n_hits / n_pred if n_pred else 1.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_gt": float(n_gt),
        "n_pred": float(n_pred),
        "n_hits": float(n_hits),
    }


def compute_delays(
    gt_events: list[GTEvent],
    gt_matches: dict[str, str | None],
    pred_events: list[PredEvent],
) -> dict[str, Any]:
    """Compute detection delay statistics for matched events."""
    pred_by_id = {event.pred_event_id: event for event in pred_events}
    delays = [
        max(0, pred_by_id[match_id].first_detect_ts_ns - gt_event.start_ts_ns)
        for gt_event in gt_events
        if (match_id := gt_matches.get(gt_event.event_id)) is not None
    ]
    if not delays:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
            "n": 0,
            "delays": [],
        }

    values = np.asarray(delays, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": len(delays),
        "delays": delays,
    }


def compute_false_alarm_rates(
    normal_run_ids: list[str],
    pred_events_by_run: dict[str, list[PredEvent]],
    run_durations_ns: dict[str, int],
    window_count_by_run: dict[str, int],
) -> dict[str, float]:
    """Compute false-event rates over normal runs."""
    total_false_events = sum(len(pred_events_by_run.get(run_id, [])) for run_id in normal_run_ids)
    total_duration_ns = sum(run_durations_ns.get(run_id, 0) for run_id in normal_run_ids)
    total_windows = sum(window_count_by_run.get(run_id, 0) for run_id in normal_run_ids)
    total_duration_hours = total_duration_ns / 3.6e12
    total_duration_days = total_duration_hours / 24.0
    return {
        "false_events_per_hour": total_false_events / total_duration_hours
        if total_duration_hours
        else 0.0,
        "false_events_per_day": total_false_events / total_duration_days
        if total_duration_days
        else 0.0,
        "false_events_per_10k_windows": total_false_events / total_windows * 10_000
        if total_windows
        else 0.0,
        "total_false_events": float(total_false_events),
        "total_duration_hours": float(total_duration_hours),
        "total_windows": float(total_windows),
    }


def compute_point_prf_for_scores(
    score_timestamps_by_run: dict[str, np.ndarray],
    score_values_by_run: dict[str, np.ndarray],
    gt_events: list[GTEvent],
    threshold: float,
    *,
    point_adjusted: bool = False,
) -> dict[str, float]:
    """Compute point-wise or point-adjusted precision, recall, and F1."""
    gt_by_run: dict[str, list[GTEvent]] = {}
    for event in gt_events:
        gt_by_run.setdefault(event.run_id, []).append(event)

    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    for run_id in sorted(score_timestamps_by_run):
        timestamps = score_timestamps_by_run[run_id]
        predictions = score_values_by_run[run_id] >= threshold
        labels = labels_for_events(timestamps, gt_by_run.get(run_id, []))
        if point_adjusted:
            predictions = point_adjust_predictions(
                timestamps,
                predictions,
                gt_by_run.get(run_id, []),
            )
        y_true_parts.append(labels)
        y_pred_parts.append(predictions)

    if not y_true_parts:
        return _binary_prf(np.asarray([], dtype=bool), np.asarray([], dtype=bool))
    return _binary_prf(np.concatenate(y_true_parts), np.concatenate(y_pred_parts))


def labels_for_events(timestamps: np.ndarray, events: list[GTEvent]) -> np.ndarray:
    """Return boolean point labels for timestamps covered by ground-truth events."""
    labels = np.zeros(len(timestamps), dtype=bool)
    for event in events:
        labels |= (timestamps >= event.start_ts_ns) & (timestamps < event.end_ts_ns)
    return labels


def point_adjust_predictions(
    timestamps: np.ndarray,
    predictions: np.ndarray,
    events: list[GTEvent],
) -> np.ndarray:
    """Expand predictions to full ground-truth event ranges when an event is hit."""
    adjusted = np.asarray(predictions, dtype=bool).copy()
    for event in events:
        event_mask = (timestamps >= event.start_ts_ns) & (timestamps < event.end_ts_ns)
        if np.any(adjusted & event_mask):
            adjusted[event_mask] = True
    return adjusted


def compute_affiliation_prf(
    gt_events: list[GTEvent],
    pred_events: list[PredEvent],
) -> dict[str, Any]:
    """Compute overlap-based affiliation-style interval precision, recall, and F1."""
    precision_values = [
        _best_overlap_fraction(
            pred_event.start_ts_ns,
            pred_event.end_ts_ns,
            [
                (event.start_ts_ns, event.end_ts_ns)
                for event in gt_events
                if event.run_id == pred_event.run_id
            ],
        )
        for pred_event in pred_events
    ]
    recall_values = [
        _best_overlap_fraction(
            gt_event.start_ts_ns,
            gt_event.end_ts_ns,
            [
                (event.start_ts_ns, event.end_ts_ns)
                for event in pred_events
                if event.run_id == gt_event.run_id
            ],
        )
        for gt_event in gt_events
    ]
    precision = (
        float(np.mean(np.asarray(precision_values, dtype=np.float64)))
        if precision_values
        else (1.0 if not gt_events else 0.0)
    )
    recall = (
        float(np.mean(np.asarray(recall_values, dtype=np.float64)))
        if recall_values
        else (1.0 if not pred_events else 0.0)
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_gt": float(len(gt_events)),
        "n_pred": float(len(pred_events)),
        "definition": "best_overlap_fraction_interval_prf",
    }


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    precision = tp / (tp + fp) if tp + fp else (1.0 if not np.any(y_true) else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "support": float(len(y_true)),
    }


def _best_overlap_fraction(
    start_ts_ns: int,
    end_ts_ns: int,
    candidate_intervals: list[tuple[int, int]],
) -> float:
    duration = max(int(end_ts_ns) - int(start_ts_ns), 1)
    best = 0
    for candidate_start, candidate_end in candidate_intervals:
        overlap = max(
            0,
            min(int(end_ts_ns), int(candidate_end)) - max(int(start_ts_ns), int(candidate_start)),
        )
        best = max(best, overlap)
    return float(best / duration)


def _pred_event(
    run_id: str,
    index: int,
    start_ts_ns: int,
    last_ts_ns: int,
    end_offset_ns: int,
) -> PredEvent:
    safe_run_id = run_id.replace("/", "_").replace("\\", "_")
    return PredEvent(
        run_id=run_id,
        pred_event_id=f"pred_{safe_run_id}_{index:04d}",
        start_ts_ns=start_ts_ns,
        end_ts_ns=last_ts_ns + end_offset_ns,
        first_detect_ts_ns=start_ts_ns,
    )
