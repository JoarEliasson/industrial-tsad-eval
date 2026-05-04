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
