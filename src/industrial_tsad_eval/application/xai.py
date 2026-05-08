"""Explanation-quality evaluation services."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from industrial_tsad_eval.application.evidence import (
    ValidateEvidence,
    ValidateGroundTruthTagMap,
    _feature_columns,
    _RobustBaseline,
)
from industrial_tsad_eval.domain.errors import XAIEvaluationError
from industrial_tsad_eval.domain.evidence import (
    EvidenceBundle,
    GroundTruthTagMap,
    XAIEvaluationResult,
)
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.evidence_repository import LocalEvidenceRepository
from industrial_tsad_eval.infrastructure.json_utils import read_json
from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository


@dataclass(frozen=True)
class EvaluateEvidenceConfig:
    """Configuration for XAI/evidence evaluation."""

    prepared: str | Path
    evidence: str | Path
    gt_map: str | Path
    out: str | Path
    ks: list[int]
    protocol: str = "naive"


class EvaluateEvidence:
    """Evaluate Evidence Bundle v1 artifacts with deterministic XAI metrics."""

    def __init__(self, config: EvaluateEvidenceConfig):
        self.config = config

    def run(self) -> XAIEvaluationResult:
        """Compute explanation-quality metrics and write report artifacts."""
        if not self.config.ks or any(k <= 0 for k in self.config.ks):
            raise ValueError("ks must contain one or more positive integers.")
        evidence_report = ValidateEvidence(self.config.prepared, self.config.evidence).run()
        if not evidence_report.ok:
            raise XAIEvaluationError(f"Evidence validation failed: {evidence_report.errors}")
        gt_report = ValidateGroundTruthTagMap(self.config.gt_map).run()
        if not gt_report.ok:
            raise XAIEvaluationError(f"GT tag-map validation failed: {gt_report.errors}")

        prepared_repository = LocalPreparedDatasetRepository(self.config.prepared)
        evidence_repository = LocalEvidenceRepository(self.config.evidence)
        evidence_manifest = evidence_repository.manifest()
        evidence_source = str(evidence_manifest.get("event_source", "unknown"))
        tag_map = GroundTruthTagMap.from_dict(read_json(Path(self.config.gt_map)))
        features = _feature_columns(prepared_repository)
        baseline = _RobustBaseline.fit(prepared_repository, self.config.protocol, features)
        bundles = evidence_repository.discover()
        ks = sorted(set(self.config.ks))
        max_k = max(ks)

        skipped = {
            "no_gt_key": 0,
            "missing_gt_entry": 0,
            "empty_gt_tags": 0,
            "masking": 0,
            "stability": 0,
        }
        hit_counts = {k: 0 for k in ks}
        recall_sums = {k: 0.0 for k in ks}
        valid_count = 0
        rows: list[dict[str, Any]] = []
        variable_drops: list[float] = []
        window_drops: list[float] = []
        stability_values: list[float] = []

        for bundle in bundles:
            row = _base_row(bundle)
            gt_tags, skip_reason = _gt_tags_for_bundle(bundle, tag_map)
            if skip_reason is None and gt_tags is not None:
                valid_count += 1
                predictions = [item.variable for item in bundle.top_variables]
                for k in ks:
                    overlap = set(predictions[:k]) & gt_tags
                    hit = 1 if overlap else 0
                    recall = len(overlap) / len(gt_tags)
                    hit_counts[k] += hit
                    recall_sums[k] += recall
                    row[f"hit_at_{k}"] = hit
                    row[f"recall_at_{k}"] = recall
            else:
                skipped[str(skip_reason)] += 1
                row["skip_reason"] = skip_reason

            masking = _masking_metrics(prepared_repository, baseline, bundle, max_k)
            row.update(masking)
            if masking["masking_status"] == "computed":
                variable_drops.append(float(masking["variable_drop_pct"]))
                window_drops.append(float(masking["window_drop_pct"]))
            else:
                skipped["masking"] += 1

            stability = _stability(bundle, max_k)
            row["stability_jaccard"] = stability
            if stability is None:
                skipped["stability"] += 1
            else:
                stability_values.append(stability)
            rows.append(row)

        metrics = {
            "dataset": prepared_repository.dataset_name,
            "protocol": self.config.protocol,
            "evidence_source": evidence_source,
            "bundle_count": len(bundles),
            "valid_bundle_count": valid_count,
            "hitrate_at_k": {
                str(k): hit_counts[k] / valid_count if valid_count else 0.0 for k in ks
            },
            "recall_at_k": {
                str(k): recall_sums[k] / valid_count if valid_count else 0.0 for k in ks
            },
            "masking": {
                "status": "computed" if variable_drops else "skipped",
                "computed_count": len(variable_drops),
                "skipped_count": skipped["masking"],
                "variable_drop_pct_mean": _mean_or_none(variable_drops),
                "window_drop_pct_mean": _mean_or_none(window_drops),
                "baseline": "train-val-robust-zscore",
            },
            "stability": {
                "status": "computed" if stability_values else "skipped",
                "computed_count": len(stability_values),
                "skipped_count": skipped["stability"],
                "mean_jaccard": _mean_or_none(stability_values),
            },
            "skipped": skipped,
        }
        writer = LocalArtifactWriter(self.config.out)
        writer.write_json("metrics.json", metrics)
        writer.write_text("bundle_metrics.csv", _csv(rows))
        writer.write_text("summary.csv", _csv([_summary_row(metrics)]))
        writer.write_json("skipped.json", skipped)
        return XAIEvaluationResult(
            dataset=prepared_repository.dataset_name,
            evidence_dir=str(self.config.evidence),
            out_dir=str(self.config.out),
            metrics=metrics,
            bundle_metrics=rows,
            skipped=skipped,
        )


def _gt_tags_for_bundle(
    bundle: EvidenceBundle,
    tag_map: GroundTruthTagMap,
) -> tuple[set[str] | None, str | None]:
    key = bundle.matched_gt_event_id or bundle.source_event_id or bundle.event_id
    if not key:
        return None, "no_gt_key"
    if key not in tag_map.entries:
        return None, "missing_gt_entry"
    tags = {str(tag) for tag in tag_map.entries[key] if str(tag)}
    if not tags:
        return None, "empty_gt_tags"
    return tags, None


def _masking_metrics(
    repository: LocalPreparedDatasetRepository,
    baseline: _RobustBaseline,
    bundle: EvidenceBundle,
    max_k: int,
) -> dict[str, Any]:
    frame = repository.read_run(bundle.run_id)
    ts_ns = frame["ts_ns"].to_numpy(dtype=np.int64)
    values = frame.reindex(columns=baseline.features, fill_value=0.0).to_numpy(dtype=np.float64)
    event_mask = (ts_ns >= bundle.event_start_ts_ns) & (ts_ns < bundle.event_end_ts_ns)
    if not np.any(event_mask):
        return {"masking_status": "skipped", "masking_reason": "empty_event_window"}
    original = _robust_score(values[event_mask], baseline)
    if original <= 1e-12:
        return {"masking_status": "skipped", "masking_reason": "zero_original_score"}

    selected_variables = [item.variable for item in bundle.top_variables[:max_k]]
    variable_indices = [
        baseline.features.index(variable)
        for variable in selected_variables
        if variable in baseline.features
    ]
    if not variable_indices:
        return {"masking_status": "skipped", "masking_reason": "no_known_top_variables"}

    variable_masked = values.copy()
    event_indices = np.where(event_mask)[0]
    variable_index_array = np.asarray(variable_indices, dtype=np.int64)
    variable_masked[np.ix_(event_indices, variable_index_array)] = baseline.median[
        variable_index_array
    ]
    variable_score = _robust_score(variable_masked[event_mask], baseline)

    window_mask = np.zeros(len(ts_ns), dtype=bool)
    for window in bundle.top_time_windows:
        window_mask |= (ts_ns >= window.start_ts_ns) & (ts_ns <= window.end_ts_ns)
    if not np.any(window_mask):
        window_mask = event_mask
    window_masked = values.copy()
    window_masked[window_mask] = baseline.median
    window_score = _robust_score(window_masked[event_mask], baseline)

    return {
        "masking_status": "computed",
        "masking_reason": None,
        "original_surrogate_score": original,
        "variable_masked_score": variable_score,
        "window_masked_score": window_score,
        "variable_drop_pct": (original - variable_score) / original * 100.0,
        "window_drop_pct": (original - window_score) / original * 100.0,
    }


def _robust_score(values: np.ndarray, baseline: _RobustBaseline) -> float:
    if values.size == 0:
        return 0.0
    z_scores = np.abs((values - baseline.median) / baseline.scale)
    return float(np.mean(np.max(z_scores, axis=1)))


def _stability(bundle: EvidenceBundle, max_k: int) -> float | None:
    rankings = [
        [str(item) for item in row.get("top_variables", [])[:max_k]]
        for row in bundle.local_rankings
        if isinstance(row.get("top_variables"), list)
    ]
    if len(rankings) < 2:
        return None
    values: list[float] = []
    for left, right in zip(rankings, rankings[1:], strict=False):
        left_set = set(left)
        right_set = set(right)
        union = left_set | right_set
        values.append(len(left_set & right_set) / len(union) if union else 1.0)
    return _mean_or_none(values)


def _base_row(bundle: EvidenceBundle) -> dict[str, Any]:
    return {
        "run_id": bundle.run_id,
        "event_id": bundle.event_id,
        "event_source": bundle.event_source,
        "matched_gt_event_id": bundle.matched_gt_event_id,
        "is_matched_pred_event": bundle.is_matched_pred_event,
        "top_variables": "|".join(item.variable for item in bundle.top_variables),
    }


def _summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    masking = dict(metrics.get("masking", {}))
    stability = dict(metrics.get("stability", {}))
    return {
        "dataset": metrics.get("dataset"),
        "protocol": metrics.get("protocol"),
        "evidence_source": metrics.get("evidence_source"),
        "bundle_count": metrics.get("bundle_count"),
        "valid_bundle_count": metrics.get("valid_bundle_count"),
        "hitrate_at_k": metrics.get("hitrate_at_k"),
        "recall_at_k": metrics.get("recall_at_k"),
        "variable_drop_pct_mean": masking.get("variable_drop_pct_mean"),
        "window_drop_pct_mean": masking.get("window_drop_pct_mean"),
        "stability_mean_jaccard": stability.get("mean_jaccard"),
    }


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


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None
