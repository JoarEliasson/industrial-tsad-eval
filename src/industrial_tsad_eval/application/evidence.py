"""Evidence Bundle generation and validation services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.errors import EvidenceError
from industrial_tsad_eval.domain.evidence import (
    EVIDENCE_FORMAT_VERSION,
    EVIDENCE_MANIFEST_VERSION,
    GT_TAG_MAP_VERSION,
    EvidenceBundle,
    EvidenceIndexRow,
    EvidenceSource,
    EvidenceTimeWindow,
    EvidenceVariable,
    GroundTruthTagMap,
)
from industrial_tsad_eval.domain.validation import ValidationReport
from industrial_tsad_eval.infrastructure.evidence_repository import LocalEvidenceRepository
from industrial_tsad_eval.infrastructure.explanation_repository import LocalExplanationRepository
from industrial_tsad_eval.infrastructure.json_utils import read_json, write_json
from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository
from industrial_tsad_eval.infrastructure.score_repository import LocalScoreRepository

GT_TAG_METADATA_KEYS = ("affected_tags", "root_cause_tags", "tags")


@dataclass(frozen=True)
class GenerateEvidenceResult:
    """Summary of evidence generation."""

    dataset: str
    event_source: EvidenceSource
    bundle_count: int
    evidence_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "dataset": self.dataset,
            "event_source": self.event_source,
            "bundle_count": self.bundle_count,
            "evidence_dir": self.evidence_dir,
        }


@dataclass(frozen=True)
class GroundTruthTagMapResult:
    """Summary of GT tag-map building."""

    dataset: str
    event_count: int
    mapped_count: int
    out: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "dataset": self.dataset,
            "event_count": self.event_count,
            "mapped_count": self.mapped_count,
            "out": self.out,
        }


@dataclass(frozen=True)
class _SourceEvent:
    run_id: str
    event_id: str
    start_ts_ns: int
    end_ts_ns: int
    event_source: EvidenceSource
    source_event_id: str | None
    matched_gt_event_id: str | None
    is_matched_pred_event: bool


@dataclass(frozen=True)
class _RobustBaseline:
    features: list[str]
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        repository: LocalPreparedDatasetRepository,
        protocol: str,
        features: list[str],
    ) -> _RobustBaseline:
        split = _protocol_split(repository.splits(), protocol)
        run_ids = split["train_runs"] + split["val_runs"]
        arrays = [
            repository.read_run(run_id)
            .reindex(columns=features, fill_value=0.0)
            .to_numpy(dtype=np.float64)
            for run_id in run_ids
        ]
        if not arrays:
            run_id = repository.run_ids()[0]
            arrays = [
                repository.read_run(run_id)
                .reindex(columns=features, fill_value=0.0)
                .to_numpy(dtype=np.float64)
            ]
        values = np.concatenate(arrays, axis=0)
        median = np.median(values, axis=0)
        mad = np.median(np.abs(values - median), axis=0)
        scale = np.where(1.4826 * mad < 1e-8, 1.0, 1.4826 * mad)
        return cls(
            features=features, median=median.astype(np.float64), scale=scale.astype(np.float64)
        )


class GenerateEvidence:
    """Generate detector-agnostic Evidence Bundle v1 artifacts."""

    def __init__(
        self,
        *,
        prepared: str | Path,
        scores: str | Path,
        out: str | Path,
        eval_dir: str | Path | None = None,
        event_source: str = "oracle",
        protocol: str = "naive",
        top_k: int = 5,
        max_events: int = 100,
        explanation_source: str = "auto",
        native_missing_policy: str = "skip_bundle",
    ):
        self.prepared = Path(prepared)
        self.scores = Path(scores)
        self.out = Path(out)
        self.eval_dir = Path(eval_dir) if eval_dir is not None else None
        self.event_source = _event_source(event_source)
        self.protocol = protocol
        self.top_k = top_k
        self.max_events = max_events
        self.explanation_source = _explanation_source(explanation_source)
        self.native_missing_policy = _native_missing_policy(native_missing_policy)

    def run(self) -> GenerateEvidenceResult:
        """Generate bundles and write evidence artifacts."""
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        if self.max_events <= 0:
            raise ValueError("max_events must be greater than 0.")
        prepared_report = ValidatePreparedDataset(self.prepared).run()
        if not prepared_report.ok:
            raise EvidenceError(f"Prepared dataset validation failed: {prepared_report.errors}")
        score_report = ValidateScores(self.prepared, self.scores).run()
        if not score_report.ok:
            raise EvidenceError(f"Score validation failed: {score_report.errors}")

        prepared_repository = LocalPreparedDatasetRepository(self.prepared)
        score_repository = LocalScoreRepository(self.scores)
        explanation_repository = LocalExplanationRepository(self.scores / "explanations")
        features = _feature_columns(prepared_repository)
        source_events = self._source_events(prepared_repository)[: self.max_events]
        native_cache: dict[str, Any] = {}
        baseline_cache: dict[str, _RobustBaseline] = {}
        bundles = [
            _build_bundle(
                prepared_repository=prepared_repository,
                score_repository=score_repository,
                baseline_cache=baseline_cache,
                explanation_repository=explanation_repository,
                explanation_source=self.explanation_source,
                native_missing_policy=self.native_missing_policy,
                source_event=source_event,
                top_k=self.top_k,
                protocol=self.protocol,
                scores_dir=self.scores,
                eval_dir=self.eval_dir,
                features=features,
                native_cache=native_cache,
            )
            for source_event in source_events
        ]

        LocalEvidenceRepository(self.out).write_bundle_set(
            dataset=prepared_repository.dataset_name,
            event_source=self.event_source,
            bundles=bundles,
        )
        return GenerateEvidenceResult(
            dataset=prepared_repository.dataset_name,
            event_source=self.event_source,
            bundle_count=len(bundles),
            evidence_dir=str(self.out),
        )

    def _source_events(self, repository: LocalPreparedDatasetRepository) -> list[_SourceEvent]:
        if self.event_source == "oracle":
            return _oracle_events(repository, self.protocol)
        if self.eval_dir is None:
            raise EvidenceError("Operational evidence generation requires eval_dir.")
        return _operational_events(self.eval_dir)


class ValidateEvidence:
    """Validate Evidence Bundle v1 artifacts against a prepared dataset."""

    def __init__(self, prepared: str | Path, evidence: str | Path):
        self.prepared = Path(prepared)
        self.evidence = Path(evidence)

    def run(self) -> ValidationReport:
        """Validate evidence manifest, index, and bundles."""
        errors: list[str] = []
        warnings: list[str] = []
        prepared_report = ValidatePreparedDataset(self.prepared).run()
        if not prepared_report.ok:
            errors.extend(f"Prepared dataset: {error}" for error in prepared_report.errors)
            return ValidationReport("evidence", str(self.evidence), errors, warnings)

        repository = LocalPreparedDatasetRepository(self.prepared)
        evidence_repository = LocalEvidenceRepository(self.evidence)
        prepared_features = set(_feature_columns(repository))
        try:
            manifest = evidence_repository.manifest()
            if manifest.get("format_version") != EVIDENCE_MANIFEST_VERSION:
                errors.append("manifest.json: unsupported format_version.")
            rows = evidence_repository.index_rows()
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            return ValidationReport("evidence", str(self.evidence), errors, warnings)

        for row in rows:
            try:
                bundle = evidence_repository.read_bundle(row)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                errors.append(f"{row.relative_path}: {type(exc).__name__}: {exc}")
                continue
            _validate_bundle(bundle, row, prepared_features, errors, warnings)

        expected_count = manifest.get("bundle_count")
        if isinstance(expected_count, int) and expected_count != len(rows):
            errors.append(
                f"manifest.json: bundle_count={expected_count} but index has {len(rows)} rows."
            )
        return ValidationReport("evidence", str(self.evidence), errors, warnings)


class BuildGroundTruthTagMap:
    """Build an event-id keyed GT tag map from prepared event metadata."""

    def __init__(self, *, prepared: str | Path, out: str | Path):
        self.prepared = Path(prepared)
        self.out = Path(out)

    def run(self) -> GroundTruthTagMapResult:
        """Extract event tags and write a GT tag map."""
        report = ValidatePreparedDataset(self.prepared).run()
        if not report.ok:
            raise EvidenceError(f"Prepared dataset validation failed: {report.errors}")
        repository = LocalPreparedDatasetRepository(self.prepared)
        events = repository.read_events()
        entries = {
            event.event_id: tags for event in events if (tags := _metadata_tags(event.metadata))
        }
        tag_map = GroundTruthTagMap(
            dataset=repository.dataset_name,
            key_mode="event_id",
            entries=entries,
        )
        write_json(self.out, tag_map.to_dict())
        return GroundTruthTagMapResult(
            dataset=repository.dataset_name,
            event_count=len(events),
            mapped_count=len(entries),
            out=str(self.out),
        )


class ValidateGroundTruthTagMap:
    """Validate a ground-truth tag map file."""

    def __init__(self, gt_map: str | Path):
        self.gt_map = Path(gt_map)

    def run(self) -> ValidationReport:
        """Validate GT tag-map shape."""
        errors: list[str] = []
        warnings: list[str] = []
        try:
            payload = read_json(self.gt_map)
            tag_map = GroundTruthTagMap.from_dict(payload)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return ValidationReport("gt-tag-map", str(self.gt_map), [str(exc)], warnings)
        if payload.get("format_version") != GT_TAG_MAP_VERSION:
            errors.append("Unsupported gt-map format_version.")
        if not tag_map.dataset:
            errors.append("Missing dataset.")
        if tag_map.key_mode != "event_id":
            warnings.append(f"Non-default key_mode: {tag_map.key_mode}")
        if not tag_map.entries:
            warnings.append("GT tag map has no entries.")
        for key, tags in tag_map.entries.items():
            if not tags:
                errors.append(f"Entry {key!r} has no tags.")
        return ValidationReport("gt-tag-map", str(self.gt_map), errors, warnings)


def _build_bundle(
    *,
    prepared_repository: LocalPreparedDatasetRepository,
    score_repository: LocalScoreRepository,
    baseline_cache: dict[str, _RobustBaseline],
    explanation_repository: LocalExplanationRepository,
    explanation_source: str,
    native_missing_policy: str,
    source_event: _SourceEvent,
    top_k: int,
    protocol: str,
    scores_dir: Path,
    eval_dir: Path | None,
    features: list[str],
    native_cache: dict[str, Any],
) -> EvidenceBundle:
    native = _native_explanation(
        explanation_repository,
        source_event,
        top_k,
        explanation_source,
        native_missing_policy,
        native_cache,
    )
    if native is not None:
        top_variables, top_time_windows, local_rankings, native_provenance = native
        return EvidenceBundle(
            dataset=prepared_repository.dataset_name,
            run_id=source_event.run_id,
            event_id=source_event.event_id,
            event_source=source_event.event_source,
            source_event_id=source_event.source_event_id,
            matched_gt_event_id=source_event.matched_gt_event_id,
            is_matched_pred_event=source_event.is_matched_pred_event,
            event_start_ts_ns=source_event.start_ts_ns,
            event_end_ts_ns=source_event.end_ts_ns,
            top_variables=top_variables,
            top_time_windows=top_time_windows,
            score_context=_score_context(score_repository, source_event),
            local_rankings=local_rankings,
            provenance={
                **native_provenance,
                "protocol": protocol,
                "top_k": top_k,
                "scores_dir": str(scores_dir),
                "eval_dir": str(eval_dir) if eval_dir is not None else None,
            },
        )

    baseline = _cached_baseline(baseline_cache, prepared_repository, protocol, features)
    frame = prepared_repository.read_run(source_event.run_id)
    ts_ns = frame["ts_ns"].to_numpy(dtype=np.int64)
    event_mask = (ts_ns >= source_event.start_ts_ns) & (ts_ns < source_event.end_ts_ns)
    if not np.any(event_mask):
        nearest = int(np.searchsorted(ts_ns, source_event.start_ts_ns, side="left"))
        nearest = min(max(nearest, 0), max(len(ts_ns) - 1, 0))
        event_mask = np.zeros(len(ts_ns), dtype=bool)
        if len(ts_ns):
            event_mask[nearest] = True
    values = frame.reindex(columns=baseline.features, fill_value=0.0).to_numpy(dtype=np.float64)
    event_values = values[event_mask]
    event_ts = ts_ns[event_mask]
    z_scores = np.abs((event_values - baseline.median) / baseline.scale)
    variable_importance = (
        z_scores.mean(axis=0) if len(z_scores) else np.zeros(len(baseline.features))
    )
    ranked_indices = np.argsort(-variable_importance)[: min(top_k, len(baseline.features))]
    top_variables = [
        EvidenceVariable(
            variable=baseline.features[int(index)],
            rank=rank,
            importance=float(variable_importance[int(index)]),
            mean_abs_z=float(variable_importance[int(index)]),
        )
        for rank, index in enumerate(ranked_indices, start=1)
    ]
    top_time_windows = _top_time_windows(
        event_ts,
        z_scores,
        source_event.start_ts_ns,
        source_event.end_ts_ns,
    )
    return EvidenceBundle(
        dataset=prepared_repository.dataset_name,
        run_id=source_event.run_id,
        event_id=source_event.event_id,
        event_source=source_event.event_source,
        source_event_id=source_event.source_event_id,
        matched_gt_event_id=source_event.matched_gt_event_id,
        is_matched_pred_event=source_event.is_matched_pred_event,
        event_start_ts_ns=source_event.start_ts_ns,
        event_end_ts_ns=source_event.end_ts_ns,
        top_variables=top_variables,
        top_time_windows=top_time_windows,
        score_context=_score_context(score_repository, source_event),
        local_rankings=_local_rankings(event_ts, z_scores, baseline.features, top_k),
        provenance={
            "generator": "robust-zscore-v1",
            "explanation_source": "robust",
            "fallback_from_native": explanation_source == "auto",
            "protocol": protocol,
            "top_k": top_k,
            "scores_dir": str(scores_dir),
            "eval_dir": str(eval_dir) if eval_dir is not None else None,
        },
    )


def _native_explanation(
    repository: LocalExplanationRepository,
    source_event: _SourceEvent,
    top_k: int,
    explanation_source: str,
    native_missing_policy: str,
    native_cache: dict[str, Any],
) -> (
    tuple[
        list[EvidenceVariable],
        list[EvidenceTimeWindow],
        list[dict[str, Any]],
        dict[str, Any],
    ]
    | None
):
    if explanation_source == "robust":
        return None
    try:
        explanations = native_cache.get(source_event.run_id)
        if explanations is None:
            explanations = repository.read_run_explanations(source_event.run_id)
            native_cache[source_event.run_id] = explanations
    except FileNotFoundError:
        if explanation_source == "native":
            raise EvidenceError(
                f"Native explanation artifacts are required but missing for run "
                f"{source_event.run_id!r}."
            ) from None
        return None
    window = _native_event_rows(explanations, source_event)
    if window.empty:
        if explanation_source == "native" and native_missing_policy == "fail":
            raise EvidenceError(
                f"Native explanation artifacts contain no rows overlapping event "
                f"{source_event.event_id!r}."
            )
        if explanation_source == "native" and native_missing_policy == "fallback_robust":
            return None
        if explanation_source == "native":
            return _skipped_native_explanation(repository, source_event)
        return None

    method = str(window["method"].iloc[0]) if "method" in window.columns else "native"
    grouped = (
        window.groupby("variable", as_index=False)["importance"]
        .mean()
        .sort_values(["importance", "variable"], ascending=[False, True])
        .head(top_k)
    )
    top_variables = [
        EvidenceVariable(
            variable=str(row["variable"]),
            rank=index,
            importance=float(row["importance"]),
            mean_abs_z=float(row["importance"]),
        )
        for index, row in enumerate(grouped.to_dict("records"), start=1)
    ]
    local_rankings = [
        {
            "ts_ns": int(ts_value),
            "top_variables": [
                str(item["variable"])
                for item in frame.sort_values(
                    ["rank", "importance"], ascending=[True, False]
                ).to_dict("records")[:top_k]
            ],
        }
        for ts_value, frame in window.groupby("ts_ns")
    ]
    top_time_windows = _native_time_windows(window, source_event)
    return (
        top_variables,
        top_time_windows,
        local_rankings,
        {
            "generator": "native-explanation-v1",
            "explanation_source": "native",
            "explainer_method": method,
            "fallback_from_native": False,
            "explanations_dir": str(repository.root),
            "coverage_status": "native_event_overlap",
        },
    )


def _native_event_rows(explanations: Any, source_event: _SourceEvent) -> Any:
    point_mask = (explanations["ts_ns"] >= source_event.start_ts_ns) & (
        explanations["ts_ns"] < source_event.end_ts_ns
    )
    if "window_start_ts_ns" in explanations.columns and "window_end_ts_ns" in explanations.columns:
        window_mask = (explanations["window_start_ts_ns"] < source_event.end_ts_ns) & (
            explanations["window_end_ts_ns"] > source_event.start_ts_ns
        )
        return explanations.loc[point_mask | window_mask].copy()
    return explanations.loc[point_mask].copy()


def _skipped_native_explanation(
    repository: LocalExplanationRepository,
    source_event: _SourceEvent,
) -> tuple[
    list[EvidenceVariable],
    list[EvidenceTimeWindow],
    list[dict[str, Any]],
    dict[str, Any],
]:
    return (
        [],
        [
            EvidenceTimeWindow(
                start_ts_ns=source_event.start_ts_ns,
                end_ts_ns=source_event.end_ts_ns,
                rank=1,
                importance=0.0,
            )
        ],
        [],
        {
            "generator": "native-explanation-v1",
            "explanation_source": "native",
            "explainer_method": None,
            "fallback_from_native": False,
            "explanations_dir": str(repository.root),
            "coverage_status": "skipped_no_native_overlap",
            "skip_reason": "native_explanation_missing_event_overlap",
        },
    )


def _cached_baseline(
    baseline_cache: dict[str, _RobustBaseline],
    repository: LocalPreparedDatasetRepository,
    protocol: str,
    features: list[str],
) -> _RobustBaseline:
    baseline = baseline_cache.get("baseline")
    if baseline is None:
        baseline = _RobustBaseline.fit(repository, protocol, features)
        baseline_cache["baseline"] = baseline
    return baseline


def _native_time_windows(
    rows: Any,
    source_event: _SourceEvent,
) -> list[EvidenceTimeWindow]:
    if "window_start_ts_ns" not in rows.columns or "window_end_ts_ns" not in rows.columns:
        return [
            EvidenceTimeWindow(
                start_ts_ns=source_event.start_ts_ns,
                end_ts_ns=source_event.end_ts_ns,
                rank=1,
                importance=float(rows["importance"].mean()),
            )
        ]
    grouped = (
        rows.groupby(["window_start_ts_ns", "window_end_ts_ns"], as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .head(3)
    )
    return [
        EvidenceTimeWindow(
            start_ts_ns=int(row["window_start_ts_ns"]),
            end_ts_ns=int(row["window_end_ts_ns"]),
            rank=index,
            importance=float(row["importance"]),
        )
        for index, row in enumerate(grouped.to_dict("records"), start=1)
    ]


def _top_time_windows(
    event_ts: np.ndarray,
    z_scores: np.ndarray,
    start_ts_ns: int,
    end_ts_ns: int,
) -> list[EvidenceTimeWindow]:
    if len(event_ts) == 0:
        return [
            EvidenceTimeWindow(
                start_ts_ns=start_ts_ns,
                end_ts_ns=end_ts_ns,
                rank=1,
                importance=0.0,
            )
        ]
    return [
        EvidenceTimeWindow(
            start_ts_ns=int(event_ts[0]),
            end_ts_ns=int(event_ts[-1]),
            rank=1,
            importance=float(np.mean(z_scores)) if z_scores.size else 0.0,
        )
    ]


def _local_rankings(
    event_ts: np.ndarray,
    z_scores: np.ndarray,
    features: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if len(event_ts) == 0 or z_scores.size == 0:
        return rows
    stride = max(len(event_ts) // 64, 1)
    for position in range(0, len(event_ts), stride):
        ranked = np.argsort(-z_scores[position])[: min(top_k, len(features))]
        rows.append(
            {
                "ts_ns": int(event_ts[position]),
                "top_variables": [features[int(index)] for index in ranked],
            }
        )
    return rows


def _score_context(
    score_repository: LocalScoreRepository,
    source_event: _SourceEvent,
) -> dict[str, Any]:
    try:
        scores = score_repository.read_run_scores(source_event.run_id)
    except FileNotFoundError:
        return {"status": "missing_run_scores"}
    window = scores.loc[
        (scores["ts_ns"] >= source_event.start_ts_ns) & (scores["ts_ns"] < source_event.end_ts_ns)
    ]
    if window.empty:
        return {"status": "no_scores_in_event_window", "score_count": 0}
    return {
        "status": "computed",
        "score_count": int(len(window)),
        "max_score": float(window["score"].max()),
        "mean_score": float(window["score"].mean()),
    }


def _validate_bundle(
    bundle: EvidenceBundle,
    row: EvidenceIndexRow,
    prepared_features: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if bundle.format_version != EVIDENCE_FORMAT_VERSION:
        errors.append(f"{row.relative_path}: unsupported format_version.")
    if bundle.run_id != row.run_id or bundle.event_id != row.event_id:
        errors.append(f"{row.relative_path}: index identity does not match bundle.")
    if bundle.event_end_ts_ns < bundle.event_start_ts_ns:
        errors.append(f"{row.relative_path}: event end precedes event start.")
    variables = [item.variable for item in bundle.top_variables]
    unknown = sorted(set(variables) - prepared_features)
    if unknown:
        errors.append(f"{row.relative_path}: unknown top variables: {unknown[:10]}")
    if not variables:
        warnings.append(f"{row.relative_path}: no top variables.")


def _oracle_events(
    repository: LocalPreparedDatasetRepository,
    protocol: str,
) -> list[_SourceEvent]:
    split = _protocol_split(repository.splits(), protocol)
    test_runs = set(split["test_runs"])
    events = [
        event for event in repository.read_events() if not test_runs or event.run_id in test_runs
    ]
    return [
        _SourceEvent(
            run_id=event.run_id,
            event_id=event.event_id,
            start_ts_ns=event.start_ts_ns,
            end_ts_ns=event.end_ts_ns,
            event_source="oracle",
            source_event_id=event.event_id,
            matched_gt_event_id=event.event_id,
            is_matched_pred_event=True,
        )
        for event in sorted(events, key=lambda item: (item.run_id, item.start_ts_ns, item.event_id))
    ]


def _operational_events(eval_dir: Path) -> list[_SourceEvent]:
    matches_path = eval_dir / "event_matches.json"
    if not matches_path.exists():
        raise FileNotFoundError(f"Operational evidence requires {matches_path}.")
    payload = read_json(matches_path)
    pred_matches = {
        str(key): (str(value) if value is not None else None)
        for key, value in dict(payload.get("pred_matches", {})).items()
    }
    source_events: list[_SourceEvent] = []
    for raw_event in payload.get("pred_events", []):
        if not isinstance(raw_event, dict):
            continue
        event_id = str(raw_event.get("pred_event_id", raw_event.get("event_id", "")))
        matched_gt_event_id = pred_matches.get(event_id)
        start_value = raw_event.get("start_ts_ns", raw_event.get("start_ns"))
        end_value = raw_event.get("end_ts_ns", raw_event.get("end_ns"))
        if start_value is None or end_value is None:
            raise ValueError("Operational pred_events require start/end timestamps.")
        source_events.append(
            _SourceEvent(
                run_id=str(raw_event["run_id"]),
                event_id=event_id,
                start_ts_ns=int(start_value),
                end_ts_ns=int(end_value),
                event_source="operational",
                source_event_id=event_id,
                matched_gt_event_id=matched_gt_event_id,
                is_matched_pred_event=matched_gt_event_id is not None,
            )
        )
    return sorted(source_events, key=lambda item: (item.run_id, item.start_ts_ns, item.event_id))


def _feature_columns(repository: LocalPreparedDatasetRepository) -> list[str]:
    tags = repository.schema().get("tags", [])
    columns = [
        str(tag["browse_path"]) for tag in tags if isinstance(tag, dict) and tag.get("browse_path")
    ]
    if columns:
        return columns
    first_run = repository.run_ids()[0]
    frame = repository.read_run(first_run)
    return [str(column) for column in frame.columns if column != "ts_ns"]


def _protocol_split(splits: dict[str, Any], protocol: str) -> dict[str, list[str]]:
    selected = splits.get(protocol, splits.get("naive", splits))
    if not isinstance(selected, dict):
        raise ValueError(f"Split protocol {protocol!r} is not an object.")
    return {
        "train_runs": [str(run_id) for run_id in selected.get("train_runs", [])],
        "val_runs": [str(run_id) for run_id in selected.get("val_runs", [])],
        "test_runs": [str(run_id) for run_id in selected.get("test_runs", [])],
    }


def _metadata_tags(metadata: dict[str, Any]) -> list[str]:
    for key in GT_TAG_METADATA_KEYS:
        tags = _coerce_tags(metadata.get(key))
        if tags:
            return tags
    return _coerce_tags(metadata.get("tag"))


def _coerce_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return sorted({str(item) for item in value if str(item)})
    return [str(value)]


def _event_source(value: str) -> EvidenceSource:
    normalized = value.strip().lower()
    if normalized not in {"oracle", "operational"}:
        raise ValueError("event_source must be either 'oracle' or 'operational'.")
    return cast(EvidenceSource, normalized)


def _explanation_source(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"auto", "native", "robust"}:
        raise ValueError("explanation_source must be one of: auto, native, robust.")
    return normalized


def _native_missing_policy(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"skip_bundle", "fallback_robust", "fail"}:
        raise ValueError(
            "native_missing_policy must be one of: skip_bundle, fallback_robust, fail."
        )
    return normalized
