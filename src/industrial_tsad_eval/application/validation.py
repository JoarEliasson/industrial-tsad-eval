"""Contract validation use cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.validation import ValidationReport
from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository
from industrial_tsad_eval.infrastructure.score_repository import LocalScoreRepository


class ValidatePreparedDataset:
    """Validate a Prepared Format v1 dataset directory."""

    def __init__(self, prepared: str | Path):
        self.prepared = Path(prepared)

    def run(self) -> ValidationReport:
        """Return a structured validation report."""
        errors: list[str] = []
        warnings: list[str] = []
        required_files = [
            "meta/manifest.json",
            "meta/schema.json",
            "meta/splits.json",
            "events/events.jsonl",
        ]

        for relative_path in required_files:
            if not (self.prepared / relative_path).exists():
                errors.append(f"Missing required file: {relative_path}")

        runs_root = self.prepared / "runs"
        if not runs_root.exists():
            errors.append("Missing runs/ directory")
            return ValidationReport(self.prepared.name, str(self.prepared), errors, warnings)

        schema_tags = _schema_browse_paths(self.prepared / "meta" / "schema.json")
        run_ranges: dict[str, tuple[int, int, int]] = {}
        for run_dir in sorted(path.parent for path in runs_root.rglob("timeseries.parquet")):
            run_id = "/".join(run_dir.relative_to(runs_root).parts)
            frame = pd.read_parquet(run_dir / "timeseries.parquet")
            meta_path = run_dir / "run_meta.json"
            if not meta_path.exists():
                errors.append(f"Run {run_id}: missing run_meta.json")
            _validate_run_frame(run_id, frame, schema_tags, errors, warnings)
            if len(frame) > 0 and "ts_ns" in frame.columns:
                run_ranges[run_id] = (
                    int(frame["ts_ns"].iloc[0]),
                    int(frame["ts_ns"].iloc[-1]),
                    int(len(frame)),
                )

        _validate_events(self.prepared / "events" / "events.jsonl", run_ranges, errors)
        return ValidationReport(self.prepared.name, str(self.prepared), errors, warnings)


class ValidateScores:
    """Validate Score Contract v1 artifacts against a prepared dataset."""

    def __init__(self, prepared: str | Path, scores: str | Path):
        self.prepared_repository = LocalPreparedDatasetRepository(prepared)
        self.score_repository = LocalScoreRepository(scores)

    def run(self) -> ValidationReport:
        """Return a structured validation report."""
        errors: list[str] = []
        warnings: list[str] = []
        valid_runs = set(self.prepared_repository.run_ids())
        subject = f"scores:{self.prepared_repository.dataset_name}"

        try:
            score_files = self.score_repository.discover()
        except Exception as exc:
            return ValidationReport(subject, str(self.score_repository.root), [str(exc)], warnings)

        if not score_files:
            errors.append(f"No score files found in {self.score_repository.root}")

        for run_id, score_path in sorted(score_files.items()):
            if run_id not in valid_runs:
                errors.append(f"Score file references unknown run_id: {run_id}")
                continue
            if not score_path.exists():
                errors.append(f"Score file path does not exist: {score_path}")
                continue
            try:
                frame = pd.read_parquet(score_path)
            except Exception as exc:
                errors.append(f"Run {run_id}: failed to read parquet: {exc}")
                continue
            _validate_score_frame(run_id, frame, errors, warnings)

        return ValidationReport(subject, str(self.score_repository.root), errors, warnings)


def _schema_browse_paths(schema_path: Path) -> set[str]:
    if not schema_path.exists():
        return set()
    payload = _read_json(schema_path)
    tags = payload.get("tags", [])
    return {str(tag.get("browse_path")) for tag in tags if isinstance(tag, dict)}


def _validate_run_frame(
    run_id: str,
    frame: pd.DataFrame,
    schema_tags: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if "ts_ns" not in frame.columns:
        errors.append(f"Run {run_id}: missing ts_ns")
        return
    if frame["ts_ns"].dtype != np.int64:
        errors.append(f"Run {run_id}: ts_ns dtype must be int64, got {frame['ts_ns'].dtype}")

    timestamps = frame["ts_ns"].to_numpy(dtype=np.int64)
    if len(timestamps) > 1:
        diffs = np.diff(timestamps)
        if not np.all(diffs >= 0):
            errors.append(f"Run {run_id}: ts_ns not monotonic non-decreasing")
        if np.any(diffs == 0):
            warnings.append(f"Run {run_id}: duplicate timestamps detected")

    data_columns = {str(column) for column in frame.columns if column != "ts_ns"}
    for column in data_columns:
        if pd.api.types.is_string_dtype(frame[column]) or pd.api.types.is_object_dtype(
            frame[column]
        ):
            errors.append(
                f"Run {run_id}: column {column} is non-numeric dtype {frame[column].dtype}"
            )

    if schema_tags:
        missing_from_schema = sorted(data_columns - schema_tags)
        missing_from_run = sorted(schema_tags - data_columns)
        if missing_from_schema:
            errors.append(
                f"Run {run_id}: columns not present in schema.json: {missing_from_schema[:10]}"
            )
        if missing_from_run:
            warnings.append(f"Run {run_id}: schema tags missing from run: {missing_from_run[:10]}")


def _validate_events(
    events_path: Path,
    run_ranges: dict[str, tuple[int, int, int]],
    errors: list[str],
) -> None:
    if not events_path.exists():
        return
    with events_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"events.jsonl line {line_number}: invalid JSON")
                continue
            _validate_event(event, line_number, run_ranges, errors)


def _validate_event(
    event: dict[str, Any],
    line_number: int,
    run_ranges: dict[str, tuple[int, int, int]],
    errors: list[str],
) -> None:
    run_id = event.get("run_id")
    if run_id not in run_ranges:
        errors.append(f"events.jsonl line {line_number}: unknown run_id {run_id}")
        return
    start = event.get("start_ts_ns")
    end = event.get("end_ts_ns")
    if not isinstance(start, int) or not isinstance(end, int):
        errors.append(f"events.jsonl line {line_number}: start_ts_ns/end_ts_ns must be int")
        return
    run_start, run_end, _row_count = run_ranges[str(run_id)]
    if not run_start <= start <= run_end:
        errors.append(f"events.jsonl line {line_number}: start_ts_ns out of run range")
    if end < start:
        errors.append(f"events.jsonl line {line_number}: end_ts_ns < start_ts_ns")


def _validate_score_frame(
    run_id: str,
    frame: pd.DataFrame,
    errors: list[str],
    warnings: list[str],
) -> None:
    for column in ("ts_ns", "score"):
        if column not in frame.columns:
            errors.append(f"Run {run_id}: missing required column {column!r}")
            return
    if not pd.api.types.is_numeric_dtype(frame["ts_ns"]):
        errors.append(f"Run {run_id}: ts_ns dtype must be numeric, got {frame['ts_ns'].dtype}")
    if not pd.api.types.is_numeric_dtype(frame["score"]):
        errors.append(f"Run {run_id}: score dtype must be numeric, got {frame['score'].dtype}")

    timestamps = frame["ts_ns"].to_numpy(dtype=np.int64)
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) >= 0):
        errors.append(f"Run {run_id}: ts_ns not monotonic non-decreasing")

    if frame["score"].isna().any():
        errors.append(f"Run {run_id}: score contains NaNs")
    if np.isinf(frame["score"]).any():
        errors.append(f"Run {run_id}: score contains infinity")

    if len(timestamps) > 2:
        diffs = np.diff(timestamps)
        positive = diffs[diffs > 0]
        if len(positive) > 0 and np.std(positive) > 0.5 * np.median(positive):
            warnings.append(f"Run {run_id}: timestamp spacing is highly irregular")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload
