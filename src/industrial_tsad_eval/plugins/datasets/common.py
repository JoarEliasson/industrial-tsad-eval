"""Shared helpers for local raw-to-prepared dataset adapters."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.domain.errors import PreparationError
from industrial_tsad_eval.infrastructure.data_utils import (
    coerce_numeric_frame,
    downcast_numeric,
    infer_period_ns,
    read_table,
    synthesize_timestamps,
    to_unix_ns,
)
from industrial_tsad_eval.infrastructure.prepared_writer import (
    ensure_prepared_dirs,
    tag_payload,
    write_events,
    write_manifest,
    write_provenance,
    write_run,
    write_schema,
    write_splits,
)

TABLE_SUFFIXES = {".csv", ".parquet", ".xls", ".xlsx"}
TIME_COLUMNS = {"ts_ns", "timestamp", "time", "datetime", "date_time", "date", "t"}


@dataclass(frozen=True)
class PreparedRun:
    """One run ready to be persisted in Prepared Format v1."""

    run_id: str
    frame: pd.DataFrame
    split: str
    source: str
    period_ns: int | None = None


def discover_table_files(raw: Path) -> list[Path]:
    """Return supported local table files below a raw path."""
    if raw.is_file() and raw.suffix.lower() in TABLE_SUFFIXES:
        return [raw]
    if not raw.exists():
        raise PreparationError(f"Raw dataset path does not exist: {raw}")
    files = [
        path for path in raw.rglob("*") if path.is_file() and path.suffix.lower() in TABLE_SUFFIXES
    ]
    if not files:
        raise PreparationError(f"No supported table files found below {raw}.")
    return sorted(files)


def read_first_matching_table(
    raw: Path, predicate: Callable[[Path], bool]
) -> tuple[Path, pd.DataFrame]:
    """Read the first supported table matching a file predicate."""
    for path in discover_table_files(raw):
        if predicate(path):
            return path, read_table(path)
    raise PreparationError(f"No supported table file matched below {raw}.")


def detect_column(frame: pd.DataFrame, names: set[str]) -> str | None:
    """Find a column by case-insensitive normalized name."""
    normalized = {_normalize_column_name(column): str(column) for column in frame.columns}
    for name in names:
        match = normalized.get(_normalize_column_name(name))
        if match is not None:
            return match
    return None


def detect_label_column(frame: pd.DataFrame) -> str | None:
    """Find a common binary anomaly label column."""
    return detect_column(
        frame,
        {
            "attack",
            "attacks",
            "fault",
            "is_attack",
            "is_anomaly",
            "label",
            "normal/attack",
            "target",
            "y",
        },
    )


def build_prepared_frame(
    frame: pd.DataFrame,
    *,
    prefix: str,
    config: DatasetAdapterConfig,
    exclude_columns: set[str],
    base_offset_rows: int = 0,
    rename: Callable[[str], str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], int | None]:
    """Create a numeric Prepared Format run frame."""
    timestamp_column = detect_column(frame, TIME_COLUMNS)
    if timestamp_column is not None and timestamp_column not in exclude_columns:
        ts_ns = to_unix_ns(frame[timestamp_column])
        exclude_columns = set(exclude_columns) | {timestamp_column}
    else:
        ts_ns = (
            synthesize_timestamps(
                len(frame),
                config.base_epoch_iso,
                config.default_period_ms,
            )
            + int(base_offset_rows) * int(config.default_period_ms) * 1_000_000
        )

    selected_columns = [
        str(column)
        for column in frame.columns
        if str(column) not in exclude_columns and not str(column).startswith("Unnamed:")
    ]
    if not selected_columns:
        raise PreparationError("No feature columns remained after removing metadata columns.")

    features, enum_maps = coerce_numeric_frame(frame[selected_columns])
    features = features.ffill().bfill().fillna(0.0)
    features = downcast_numeric(features)
    renamed = [
        _unique_feature_name(prefix, rename(column) if rename else column)
        for column in selected_columns
    ]
    features.columns = renamed

    prepared = pd.concat(
        [pd.Series(ts_ns.astype(np.int64), name="ts_ns"), features.reset_index(drop=True)],
        axis=1,
    )
    return (
        downcast_numeric(prepared),
        {renamed[index]: enum_map for index, enum_map in enumerate(enum_maps.values())},
        infer_period_ns(ts_ns),
    )


def write_prepared_dataset(
    *,
    prepared: Path,
    dataset_name: str,
    source_notes: str,
    runs: list[PreparedRun],
    events: list[dict[str, Any]],
    config: DatasetAdapterConfig,
    warnings: list[str] | None = None,
    provenance_extra: dict[str, Any] | None = None,
) -> DatasetAdapterResult:
    """Persist prepared runs and metadata through the shared writer helpers."""
    if not runs:
        raise PreparationError(f"Adapter {dataset_name} did not produce any runs.")

    ensure_prepared_dirs(prepared)
    tags = _schema_tags(runs)
    run_ids = [run.run_id for run in runs]
    train_runs = [run.run_id for run in runs if run.split == "train"]
    val_runs = [run.run_id for run in runs if run.split == "val"]
    test_runs = [run.run_id for run in runs if run.split == "test"]
    if not train_runs:
        train_runs = run_ids[:1]
    if not test_runs:
        test_runs = run_ids[len(train_runs) :]

    for run in runs:
        write_run(
            prepared,
            run.run_id,
            run.frame,
            {
                "run_id": run.run_id,
                "split": run.split,
                "source": run.source,
                "period_ns": run.period_ns,
                "start_ts_ns": int(run.frame["ts_ns"].iloc[0]),
                "end_ts_ns": int(run.frame["ts_ns"].iloc[-1]),
                "rows": int(len(run.frame)),
            },
        )

    splits = _splits(train_runs, val_runs, test_runs)
    write_schema(prepared, tags)
    write_events(prepared, events)
    write_splits(prepared, splits)
    write_manifest(
        root=prepared,
        dataset_name=dataset_name,
        source_notes=source_notes,
        timebase={"column": "ts_ns", "unit": "ns", "timezone": "UTC"},
        run_ids=run_ids,
        extra={"adapter_config": _adapter_config_payload(config)},
    )
    provenance = {"adapter": dataset_name, "source": source_notes}
    if provenance_extra:
        provenance.update(provenance_extra)
    write_provenance(prepared, provenance)
    return DatasetAdapterResult(
        dataset_name=dataset_name,
        prepared_path=str(prepared),
        run_count=len(runs),
        event_count=len(events),
        warnings=list(warnings or []),
    )


def prefixed_feature(prefix: str, column: str) -> str:
    """Normalize a raw column name to an OPC-UA-like browse path."""
    return f"{prefix}/{_clean_path_segment(column)}"


def _schema_tags(runs: list[PreparedRun]) -> list[dict[str, Any]]:
    dtype_by_column: dict[str, str] = {}
    for run in runs:
        for column in run.frame.columns:
            column_text = str(column)
            if column_text == "ts_ns":
                continue
            dtype_by_column.setdefault(column_text, str(run.frame[column].dtype))

    return [
        tag_payload(
            browse_path=column,
            dtype=dtype,
            kind="actuator" if _is_actuator(column) else "sensor",
            group="/".join(column.split("/")[:-1]),
        )
        for column, dtype in sorted(dtype_by_column.items())
    ]


def _splits(
    train_runs: list[str],
    val_runs: list[str],
    test_runs: list[str],
) -> dict[str, dict[str, list[str]]]:
    split = {"train_runs": train_runs, "val_runs": val_runs, "test_runs": test_runs}
    return {"naive": split, "all_in_one": split, "zero_shot": split}


def _adapter_config_payload(config: DatasetAdapterConfig) -> dict[str, Any]:
    return {
        "base_epoch_iso": config.base_epoch_iso,
        "default_period_ms": config.default_period_ms,
        "strict": config.strict,
        "extra": dict(config.extra),
    }


def _unique_feature_name(prefix: str, column: str) -> str:
    return prefixed_feature(prefix, column)


def _is_actuator(column: str) -> bool:
    lowered = column.lower()
    return any(token in lowered for token in ("/xmv", "/mv", "/valve", "/actuator", "/pump"))


def _normalize_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())


def _clean_path_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return cleaned or "value"
