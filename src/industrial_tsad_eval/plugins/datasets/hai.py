"""HAI local dataset adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.infrastructure.data_utils import (
    binary_labels_from_series,
    read_table,
    segments_from_binary,
)
from industrial_tsad_eval.plugins.datasets.common import (
    TIME_COLUMNS,
    PreparedRun,
    build_prepared_frame,
    detect_column,
    detect_label_column,
    discover_table_files,
    write_prepared_dataset,
)

PAIRED_LABEL_COLUMN = "__itse_attack_label"
LABEL_TEST_RE = re.compile(r"(?:^|[-_])label[-_]?test(\d+)", re.IGNORECASE)
DATA_TEST_RE = re.compile(r"(?:^|[-_])(hai[-_]?)?(end[-_]?)?test(\d+)", re.IGNORECASE)


class HAIDatasetAdapterPlugin:
    """Prepare local HAI CSV layouts into Prepared Format v1."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "hai"

    @property
    def dataset_name(self) -> str:
        """Return the prepared dataset directory name."""
        return "HAI"

    def describe_expected_raw_layout(self) -> str:
        """Describe accepted local raw inputs."""
        return (
            "Directory containing HAI CSV/parquet files, optionally in version subdirectories. "
            "Filenames infer train/test splits and attack/label columns define events."
        )

    def prepare(
        self,
        *,
        raw: Path,
        prepared: Path,
        config: DatasetAdapterConfig,
    ) -> DatasetAdapterResult:
        """Convert HAI tables to Prepared Format v1."""
        runs: list[PreparedRun] = []
        events: list[dict[str, Any]] = []
        table_files = discover_table_files(raw)
        paired_labels = _paired_label_files(table_files)

        data_files = [path for path in table_files if not _is_label_file(path)]
        for index, table_path in enumerate(data_files, start=1):
            table = read_table(table_path)
            label_column = detect_label_column(table)
            split = _split_from_name(table_path)
            if label_column is None:
                label_path = paired_labels.get(_pair_key(table_path))
                if label_path is not None:
                    table = _join_label_file(table, read_table(label_path))
                    label_column = PAIRED_LABEL_COLUMN
                    split = "test"
            run_id = f"hai/{split}/{table_path.stem}_{index:03d}"
            exclude = {label_column} if label_column else set()
            prepared_frame, _enum_maps, period_ns = build_prepared_frame(
                table,
                prefix="Plant/HAI",
                config=config,
                exclude_columns=exclude,
                rename=_hai_feature_name,
                base_offset_rows=index * 100_000,
            )
            runs.append(
                PreparedRun(
                    run_id=run_id,
                    frame=prepared_frame,
                    split=split,
                    source=str(table_path),
                    period_ns=period_ns,
                )
            )
            if label_column is not None:
                labels = binary_labels_from_series(table[label_column])
                events.extend(_events_from_labels(run_id, prepared_frame, labels, table_path.stem))

        runs = _align_feature_schema(runs)
        return write_prepared_dataset(
            prepared=prepared,
            dataset_name=self.dataset_name,
            source_notes="Prepared from local HAI exports.",
            runs=runs,
            events=events,
            config=config,
        )


def _split_from_name(path: Path) -> str:
    lowered = path.name.lower()
    if any(token in lowered for token in ("test", "attack", "anomaly")):
        return "test"
    if "val" in lowered:
        return "val"
    return "train"


def _paired_label_files(files: list[Path]) -> dict[tuple[Path, str], Path]:
    paired: dict[tuple[Path, str], Path] = {}
    for path in files:
        if _is_label_file(path):
            key = _pair_key(path)
            if key[1]:
                paired[key] = path
    return paired


def _is_label_file(path: Path) -> bool:
    return LABEL_TEST_RE.search(path.stem) is not None


def _pair_key(path: Path) -> tuple[Path, str]:
    label_match = LABEL_TEST_RE.search(path.stem)
    if label_match is not None:
        return (path.parent.resolve(), f"test{label_match.group(1)}")
    data_match = DATA_TEST_RE.search(path.stem)
    if data_match is not None:
        return (path.parent.resolve(), f"test{data_match.group(3)}")
    return (path.parent.resolve(), "")


def _join_label_file(data: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    label_column = detect_label_column(labels)
    if label_column is None:
        return data

    merged = data.copy()
    if len(labels) == len(data):
        merged[PAIRED_LABEL_COLUMN] = binary_labels_from_series(labels[label_column])
        return merged

    data_time = detect_column(data, TIME_COLUMNS)
    label_time = detect_column(labels, TIME_COLUMNS)
    if data_time is None or label_time is None:
        merged[PAIRED_LABEL_COLUMN] = 0
        return merged

    label_frame = pd.DataFrame(
        {
            data_time: labels[label_time].astype(str).str.strip(),
            PAIRED_LABEL_COLUMN: binary_labels_from_series(labels[label_column]),
        }
    )
    merged["_itse_join_time"] = merged[data_time].astype(str).str.strip()
    label_frame["_itse_join_time"] = label_frame[data_time].astype(str).str.strip()
    joined = merged.merge(
        label_frame[["_itse_join_time", PAIRED_LABEL_COLUMN]],
        on="_itse_join_time",
        how="left",
    ).drop(columns=["_itse_join_time"])
    joined[PAIRED_LABEL_COLUMN] = joined[PAIRED_LABEL_COLUMN].fillna(0).astype(np.int64)
    return joined


def _hai_feature_name(column: str) -> str:
    parts = [part for part in column.replace("-", "_").split("_") if part]
    if len(parts) >= 2 and parts[0].upper().startswith("P"):
        return "/".join([parts[0].upper(), "_".join(parts[1:])])
    return column


def _align_feature_schema(runs: list[PreparedRun]) -> list[PreparedRun]:
    feature_columns = sorted(
        {str(column) for run in runs for column in run.frame.columns if str(column) != "ts_ns"}
    )
    aligned: list[PreparedRun] = []
    for run in runs:
        frame = run.frame.copy()
        missing = [column for column in feature_columns if column not in frame.columns]
        if missing:
            frame = pd.concat(
                [
                    frame,
                    pd.DataFrame(0.0, index=frame.index, columns=missing),
                ],
                axis=1,
            )
        frame = frame[["ts_ns", *feature_columns]].copy()
        aligned.append(
            PreparedRun(
                run_id=run.run_id,
                frame=frame,
                split=run.split,
                source=run.source,
                period_ns=run.period_ns,
            )
        )
    return aligned


def _events_from_labels(
    run_id: str,
    frame: pd.DataFrame,
    labels: np.ndarray,
    source_id: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(segments_from_binary(labels), start=1):
        events.append(
            {
                "event_id": f"{source_id}_attack_{index:03d}",
                "run_id": run_id,
                "start_ts_ns": int(frame["ts_ns"].iloc[start]),
                "end_ts_ns": int(frame["ts_ns"].iloc[end]),
                "event_type": "attack",
                "metadata": {"source": source_id},
            }
        )
    return events
