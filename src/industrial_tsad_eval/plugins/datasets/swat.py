"""SWaT local dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.infrastructure.data_utils import (
    binary_labels_from_series,
    segments_from_binary,
)
from industrial_tsad_eval.plugins.datasets.common import (
    PreparedRun,
    build_prepared_frame,
    detect_label_column,
    discover_table_files,
    write_prepared_dataset,
)


class SWaTDatasetAdapterPlugin:
    """Prepare local SWaT normal and attack tables into Prepared Format v1."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "swat"

    @property
    def dataset_name(self) -> str:
        """Return the prepared dataset directory name."""
        return "SWaT"

    def describe_expected_raw_layout(self) -> str:
        """Describe accepted local raw inputs."""
        return (
            "Directory containing SWaT normal/train and attack/test CSV, parquet, or xlsx files. "
            "A label, attack, anomaly, or Normal/Attack column is used for event extraction."
        )

    def prepare(
        self,
        *,
        raw: Path,
        prepared: Path,
        config: DatasetAdapterConfig,
    ) -> DatasetAdapterResult:
        """Convert SWaT tables to Prepared Format v1."""
        runs: list[PreparedRun] = []
        events: list[dict[str, Any]] = []

        for index, table_path in enumerate(discover_table_files(raw), start=1):
            table = pd.read_parquet(table_path) if table_path.suffix.lower() == ".parquet" else None
            if table is None:
                from industrial_tsad_eval.infrastructure.data_utils import read_table

                table = read_table(table_path)
            label_column = detect_label_column(table)
            split = _split_from_name(table_path)
            run_id = f"swat/{split}/{table_path.stem}_{index:03d}"
            exclude = {label_column} if label_column else set()
            prepared_frame, _enum_maps, period_ns = build_prepared_frame(
                table,
                prefix="Plant/SWaT",
                config=config,
                exclude_columns=exclude,
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

        return write_prepared_dataset(
            prepared=prepared,
            dataset_name=self.dataset_name,
            source_notes="Prepared from local SWaT exports.",
            runs=runs,
            events=events,
            config=config,
        )


def _split_from_name(path: Path) -> str:
    lowered = path.name.lower()
    if any(token in lowered for token in ("attack", "anomaly", "test")):
        return "test"
    return "train"


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
