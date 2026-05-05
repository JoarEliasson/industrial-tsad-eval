"""Tennessee Eastman Process local dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.infrastructure.data_utils import segments_from_binary
from industrial_tsad_eval.plugins.datasets.common import (
    PreparedRun,
    build_prepared_frame,
    discover_table_files,
    write_prepared_dataset,
)


class TEPDatasetAdapterPlugin:
    """Prepare local TEP CSV exports into Prepared Format v1."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "tep"

    @property
    def dataset_name(self) -> str:
        """Return the prepared dataset directory name."""
        return "TEP"

    def describe_expected_raw_layout(self) -> str:
        """Describe accepted local raw inputs."""
        return (
            "Directory with TEP CSV files. Columns such as faultNumber, simulationRun, sample, "
            "XMEAS(1)/xmeas_1, and XMV(1)/xmv_1 are recognized."
        )

    def prepare(
        self,
        *,
        raw: Path,
        prepared: Path,
        config: DatasetAdapterConfig,
    ) -> DatasetAdapterResult:
        """Convert local TEP CSV tables to Prepared Format v1."""
        runs: list[PreparedRun] = []
        events: list[dict[str, Any]] = []
        warnings: list[str] = []

        for table_path in discover_table_files(raw):
            if table_path.suffix.lower() != ".csv":
                warnings.append(
                    f"Skipped unsupported TEP table {table_path.name}; CSV is supported now."
                )
                continue
            table = pd.read_csv(table_path, low_memory=False)
            for run_index, run_frame in _group_tep_runs(table):
                fault_id = _fault_id(run_frame)
                split = _tep_split(table_path, fault_id)
                run_id = _run_id(table_path, run_index, fault_id, split)
                prepared_frame, _enum_maps, period_ns = build_prepared_frame(
                    run_frame,
                    prefix="Plant/TEP",
                    config=config,
                    exclude_columns=_metadata_columns(run_frame),
                    rename=_tep_feature_name,
                    base_offset_rows=len(runs) * 100_000,
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
                events.extend(_fault_events(run_id, prepared_frame, run_frame, fault_id))

        return write_prepared_dataset(
            prepared=prepared,
            dataset_name=self.dataset_name,
            source_notes="Prepared from local Tennessee Eastman Process exports.",
            runs=runs,
            events=events,
            config=config,
            warnings=warnings,
        )


def _group_tep_runs(frame: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    run_column = _find(frame, {"simulationRun", "simulation_run", "run", "run_id"})
    if run_column is None:
        return [("001", frame.reset_index(drop=True))]
    return [
        (str(run_id), group.reset_index(drop=True))
        for run_id, group in frame.groupby(run_column, sort=True)
    ]


def _fault_id(frame: pd.DataFrame) -> int:
    fault_column = _find(frame, {"faultNumber", "fault_number", "fault", "Fault"})
    if fault_column is None:
        return 0
    value = pd.to_numeric(frame[fault_column], errors="coerce").dropna()
    return int(value.iloc[0]) if not value.empty else 0


def _tep_split(path: Path, fault_id: int) -> str:
    lowered = path.name.lower()
    if fault_id == 0 or any(token in lowered for token in ("train", "normal", "d00", "fault_00")):
        return "train"
    return "test"


def _run_id(path: Path, run_index: str, fault_id: int, split: str) -> str:
    source = "train" if split == "train" else "test"
    return f"tep/{source}/fault_{fault_id:02d}/run_{_safe_index(run_index)}_{path.stem}"


def _metadata_columns(frame: pd.DataFrame) -> set[str]:
    names = {
        "faultNumber",
        "fault_number",
        "fault",
        "simulationRun",
        "simulation_run",
        "run",
        "run_id",
        "sample",
    }
    return {column for column in frame.columns if str(column) in names}


def _tep_feature_name(column: str) -> str:
    lowered = column.lower().replace("(", "_").replace(")", "").replace(".", "_")
    compact = lowered.replace(" ", "").replace("-", "_")
    if compact.startswith("xmeas"):
        suffix = compact.replace("xmeas", "").strip("_")
        return f"XMEAS_{int(suffix):02d}" if suffix.isdigit() else column
    if compact.startswith("xmv"):
        suffix = compact.replace("xmv", "").strip("_")
        return f"XMV_{int(suffix):02d}" if suffix.isdigit() else column
    return column


def _fault_events(
    run_id: str,
    prepared_frame: pd.DataFrame,
    raw_frame: pd.DataFrame,
    fault_id: int,
) -> list[dict[str, Any]]:
    if fault_id == 0 or len(prepared_frame) == 0:
        return []
    sample_column = _find(raw_frame, {"sample"})
    if sample_column is not None:
        labels = (
            pd.to_numeric(raw_frame[sample_column], errors="coerce").fillna(0) > 160
        ).to_numpy()
    else:
        start = min(len(prepared_frame) // 3, max(len(prepared_frame) - 1, 0))
        labels = np.zeros(len(prepared_frame), dtype=np.int64)
        labels[start:] = 1

    events: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(segments_from_binary(labels), start=1):
        events.append(
            {
                "event_id": f"{run_id.replace('/', '_')}_fault_{fault_id:02d}_{index:03d}",
                "run_id": run_id,
                "start_ts_ns": int(prepared_frame["ts_ns"].iloc[start]),
                "end_ts_ns": int(prepared_frame["ts_ns"].iloc[end]),
                "event_type": "fault",
                "metadata": {"fault_id": fault_id},
            }
        )
    return events


def _find(frame: pd.DataFrame, names: set[str]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for name in names:
        match = lowered.get(name.lower())
        if match is not None:
            return match
    return None


def _safe_index(value: str) -> str:
    cleaned = "".join(character for character in value if character.isalnum() or character in "._-")
    return cleaned or "001"
