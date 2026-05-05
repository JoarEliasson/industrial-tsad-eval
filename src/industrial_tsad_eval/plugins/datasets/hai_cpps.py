"""HAI-CPPS local scenario dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.infrastructure.data_utils import read_table
from industrial_tsad_eval.plugins.datasets.common import (
    PreparedRun,
    build_prepared_frame,
    discover_table_files,
    write_prepared_dataset,
)


class HAICPPSDatasetAdapterPlugin:
    """Prepare HAI-CPPS scenario directories into Prepared Format v1."""

    @property
    def name(self) -> str:
        """Return the registry name."""
        return "hai-cpps"

    @property
    def dataset_name(self) -> str:
        """Return the prepared dataset directory name."""
        return "HAI_CPPS"

    def describe_expected_raw_layout(self) -> str:
        """Describe accepted local raw inputs."""
        return (
            "Directory with HAI-CPPS scenario subdirectories. Each scenario should contain a "
            "continuous CSV/parquet table and may include sim_setup.json with anomaly metadata."
        )

    def prepare(
        self,
        *,
        raw: Path,
        prepared: Path,
        config: DatasetAdapterConfig,
    ) -> DatasetAdapterResult:
        """Convert HAI-CPPS scenarios to Prepared Format v1."""
        runs: list[PreparedRun] = []
        events: list[dict[str, Any]] = []

        for index, table_path in enumerate(discover_table_files(raw), start=1):
            if table_path.name.lower() == "sim_setup.json":
                continue
            table = read_table(table_path)
            scenario_dir = table_path.parent
            setup = _load_setup(scenario_dir / "sim_setup.json")
            split = _split_from_context(table_path, setup)
            run_id = f"hai-cpps/{split}/{scenario_dir.name}_{index:03d}"
            exclude = _setup_label_columns(table)
            prepared_frame, _enum_maps, period_ns = build_prepared_frame(
                table,
                prefix="Plant/HAI_CPPS",
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
            events.extend(_events_from_setup(run_id, prepared_frame, setup, table_path))

        return write_prepared_dataset(
            prepared=prepared,
            dataset_name=self.dataset_name,
            source_notes="Prepared from local HAI-CPPS scenario exports.",
            runs=runs,
            events=events,
            config=config,
        )


def _load_setup(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _split_from_context(path: Path, setup: dict[str, Any]) -> str:
    text = " ".join([path.name, path.parent.name, json.dumps(setup, sort_keys=True)]).lower()
    if any(token in text for token in ("anom", "attack", "fault")):
        return "test"
    return "train"


def _setup_label_columns(frame: pd.DataFrame) -> set[str]:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    return {
        column
        for key, column in lowered.items()
        if key in {"label", "attack", "anomaly", "fault", "is_attack", "is_anomaly"}
    }


def _events_from_setup(
    run_id: str,
    frame: pd.DataFrame,
    setup: dict[str, Any],
    table_path: Path,
) -> list[dict[str, Any]]:
    if len(frame) == 0 or _split_from_context(table_path, setup) != "test":
        return []

    start_index = _setup_int(
        setup, ("attack_start", "anomaly_start", "fault_start", "onset"), len(frame) // 3
    )
    duration = _setup_int(
        setup, ("duration", "attack_duration", "anomaly_duration"), max(1, len(frame) // 5)
    )
    start_index = int(np.clip(start_index, 0, len(frame) - 1))
    end_index = int(np.clip(start_index + duration, start_index, len(frame) - 1))
    return [
        {
            "event_id": f"{table_path.parent.name}_event_001",
            "run_id": run_id,
            "start_ts_ns": int(frame["ts_ns"].iloc[start_index]),
            "end_ts_ns": int(frame["ts_ns"].iloc[end_index]),
            "event_type": "attack",
            "metadata": {
                "scenario": table_path.parent.name,
                "setup": setup,
                "onset_clamped": start_index,
            },
        }
    ]


def _setup_int(payload: dict[str, Any], keys: tuple[str, ...], default: int) -> int:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, int | float | str):
            try:
                return int(value)
            except ValueError:
                continue
    return default
