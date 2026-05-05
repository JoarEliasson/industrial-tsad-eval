"""SWaT dataset adapter plugin."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.infrastructure.data_utils import (
    binary_labels_from_series,
    coerce_numeric_frame,
    downcast_numeric,
    infer_period_ns,
    read_table,
    segments_from_binary,
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


class SWaTDatasetAdapterPlugin:
    """Prepare local SWaT files into Prepared Format v1."""

    @property
    def name(self) -> str:
        """Return the adapter registry name."""
        return "swat"

    @property
    def dataset_name(self) -> str:
        """Return the prepared dataset directory name."""
        return "SWaT"

    def describe_expected_raw_layout(self) -> str:
        """Describe accepted local raw SWaT layouts."""
        return (
            "Provide local CSV, parquet, or Excel files. Filenames containing normal are "
            "treated as train runs; names containing attack, attacked, anomaly, or anomalous "
            "are treated as test runs. A label column such as Normal/Attack, attack, label, "
            "or anomaly is used to build contiguous attack events."
        )

    def prepare(
        self,
        *,
        raw: Path,
        prepared: Path,
        config: DatasetAdapterConfig,
    ) -> DatasetAdapterResult:
        """Write SWaT Prepared Format v1 artifacts."""
        ensure_prepared_dirs(prepared)
        files = _candidate_files(raw)
        if not files:
            raise FileNotFoundError(f"No SWaT CSV, parquet, or Excel files found under {raw}.")

        train_files, test_files = _split_files(files)
        plan = _build_plan(train_files, test_files)
        tags: dict[str, dict[str, Any]] = {}
        events: list[dict[str, Any]] = []
        train_runs: list[str] = []
        test_runs: list[str] = []
        run_ids: list[str] = []
        warnings: list[str] = []
        effective_period_ns: int | None = None

        remove_startup = bool(config.extra.get("remove_startup", True))
        startup_remove_rows = int(config.extra.get("startup_remove_rows", 0))

        for split_kind, path, run_prefix in plan:
            frame = read_table(path)
            parsed = _parse_swat_frame(frame, path, config)
            effective_period_ns = parsed.period_ns or effective_period_ns
            if parsed.used_synthesized_time:
                warnings.append(f"{path.name}: synthesized timestamps from adapter config.")

            yielded_runs = _yield_runs_for_plan(split_kind, run_prefix, parsed)
            for run_id, run_kind, run_frame, labels in yielded_runs:
                if run_kind == "train":
                    train_runs.append(run_id)
                    if remove_startup and startup_remove_rows > 0:
                        run_frame = run_frame.iloc[startup_remove_rows:].reset_index(drop=True)
                        labels = labels[startup_remove_rows:]
                else:
                    test_runs.append(run_id)
                run_ids.append(run_id)

                segment_count = 0
                if run_kind == "test" and np.any(labels > 0):
                    segment_count = _append_label_events(
                        events=events,
                        run_id=run_id,
                        labels=labels,
                        frame=run_frame,
                        label_col=parsed.label_col,
                        period_ns=parsed.period_ns,
                    )

                for original, browse_path in parsed.rename_map.items():
                    if browse_path not in run_frame.columns:
                        continue
                    if browse_path not in tags:
                        tags[browse_path] = tag_payload(
                            browse_path=browse_path,
                            dtype=str(run_frame[browse_path].dtype),
                            kind=_infer_kind_swat(browse_path),
                            group=_infer_group_swat(browse_path),
                            enum_map=parsed.enum_maps.get(original),
                        )

                write_run(
                    prepared,
                    run_id,
                    run_frame,
                    {
                        "run_id": run_id,
                        "source_file": path.name,
                        "time_col": parsed.time_col or "synthesized",
                        "label_col": parsed.label_col or "none",
                        "attack_fraction": float(np.mean(labels > 0)) if len(labels) else 0.0,
                        "attack_segments_detected": segment_count,
                        "remove_startup": remove_startup if run_kind == "train" else False,
                        "rows": int(run_frame.shape[0]),
                        "columns": int(run_frame.shape[1]),
                        "inferred_period_ns": parsed.period_ns,
                        "fill_policy": "ffill_then_zero",
                    },
                )

        splits = {
            "naive": {"train_runs": train_runs, "val_runs": [], "test_runs": test_runs},
            "all_in_one": {"train_runs": train_runs, "val_runs": [], "test_runs": test_runs},
            "zero_shot": {
                "train_runs": train_runs,
                "val_runs": [],
                "test_runs": test_runs,
                "notes": "Mapped to normal/train and attack/test file split.",
            },
        }
        write_schema(prepared, list(tags.values()))
        write_events(prepared, events)
        write_splits(prepared, splits)
        write_provenance(
            prepared,
            {
                "dataset": "SWaT",
                "notes": (
                    "Prepared from user-supplied local SWaT files. Raw data is not redistributed."
                ),
            },
        )
        write_manifest(
            root=prepared,
            dataset_name=self.dataset_name,
            source_notes="SWaT local files normalized to tag time-series.",
            timebase={
                "column": "ts_ns",
                "unit": "ns",
                "timezone": "UTC",
                "origin": "unix_epoch",
                "nominal_period_ns": effective_period_ns,
            },
            run_ids=run_ids,
            extra={
                "remove_startup": remove_startup,
                "startup_remove_rows": startup_remove_rows,
            },
        )
        return DatasetAdapterResult(
            dataset_name=self.dataset_name,
            prepared_path=str(prepared),
            run_count=len(run_ids),
            event_count=len(events),
            warnings=warnings,
        )


class _ParsedSWaTFrame:
    def __init__(
        self,
        *,
        frame: pd.DataFrame,
        labels: np.ndarray,
        time_col: str | None,
        label_col: str | None,
        rename_map: dict[str, str],
        enum_maps: dict[str, dict[str, int]],
        period_ns: int | None,
        used_synthesized_time: bool,
    ):
        self.frame = frame
        self.labels = labels
        self.time_col = time_col
        self.label_col = label_col
        self.rename_map = rename_map
        self.enum_maps = enum_maps
        self.period_ns = period_ns
        self.used_synthesized_time = used_synthesized_time


def _candidate_files(raw: Path) -> list[Path]:
    return sorted(
        path
        for path in raw.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet", ".xls", ".xlsx"}
    )


def _split_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    train: list[Path] = []
    test: list[Path] = []
    for path in files:
        name = path.name.lower()
        if any(token in name for token in ("attack", "attacked", "anomaly", "anomalous")):
            test.append(path)
        elif "normal" in name:
            train.append(path)
    if not train and not test:
        ordered = sorted(files, key=lambda item: item.name)
        if len(ordered) == 1:
            test = [ordered[0]]
        else:
            train = [ordered[0]]
            test = ordered[1:]
    return train, test


def _build_plan(
    train_files: list[Path],
    test_files: list[Path],
) -> list[tuple[str, Path, str]]:
    if not train_files and len(test_files) == 1:
        return [("split_from_test", test_files[0], "")]
    plan: list[tuple[str, Path, str]] = []
    for index, path in enumerate(train_files, start=1):
        plan.append(("train", path, f"swat/train/normal_run_{index:03d}"))
    for index, path in enumerate(test_files, start=1):
        plan.append(("test", path, f"swat/test/attack_run_{index:03d}"))
    return plan


def _parse_swat_frame(
    frame: pd.DataFrame,
    path: Path,
    config: DatasetAdapterConfig,
) -> _ParsedSWaTFrame:
    time_col = _first_matching_column(frame, {"timestamp", "time", "datetime"})
    used_synthesized_time = time_col is None
    if time_col is None:
        ts_ns = synthesize_timestamps(len(frame), config.base_epoch_iso, config.default_period_ms)
    else:
        ts_ns = to_unix_ns(frame[time_col])

    label_col = _first_matching_column(
        frame,
        {"normal/attack", "attack", "attacks", "label", "labels", "anomaly"},
    )
    labels = (
        binary_labels_from_series(frame[label_col])
        if label_col is not None
        else np.zeros(len(frame), dtype=np.int64)
    )

    drop_columns = {column for column in (time_col, label_col) if column is not None}
    drop_columns.update(
        column
        for column in frame.columns
        if str(column).strip() == ""
        or str(column).strip().lower().startswith("unnamed")
        or str(column).strip().lower() == "index"
    )
    feature_frame = frame.drop(columns=list(drop_columns), errors="ignore")
    feature_frame, enum_maps = coerce_numeric_frame(feature_frame)
    feature_frame = feature_frame.ffill().fillna(0)
    rename_map = {str(column): _swat_browse_path(str(column)) for column in feature_frame.columns}
    feature_frame = feature_frame.rename(columns=rename_map)

    out_frame = pd.concat([pd.Series(ts_ns, name="ts_ns"), feature_frame], axis=1)
    out_frame["ts_ns"] = out_frame["ts_ns"].astype("int64")
    order = np.argsort(out_frame["ts_ns"].to_numpy(dtype=np.int64))
    out_frame = out_frame.iloc[order].reset_index(drop=True)
    labels = labels[order]
    out_frame = downcast_numeric(out_frame)

    return _ParsedSWaTFrame(
        frame=out_frame,
        labels=labels,
        time_col=str(time_col) if time_col is not None else None,
        label_col=str(label_col) if label_col is not None else None,
        rename_map=rename_map,
        enum_maps=enum_maps,
        period_ns=infer_period_ns(out_frame["ts_ns"].to_numpy(dtype=np.int64)),
        used_synthesized_time=used_synthesized_time,
    )


def _yield_runs_for_plan(
    split_kind: str,
    run_prefix: str,
    parsed: _ParsedSWaTFrame,
) -> list[tuple[str, str, pd.DataFrame, np.ndarray]]:
    if split_kind != "split_from_test":
        return [(run_prefix, split_kind, parsed.frame, parsed.labels)]

    attack_indices = np.where(parsed.labels == 1)[0]
    split_index = int(attack_indices[0]) if len(attack_indices) else max(1, len(parsed.frame) // 5)
    split_index = max(1, split_index)
    return [
        (
            "swat/train/normal_run_001",
            "train",
            parsed.frame.iloc[:split_index].reset_index(drop=True),
            parsed.labels[:split_index],
        ),
        (
            "swat/test/attack_run_001",
            "test",
            parsed.frame.iloc[split_index:].reset_index(drop=True),
            parsed.labels[split_index:],
        ),
    ]


def _append_label_events(
    *,
    events: list[dict[str, Any]],
    run_id: str,
    labels: np.ndarray,
    frame: pd.DataFrame,
    label_col: str | None,
    period_ns: int | None,
) -> int:
    timestamps = frame["ts_ns"].to_numpy(dtype=np.int64)
    segments = segments_from_binary(labels)
    effective_period_ns = period_ns or infer_period_ns(timestamps)
    for index, (start, end) in enumerate(segments):
        end_ts = (
            int(timestamps[end] + effective_period_ns)
            if effective_period_ns is not None
            else int(timestamps[end])
        )
        events.append(
            {
                "event_id": f"{run_id.replace('/', '_')}_attack_{index:04d}",
                "run_id": run_id,
                "start_ts_ns": int(timestamps[start]),
                "end_ts_ns": end_ts,
                "event_type": "attack",
                "metadata": {
                    "label_col": label_col or "none",
                    "end_is_exclusive": effective_period_ns is not None,
                    "inferred_period_ns": effective_period_ns,
                    "segment_index": index,
                },
            }
        )
    return len(segments)


def _first_matching_column(frame: pd.DataFrame, candidates: set[str]) -> str | None:
    for column in frame.columns:
        if str(column).strip().lower() in candidates:
            return str(column)
    return None


def _swat_browse_path(tag: str) -> str:
    return f"Plant/SWaT/{tag.strip()}"


def _infer_group_swat(browse_path: str) -> str:
    tag = browse_path.split("/")[-1]
    match = re.search(r"(\d{3})$", tag)
    return f"P{match.group(1)[0]}" if match else ""


def _infer_kind_swat(browse_path: str) -> str:
    tag = browse_path.split("/")[-1].upper()
    if tag.startswith(("MV", "P", "UV")):
        return "actuator"
    return "sensor"
