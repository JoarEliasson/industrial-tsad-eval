"""HAI dataset adapter plugin."""

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
    segments_from_binary,
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


class HAIDatasetAdapterPlugin:
    """Prepare local HAI CSV layouts into Prepared Format v1."""

    @property
    def name(self) -> str:
        """Return the adapter registry name."""
        return "hai"

    @property
    def dataset_name(self) -> str:
        """Return the prepared dataset directory name."""
        return "HAI"

    def describe_expected_raw_layout(self) -> str:
        """Describe accepted local raw HAI layouts."""
        return (
            "Provide HAI CSV files directly or inside version directories such as hai-22.04. "
            "Set extra.version to choose a version when multiple are present. CSVs should "
            "contain a time or timestamp column plus one or more attack* label columns."
        )

    def prepare(
        self,
        *,
        raw: Path,
        prepared: Path,
        config: DatasetAdapterConfig,
    ) -> DatasetAdapterResult:
        """Write HAI Prepared Format v1 artifacts."""
        ensure_prepared_dirs(prepared)
        version, csv_root = _resolve_hai_root(raw, config)
        csvs = sorted(
            path
            for path in csv_root.rglob("*.csv")
            if path.is_file() and not path.name.lower().startswith("label")
        )
        if not csvs:
            raise FileNotFoundError(f"No HAI CSV files found under {csv_root}.")

        supervised = bool(config.extra.get("supervised", False))
        tags: dict[str, dict[str, Any]] = {}
        events: list[dict[str, Any]] = []
        run_ids: list[str] = []

        for csv_path in csvs:
            run_id = _run_id_for_csv(csv_path, version)
            run_ids.append(run_id)
            frame = pd.read_csv(csv_path, sep=None, engine="python")
            if frame.shape[1] < 2:
                raise ValueError(f"{csv_path.name}: expected at least a time column and one tag.")

            time_col = _time_column(frame)
            ts_ns = to_unix_ns(frame[time_col])
            feature_frame = frame.drop(columns=[time_col])
            label_cols = [
                str(column) for column in feature_frame.columns if _is_label_col(str(column))
            ]
            labels = _combined_labels(feature_frame, label_cols)
            feature_frame = feature_frame.drop(columns=label_cols, errors="ignore")
            feature_frame, enum_maps = coerce_numeric_frame(feature_frame)
            rename_map = {
                str(column): _hai_browse_path(str(column)) for column in feature_frame.columns
            }
            feature_frame = feature_frame.rename(columns=rename_map)

            is_train = bool(re.search(r"(?i)train", run_id))
            include_events = labels is not None and (supervised or not is_train)
            if labels is not None and is_train and not supervised:
                normal_mask = labels == 0
                feature_frame = feature_frame.iloc[normal_mask].reset_index(drop=True)
                ts_ns = ts_ns[normal_mask]
                labels = labels[normal_mask]
            else:
                feature_frame = feature_frame.reset_index(drop=True)

            out_frame = pd.concat([pd.Series(ts_ns, name="ts_ns"), feature_frame], axis=1)
            out_frame["ts_ns"] = out_frame["ts_ns"].astype("int64")
            out_frame = downcast_numeric(out_frame)
            period_ns = infer_period_ns(out_frame["ts_ns"].to_numpy(dtype=np.int64))

            if include_events and labels is not None:
                _append_hai_events(
                    events=events,
                    run_id=run_id,
                    labels=labels,
                    timestamps=out_frame["ts_ns"].to_numpy(dtype=np.int64),
                    period_ns=period_ns,
                    label_cols=label_cols,
                    version=version,
                )

            for original, browse_path in rename_map.items():
                if browse_path not in out_frame.columns:
                    continue
                tags[browse_path] = tag_payload(
                    browse_path=browse_path,
                    dtype=str(out_frame[browse_path].dtype),
                    kind=_infer_kind_hai(browse_path),
                    group=_infer_group_hai(browse_path),
                    enum_map=enum_maps.get(original),
                )

            write_run(
                prepared,
                run_id,
                out_frame,
                {
                    "run_id": run_id,
                    "source_file": str(csv_path),
                    "time_col": str(time_col),
                    "label_cols": label_cols,
                    "has_anomalies": bool(labels is not None and np.any(labels > 0)),
                    "rows": int(out_frame.shape[0]),
                    "columns": int(out_frame.shape[1]),
                    "inferred_period_ns": period_ns,
                    "version": version,
                },
            )

        splits = _hai_splits(run_ids, events)
        write_schema(prepared, list(tags.values()))
        write_events(prepared, events)
        write_splits(prepared, splits)
        provenance: dict[str, Any] = {
            "dataset": "HAI",
            "notes": "Prepared from user-supplied local HAI files.",
        }
        if version is not None:
            provenance["version"] = version
        write_provenance(prepared, provenance)
        write_manifest(
            root=prepared,
            dataset_name=self.dataset_name,
            source_notes="HAI CSV files normalized to tag time-series.",
            timebase={
                "column": "ts_ns",
                "unit": "ns",
                "timezone": "UTC",
                "origin": "unix_epoch",
            },
            run_ids=run_ids,
            extra={"version": version, "supervised": supervised},
        )
        return DatasetAdapterResult(
            dataset_name=self.dataset_name,
            prepared_path=str(prepared),
            run_count=len(run_ids),
            event_count=len(events),
            warnings=[],
        )


def _resolve_hai_root(raw: Path, config: DatasetAdapterConfig) -> tuple[str | None, Path]:
    version_dirs = sorted(
        path
        for path in raw.iterdir()
        if path.is_dir() and re.match(r"(?:hai|haiend)-\d{2}\.\d{2}$", path.name)
    )
    if not version_dirs:
        return None, raw

    requested = str(config.extra.get("version", "hai-22.04"))
    by_name = {path.name: path for path in version_dirs}
    if requested not in by_name:
        raise FileNotFoundError(
            f"Requested HAI version {requested!r} not found. Available: {sorted(by_name)}"
        )
    return requested, by_name[requested]


def _run_id_for_csv(path: Path, version: str | None) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)
    return f"{version}/{stem}" if version is not None else stem


def _time_column(frame: pd.DataFrame) -> str:
    for candidate in ("time", "timestamp", "Timestamp"):
        if candidate in frame.columns:
            return str(candidate)
    return str(frame.columns[0])


def _is_label_col(column: str) -> bool:
    return bool(re.match(r"(?i)^(attack|label|anomaly)", column.strip()))


def _combined_labels(frame: pd.DataFrame, label_cols: list[str]) -> np.ndarray | None:
    if not label_cols:
        return None
    labels = np.zeros(len(frame), dtype=np.int64)
    for column in label_cols:
        labels = np.maximum(labels, binary_labels_from_series(frame[column]))
    return labels


def _append_hai_events(
    *,
    events: list[dict[str, Any]],
    run_id: str,
    labels: np.ndarray,
    timestamps: np.ndarray,
    period_ns: int | None,
    label_cols: list[str],
    version: str | None,
) -> None:
    for index, (start, end) in enumerate(segments_from_binary(labels)):
        end_ts = int(timestamps[end] + period_ns) if period_ns is not None else int(timestamps[end])
        attack_type, target_process = _label_metadata(label_cols)
        events.append(
            {
                "event_id": f"{run_id.replace('/', '_')}_attack_{index:04d}",
                "run_id": run_id,
                "start_ts_ns": int(timestamps[start]),
                "end_ts_ns": end_ts,
                "event_type": "attack",
                "metadata": {
                    "label_columns": label_cols,
                    "end_is_exclusive": period_ns is not None,
                    "inferred_period_ns": period_ns,
                    "attack_type": attack_type,
                    "target_process": target_process,
                    "source_dataset_version": version,
                    "version": version,
                    "segment_index": index,
                    "metadata_source": {
                        "attack_type": "label column suffix" if attack_type else None,
                        "target_process": "label column suffix" if target_process else None,
                        "version": "adapter config or version directory" if version else None,
                    },
                },
            }
        )


def _label_metadata(label_cols: list[str]) -> tuple[str | None, str | None]:
    attack_type: str | None = None
    target_process: str | None = None
    for label_col in label_cols:
        match = re.match(r"(?i)^attack[_\s]*(.+)$", label_col)
        if match:
            suffix = match.group(1).strip()
            attack_type = attack_type or suffix
            if re.match(r"^P\d+$", suffix):
                target_process = suffix
    return attack_type, target_process


def _hai_splits(run_ids: list[str], events: list[dict[str, Any]]) -> dict[str, Any]:
    train_candidates = [run_id for run_id in sorted(run_ids) if re.search(r"(?i)train", run_id)]
    test_candidates = [run_id for run_id in sorted(run_ids) if re.search(r"(?i)test", run_id)]
    if not train_candidates:
        train_candidates = sorted(run_ids)

    anomalous_runs = {str(event["run_id"]) for event in events}
    normal_train = [run_id for run_id in train_candidates if run_id not in anomalous_runs]
    train_base = normal_train if normal_train else train_candidates
    if len(train_base) >= 2:
        train_runs = train_base[:-1]
        val_runs = [train_base[-1]]
    else:
        train_runs = train_base
        val_runs = []
    test_runs = test_candidates or sorted(anomalous_runs)
    return {
        "naive": {"train_runs": train_runs, "val_runs": val_runs, "test_runs": test_runs},
        "all_in_one": {"train_runs": train_base, "val_runs": val_runs, "test_runs": test_runs},
        "zero_shot": {
            "train_runs": train_runs,
            "val_runs": val_runs,
            "test_runs": test_runs,
            "notes": "Best-effort run split inferred from filenames.",
        },
    }


def _hai_browse_path(column: str) -> str:
    match = re.match(r"^(P\d+)_(.+)$", column)
    if match:
        return f"Plant/HAI/{match.group(1)}/{match.group(2)}"
    return f"Plant/HAI/{column}"


def _infer_group_hai(browse_path: str) -> str:
    parts = browse_path.split("/")
    return parts[2] if len(parts) >= 4 and parts[0] == "Plant" and parts[1] == "HAI" else ""


def _infer_kind_hai(browse_path: str) -> str:
    last = browse_path.split("/")[-1].upper()
    if last.startswith("MV") or "VALVE" in last:
        return "actuator"
    if "MODE" in last or "STATE" in last:
        return "state"
    return "sensor"
