"""Tennessee Eastman Process dataset adapter plugin."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.infrastructure.data_utils import downcast_numeric, parse_iso_to_ns
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


class TEPDatasetAdapterPlugin:
    """Prepare local TEP raw files into Prepared Format v1."""

    @property
    def name(self) -> str:
        """Return the adapter registry name."""
        return "tep"

    @property
    def dataset_name(self) -> str:
        """Return the prepared dataset directory name."""
        return "TEP"

    def describe_expected_raw_layout(self) -> str:
        """Describe accepted local raw TEP layouts."""
        return (
            "Provide local TEP CSV, MAT, or RData files. CSVs may be combined files with "
            "faultNumber, simulationRun, sample, XMEAS*, and XMV* columns, or split files "
            "such as TEP_FaultFree_Training.csv and TEP_Faulty_Testing_01.csv."
        )

    def prepare(
        self,
        *,
        raw: Path,
        prepared: Path,
        config: DatasetAdapterConfig,
    ) -> DatasetAdapterResult:
        """Write TEP Prepared Format v1 artifacts."""
        ensure_prepared_dirs(prepared)
        base_ns = parse_iso_to_ns(config.base_epoch_iso)
        period_ns = _tep_period_ns(config)
        bundles = _load_tep_bundle(raw)

        tags: dict[str, dict[str, Any]] = {}
        events: list[dict[str, Any]] = []
        run_ids: list[str] = []
        train_runs: list[str] = []
        test_runs: list[str] = []
        warnings: list[str] = []

        for split_name, frame in bundles:
            is_training = "training" in split_name.lower() or "train" in split_name.lower()
            split_fault = _fault_from_split_name(split_name)
            normalized = frame.copy()
            for simulation_run, run_frame in normalized.groupby("simulationRun"):
                run_frame = run_frame.sort_values("sample").reset_index(drop=True)
                fault_id = int(run_frame["faultNumber"].iloc[0])
                if split_fault is not None:
                    fault_id = split_fault
                subset = _subset_name(is_training, fault_id)
                run_id = f"tep/{subset}/fault_{fault_id:02d}/run_{int(simulation_run):03d}"
                run_ids.append(run_id)
                if is_training:
                    train_runs.append(run_id)
                else:
                    test_runs.append(run_id)

                samples = run_frame["sample"].to_numpy(dtype=np.int64)
                feature_columns = [
                    str(column)
                    for column in run_frame.columns
                    if str(column) not in {"faultNumber", "simulationRun", "sample"}
                ]
                feature_frame = run_frame[feature_columns].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                rename_map = {column: _tep_browse_path(column) for column in feature_columns}
                feature_frame = feature_frame.rename(columns=rename_map).reset_index(drop=True)
                ts_ns = base_ns + (samples - 1) * period_ns
                out_frame = pd.concat(
                    [pd.Series(ts_ns, name="ts_ns"), feature_frame],
                    axis=1,
                )
                out_frame = downcast_numeric(out_frame)

                for original, browse_path in rename_map.items():
                    if browse_path in out_frame.columns:
                        tags[browse_path] = tag_payload(
                            browse_path=browse_path,
                            dtype=str(out_frame[browse_path].dtype),
                            kind="sensor" if "/XMEAS_" in browse_path else "actuator",
                            group="XMEAS" if "/XMEAS_" in browse_path else "XMV",
                            description=original,
                        )

                if fault_id != 0:
                    event = _fault_event(
                        run_id=run_id,
                        fault_id=fault_id,
                        is_training=is_training,
                        samples=samples,
                        ts_ns=ts_ns,
                        period_ns=period_ns,
                        subset=subset,
                    )
                    if event is not None:
                        events.append(event)
                    elif config.strict:
                        warnings.append(f"Run {run_id}: fault start sample is outside the run.")

                write_run(
                    prepared,
                    run_id,
                    out_frame,
                    {
                        "run_id": run_id,
                        "split_source": split_name,
                        "fault_id": fault_id,
                        "is_training": is_training,
                        "period_ns": period_ns,
                        "base_ts_ns": base_ns,
                        "base_epoch_iso": config.base_epoch_iso,
                        "rows": int(out_frame.shape[0]),
                        "columns": int(out_frame.shape[1]),
                    },
                )

        split_payload = _build_splits(train_runs, test_runs)
        write_schema(prepared, list(tags.values()))
        write_events(prepared, events)
        write_splits(prepared, split_payload)
        write_provenance(
            prepared,
            {
                "dataset": "TEP",
                "notes": "Prepared from user-supplied local TEP raw files.",
            },
        )
        write_manifest(
            root=prepared,
            dataset_name=self.dataset_name,
            source_notes="Tennessee Eastman Process simulations normalized to tag time-series.",
            timebase={
                "column": "ts_ns",
                "unit": "ns",
                "timezone": "UTC",
                "origin": "unix_epoch",
                "nominal_period_ns": period_ns,
            },
            run_ids=run_ids,
            extra={"base_ts_ns": base_ns, "period_ns": period_ns},
        )
        return DatasetAdapterResult(
            dataset_name=self.dataset_name,
            prepared_path=str(prepared),
            run_count=len(run_ids),
            event_count=len(events),
            warnings=warnings,
        )


def _tep_period_ns(config: DatasetAdapterConfig) -> int:
    if "period_ns" in config.extra:
        return int(config.extra["period_ns"])
    if "period_ms" in config.extra:
        return int(config.extra["period_ms"]) * 1_000_000
    return 180 * 1_000_000_000


def _subset_name(is_training: bool, fault_id: int) -> str:
    if is_training and fault_id == 0:
        return "ff_train"
    if not is_training and fault_id == 0:
        return "ff_test"
    if is_training:
        return "f_train"
    return "f_test"


def _build_splits(train_runs: list[str], test_runs: list[str]) -> dict[str, Any]:
    train_ff = sorted(run_id for run_id in train_runs if "fault_00" in run_id)
    train_faulty = sorted(run_id for run_id in train_runs if "fault_00" not in run_id)
    test_ff = sorted(run_id for run_id in test_runs if "fault_00" in run_id)
    test_faulty = sorted(run_id for run_id in test_runs if "fault_00" not in run_id)

    if len(train_ff) >= 3:
        val_runs = [train_ff[-1]]
        selected_train = train_ff[:-1]
    else:
        val_runs = []
        selected_train = train_ff
    selected_test = test_ff + test_faulty + train_faulty
    return {
        "naive": {
            "train_runs": selected_train,
            "val_runs": val_runs,
            "test_runs": selected_test,
        },
        "all_in_one": {
            "train_runs": selected_train,
            "val_runs": val_runs,
            "test_runs": selected_test,
        },
        "zero_shot": {
            "train_runs": train_ff,
            "val_runs": [],
            "test_runs": test_faulty + train_faulty,
            "notes": "Train on fault-free runs and test on faulty runs.",
        },
    }


def _fault_event(
    *,
    run_id: str,
    fault_id: int,
    is_training: bool,
    samples: np.ndarray,
    ts_ns: np.ndarray,
    period_ns: int,
    subset: str,
) -> dict[str, Any] | None:
    start_sample = 21 if is_training else 161
    candidate_indices = np.where(samples >= start_sample)[0]
    if len(candidate_indices) == 0:
        return None
    start_index = int(candidate_indices[0])
    return {
        "event_id": f"{run_id.replace('/', '_')}_fault_{fault_id:02d}",
        "run_id": run_id,
        "start_ts_ns": int(ts_ns[start_index]),
        "end_ts_ns": int(ts_ns[-1] + period_ns),
        "event_type": "fault",
        "metadata": {
            "fault_number": fault_id,
            "subset": subset,
            "end_is_exclusive": True,
            "period_ns": period_ns,
            "start_sample": start_sample,
        },
    }


def _load_tep_bundle(raw: Path) -> list[tuple[str, pd.DataFrame]]:
    loaders = (_load_csv_bundle, _load_mat_bundle, _load_rdata_bundle)
    skipped: list[str] = []
    for loader in loaders:
        try:
            loaded = loader(raw)
        except ImportError as exc:
            skipped.append(str(exc))
            continue
        if loaded:
            return loaded
    suffix = f" Skipped optional formats: {', '.join(skipped)}." if skipped else ""
    raise FileNotFoundError(
        "No usable TEP files found. Expected CSV, MAT, or RData files with TEP columns." + suffix
    )


def _load_csv_bundle(raw: Path) -> list[tuple[str, pd.DataFrame]]:
    csvs = sorted(path for path in raw.rglob("*.csv") if path.is_file())
    loaded: list[tuple[str, pd.DataFrame]] = []
    for path in csvs:
        frame = _normalize_tep_frame(pd.read_csv(path, low_memory=False), path)
        split_name = _split_name_from_filename(path)
        if split_name is not None:
            loaded.append((split_name, frame))
            continue
        subset_name = "Training" if "train" in path.name.lower() else "Testing"
        for fault_value, group in frame.groupby("faultNumber"):
            fault_id = int(fault_value)
            if fault_id == 0:
                loaded.append((f"FaultFree_{subset_name}", group.copy()))
            else:
                loaded.append((f"Faulty_{subset_name}_{fault_id:02d}", group.copy()))
    return loaded


def _load_mat_bundle(raw: Path) -> list[tuple[str, pd.DataFrame]]:
    mat_files = sorted(path for path in raw.rglob("*.mat") if path.is_file())
    if not mat_files:
        return []
    try:
        import scipy.io as scipy_io
    except ImportError as exc:
        raise ImportError("MAT requires scipy") from exc

    loaded: list[tuple[str, pd.DataFrame]] = []
    for path in mat_files:
        split_name = _split_name_from_filename(path)
        if split_name is None:
            continue
        try:
            mat_obj = scipy_io.loadmat(path)
        except NotImplementedError:
            mat_obj = _load_hdf5_mat(path)
        frame = _normalize_mat_bundle(mat_obj, path)
        if "Faulty" in split_name:
            for fault_value, group in frame.groupby("faultNumber"):
                fault_id = int(fault_value)
                if fault_id != 0:
                    loaded.append((f"{split_name}_{fault_id:02d}", group.copy()))
        else:
            loaded.append((split_name, frame))
    return loaded


def _load_rdata_bundle(raw: Path) -> list[tuple[str, pd.DataFrame]]:
    rdata_files = sorted(path for path in raw.rglob("*.RData") if path.is_file())
    if not rdata_files:
        return []
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError("RData requires pyreadr") from exc

    loaded: list[tuple[str, pd.DataFrame]] = []
    for path in rdata_files:
        split_name = _split_name_from_filename(path)
        if split_name is None:
            continue
        result = pyreadr.read_r(str(path))
        if not result:
            continue
        frame = _normalize_tep_frame(next(iter(result.values())), path)
        if "Faulty" in split_name:
            for fault_value, group in frame.groupby("faultNumber"):
                fault_id = int(fault_value)
                if fault_id != 0:
                    loaded.append((f"{split_name}_{fault_id:02d}", group.copy()))
        else:
            loaded.append((split_name, frame))
    return loaded


def _load_hdf5_mat(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("MAT v7.3 requires h5py") from exc
    with h5py.File(path, "r") as handle:
        return {key: np.asarray(value) for key, value in handle.items()}


def _normalize_mat_bundle(mat_obj: dict[str, Any], path: Path) -> pd.DataFrame:
    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in mat_obj.items():
        if str(key).startswith("__"):
            continue
        array = np.asarray(value)
        if array.ndim == 2:
            candidates.append((str(key), array))
    if not candidates:
        raise ValueError(f"{path.name}: no 2D numeric arrays found in MAT file.")

    _key, array = max(candidates, key=lambda item: item[1].size)
    if array.shape[0] in {54, 55} and array.shape[1] not in {54, 55}:
        array = array.T
    if array.shape[1] not in {54, 55}:
        raise ValueError(
            f"{path.name}: expected 54 or 55 TEP columns in MAT array, got {array.shape}."
        )

    columns = ["faultNumber", "simulationRun"]
    if array.shape[1] == 55:
        columns.append("sample")
    columns.extend(f"xmeas_{index}" for index in range(1, 42))
    columns.extend(f"xmv_{index}" for index in range(1, 12))
    frame = pd.DataFrame(array, columns=columns)
    if "sample" not in frame.columns:
        frame["sample"] = frame.groupby(["faultNumber", "simulationRun"]).cumcount() + 1
    return _normalize_tep_frame(frame, path)


def _normalize_tep_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    rename: dict[Any, str] = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        if "fault" in key and "num" in key:
            rename[column] = "faultNumber"
        elif "sim" in key and "run" in key:
            rename[column] = "simulationRun"
        elif key == "sample":
            rename[column] = "sample"
        elif match := re.match(r"xmeas[_\s(]*(\d+)", key):
            rename[column] = f"xmeas_{int(match.group(1))}"
        elif match := re.match(r"xmv[_\s(]*(\d+)", key):
            rename[column] = f"xmv_{int(match.group(1))}"

    normalized = frame.rename(columns=rename)
    required = ["faultNumber", "simulationRun", "sample"]
    required.extend(f"xmeas_{index}" for index in range(1, 42))
    required.extend(f"xmv_{index}" for index in range(1, 12))
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"CSV file {path.name} is missing required TEP columns: {missing[:10]}")
    return normalized[required].copy()


def _split_name_from_filename(path: Path) -> str | None:
    name = path.name.lower()
    if "faultfree" in name and "train" in name:
        return "FaultFree_Training"
    if "faultfree" in name and "test" in name:
        return "FaultFree_Testing"
    match = re.match(r".*faulty[_-]?(training|testing)[_-]?(\d{2})?", name)
    if match:
        suffix = f"_{int(match.group(2)):02d}" if match.group(2) else ""
        return f"Faulty_{match.group(1).title()}{suffix}"
    return None


def _fault_from_split_name(split_name: str) -> int | None:
    match = re.search(r"Faulty_(?:Training|Testing)_(\d{2})", split_name)
    return int(match.group(1)) if match else None


def _tep_browse_path(column: str) -> str:
    key = column.strip().lower()
    if match := re.match(r"xmeas[_\s(]*(\d+)", key):
        return f"Plant/TEP/XMEAS_{int(match.group(1)):02d}"
    if match := re.match(r"xmv[_\s(]*(\d+)", key):
        return f"Plant/TEP/XMV_{int(match.group(1)):02d}"
    return f"Plant/TEP/{column.upper()}"
