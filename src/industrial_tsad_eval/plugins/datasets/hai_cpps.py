"""HAI-CPPS dataset adapter plugin."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.infrastructure.data_utils import (
    coerce_numeric_frame,
    downcast_numeric,
    parse_iso_to_ns,
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


class HAICPPSDatasetAdapterPlugin:
    """Prepare local HAI-CPPS scenario directories into Prepared Format v1."""

    @property
    def name(self) -> str:
        """Return the adapter registry name."""
        return "hai-cpps"

    @property
    def dataset_name(self) -> str:
        """Return the prepared dataset directory name."""
        return "HAI-CPPS"

    def describe_expected_raw_layout(self) -> str:
        """Describe accepted local raw HAI-CPPS layouts."""
        return (
            "Provide scenario directories. Each run directory should contain one or more "
            "CSV files for the selected mode, such as continuous, and may include "
            "sim_setup.json with period metadata. Directory or file names containing "
            "anom/anomalous are treated as anomalous runs."
        )

    def prepare(
        self,
        *,
        raw: Path,
        prepared: Path,
        config: DatasetAdapterConfig,
    ) -> DatasetAdapterResult:
        """Write HAI-CPPS Prepared Format v1 artifacts."""
        ensure_prepared_dirs(prepared)
        mode = str(config.extra.get("mode", "continuous")).lower()
        include_states = bool(config.extra.get("include_states", False))
        injection_step = int(config.extra.get("injection_step", 2500))
        base_ns = parse_iso_to_ns(config.base_epoch_iso)
        default_period_ns = int(config.default_period_ms) * 1_000_000

        run_dirs = sorted(path for path in raw.iterdir() if path.is_dir())
        if not run_dirs:
            raise FileNotFoundError(f"No HAI-CPPS run directories found under {raw}.")

        tags: dict[str, dict[str, Any]] = {}
        events: list[dict[str, Any]] = []
        run_ids: list[str] = []
        normal_runs: list[str] = []
        anomalous_runs: list[str] = []

        for run_dir in run_dirs:
            run_id = re.sub(r"[^A-Za-z0-9._-]+", "_", run_dir.name)
            run_ids.append(run_id)
            sim_setup = _read_sim_setup(run_dir / "sim_setup.json")
            period_ns = _period_ns_from_sim_setup(sim_setup) or default_period_ns
            csv_path = _select_mode_csv(run_dir, mode=mode, include_states=include_states)
            if csv_path is None:
                raise FileNotFoundError(
                    f"{run_dir}: no CSV found for mode={mode} include_states={include_states}."
                )

            raw_frame = pd.read_csv(csv_path, low_memory=False)
            time_col = _time_column(raw_frame)
            if time_col is None:
                time_source = "inferred_steps"
                ts_ns = base_ns + np.arange(len(raw_frame), dtype=np.int64) * period_ns
                feature_frame = raw_frame
            else:
                time_source = _time_source(raw_frame[time_col])
                ts_ns = to_unix_ns(
                    raw_frame[time_col],
                    assume_timestep_period_ns=period_ns,
                    base_ts_ns=base_ns,
                )
                feature_frame = raw_frame.drop(columns=[time_col])

            feature_frame, enum_maps = coerce_numeric_frame(feature_frame)
            rename_map = {
                str(column): _browse_path_from_channel(str(column))
                for column in feature_frame.columns
            }
            feature_frame = feature_frame.rename(columns=rename_map)
            out_frame = pd.concat([pd.Series(ts_ns, name="ts_ns"), feature_frame], axis=1)
            out_frame["ts_ns"] = out_frame["ts_ns"].astype("int64")
            out_frame = downcast_numeric(out_frame)

            is_anomalous = bool(
                re.search(r"(?i)anom|anomal", run_dir.name)
                or re.search(r"(?i)anom|anomal", csv_path.name)
            )
            parsed = _parse_anomalous_dir_name(run_dir.name)
            if is_anomalous:
                anomalous_runs.append(run_id)
                events.append(
                    _cpps_event(
                        run_id=run_id,
                        frame=out_frame,
                        period_ns=period_ns,
                        injection_step=injection_step,
                        sim_setup=sim_setup,
                        parsed=parsed,
                        include_states=include_states,
                        mode=mode,
                        raw_dir_name=run_dir.name,
                    )
                )
            else:
                normal_runs.append(run_id)

            for original, browse_path in rename_map.items():
                if browse_path not in out_frame.columns:
                    continue
                tags[browse_path] = tag_payload(
                    browse_path=browse_path,
                    dtype=str(out_frame[browse_path].dtype),
                    kind=_infer_kind_cpps(browse_path),
                    group=_infer_group_cpps(browse_path),
                    enum_map=enum_maps.get(original),
                )

            run_meta: dict[str, Any] = {
                "run_id": run_id,
                "source_dir": str(run_dir),
                "source_file": str(csv_path),
                "raw_dir_name": run_dir.name,
                "mode": mode,
                "include_states": include_states,
                "rows": int(out_frame.shape[0]),
                "columns": int(out_frame.shape[1]),
                "period_ns": period_ns,
                "base_epoch_iso": config.base_epoch_iso,
                "sim_setup": sim_setup,
                "is_anomalous": is_anomalous,
                "time_source": time_source,
                "base_ts_ns": int(out_frame["ts_ns"].iloc[0]) if len(out_frame) else base_ns,
                "scenario_id": parsed["scenario_id"] if is_anomalous else run_id,
            }
            if is_anomalous:
                run_meta.update(
                    {
                        "anomaly_type": "injection",
                        "target_module": parsed["target_module"],
                        "target_component": parsed["target_component"],
                    }
                )
            write_run(prepared, run_id, out_frame, run_meta)

        splits = _cpps_splits(normal_runs, anomalous_runs)
        write_schema(prepared, list(tags.values()))
        write_events(prepared, events)
        write_splits(prepared, splits)
        write_provenance(
            prepared,
            {
                "dataset": "HAI-CPPS",
                "docs": "https://j-ehrhardt.github.io/hai-cpps-benchmark/",
                "notes": "Prepared from user-supplied local HAI-CPPS scenario directories.",
            },
        )
        write_manifest(
            root=prepared,
            dataset_name=self.dataset_name,
            source_notes="HAI-CPPS scenario directories normalized to tag time-series.",
            timebase={
                "column": "ts_ns",
                "unit": "ns",
                "timezone": "UTC",
                "origin": "unix_epoch",
                "synthetic": True,
                "base_epoch_iso": config.base_epoch_iso,
            },
            run_ids=run_ids,
            extra={
                "mode": mode,
                "include_states": include_states,
                "default_period_ns": default_period_ns,
                "injection_step_default": injection_step,
            },
        )
        return DatasetAdapterResult(
            dataset_name=self.dataset_name,
            prepared_path=str(prepared),
            run_count=len(run_ids),
            event_count=len(events),
            warnings=[],
        )


def _read_sim_setup(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _time_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("time", "t", "timestamp", "step", "timestep"):
        if candidate in frame.columns:
            return str(candidate)
    return None


def _time_source(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        max_value = float(series.max()) if len(series) else 0.0
        return "timestep" if max_value < 1e8 else "epoch_numeric"
    return "datetime"


def _period_ns_from_sim_setup(sim_setup: dict[str, Any]) -> int | None:
    for key in ("period_ns", "dt_ns"):
        if isinstance(sim_setup.get(key), int | float):
            return int(sim_setup[key])
    for key in ("period_ms", "dt_ms", "sample_time_ms"):
        if isinstance(sim_setup.get(key), int | float):
            return int(float(sim_setup[key]) * 1_000_000)
    for key in ("dt", "sample_time_s", "sample_time_sec"):
        if isinstance(sim_setup.get(key), int | float):
            return int(float(sim_setup[key]) * 1_000_000_000)
    return None


def _select_mode_csv(run_dir: Path, *, mode: str, include_states: bool) -> Path | None:
    files = sorted(path for path in run_dir.iterdir() if path.is_file() and path.suffix == ".csv")
    if not files:
        return None

    def score(path: Path) -> int:
        name = path.name.lower()
        value = 0
        if mode in name:
            value += 10
        if include_states and "_s.csv" in name:
            value += 6
        if not include_states and "_s.csv" in name:
            value -= 5
        if "anom" in name or "anomal" in name:
            value += 2
        if "normal" in name:
            value += 1
        if name.endswith("_ds.csv"):
            value += 1
        return value

    return sorted(files, key=lambda path: (-score(path), path.name))[0]


def _cpps_event(
    *,
    run_id: str,
    frame: pd.DataFrame,
    period_ns: int,
    injection_step: int,
    sim_setup: dict[str, Any],
    parsed: dict[str, Any],
    include_states: bool,
    mode: str,
    raw_dir_name: str,
) -> dict[str, Any]:
    clamped = injection_step >= len(frame)
    effective_step = max(0, len(frame) - 1) if clamped else injection_step
    start_ts = int(frame["ts_ns"].iloc[effective_step])
    end_ts = int(frame["ts_ns"].iloc[-1] + period_ns) if len(frame) > 1 else start_ts
    has_reset = bool(sim_setup.get("has_reset", sim_setup.get("reset", False)))
    return {
        "event_id": f"{run_id}_fault_0000",
        "run_id": run_id,
        "start_ts_ns": start_ts,
        "end_ts_ns": end_ts,
        "event_type": "fault",
        "metadata": {
            "injection_step": injection_step,
            "onset_step": effective_step,
            "injection_step_clamped": clamped,
            "effective_injection_step": effective_step,
            "period_ns": period_ns,
            "sim_setup": sim_setup,
            "end_is_exclusive": True,
            "scenario_id": parsed["scenario_id"],
            "anomaly_type": "injection",
            "target_module": parsed["target_module"],
            "target_component": parsed["target_component"],
            "has_reset": has_reset,
            "states_included": include_states,
            "mode": mode,
            "raw_dir_name": raw_dir_name,
            "metadata_source": {
                "scenario_id": "parsed from directory name prefix",
                "anomaly_type": "dataset documentation",
                "target_module": "parsed from directory name"
                if parsed["parse_confident"]
                else "not determined",
                "target_component": "parsed from directory name"
                if parsed["target_component"]
                else "not determined",
                "has_reset": "sim_setup.json has_reset/reset key",
                "mode": "adapter config extra.mode",
                "raw_dir_name": "raw directory name",
            },
        },
    }


def _parse_anomalous_dir_name(name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "scenario_id": name,
        "target_module": None,
        "target_component": None,
        "parse_confident": False,
        "raw_dir_name": name,
    }
    parts = re.split(r"_anom(?:al(?:ous|y)?)?_?", name, maxsplit=1)
    if len(parts) < 2:
        return result
    prefix_parts = parts[0].split("_")
    if prefix_parts:
        result["scenario_id"] = prefix_parts[0]
    if len(prefix_parts) >= 2:
        component = prefix_parts[1]
        result["target_component"] = component
        match = re.match(r"^([A-Za-z]+)", component)
        if match:
            result["target_module"] = match.group(1)
            result["parse_confident"] = True
    return result


def _cpps_splits(normal_runs: list[str], anomalous_runs: list[str]) -> dict[str, Any]:
    normal_sorted = sorted(normal_runs)
    if len(normal_sorted) >= 2:
        train_runs = normal_sorted[:-1]
        val_runs = [normal_sorted[-1]]
    else:
        train_runs = normal_sorted
        val_runs = []
    test_runs = sorted(anomalous_runs)
    return {
        "naive": {"train_runs": train_runs, "val_runs": val_runs, "test_runs": test_runs},
        "all_in_one": {"train_runs": normal_sorted, "val_runs": val_runs, "test_runs": test_runs},
        "zero_shot": {
            "train_runs": train_runs,
            "val_runs": val_runs,
            "test_runs": test_runs,
            "notes": "Best-effort split inferred from anomalous scenario names.",
        },
    }


def _browse_path_from_channel(channel: str) -> str:
    parts = channel.split("_")
    if len(parts) >= 3:
        return f"Plant/CPPS/{parts[0]}/{parts[1]}/{'_'.join(parts[2:])}"
    if len(parts) == 2:
        return f"Plant/CPPS/{parts[0]}/{parts[1]}"
    return f"Plant/CPPS/{channel}"


def _infer_group_cpps(browse_path: str) -> str:
    parts = browse_path.split("/")
    return parts[2] if len(parts) >= 3 else ""


def _infer_kind_cpps(browse_path: str) -> str:
    last = browse_path.split("/")[-1].lower()
    if "state" in last or "mode" in last or last.endswith("_s"):
        return "state"
    if "valve" in last or "pump" in last or last.startswith("mv") or last.startswith("p"):
        return "actuator"
    return "sensor"
