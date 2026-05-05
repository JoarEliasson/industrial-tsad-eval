"""Synthetic Prepared Format fixture generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from industrial_tsad_eval.infrastructure.json_utils import write_json, write_jsonl

FEATURES = [
    "Plant/AreaA/Tank101/LIT101",
    "Plant/AreaA/Tank101/FIT101",
    "Plant/AreaA/Tank101/MV101",
    "Plant/AreaA/Pump101/Speed",
]


def make_opcua_fixture(out: str | Path) -> Path:
    """Generate a tiny OPC-UA-like Prepared Format v1 dataset."""
    dataset_root = Path(out) / "OPCUA_SYNTH"
    _ensure_prepared_dirs(dataset_root)

    base_ns = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000_000_000)
    period_ns = 1_000_000_000
    runs = {
        "opcua/train/normal_001": _make_run(base_ns, period_ns, 0, anomaly=False),
        "opcua/val/normal_001": _make_run(base_ns, period_ns, 300, anomaly=False),
        "opcua/test/fault_001": _make_run(base_ns, period_ns, 600, anomaly=True),
    }

    for run_id, frame in runs.items():
        run_dir = dataset_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(run_dir / "timeseries.parquet", index=False)
        write_json(
            run_dir / "run_meta.json",
            {
                "run_id": run_id,
                "period_ns": period_ns,
                "start_ts_ns": int(frame["ts_ns"].iloc[0]),
                "end_ts_ns": int(frame["ts_ns"].iloc[-1] + period_ns),
                "rows": int(len(frame)),
            },
        )

    event_start_index = 120
    event_end_index = 155
    test_frame = runs["opcua/test/fault_001"]
    event = {
        "event_id": "opcua_fault_001",
        "run_id": "opcua/test/fault_001",
        "start_ts_ns": int(test_frame["ts_ns"].iloc[event_start_index]),
        "end_ts_ns": int(test_frame["ts_ns"].iloc[event_end_index] + period_ns),
        "event_type": "fault",
        "metadata": {
            "description": "Synthetic level-control disturbance",
            "affected_tags": FEATURES[:3],
            "end_is_exclusive": True,
        },
    }

    write_jsonl(dataset_root / "events" / "events.jsonl", [event])
    write_json(dataset_root / "meta" / "schema.json", _schema_payload())
    write_json(
        dataset_root / "meta" / "splits.json",
        {
            "naive": {
                "train_runs": ["opcua/train/normal_001"],
                "val_runs": ["opcua/val/normal_001"],
                "test_runs": ["opcua/test/fault_001"],
            },
            "all_in_one": {
                "train_runs": ["opcua/train/normal_001"],
                "val_runs": ["opcua/val/normal_001"],
                "test_runs": ["opcua/test/fault_001"],
            },
        },
    )
    write_json(
        dataset_root / "meta" / "manifest.json",
        {
            "dataset": "OPCUA_SYNTH",
            "prepared_format": "Prepared Format v1",
            "format_version": "tagtimeseries-v1",
            "timebase": {"column": "ts_ns", "unit": "ns", "timezone": "UTC"},
            "runs": {"count": len(runs), "run_ids": sorted(runs)},
            "source_notes": "Generated synthetic OPC-UA-like process telemetry.",
        },
    )
    write_json(
        dataset_root / "meta" / "provenance.json",
        {"generator": "industrial_tsad_eval.infrastructure.examples.make_opcua_fixture"},
    )
    return dataset_root


def _ensure_prepared_dirs(root: Path) -> None:
    for relative in ("meta", "runs", "events", "exports"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _make_run(base_ns: int, period_ns: int, offset: int, anomaly: bool) -> pd.DataFrame:
    sample_count = 220
    index = np.arange(sample_count, dtype=np.float64)
    rng = np.random.default_rng(2026 + offset)
    level = 50.0 + 2.0 * np.sin(index / 18.0) + rng.normal(0.0, 0.05, sample_count)
    flow = 12.0 + 0.4 * np.cos(index / 15.0) + rng.normal(0.0, 0.03, sample_count)
    valve = 45.0 + 1.5 * np.sin(index / 24.0)
    speed = 1750.0 + 8.0 * np.cos(index / 31.0)

    if anomaly:
        fault_slice = slice(120, 156)
        level[fault_slice] += np.linspace(3.0, 8.0, 36)
        flow[fault_slice] -= 1.8
        valve[fault_slice] += 9.0

    ts_ns = base_ns + (offset + index.astype(np.int64)) * period_ns
    return pd.DataFrame(
        {
            "ts_ns": ts_ns.astype(np.int64),
            FEATURES[0]: level.astype(np.float32),
            FEATURES[1]: flow.astype(np.float32),
            FEATURES[2]: valve.astype(np.float32),
            FEATURES[3]: speed.astype(np.float32),
        }
    )


def _schema_payload() -> dict[str, Any]:
    tags = [
        {
            "browse_path": feature,
            "node_id": f"ns=2;s={feature}",
            "opcua_type": "Float",
            "dtype": "float32",
            "kind": "sensor" if "MV" not in feature else "actuator",
        }
        for feature in FEATURES
    ]
    return {
        "format_version": "tagtimeseries-v1",
        "time": {"column": "ts_ns", "unit": "ns", "timezone": "UTC"},
        "tags": tags,
    }
