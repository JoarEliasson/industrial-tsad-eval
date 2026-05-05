"""Prepared Format v1 write helpers used by dataset adapters."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from industrial_tsad_eval.infrastructure.json_utils import write_json, write_jsonl


def ensure_prepared_dirs(root: Path) -> None:
    """Create the standard Prepared Format v1 directory tree."""
    for relative in ("meta", "runs", "events", "exports"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def stable_node_id(browse_path: str) -> str:
    """Return a deterministic OPC-UA-like NodeId for a browse path."""
    return f"ns=2;s={browse_path}"


def opcua_type_from_dtype(dtype: str) -> str:
    """Map pandas/numpy dtypes to common OPC-UA scalar names."""
    value = dtype.lower()
    if value in {"bool", "boolean", "uint8"}:
        return "Boolean"
    if value == "int16":
        return "Int16"
    if value == "int32":
        return "Int32"
    if value == "int64":
        return "Int64"
    if value == "float32":
        return "Float"
    return "Double"


def tag_payload(
    *,
    browse_path: str,
    dtype: str,
    kind: str,
    group: str = "",
    enum_map: dict[str, int] | None = None,
    description: str = "",
    unit: str = "",
) -> dict[str, Any]:
    """Build a schema tag payload."""
    payload: dict[str, Any] = {
        "browse_path": browse_path,
        "node_id": stable_node_id(browse_path),
        "opcua_type": opcua_type_from_dtype(dtype),
        "dtype": dtype,
        "kind": kind,
        "group": group,
        "description": description,
        "unit": unit,
    }
    if enum_map is not None:
        payload["enum_map"] = enum_map
    return payload


def write_schema(root: Path, tags: list[dict[str, Any]]) -> None:
    """Write `meta/schema.json` sorted by browse path."""
    write_json(
        root / "meta" / "schema.json",
        {
            "format_version": "tagtimeseries-v1",
            "time": {"column": "ts_ns", "unit": "ns", "timezone": "UTC"},
            "tags": sorted(tags, key=lambda tag: str(tag["browse_path"])),
        },
    )


def write_manifest(
    *,
    root: Path,
    dataset_name: str,
    source_notes: str,
    timebase: dict[str, Any],
    run_ids: list[str],
    extra: dict[str, Any] | None = None,
) -> None:
    """Write `meta/manifest.json`."""
    manifest: dict[str, Any] = {
        "dataset": dataset_name,
        "prepared_format": "Prepared Format v1",
        "format_version": "tagtimeseries-v1",
        "timebase": timebase,
        "runs": {"count": len(run_ids), "run_ids": sorted(run_ids)},
        "source_notes": source_notes,
    }
    if extra:
        manifest.update(extra)
    write_json(root / "meta" / "manifest.json", manifest)


def write_splits(root: Path, splits: dict[str, Any]) -> None:
    """Write `meta/splits.json`."""
    write_json(root / "meta" / "splits.json", splits)


def write_provenance(root: Path, provenance: dict[str, Any]) -> None:
    """Write `meta/provenance.json`."""
    write_json(root / "meta" / "provenance.json", provenance)


def write_events(root: Path, events: list[dict[str, Any]]) -> None:
    """Write `events/events.jsonl`."""
    write_jsonl(root / "events" / "events.jsonl", events)


def write_run(root: Path, run_id: str, frame: pd.DataFrame, run_meta: dict[str, Any]) -> None:
    """Write one run parquet and metadata file."""
    run_dir = root / "runs" / _safe_run_path(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(run_dir / "timeseries.parquet", index=False)
    write_json(run_dir / "run_meta.json", run_meta)


def _safe_run_path(run_id: str) -> Path:
    pure = PurePosixPath(run_id)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"Invalid run_id path: {run_id!r}")
    return Path(*pure.parts)
