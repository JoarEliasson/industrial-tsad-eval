"""Shared timestamp, label, and numeric coercion helpers for adapters."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_iso_to_ns(iso: str) -> int:
    """Parse an ISO-8601 timestamp to Unix nanoseconds."""
    value = iso.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def to_unix_ns(
    values: pd.Series | pd.Index | Sequence[Any],
    *,
    assume_timestep_period_ns: int | None = None,
    base_ts_ns: int | None = None,
) -> np.ndarray:
    """Convert timestamp-like values to int64 Unix nanoseconds."""
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_numeric_dtype(series):
        numeric = series.to_numpy(dtype=np.float64)
        max_value = float(np.nanmax(numeric)) if len(numeric) else 0.0
        if max_value < 1e8 and assume_timestep_period_ns is not None and base_ts_ns is not None:
            return base_ts_ns + numeric.astype(np.int64) * assume_timestep_period_ns
        if max_value > 1e17:
            return numeric.astype(np.int64)
        if max_value > 1e14:
            return (numeric * 1_000).astype(np.int64)
        if max_value > 1e11:
            return (numeric * 1_000_000).astype(np.int64)
        return (numeric * 1_000_000_000).astype(np.int64)

    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Failed to parse timestamp values to UTC datetimes.")
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return ((parsed - epoch) // pd.Timedelta("1ns")).to_numpy(dtype=np.int64)


def synthesize_timestamps(row_count: int, base_epoch_iso: str, period_ms: int) -> np.ndarray:
    """Generate monotonically increasing Unix nanosecond timestamps."""
    base_ns = parse_iso_to_ns(base_epoch_iso)
    period_ns = int(period_ms) * 1_000_000
    return base_ns + np.arange(row_count, dtype=np.int64) * period_ns


def infer_period_ns(ts_ns: np.ndarray) -> int | None:
    """Infer a nominal sample period from timestamp differences."""
    if len(ts_ns) < 3:
        return None
    diffs = np.diff(ts_ns.astype(np.int64))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return None
    return int(np.median(positive))


def segments_from_binary(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive contiguous one-valued label segments."""
    binary = np.asarray(labels).astype(np.int64)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(binary):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            segments.append((start, index - 1))
            start = None
    if start is not None:
        segments.append((start, len(binary) - 1))
    return segments


def binary_labels_from_series(series: pd.Series) -> np.ndarray:
    """Normalize common industrial attack-label encodings to 0/1 integers."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).to_numpy(dtype=np.float64) > 0).astype(np.int64)
    values = series.astype(str).str.strip().str.lower()
    positive = {
        "1",
        "abnormal",
        "anomaly",
        "attack",
        "attacked",
        "fault",
        "true",
        "yes",
    }
    return values.isin(positive).to_numpy(dtype=np.int64)


def coerce_numeric_series(series: pd.Series) -> tuple[pd.Series, dict[str, int] | None]:
    """Coerce a series to numeric data, encoding categoricals when needed."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype("uint8"), None
    if pd.api.types.is_numeric_dtype(series):
        return series, None

    values = series.astype(str).str.strip()
    lowered = values.str.lower()
    boolean_values = {"true", "false", "0", "1", "yes", "no", "on", "off"}
    if lowered.isin(boolean_values).all():
        mapping = {
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0,
            "yes": 1,
            "no": 0,
            "on": 1,
            "off": 0,
        }
        return lowered.map(mapping).astype("uint8"), None

    coerced = pd.to_numeric(series, errors="coerce")
    if not coerced.isna().all() and not coerced.isna().any():
        return coerced, None

    codes, uniques = pd.factorize(values, sort=True)
    enum_map = {str(value): int(index) for index, value in enumerate(uniques)}
    return pd.Series(codes, index=series.index, dtype="int32"), enum_map


def coerce_numeric_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Coerce all columns in a dataframe to numeric data."""
    output = frame.copy()
    enum_maps: dict[str, dict[str, int]] = {}
    for column in list(output.columns):
        series, enum_map = coerce_numeric_series(output[column])
        output[column] = series
        if enum_map is not None:
            enum_maps[str(column)] = enum_map
    return output, enum_maps


def downcast_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric non-time columns for compact parquet output."""
    output = frame.copy()
    for column in output.columns:
        if str(column) == "ts_ns":
            output[column] = output[column].astype("int64")
            continue
        series = output[column]
        if pd.api.types.is_float_dtype(series):
            output[column] = pd.to_numeric(series, downcast="float").astype("float32")
        elif pd.api.types.is_integer_dtype(series):
            output[column] = pd.to_numeric(series, downcast="integer")
            if str(output[column].dtype) == "int64":
                minimum = int(output[column].min())
                maximum = int(output[column].max())
                if -2_147_483_648 <= minimum <= maximum <= 2_147_483_647:
                    output[column] = output[column].astype("int32")
        elif pd.api.types.is_bool_dtype(series):
            output[column] = series.astype("uint8")
    return output


def read_table(path: Path) -> pd.DataFrame:
    """Read a local CSV, parquet, or Excel table."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")
