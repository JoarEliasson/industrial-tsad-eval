# Data Contracts

## Prepared Format v1

Prepared Format v1 is the normalized dataset boundary. A dataset directory must
contain:

```text
meta/manifest.json
meta/schema.json
meta/splits.json
events/events.jsonl
runs/<run_id>/timeseries.parquet
runs/<run_id>/run_meta.json
```

Timeseries parquet files must include:

- `ts_ns`: `int64` Unix timestamp in nanoseconds.
- one numeric column per tag in `meta/schema.json`.

`schema.json` contains a `tags` list. Each tag should include a stable
`browse_path`; OPC-UA exports can also use `node_id` and `opcua_type`.

`splits.json` defines protocol-specific run groups:

```json
{
  "naive": {
    "train_runs": ["..."],
    "val_runs": ["..."],
    "test_runs": ["..."]
  }
}
```

`events/events.jsonl` stores one event per line with:

- `event_id`
- `run_id`
- `start_ts_ns`
- `end_ts_ns`
- `event_type`
- optional `metadata`

## Score Contract v1

Score artifacts are stored in a score directory. Each run has one parquet file
with:

- `ts_ns`: numeric timestamp in nanoseconds.
- `score`: numeric anomaly score; larger values are more anomalous.

A `manifest.json` can map run IDs to parquet filenames. If no manifest exists,
filenames are decoded by replacing `__` with `/`.

`model_meta.json` records detector configuration and run provenance.

## Dataset Adapter Contract

Dataset adapters are plugins that convert local raw data into Prepared Format
v1. They receive:

- `raw`: directory containing user-provided raw data.
- `prepared`: exact prepared dataset root to write.
- `DatasetAdapterConfig`: `base_epoch_iso`, `default_period_ms`, `strict`, and
  dataset-specific `extra` values.

They return `DatasetAdapterResult` with dataset name, prepared path, run count,
event count, and warnings. The application layer validates the produced dataset
before promotion.
