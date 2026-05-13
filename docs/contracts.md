# Data Contracts

## Prepared Format v1

Prepared Format v1 is the normalized dataset boundary. Writer helpers live in
`src/industrial_tsad_eval/infrastructure/prepared_writer.py:13`; reads go
through `LocalPreparedDatasetRepository`
(`infrastructure/prepared_repository.py:15`); the contract is enforced by
`ValidatePreparedDataset` (`application/validation.py:20`). A dataset directory
must contain:

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

Score artifacts are stored in a score directory and produced by `ScoreRuns`
(`src/industrial_tsad_eval/application/scoring.py:29`) through
`LocalScoreRepository` (`infrastructure/score_repository.py:14`); validation is
in `ValidateScores` (`application/validation.py:68`). Each run has one parquet
file with:

- `ts_ns`: numeric timestamp in nanoseconds.
- `score`: numeric anomaly score; larger values are more anomalous.

A `manifest.json` can map run IDs to parquet filenames. If no manifest exists,
filenames are decoded by replacing `__` with `/`.

`model_meta.json` records detector configuration and run provenance.

## Raw Provenance v1

Raw acquisition writes `raw_provenance.json` in the acquired raw root via
`write_raw_provenance` (`src/industrial_tsad_eval/infrastructure/acquisition.py:117`);
the SHA256 inventory is produced by `file_inventory` (`:92`):

```text
<raw>/<dataset_name>/raw_provenance.json
```

The JSON object includes:

- `contract_version`: `raw-provenance-v1`
- `source_name`
- `dataset_name`
- `method`
- optional `manual_path` and `ref`
- `file_count`
- `files`: relative path, size in bytes, and SHA256 per raw file
- `warnings`

The inventory excludes `raw_provenance.json` itself. Raw provenance validates the
acquisition boundary only; Prepared Format validation happens after dataset
preparation.

## Evidence Bundle v1

Evidence artifacts explain one oracle or operational event. The bundle dataclass
is `EvidenceBundle` (`src/industrial_tsad_eval/domain/evidence.py:44`), written
through `LocalEvidenceRepository` (`infrastructure/evidence_repository.py:13`)
by `GenerateEvidence` (`application/evidence.py:121`) and checked by
`ValidateEvidence` (`application/evidence.py:208`). An evidence root contains:

```text
manifest.json
index.jsonl
bundles/<safe_run_id>/<safe_event_id>/evidence.json
```

Each bundle includes event identity, `event_source`, event bounds, optional
matched GT event id, ranked `top_variables`, ranked `top_time_windows`, score
context, local rankings, and provenance.

Ground-truth tag maps use `gt-tag-map-v1` (`GroundTruthTagMap`,
`domain/evidence.py:156`) built by `BuildGroundTruthTagMap`
(`application/evidence.py:252`) and checked by `ValidateGroundTruthTagMap`
(`application/evidence.py:283`).

## Operator Card v1

Operator-card artifacts are stored in an operator-card root and written by
`LocalOperatorCardRepository`
(`src/industrial_tsad_eval/infrastructure/operator_repository.py:17`); the
Markdown view is rendered by `render_operator_card_markdown` (`:93`):

```text
manifest.json
index.jsonl
retrieval/retrieval_result.json
cards/<safe_event_id>/operator_card.json
cards/<safe_event_id>/operator_card.md
```

`operator_card.json` uses `operator-card-v1` (`OperatorCard` dataclass,
`src/industrial_tsad_eval/domain/operator.py:100`) and includes event identity,
`status`, situation summary, evidence highlights, checks, recommended actions,
escalation criteria, citations, diagnostics, and provenance. `answered` cards
must carry citations. `abstained` cards must carry an abstention reason —
enforced by `ValidateOperatorCards` (`application/operator.py:253`).

## assistant replay Replay Artifacts v1

assistant replay suites preserve the assistant-evaluation contract used for
thesis-style reproduction. Suite manifests use `ReplaySuiteManifest`
(`src/industrial_tsad_eval/domain/assistant_replay.py:177`); per-run metrics use
`AssistantRunMetrics` (`:194`); the suite-level summary uses
`AssistantReplayAggregateMetrics` (`:222`). Writers are at
`infrastructure/assistant_replay_repository.py:19`/`:50`/`:82`:

```text
<assistant_out>/
  cases/<case_id>/case.json
  cases/index.jsonl
  suites/suite_manifest.json
  runs/<case_id>/
    retrieval_result.json
    provider_request.json
    provider_response.json
    planner_output.json
    referee_output.json
    run_log.json
    rendered_response.md
  assistant_summary.json
  assistant_summary.csv
```

`case.json` stores event identity, query text, expected retrieval event ids, and
minimum supported-claim expectations. `assistant_summary.json` stores
thesis-compatible claim/citation proxy metrics.

## Dataset Adapter Contract

Dataset adapters are plugins that convert local raw data into Prepared Format
v1. The port is `DatasetAdapterPlugin`
(`src/industrial_tsad_eval/ports/dataset_adapters.py:11`). They receive:

- `raw`: directory containing user-provided raw data.
- `prepared`: exact prepared dataset root to write.
- `DatasetAdapterConfig` (`domain/datasets.py:10`): `base_epoch_iso`,
  `default_period_ms`, `strict`, and dataset-specific `extra` values.

They return `DatasetAdapterResult` (`domain/datasets.py:20`) with dataset name,
prepared path, run count, event count, and warnings. The application layer
(`PrepareDataset`, `application/preparation.py:15`) validates the produced
dataset before promotion.

## Dataset Source Contract

Dataset sources are plugins that materialize raw files into a local cache. The
port is `DatasetSourcePlugin`
(`src/industrial_tsad_eval/ports/dataset_sources.py:11`). A source provides:

- stable `name`, such as `swat`
- raw `dataset_name`, such as `SWaT`
- `supported_methods()`
- `describe()`
- `acquire(target, config)`

The application layer owns staging, overwrite policy, provenance writing, and
promotion.
