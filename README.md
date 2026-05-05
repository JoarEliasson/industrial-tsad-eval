# Industrial TSAD Eval

Industrial TSAD Eval is a small, product-oriented toolkit for evaluating
industrial time-series anomaly detection systems. It standardizes datasets into
a stable Prepared Format, writes detector outputs through a Score Contract, and
evaluates event-level detection quality with reproducible artifact outputs.

The project is intentionally structured around ports and plugins. The default
distribution ships dataset adapter plugins, a compact ForecastRidge detector
plugin, and a generated OPC-UA-like fixture so the full pipeline can run without
vendored industrial data.

## Quickstart

```powershell
python -m pip install -e ".[dev]"

itse examples make-opcua-fixture --out examples/generated
itse prepared validate --prepared examples/generated/OPCUA_SYNTH
itse score run --prepared examples/generated/OPCUA_SYNTH --detector forecast-ridge --out out/scores
itse scores validate --prepared examples/generated/OPCUA_SYNTH --scores out/scores
itse eval run --prepared examples/generated/OPCUA_SYNTH --scores out/scores --out out/eval
```

Dataset adapters are available for local raw TEP, SWaT, HAI, and HAI-CPPS data:

```powershell
python -m pip install -e ".[dev,datasets]"

itse prepared adapters
itse prepared describe --dataset swat
itse prepared prepare --dataset swat --raw data/raw/SWaT --out prepared --extra-json "{\"remove_startup\": false}"
itse prepared validate --prepared prepared/SWaT
```

The final command writes:

- `metrics.json`
- `event_matches.json`
- `threshold.json`

## Architecture

The package uses a hexagonal structure:

- `domain`: contracts, events, policies, validation reports, metric functions.
- `ports`: dataset adapter, detector, repository, and artifact writer interfaces.
- `application`: use cases such as prepare, validate, score, and evaluate.
- `infrastructure`: local parquet/json repositories, prepared writers, and fixture generation.
- `plugins`: dataset adapter and detector implementations plus registry wiring.
- `interfaces/cli`: Typer/Rich command-line interface.

Core code does not import CLI/UI libraries. Workflows are exposed as application
services, while the CLI only parses arguments, calls a service, and renders
results.

## Plugin Model

Detector plugins implement a small factory interface:

- stable `name`
- `create(config)`
- returned detector with `train`, `score_run`, and `metadata`

The first plugin is `forecast-ridge`. Additional detectors can be registered
without changing application or CLI workflows.

Dataset adapter plugins implement:

- stable `name`
- `dataset_name` for the prepared output directory
- `describe_expected_raw_layout()`
- `prepare(raw, prepared, config)`

The built-in adapters are `tep`, `swat`, `hai`, and `hai-cpps`. Users provide
local raw data paths; dataset acquisition and credentials are intentionally out
of scope for this package.

## Data Contracts

Prepared Format v1 expects:

```text
<dataset>/
  meta/manifest.json
  meta/schema.json
  meta/splits.json
  events/events.jsonl
  runs/<run_id>/timeseries.parquet
  runs/<run_id>/run_meta.json
```

Score Contract v1 expects one parquet per run with at least:

- `ts_ns`: numeric timestamp in nanoseconds
- `score`: numeric anomaly score where higher means more anomalous

See [docs/contracts.md](docs/contracts.md) for details.

## Development

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```
