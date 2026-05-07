# Industrial TSAD Eval

Industrial TSAD Eval is a research toolkit for evaluating
industrial time-series anomaly detection systems. It standardizes datasets into
a stable Prepared Format, writes detector outputs through a Score Contract, and
evaluates event-level detection quality with reproducible artifact outputs.

The project is intentionally structured around ports and plugins. The default
distribution ships dataset adapter plugins, a compact ForecastRidge detector,
optional torch-backed detector plugins, and a generated OPC-UA-like fixture so
the full pipeline can run without vendored industrial data.

## Quickstart

```powershell
python -m pip install -e ".[dev]"

itse examples make-opcua-fixture --out examples/generated
itse prepared validate --prepared examples/generated/OPCUA_SYNTH
itse score run --prepared examples/generated/OPCUA_SYNTH --detector forecast-ridge --out out/scores
itse scores validate --prepared examples/generated/OPCUA_SYNTH --scores out/scores
itse eval run --prepared examples/generated/OPCUA_SYNTH --scores out/scores --out out/eval
```

Torch-backed detectors are optional:

```powershell
python -m pip install -e ".[torch]"

itse score detectors
itse score run --prepared examples/generated/OPCUA_SYNTH --detector forecast-lstm --out out/lstm-scores --parameters-json "{\"window\": 16, \"train_stride\": 8, \"score_stride\": 8, \"epochs\": 1, \"device\": \"cpu\"}"
```

Use the PyTorch selector wheel instructions for CUDA or XPU installations when
the default `torch` wheel is not the desired runtime.

Dataset adapters are available for local raw TEP, SWaT, HAI, and HAI-CPPS data:

```powershell
python -m pip install -e ".[dev,datasets]"

itse prepared adapters
itse prepared describe --dataset swat
itse prepared prepare --dataset swat --raw data/raw/SWaT --out prepared --extra-json "{\"remove_startup\": false}"
itse prepared validate --prepared prepared/SWaT
```

Raw acquisition helpers keep manual imports and optional downloads separate from
preparation:

```powershell
python -m pip install -e ".[dev,datasets,acquisition]"

itse data sources
itse data describe --source swat
itse data acquire --source swat --method manual --manual data/downloads/SWaT --out data/raw
itse data validate --source swat --raw data/raw/SWaT
itse prepared prepare --dataset swat --raw data/raw/SWaT --out prepared
```

The setup flow can be rehearsed without gated datasets:

```powershell
itse examples make-thesis-raw-fixtures --out out/setup-fixtures/raw
itse data acquire --source swat --method manual --manual out/setup-fixtures/raw/swat --out out/setup-fixtures/raw-cache
itse prepared prepare --dataset swat --raw out/setup-fixtures/raw-cache/SWaT --out out/setup-fixtures/prepared
```

Repeatable benchmark runs use TOML configs and existing Prepared Format
directories:

```powershell
itse bench init-config --out benchmarks/opcua.toml
itse bench plan --config benchmarks/opcua.toml
itse bench run --config benchmarks/opcua.toml --out out/benchmarks
itse bench summarize --run out/benchmarks/opcua-smoke-<timestamp>
```

System diagnostics and profiling make runs reproducible:

```powershell
itse system gpu-check --device auto --json
itse system preflight --prepared examples/generated/OPCUA_SYNTH --detector forecast-ridge --out out/preflight --strict
itse profile run --prepared examples/generated/OPCUA_SYNTH --detector forecast-ridge --out out/profiles --profile-id smoke
```

Evidence and XAI evaluation add explanation-quality artifacts:

```powershell
itse evidence generate --prepared examples/generated/OPCUA_SYNTH --scores out/scores --out out/evidence
itse evidence validate --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence
itse xai gt-map build --prepared examples/generated/OPCUA_SYNTH --out out/gt_map.json
itse xai eval --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence --gt-map out/gt_map.json --out out/xai --ks 1,3,5
```

Deterministic operator cards turn evidence into cited operator-facing summaries:

```powershell
itse operator retrieve --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence --query "what should the operator check"
itse operator card generate --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence --out out/operator-cards
itse operator card validate --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence --cards out/operator-cards
```

Thesis-style reproducibility keeps benchmark, evidence, XAI, profiling, and
assistant replay experiments behind clean application services:

```powershell
itse reproduce init-config --out config/thesis_smoke.toml --profile thesis-smoke
itse reproduce plan --config config/thesis_smoke.toml
itse reproduce run --config config/thesis_smoke.toml --out out/reproduction --run-id smoke
itse reproduce status --run out/reproduction/smoke

itse assistant providers
itse assistant preflight --config config/thesis_smoke.toml
```

For full assistant replay thesis runs, `llama.cpp` is the recommended local reproducibility
backend through its OpenAI-compatible server. Cloud providers are configured via
environment-variable names only; secrets are never written to config files.

Run a clean-repo audit before sharing or reviewing the architecture:

```powershell
itse audit run --out out/audit
```

## Architecture

The package uses a hexagonal structure:

- `domain`: contracts, events, policies, validation reports, metric functions.
- `ports`: dataset adapter, detector, LLM provider, repository, and artifact writer interfaces.
- `application`: use cases such as prepare, validate, score, evaluate, and benchmark.
- `infrastructure`: local parquet/json repositories, prepared writers, and fixtures.
- `plugins`: dataset source, dataset adapter, and detector implementations.
- `interfaces/cli`: Typer/Rich command-line interface.

Core code does not import CLI/UI libraries. Workflows are exposed as application
services, while the CLI only parses arguments, calls a service, and renders
results.

Torch imports are isolated to optional torch plugin modules. The default
registry can list torch-backed detector plugins even when torch is not installed;
training one of those detectors raises a clear optional-dependency error.

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

Raw acquisition writes `raw_provenance.json` with `raw-provenance-v1`, source,
method, warnings, and a SHA256 inventory of imported files.

Operator cards use `operator-card-v1` JSON plus Markdown views. They are
deterministic and cite Evidence Bundle v1 or local Markdown playbook chunks.
assistant replay suites use provider-backed assistant runs and deterministic referee
checks to produce thesis-compatible claim/citation metrics.

Benchmark runs create:

```text
<out>/<run_id>/
  config/benchmark.toml
  resolved_config.json
  run_manifest.json
  summary.json
  summary.csv
  experiments/<experiment_id>/
```

See [docs/contracts.md](docs/contracts.md), [docs/acquisition.md](docs/acquisition.md),
[docs/plugins.md](docs/plugins.md), [docs/benchmarks.md](docs/benchmarks.md),
[docs/system.md](docs/system.md), [docs/profiling.md](docs/profiling.md),
[docs/evidence.md](docs/evidence.md), [docs/xai.md](docs/xai.md), and
[docs/operator.md](docs/operator.md), [docs/providers.md](docs/providers.md),
[docs/assistant_replay.md](docs/assistant_replay.md),
[docs/reproduction.md](docs/reproduction.md),
[docs/thesis_runbook.md](docs/thesis_runbook.md), and
[docs/thesis_crosswalk.md](docs/thesis_crosswalk.md),
[docs/reproducibility_audit.md](docs/reproducibility_audit.md), and
[docs/optional_setup.md](docs/optional_setup.md) for details.

## Development

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```
