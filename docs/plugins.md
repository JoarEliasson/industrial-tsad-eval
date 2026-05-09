# Plugins

Plugins extend the toolkit without changing application workflows.

Operator cards are not plugins. They are deterministic application services
over Evidence Bundle v1 and optional local Markdown playbooks. Thesis assistant replay
assistant replay uses provider plugins behind the `LLMProvider` port.

## Dataset Source Plugins

Dataset source plugins acquire raw data into a local cache. A plugin provides:

- stable `name`, such as `swat`
- raw `dataset_name`, such as `SWaT`
- `supported_methods()`
- `describe()`
- `acquire(target, config)`

The application service handles staging, overwrite policy, SHA256 inventory, and
`raw_provenance.json`. Source plugins should only materialize raw files into the
provided target directory.

Built-in sources:

- `tep`: `manual`, `mathworks-http`, optional `kaggle`.
- `swat`: `manual`, optional `kaggle`.
- `hai`: `manual`, optional `kaggle`, optional `git`.
- `hai-cpps`: `manual`.

Optional online helpers are installed with:

```powershell
python -m pip install -e ".[acquisition]"
```

Acquisition does not call dataset adapters automatically:

```powershell
itse data acquire --source swat --method manual --manual data/downloads/SWaT --out data/raw
itse prepared prepare --dataset swat --raw data/raw/SWaT --out prepared
```

## Detector Plugins

A detector plugin provides:

- a stable name
- a factory method accepting detector parameters
- a detector instance with `train`, `score_run`, and `metadata`

The detector reads data through a `PreparedDatasetRepository` port and returns
Score Contract v1 rows as a dataframe. It does not write files directly.
Detectors may also implement the optional explainer port. When present,
`ScoreRuns` writes native explanation parquet files under
`scores/explanations/`, and Evidence Bundle generation can consume those
rankings.

Built-in detectors:

- `forecast-ridge`: scikit-learn ridge next-step forecasting baseline.
- `forecast-lstm`: optional torch LSTM next-step forecaster.
- `dra`: optional torch TCN forecaster with residual-gradient saliency.
- `interfusion`: optional torch HVAE detector with MC reconstruction attribution.
- `drcad`: optional torch dual-view detector with counterfactual reconstruction deltas.

Torch-backed detectors share these parameters:

- `window`, `train_stride`, `score_stride`, `max_train_windows`
- `epochs`, `batch_size`, `lr`, `seed`, `device`, `standardize`
- `explanation_top_k` for native explanation artifact size

Install optional torch support with:

```powershell
python -m pip install -e ".[torch]"
```

For CUDA or XPU runtimes, install the PyTorch wheel recommended by the PyTorch
selector, then install this package. The supported device values are `auto`,
`cpu`, `cuda`, and `xpu`.

Example benchmark detector entry:

```toml
[[detectors]]
id = "forecast-lstm-tiny"
name = "forecast-lstm"
parameters = { window = 16, train_stride = 8, score_stride = 8, epochs = 1, batch_size = 8, device = "cpu", hidden_size = 8 }
```

## Dataset Adapter Plugins

Dataset adapters convert local raw data into Prepared Format v1. A plugin
provides:

- stable `name`, such as `swat`
- prepared `dataset_name`, such as `SWaT`
- `describe_expected_raw_layout()`
- `prepare(raw, prepared, config)`

The `prepare` method writes to the exact prepared path supplied by the
application service. It should use shared prepared writer helpers, avoid CLI
libraries, and avoid deleting existing outputs. The `PrepareDataset` use case
handles staging, validation, overwrite policy, and promotion.

Built-in adapters:

- `tep`: Tennessee Eastman Process CSV, MAT, or RData.
- `swat`: SWaT CSV, parquet, or Excel.
- `hai`: HAI CSV files with optional version directory selection.
- `hai-cpps`: HAI-CPPS scenario directories with `sim_setup.json` enrichment.

Optional raw readers are installed with:

```powershell
python -m pip install -e ".[datasets]"
```

## LLM Provider Plugins

Provider plugins serve assistant replay. A plugin provides:

- stable `name`, such as `llama-cpp`
- provider metadata for `itse assistant providers`
- `default_config()`
- `create(config)` returning an `LLMProvider`

Built-in providers:

- `llama-cpp`: local OpenAI-compatible llama.cpp server.
- `openai-compatible`: generic local/cloud OpenAI-compatible endpoint.
- `openai`, `anthropic`, `google`, `xai`: cloud provider config shapes.
- `fake`: deterministic provider for CI and smoke tests.

Provider secrets are read from environment variables named in config. They are
not stored in TOML or artifacts.
