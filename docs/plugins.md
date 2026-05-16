# Plugins

Plugins extend the toolkit without changing application workflows. Discovery
goes through registries defined in
`src/industrial_tsad_eval/plugins/registry.py` (detectors, dataset adapters,
dataset sources) and `plugins/providers.py:29` (LLM providers). Default
registries are wired by `default_detector_registry()` (`registry.py:88`),
`default_dataset_adapter_registry()` (`:107`),
`default_dataset_source_registry()` (`:122`), and
`default_llm_provider_registry()` (`providers.py:461`).

Operator cards are not plugins. They are deterministic application services
over Evidence Bundle v1 and optional local Markdown playbooks. Thesis assistant replay
assistant replay uses provider plugins behind the `LLMProvider` port
(`src/industrial_tsad_eval/ports/llm.py:18`).

## Dataset Source Plugins

Dataset source plugins acquire raw data into a local cache. The port is
`DatasetSourcePlugin`
(`src/industrial_tsad_eval/ports/dataset_sources.py:11`) and the registry is
`DatasetSourceRegistry` (`plugins/registry.py:64`). A plugin provides:

- stable `name`, such as `swat`
- raw `dataset_name`, such as `SWaT`
- `supported_methods()`
- `describe()`
- `acquire(target, config)`

The application service handles staging, overwrite policy, SHA256 inventory, and
`raw_provenance.json`. Source plugins should only materialize raw files into the
provided target directory.

Built-in sources (each defined in `src/industrial_tsad_eval/plugins/sources/`):

- `tep` (`plugins/sources/tep.py:11`): `manual`, `mathworks-http`, optional `kaggle`.
- `swat` (`plugins/sources/swat.py:11`): `manual`, optional `kaggle`.
- `hai` (`plugins/sources/hai.py:11`): `manual`, optional `kaggle`, optional `git`.
- `hai-cpps` (`plugins/sources/hai_cpps.py:11`): `manual`.

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

The port is `DetectorPlugin`
(`src/industrial_tsad_eval/ports/detectors.py:40`); detector instances implement
`Detector` (`:20`) and may additionally implement `DetectorExplainer` (`:33`).
The registry is `DetectorRegistry` (`plugins/registry.py:14`). A detector plugin
provides:

- a stable name
- a factory method accepting detector parameters
- a detector instance with `train`, `score_run`, and `metadata`

The detector reads data through a `PreparedDatasetRepository` port
(`ports/repositories.py:13`) and returns Score Contract v1 rows as a dataframe.
It does not write files directly. Detectors may also implement the optional
explainer port. When present, `ScoreRuns` (`application/scoring.py:29`) writes
native explanation parquet files under `scores/explanations/` through
`LocalExplanationRepository` (`infrastructure/explanation_repository.py:16`),
and Evidence Bundle generation can consume those rankings.

Built-in detectors:

- `forecast-ridge` (`plugins/forecast_ridge.py:124`, detector at `:29`): scikit-learn ridge next-step forecasting baseline.
- `forecast-lstm` (`plugins/torch_detectors.py:635`, detector at `:201`): optional torch LSTM next-step forecaster.
- `dra` (`plugins/torch_detectors.py:653`, detector at `:271`): optional torch TCN forecaster with residual-gradient saliency.
- `interfusion` (`plugins/torch_detectors.py:671`, detector at `:402`): optional torch HVAE detector with MC reconstruction attribution.
- `drcad` (`plugins/torch_detectors.py:689`, detector at `:516`): optional torch dual-view detector with counterfactual reconstruction deltas.

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

The port is `DatasetAdapterPlugin`
(`src/industrial_tsad_eval/ports/dataset_adapters.py:11`); the registry is
`DatasetAdapterRegistry` (`plugins/registry.py:39`). Dataset adapters convert
local raw data into Prepared Format v1. A plugin provides:

- stable `name`, such as `swat`
- prepared `dataset_name`, such as `SWaT`
- `describe_expected_raw_layout()`
- `prepare(raw, prepared, config)`

The `prepare` method writes to the exact prepared path supplied by the
application service. It should use shared prepared writer helpers
(`src/industrial_tsad_eval/infrastructure/prepared_writer.py:13`), avoid CLI
libraries, and avoid deleting existing outputs (enforced by
`tests/test_architecture.py:110`). The `PrepareDataset`
(`application/preparation.py:15`) use case handles staging, validation,
overwrite policy, and promotion.

Built-in adapters:

- `tep` (`plugins/datasets/tep.py:22`): Tennessee Eastman Process CSV, MAT, or RData.
- `swat` (`plugins/datasets/swat.py:28`): SWaT CSV, parquet, or Excel.
- `hai` (`plugins/datasets/hai.py:33`): HAI CSV files with optional version directory selection.
- `hai-cpps` (`plugins/datasets/hai_cpps.py:22`): HAI-CPPS scenario directories with `sim_setup.json` enrichment.

Optional raw readers are installed with:

```powershell
python -m pip install -e ".[datasets]"
```

## LLM Provider Plugins

Provider plugins serve assistant replay. The port is `LLMProviderPlugin`
(`src/industrial_tsad_eval/ports/llm.py:35`); the runtime interface is
`LLMProvider` (`ports/llm.py:18`); the registry is `LLMProviderRegistry`
(`plugins/providers.py:29`). A plugin provides:

- stable `name`, such as `llama-cpp`
- provider metadata for `itse assistant providers`
- `default_config()`
- `create(config)` returning an `LLMProvider`

Built-in providers (`plugins/providers.py`):

- `llama-cpp` / `openai-compatible` (`:151`, runtime at `:204`): local or cloud OpenAI-compatible endpoint.
- `anthropic` (`:337`, runtime at `:367`): Anthropic Messages API.
- `openai`, `google`, `xai`: cloud provider config shapes built on the OpenAI-compatible plugin.
- `fake` (`:54`, runtime at `:79`): deterministic provider for CI and smoke tests.

Provider secrets are read from environment variables named in config. They are
not stored in TOML or artifacts. Provider SDKs are imported lazily inside the
plugin methods — enforced by `tests/test_architecture.py:202`.
