# Plugins

Plugins extend the toolkit without changing application workflows.

## Detector Plugins

A detector plugin provides:

- a stable name
- a factory method accepting detector parameters
- a detector instance with `train`, `score_run`, and `metadata`

The detector reads data through a `PreparedDatasetRepository` port and returns
Score Contract v1 rows as a dataframe. It does not write files directly.

Built-in detectors:

- `forecast-ridge`: scikit-learn ridge next-step forecasting baseline.
- `forecast-lstm`: optional torch LSTM next-step forecaster.
- `dra`: optional torch detection-only DRA Model 1 TCN forecaster.
- `interfusion`: optional torch detection-only HVAE window detector.
- `drcad`: optional torch detection-only dual-view contrastive detector.

Torch-backed detectors share these parameters:

- `window`, `train_stride`, `score_stride`, `max_train_windows`
- `epochs`, `batch_size`, `lr`, `seed`, `device`, `standardize`

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
