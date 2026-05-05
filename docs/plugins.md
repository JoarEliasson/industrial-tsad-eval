# Plugins

Plugins extend the toolkit without changing application workflows.

## Detector Plugins

A detector plugin provides:

- a stable name
- a factory method accepting detector parameters
- a detector instance with `train`, `score_run`, and `metadata`

The detector reads data through a `PreparedDatasetRepository` port and returns
Score Contract v1 rows as a dataframe. It does not write files directly.

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
