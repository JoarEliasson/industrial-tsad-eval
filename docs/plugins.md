# Plugins

Plugins extend the toolkit without changing application workflows.

## Detector Plugins

A detector plugin provides:

- a stable name
- a factory method accepting detector parameters
- a detector instance with `train`, `score_run`, and `metadata`

The detector reads data through a `PreparedDatasetRepository` port and returns
Score Contract v1 rows as a dataframe. It does not write files directly.

## Registration

The default registry is built in `industrial_tsad_eval.plugins.registry`.
Additional plugins should be registered by constructing a `DetectorRegistry` and
calling `register(plugin)` before invoking application services.

## Future Dataset Adapters

Dataset adapters should write Prepared Format v1 through repository or artifact
writer abstractions. Adapter implementations may handle source-specific parsing,
but their public output should remain the same prepared dataset contract.
