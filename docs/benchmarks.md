# Benchmarks

Benchmarks run a detector/dataset/protocol matrix against existing Prepared
Format datasets. They do not fetch or prepare raw data. The orchestrator is
`RunBenchmark` (`src/industrial_tsad_eval/application/benchmark.py:75`), which
delegates to `ScoreRuns` (`application/scoring.py:29`) and `EvaluateScores`
(`application/evaluation.py:40`) for each cell of the matrix.

## Config

```toml
[benchmark]
name = "opcua-smoke"
protocols = ["naive"]

[benchmark.evaluation]
threshold_quantile = 0.995

[[datasets]]
id = "opcua"
prepared = "examples/generated/OPCUA_SYNTH"

[[detectors]]
id = "forecast-ridge-default"
name = "forecast-ridge"
parameters = { window = 32, stride = 4, lags = 1, alpha = 1.0, standardize = true, seed = 1337 }

[[detectors]]
id = "forecast-lstm-tiny"
name = "forecast-lstm"
parameters = { window = 16, train_stride = 8, score_stride = 8, epochs = 1, batch_size = 8, device = "cpu", hidden_size = 8 }
```

Config dataclasses live in `src/industrial_tsad_eval/domain/benchmark.py`:
`BenchmarkConfig` (`:168`), `BenchmarkDatasetConfig` (`:47`),
`BenchmarkDetectorConfig` (`:59`), `BenchmarkEvaluationConfig` (`:16`),
`BenchmarkParameterOverride` (`:103`), `BenchmarkExperiment` (`:127`). TOML
loading and rendering are in `infrastructure/benchmark_config.py:39`/`:55`.
Relative `prepared` paths are resolved relative to the TOML file. Dataset ids,
detector ids, and protocols may contain letters, numbers, `.`, `_`, and `-`.
Experiment ids are generated as:

```text
<dataset-id>__<detector-id>__<protocol>
```

## Commands

```powershell
itse bench init-config --out benchmarks/opcua.toml
itse bench plan --config benchmarks/opcua.toml
itse bench run --config benchmarks/opcua.toml --out out/benchmarks
itse bench summarize --run out/benchmarks/<run_id>
```

`bench run` continues after per-experiment failures, writes all statuses, and
exits nonzero if any experiment failed.

## Artifacts

```text
<out>/<run_id>/
  config/benchmark.toml
  resolved_config.json
  run_manifest.json
  summary.json
  summary.csv
  experiments/<experiment_id>/
    status.json
    scores/
    eval/
```

The CSV summary includes the public columns needed for quick comparison:
experiment id, dataset, detector, protocol, status, threshold, event PRF,
detection-delay mean, false alarms per hour, point-adjusted F1,
affiliation-style interval PRF, artifact paths, and error.

Metric families are written as separate blocks in `eval/metrics.json`. Event
PRF remains the operational alarm metric. Point and point-adjusted metrics are
reported for comparison with range/point-adjusted TSAD literature, and
affiliation-style interval metrics report temporal-overlap behavior separately.
See Tatbul et al., "Precision and Recall for Time Series", and Huet et al.,
"Local Evaluation of Time Series Anomaly Detection Algorithms" for the metric
families that motivate these exports.
