# Benchmarks

Benchmarks run a detector/dataset/protocol matrix against existing Prepared
Format datasets. They do not fetch or prepare raw data.

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
```

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
detection-delay mean, false alarms per hour, artifact paths, and error.
