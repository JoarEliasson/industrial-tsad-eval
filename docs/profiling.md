# Profiling

Profiling wraps the existing prepared-score-evaluate workflow with lightweight
timing and resource measurement.

## Command

```powershell
itse profile run --prepared examples/generated/OPCUA_SYNTH --detector forecast-ridge --out out/profiles --profile-id smoke
```

Torch detectors can be profiled with their normal detector parameters:

```powershell
itse profile run --prepared examples/generated/OPCUA_SYNTH --detector forecast-lstm --out out/profiles --parameters-json "{\"window\": 16, \"train_stride\": 8, \"score_stride\": 8, \"epochs\": 1, \"device\": \"cpu\"}"
```

## Artifacts

```text
<out>/<profile_id>/
  machine_env.json
  preflight.json
  stages.csv
  summary.json
  budget_check.md
  artifacts/
    scores/
    eval/
```

Stages are `validate_prepared`, `score`, `validate_scores`, `evaluate`, and
`end_to_end`. Memory fields are best-effort: RSS uses optional `psutil`, Python
allocation uses `tracemalloc`, torch memory uses torch runtime APIs, and VRAM
uses optional NVML when enabled by application callers.
