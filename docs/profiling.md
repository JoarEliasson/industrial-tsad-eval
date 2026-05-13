# Profiling

Profiling wraps the existing prepared-score-evaluate workflow with lightweight
timing and resource measurement. The use case is `ProfileScoreEvaluate`
(`src/industrial_tsad_eval/application/profiling.py:42`); configuration is
`ProfileScoreEvaluateConfig` (`:28`); the per-stage monitor is `StageMonitor`
(`infrastructure/profiling.py:19`); the per-run summary is `ProfileRunResult`
(`domain/profiling.py:37`) built from `StageSample` rows (`:10`).

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
`end_to_end`. Memory fields are best-effort: RSS uses optional `psutil`
(`infrastructure/profiling.py:199`), Python allocation uses `tracemalloc`
(`:238`), torch memory uses torch runtime APIs (`:219`), and VRAM uses optional
NVML (`:207`) when enabled by application callers. `psutil` and `pynvml`
imports stay lazy — enforced by `tests/test_architecture.py:141`.
