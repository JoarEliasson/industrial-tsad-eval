# Thesis-Style Reproduction

The reproduction layer ties the productized services together into a
thesis-shaped run without importing old thesis modules. It exists to reproduce
the experimental surface while keeping the improved package boundaries.

## Profiles

Create a local smoke config:

```powershell
itse reproduce init-config --out config/thesis_smoke.toml --profile thesis-smoke
```

Create a full thesis-style config:

```powershell
itse reproduce init-config --out config/thesis_full.toml --profile thesis-full
```

The full profile expects local Prepared Format roots for TEP, SWaT, HAI, and
HAI_CPPS. Raw data and credentials are never vendored.

## Stages

`RunThesisReproduction` executes:

- prepared dataset validation
- benchmark scoring/evaluation
- evidence generation
- XAI evaluation
- optional profiling
- RQ3 replay suites
- summary aggregation and thesis crosswalk generation

Use:

```powershell
itse reproduce plan --config config/thesis_smoke.toml
itse reproduce preflight --config config/thesis_smoke.toml --out out/preflight
itse reproduce run --config config/thesis_smoke.toml --out out/reproduction --run-id smoke
itse reproduce summarize --run out/reproduction/smoke
```

## Artifacts

```text
<out>/<run_id>/
  config/
  resolved_config.json
  run_manifest.json
  preflight.json
  benchmark/
  evidence/
  xai/
  profiles/
  rq3/
  summaries/
    detection_summary.csv
    xai_summary.csv
    rq3_summary.csv
    reproducibility_matrix.json
    thesis_crosswalk.md
```

The RQ3 provider defaults to `llama-cpp` in the full profile and `fake` in the
smoke profile. The fake provider is for CI and quick contract checks only.
