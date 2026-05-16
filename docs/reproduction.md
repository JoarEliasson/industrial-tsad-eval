# Thesis-Style Reproduction

The reproduction layer ties the productized services together into a
thesis-shaped run without importing old thesis modules. It exists to reproduce
the experimental surface while keeping the improved package boundaries. The
orchestrating use cases live in
`src/industrial_tsad_eval/application/reproduction.py`:
`PlanThesisReproduction` (`:80`), `PreflightThesisReproduction` (`:106`),
`RunThesisReproduction` (`:220`), `SummarizeThesisReproduction` (`:1086`),
`DiagnoseThesisReproduction` (`:1097`). Run configuration is
`ReproductionConfig` (`domain/reproduction.py:103`); thesis-draft exports go
through `write_thesis_draft_exports`
(`application/thesis_exports.py:26`).

## Profiles

Create a local smoke config:

```powershell
itse reproduce init-config --out config/thesis_smoke.toml --profile thesis-smoke
```

Create a full thesis-style config:

```powershell
itse reproduce init-config --out config/thesis_full.toml --profile thesis-full
```

Create the bounded real-data verification profile:

```powershell
itse reproduce init-config --out config/thesis_verification.toml --profile thesis-verification
```

The verification and full profiles expect local Prepared Format roots for TEP,
SWaT, HAI, and HAI_CPPS. Raw data and credentials are never vendored. See
`docs/thesis_runbook.md` for the recommended manual run order.

`thesis-verification` keeps torch checks intentionally small. `thesis-full`
uses the thesis-aligned neural detector settings and requires native
explanation artifacts for DRA, InterFusion, and DRCAD while keeping Forecast
Ridge on the robust baseline.

TOML loading uses `load_reproduction_config`
(`src/industrial_tsad_eval/infrastructure/reproduction_config.py:376`); writing
the default template uses `write_default_reproduction_config` (`:392`); the
provider config template comes from `write_provider_config_template` (`:410`).

## Stages

`RunThesisReproduction` executes:

- prepared dataset validation
- benchmark scoring/evaluation
- evidence generation
- XAI evaluation
- optional profiling
- assistant replay suites
- summary aggregation and thesis crosswalk generation

Use:

```powershell
itse reproduce plan --config config/thesis_smoke.toml
itse reproduce preflight --config config/thesis_smoke.toml --out out/preflight
itse reproduce run --config config/thesis_smoke.toml --out out/reproduction --run-id smoke
itse reproduce status --run out/reproduction/smoke
itse reproduce summarize --run out/reproduction/smoke
```

## Chunked Runs

Long thesis-style runs can be split into compatible slices. This keeps each
execution window bounded while preserving full training, scoring, metric, XAI,
and assistant settings inside each selected cell.

Run one detector/dataset/protocol slice:

```powershell
itse reproduce run-slice --config config/thesis_full.toml --out out/reproduction --run-id swat-drcad-naive --datasets SWaT --detectors drcad --protocols naive --stages benchmark,evidence,xai,assistant
```

Run a detector family as a slice:

```powershell
itse reproduce run-slice --config config/thesis_full.toml --out out/reproduction --run-id tep-dra --datasets TEP --detectors dra --protocols naive,all_in_one,zero_shot
```

Assemble compatible slices:

```powershell
itse reproduce assemble --runs out/reproduction/swat-drcad-naive --runs out/reproduction/tep-dra --out out/reproduction --run-id thesis-full-assembled
```

Assembly validates compatible configs, prepared dataset fingerprints, detector
parameters, evaluation policy, and provider config. The assembled result pack is
explicitly marked as assembled from slices; use it when provenance-rich chunked
execution is preferred over one uninterrupted run.

For graceful cancellation, write a run-control marker before stopping the
container:

```powershell
itse reproduce stop --run out/reproduction/<run_id> --container <container-name>
docker stop <container-name>
```

## Performance Hardening

Large real-data runs now include additive fast paths that preserve the public
artifact contracts:

- Score directories still contain one Parquet per run, plus an optional
  `combined_scores.parquet` sidecar used by threshold calibration and evaluation.
- Reproduction preflight and benchmark validation reuse a prepared-validation
  cache under `out/cache/prepared-validation` when metadata and run-file
  fingerprints are unchanged.
- Torch detectors can score windows across runs in GPU batches; ForecastRidge
  uses bounded CPU parallel scoring.
- Docker runs can use the named-volume fast-I/O route documented in
  [docker.md](docker.md) to avoid slow Windows bind-mounted tiny-file reads.

`itse reproduce diagnose --run <run>` reports sidecar use, slowest experiments,
and scoring telemetry so a slow slice can be optimized without changing the
experiment budget.

## Artifacts

```text
<out>/<run_id>/
  config/
  resolved_config.json
  run_manifest.json
  progress.jsonl
  progress_snapshot.json
  preflight.json
  benchmark/
  evidence/
  xai/
  profiles/
  assistant/
  summaries/
    detection_summary.csv
    xai_summary.csv
    assistant_summary.csv
    detection_tables.csv
    explanation_results.csv
    explanation_results_split_summary.csv
    assistant_faithfulness_logs.csv
    profiling_logs.csv
    hyperparameters.toml
    scoring_config.json
    scoring_config.per_dataset/
    planner_prompt.txt
    referee_prompt.txt
    reproducibility_matrix.json
    thesis_crosswalk.md
```

The assistant replay provider defaults to `llama-cpp` in the full profile and `fake` in the
smoke profile. The full profile expects Qwen2.5-7B-Instruct GGUF Q4_K_M served
through llama.cpp at `http://127.0.0.1:8080/v1`. The older vLLM Qwen setup is a
historical baseline context, not the default path. The fake provider is for CI
and quick contract checks only.
