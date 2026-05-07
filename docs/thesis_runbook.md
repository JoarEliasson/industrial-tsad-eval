# Thesis-Style Reproduction Runbook

This runbook is the manual execution path for a reproducible thesis-style run.
It assumes the repository is already installed locally and that raw datasets are
available from approved local downloads.

## 1. Environment

```powershell
cd C:\Users\joare\PycharmProjects\industrial-tsad-eval
git status --short
python -m pip install -e ".[dev,datasets,acquisition,profile,torch]"
python -m pip install "llama-cpp-python[server]" huggingface_hub
```

Start llama.cpp/OpenAI-compatible serving in a separate terminal:

```powershell
python -m llama_cpp.server --model out\local-setup\models\<Qwen2.5-7B-Instruct-Q4_K_M.gguf> --host 127.0.0.1 --port 8080
```

Success means `itse assistant providers` lists `llama-cpp` and
`itse assistant preflight --config <config>` reports a ready provider.

## 2. Data Preparation

Use approved local raw roots:

```powershell
itse data acquire --source tep --method manual --manual <raw-TEP> --out data\raw --overwrite
itse data acquire --source swat --method manual --manual <raw-SWaT> --out data\raw --overwrite
itse data acquire --source hai --method manual --manual <raw-HAI> --out data\raw --overwrite
itse data acquire --source hai-cpps --method manual --manual <raw-HAI-CPPS> --out data\raw --overwrite

itse data validate --source tep --raw data\raw\TEP
itse data validate --source swat --raw data\raw\SWaT
itse data validate --source hai --raw data\raw\HAI
itse data validate --source hai-cpps --raw data\raw\HAI_CPPS

itse prepared prepare --dataset tep --raw data\raw\TEP --out prepared --overwrite
itse prepared prepare --dataset swat --raw data\raw\SWaT --out prepared --overwrite
itse prepared prepare --dataset hai --raw data\raw\HAI --out prepared --overwrite
itse prepared prepare --dataset hai-cpps --raw data\raw\HAI_CPPS --out prepared --overwrite

itse prepared validate --prepared prepared\TEP
itse prepared validate --prepared prepared\SWaT
itse prepared validate --prepared prepared\HAI
itse prepared validate --prepared prepared\HAI_CPPS
```

## 3. Gates And Preflight

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src

itse system gpu-check --device auto --json
itse system report --out out\reproduction\machine_env.json --device auto
itse audit run --out out\audit --audit-id thesis-run-preflight
```

Required failures should be fixed before continuing. Optional skips for missing
extra resources are acceptable only when that resource is not part of the run.

## 4. Bounded Verification

Run the bounded real-data profile before any larger matrix:

```powershell
itse reproduce init-config --out config\thesis_verification.local.toml --profile thesis-verification
itse reproduce plan --config config\thesis_verification.local.toml
itse reproduce preflight --config config\thesis_verification.local.toml --out out\preflight\thesis-verification
itse reproduce run --config config\thesis_verification.local.toml --out out\reproduction --run-id thesis-verification-YYYYMMDD
itse reproduce status --run out\reproduction\thesis-verification-YYYYMMDD
itse reproduce summarize --run out\reproduction\thesis-verification-YYYYMMDD
```

The verification profile runs all four prepared datasets and all three
protocols with `forecast-ridge`, plus a tiny `forecast-lstm` torch check over
the same matrix. It is intended to validate the workflow shape, not to produce
research-scale detector budgets.

## 5. Larger Thesis-Style Run

Only run this after the bounded verification profile passes:

```powershell
itse reproduce init-config --out config\thesis_full.local.toml --profile thesis-full
itse reproduce plan --config config\thesis_full.local.toml
itse reproduce preflight --config config\thesis_full.local.toml --out out\preflight\thesis-full
itse reproduce run --config config\thesis_full.local.toml --out out\reproduction --run-id thesis-full-YYYYMMDD
itse reproduce status --run out\reproduction\thesis-full-YYYYMMDD --watch
itse reproduce summarize --run out\reproduction\thesis-full-YYYYMMDD
```

`thesis-full` explicitly includes `forecast-ridge`, `forecast-lstm`, `dra`,
`interfusion`, and `drcad` across the configured datasets and protocols. Adjust
epochs or detector parameters in the local TOML if the target machine needs a
smaller or larger budget.

## 6. Expected Artifacts

```text
out/reproduction/<run_id>/
  progress.jsonl
  progress_snapshot.json
  run_manifest.json
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
    reproducibility_matrix.json
    thesis_crosswalk.md
```

Use `itse reproduce status --run <run>` while the command is running and
`itse reproduce summarize --run <run>` after it completes.
