# Thesis-Style Reproduction Runbook

This runbook is the manual execution path for a reproducible thesis-style run.
It assumes the repository is already installed locally and that raw datasets are
available from approved local downloads.

## 1. Environment

Use either the local Python route or the Docker route. The local route is the
fastest setup path. The Docker route standardizes the toolkit container to a
16 GB memory budget for comparable runs across machines.

### Route A: Local Python

```powershell
cd C:\Users\joare\PycharmProjects\industrial-tsad-eval
git status --short
python -m pip install -e ".[dev,datasets,acquisition,profile,torch]"
python -m pip install "llama-cpp-python[server]" huggingface_hub
```

Start llama.cpp/OpenAI-compatible serving in a separate terminal:

```powershell
.\scripts\Start-LlamaCppServer.ps1 -ModelPath out\local-setup\models\<Qwen2.5-7B-Instruct-Q4_K_M.gguf> -Gpu auto
```

Success means `itse assistant providers` lists `llama-cpp` and
`itse assistant preflight --config <config>` reports a ready provider.

### Route B: Docker Compose

Set Docker Desktop Resources > Memory to 16 GB on Windows before starting the
run. Compose also sets `mem_limit: 16g` and `memswap_limit: 16g`.

```powershell
docker compose build itse
docker compose run --rm itse itse --help
```

For GPU-aware toolkit commands, use the wrapper after the image has been built:

```powershell
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse system gpu-check --device auto --json
```

`-Gpu auto` uses `docker-compose.gpu.yml` only when Docker CUDA is available.
Use `-Gpu on` if the run should fail rather than fall back to CPU.

The Docker route expects llama.cpp to run on the host. Start it with automatic
GPU offload when available:

```powershell
.\scripts\Start-LlamaCppServer.ps1 -ModelPath out\local-setup\models\<Qwen2.5-7B-Instruct-Q4_K_M.gguf> -Gpu auto
```

Use
`http://host.docker.internal:8080/v1` in Docker-specific assistant provider
configs:

```toml
[assistant.provider]
name = "llama-cpp"
model = "Qwen2.5-7B-Instruct-GGUF-Q4_K_M"
base_url = "http://host.docker.internal:8080/v1"
timeout_s = 180.0
temperature = 0.0
top_p = 1.0
max_tokens = 700
seed = 1337
```

See `docs/docker.md` for volume mounts, resource caps, GPU routing, and smoke
commands.

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

For Docker runs, use the same commands through Compose:

```powershell
docker compose run --rm itse python -m pytest
docker compose run --rm itse python -m ruff check .
docker compose run --rm itse python -m ruff format --check .
docker compose run --rm itse python -m mypy src
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse system report --out out\reproduction\machine_env.json --device auto
```

Record the memory and thread budget before long runs:

```text
out/reproduction/resource_budget.json
```

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

Docker equivalent:

```powershell
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce init-config --out config\thesis_verification.docker.toml --profile thesis-verification
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce plan --config config\thesis_verification.docker.toml
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce preflight --config config\thesis_verification.docker.toml --out out\preflight\thesis-verification-docker
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce run --config config\thesis_verification.docker.toml --out out\reproduction --run-id thesis-verification-docker-YYYYMMDD
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce status --run out\reproduction\thesis-verification-docker-YYYYMMDD
```

The verification profile runs all four prepared datasets and all three
protocols with `forecast-ridge`, plus bounded torch checks:
`forecast-lstm` on SWaT, HAI, and HAI-CPPS `naive`, and DRA, InterFusion, and
DRCAD on SWaT `naive`. It is intended to validate the workflow shape, not to
produce research-scale detector budgets.

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

Docker equivalent:

```powershell
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce init-config --out config\thesis_full.docker.toml --profile thesis-full
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce plan --config config\thesis_full.docker.toml
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce preflight --config config\thesis_full.docker.toml --out out\preflight\thesis-full-docker
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce run --config config\thesis_full.docker.toml --out out\reproduction --run-id thesis-full-docker-YYYYMMDD
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce status --run out\reproduction\thesis-full-docker-YYYYMMDD --watch
```

`thesis-full` explicitly includes `forecast-ridge`, `forecast-lstm`, `dra`,
`interfusion`, and `drcad` across the configured datasets and protocols. Adjust
epochs or detector parameters in the local TOML if the target machine needs a
smaller or larger budget.

## 6. Chunked Thesis-Style Execution

If one uninterrupted run is too long for the machine window, run compatible
slices without reducing detector budgets:

```powershell
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce run-slice --config config\thesis_full.docker.toml --out out\reproduction --run-id swat-drcad-naive --datasets SWaT --detectors drcad --protocols naive --stages benchmark,evidence,xai,assistant
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce run-slice --config config\thesis_full.docker.toml --out out\reproduction --run-id tep-forecast-ridge --datasets TEP --detectors forecast-ridge --protocols naive,all_in_one,zero_shot --stages benchmark,evidence,xai,assistant
```

Assemble only slices produced from compatible code, config, provider settings,
detector parameters, evaluation policy, and prepared dataset fingerprints:

```powershell
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce assemble --runs out\reproduction\swat-drcad-naive --runs out\reproduction\tep-forecast-ridge --out out\reproduction --run-id thesis-full-assembled-YYYYMMDD
```

The assembled pack is provenance-rich and marked as assembled. It is useful when
execution windows are limited; a single uninterrupted run remains the cleanest
reporting route when practical.

## 7. Fast-I/O Docker Option

For TEP-scale slices on Windows, hydrate Docker named volumes before running so
thousands of small Parquet reads stay inside Docker storage:

```powershell
.\scripts\Sync-ItseDockerFastIo.ps1 -Action hydrate
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto -FastIo run --name itse-tep-drcad-naive itse itse reproduce run-slice --config config\thesis_full.docker.toml --out out\reproduction --run-id tep-drcad-naive-fastio --datasets TEP --detectors drcad --protocols naive --stages benchmark,evidence,xai,assistant
.\scripts\Sync-ItseDockerFastIo.ps1 -Action export
```

The run records `INDUSTRIAL_TSAD_DOCKER_IO_ROUTE=named-volume` in
`resource_budget.json`. The normal bind-mount route remains simpler when you
need live host-side access to every output file.

## 8. Expected Artifacts

```text
out/reproduction/<run_id>/
  progress.jsonl
  progress_snapshot.json
  resource_budget.json
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
