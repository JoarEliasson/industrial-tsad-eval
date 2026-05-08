# Docker Execution Route

Docker is an optional route for thesis-style runs when you want a repeatable
toolkit environment across machines. The default local Python route remains
valid; Docker adds a standardized container memory budget and fixed thread caps.

## Resource Budget

The Compose service sets:

- container memory: `16g`
- container memory plus swap: `16g`
- shared memory: `1g`
- `OMP_NUM_THREADS=8`
- `MKL_NUM_THREADS=8`
- `OPENBLAS_NUM_THREADS=8`
- `NUMEXPR_NUM_THREADS=8`

On Windows, Docker Desktop also needs its resource limit set to 16 GB. Compose
cannot raise the memory available to Docker Desktop if the desktop limit is
lower.

## Build

```powershell
docker compose build itse
docker compose run --rm itse itse --help
```

The image installs the toolkit with the `dev`, `datasets`, `acquisition`,
`profile`, and `torch` extras. Raw data, prepared data, model weights, and run
outputs are not copied into the image.

PyPI Linux torch wheels may include large CUDA runtime packages even when a run
ultimately uses CPU. For a smaller CPU-only toolkit image, build without the
`torch` extra and run only non-torch detectors:

```powershell
docker compose build --build-arg INSTALL_EXTRAS=dev,datasets,acquisition,profile itse
```

Use the default compose build for the verification route when bounded torch
detector checks are part of the plan.

## GPU-Aware Toolkit Runs

The default compose file stays CPU-compatible. GPU passthrough is available
through `docker-compose.gpu.yml` and the PowerShell wrapper
`scripts/Invoke-ItseDocker.ps1`.

Build the image first, then let the wrapper decide whether Docker CUDA is usable:

```powershell
docker compose build itse
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse system gpu-check --device auto --json
```

`-Gpu auto` enables the GPU override only when both conditions pass:

- `nvidia-smi` is available on the host.
- `docker run --gpus all industrial-tsad-eval:local ...` can import torch and
  reports `torch.cuda.is_available() == True`.

Use `-Gpu on` when a run should fail immediately if Docker GPU passthrough is
not configured:

```powershell
.\scripts\Invoke-ItseDocker.ps1 -Gpu on run --rm itse itse system gpu-check --device cuda --json
```

Use `-Gpu off` for a deliberately CPU-only run:

```powershell
.\scripts\Invoke-ItseDocker.ps1 -Gpu off run --rm itse itse reproduce status --run out/reproduction/<run_id>
```

The wrapper prints the exact `docker compose` command it selects. Add `-DryRun`
to inspect the route without executing it.

## Volumes

The service mounts:

```text
./config             -> /workspace/config
./data               -> /workspace/data
./prepared           -> /workspace/prepared
./out                -> /workspace/out
./examples/generated -> /workspace/examples/generated
```

Keep approved raw data under `data/`, prepared roots under `prepared/`, local
TOML configs under `config/`, and generated artifacts under `out/`.

## llama.cpp

The first Docker route expects llama.cpp to run on the host machine, not inside
the toolkit container. Start it separately with automatic GPU offload when an
NVIDIA runtime is available:

```powershell
.\scripts\Start-LlamaCppServer.ps1 -ModelPath out\local-setup\models\<Qwen2.5-7B-Instruct-Q4_K_M.gguf> -Gpu auto
```

`-Gpu auto` adds `--n_gpu_layers -1` when `nvidia-smi` is available. Use
`-Gpu on` to require offload, or `-Gpu off` to force CPU serving. The helper
does not install a GPU-enabled `llama-cpp-python` build; install the wheel or
build matching your host CUDA setup before using GPU offload.

For a background server with logs:

```powershell
.\scripts\Start-LlamaCppServer.ps1 -ModelPath out\local-setup\models\<Qwen2.5-7B-Instruct-Q4_K_M.gguf> -Gpu auto -Background
```

For Docker runs, set assistant provider configs to:

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

`host.docker.internal` is available by default on Docker Desktop. The compose
file also adds a Linux-compatible host-gateway mapping.

## Gates

```powershell
docker compose run --rm itse python -m pytest
docker compose run --rm itse python -m ruff check .
docker compose run --rm itse python -m ruff format --check .
docker compose run --rm itse python -m mypy src
```

## Smoke Run

```powershell
docker compose run --rm itse itse examples make-opcua-fixture --out examples/generated
docker compose run --rm itse itse reproduce init-config --out config/thesis_smoke.docker.toml --profile thesis-smoke
docker compose run --rm itse itse reproduce run --config config/thesis_smoke.docker.toml --out out/reproduction --run-id docker-smoke
docker compose run --rm itse itse reproduce status --run out/reproduction/docker-smoke
docker compose run --rm itse itse reproduce summarize --run out/reproduction/docker-smoke
```

The smoke profile uses the fake provider, so it does not need llama.cpp.

## Verification Run

After preparing real datasets and editing the provider base URL to
`http://host.docker.internal:8080/v1`, run:

```powershell
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce init-config --out config/thesis_verification.docker.toml --profile thesis-verification
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce plan --config config/thesis_verification.docker.toml
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce preflight --config config/thesis_verification.docker.toml --out out/preflight/thesis-verification-docker
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce run --config config/thesis_verification.docker.toml --out out/reproduction --run-id thesis-verification-docker-YYYYMMDD
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse reproduce status --run out/reproduction/thesis-verification-docker-YYYYMMDD
```

## Resource Budget Artifact

Record the intended execution budget alongside the run:

```json
{
  "format_version": "resource-budget-v1",
  "execution_route": "docker-compose",
  "container_memory_gb": 16,
  "container_swap_gb": 16,
  "thread_caps": {
    "OMP_NUM_THREADS": 8,
    "MKL_NUM_THREADS": 8,
    "OPENBLAS_NUM_THREADS": 8,
    "NUMEXPR_NUM_THREADS": 8
  },
  "docker_gpu_mode": "auto",
  "llama_base_url": "http://host.docker.internal:8080/v1",
  "llama_gpu_mode": "auto",
  "llama_gpu_layers": -1
}
```

Save this as `out/reproduction/resource_budget.json` before the long run.

## GPU Notes

GPU support has two independent parts:

- Toolkit ML detectors use PyTorch inside Docker. This requires Docker GPU
  passthrough plus a torch wheel that reports CUDA as available.
- Assistant replay uses the host llama.cpp server. This requires a GPU-enabled
  `llama-cpp-python` build and `--n_gpu_layers` offload.

Check the toolkit side with:

```powershell
.\scripts\Invoke-ItseDocker.ps1 -Gpu auto run --rm itse itse system gpu-check --device auto --json
```

Check the host llama.cpp side with:

```powershell
Invoke-WebRequest http://127.0.0.1:8080/v1/models -UseBasicParsing
```

Record the selected backend in `machine_env.json`, `resource_budget.json`, and
the run summary. On Windows, Docker Desktop GPU support requires WSL 2 backed
Docker Desktop plus current NVIDIA drivers.
