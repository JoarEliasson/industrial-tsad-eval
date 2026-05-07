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
the toolkit container. Start it separately:

```powershell
python -m llama_cpp.server --model out\local-setup\models\<Qwen2.5-7B-Instruct-Q4_K_M.gguf> --host 127.0.0.1 --port 8080
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
docker compose run --rm itse itse reproduce init-config --out config/thesis_verification.docker.toml --profile thesis-verification
docker compose run --rm itse itse reproduce plan --config config/thesis_verification.docker.toml
docker compose run --rm itse itse reproduce preflight --config config/thesis_verification.docker.toml --out out/preflight/thesis-verification-docker
docker compose run --rm itse itse reproduce run --config config/thesis_verification.docker.toml --out out/reproduction --run-id thesis-verification-docker-YYYYMMDD
docker compose run --rm itse itse reproduce status --run out/reproduction/thesis-verification-docker-YYYYMMDD
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
  "llama_base_url": "http://host.docker.internal:8080/v1"
}
```

Save this as `out/reproduction/resource_budget.json` before the long run.

## GPU Notes

The compose file intentionally does not hard-code GPU passthrough. CUDA, XPU,
and CPU-only runs require different host setup. Use:

```powershell
docker compose run --rm itse itse system gpu-check --device auto --json
```

Record the selected backend in `machine_env.json` and the run summary.
