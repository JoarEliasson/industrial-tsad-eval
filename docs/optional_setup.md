# Optional Setup

The core package runs with the default dependencies. Optional components are
enabled only when local resources are present, and the audit reports skipped
checks with exact follow-up commands.

## Torch Detectors

Install the optional torch extra for `forecast-lstm`, `dra`, `interfusion`, and
`drcad`:

```powershell
python -m pip install -e ".[torch]"
itse score detectors
```

For CUDA or XPU systems, use the PyTorch selector to install the wheel that
matches the local driver/runtime. A small CPU smoke is:

```powershell
itse examples make-opcua-fixture --out examples/generated
itse score run --prepared examples/generated/OPCUA_SYNTH --detector forecast-lstm --out out/lstm-smoke --parameters-json "{\"window\": 16, \"train_stride\": 8, \"score_stride\": 8, \"epochs\": 1, \"device\": \"cpu\"}"
```

Success means Score Contract v1 artifacts are written and
`itse scores validate` passes.

## Profiling Extras

Install memory/runtime helpers:

```powershell
python -m pip install -e ".[profile]"
itse profile run --prepared examples/generated/OPCUA_SYNTH --detector forecast-ridge --out out/profiles --profile-id smoke
```

Success means `summary.json`, `stages.csv`, and `budget_check.md` exist under
the profile directory. If the extras are absent, profiling still records timing
and Python allocation peaks where available.

## llama.cpp

The recommended local assistant replay provider is a running llama.cpp OpenAI-compatible
server with Qwen2.5-7B-Instruct GGUF Q4_K_M:

```powershell
python -m pip install "llama-cpp-python[server]" huggingface_hub
llama-server -m C:\path\to\model.gguf --host 127.0.0.1 --port 8080
itse assistant providers
itse assistant preflight --config config/thesis_full.toml
```

Success means the `llama-cpp` provider healthcheck reaches
`http://127.0.0.1:8080/v1/models`, reports `ready`, and assistant replay structured planner
and referee JSON generation works through `/v1/chat/completions`.

## OpenAI-Compatible And Cloud Providers

Use `openai-compatible` for local/cloud endpoints that implement
`/v1/chat/completions`. Cloud provider configs store environment variable names
only:

```toml
[assistant.provider]
name = "openai"
model = "gpt-4.1-mini"
api_key_env = "OPENAI_API_KEY"
temperature = 0.0
max_tokens = 700
```

The built-in provider protocol is tested with a local HTTP stub, so llama.cpp
and OpenAI-compatible request/response handling can be validated without a
downloaded model.

## Real Thesis Datasets

Raw industrial datasets are never vendored. Acquire or download them locally,
then keep acquisition separate from preparation:

```powershell
itse data acquire --source swat --method manual --manual data/downloads/SWaT --out data/raw
itse data validate --source swat --raw data/raw/SWaT
itse prepared prepare --dataset swat --raw data/raw/SWaT --out prepared
itse prepared validate --prepared prepared/SWaT
```

Repeat the same shape for `tep`, `hai`, and `hai-cpps`. The synthetic rehearsal
uses:

```powershell
itse examples make-thesis-raw-fixtures --out out/setup-fixtures/raw
```

Those generated fixtures validate the flow shape only. Real thesis-full runs
should point at approved local prepared roots.
