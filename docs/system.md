# System Diagnostics

System diagnostics capture machine and runtime facts that make benchmark and
profiling artifacts reproducible. Probes live in
`src/industrial_tsad_eval/infrastructure/system.py` and return domain dataclasses
from `domain/system.py`.

## GPU And Torch Readiness

```powershell
itse system gpu-check --device auto --json
```

The report includes detected GPU adapters (`detect_system_gpus`,
`infrastructure/system.py:84`), the recommended backend
(`recommend_backend_for_runtime`, `:74`), and a torch runtime probe
(`probe_torch_runtime`, `:96`) returning `TorchRuntimeStatus`
(`domain/system.py:27`) for `auto`, `cpu`, `cuda`, or `xpu`. Torch is optional;
when it is missing the report marks torch unavailable rather than failing
import-time.

## Machine Reports

```powershell
itse system report --out out/system/machine_env.json --device auto
```

Machine reports are produced by `capture_machine_environment`
(`src/industrial_tsad_eval/infrastructure/system.py:136`) and shaped as
`MachineEnvironment` (`domain/system.py:46`). They include OS, Python, CPU,
RAM, detected GPUs (`SystemGpu`, `domain/system.py:14`), torch runtime,
selected package versions, and best-effort git provenance.

## Preflight

```powershell
itse system preflight --prepared examples/generated/OPCUA_SYNTH --detector forecast-ridge --out out/preflight --strict
```

Preflight (`RunPreflight`,
`src/industrial_tsad_eval/application/preflight.py:34`, with input dataclass
`PreflightInput` at `:22`) validates supplied prepared datasets, detector
lookup and parameters, torch runtime readiness for torch-backed detectors, and
output writability. Per-check results use `PreflightCheck`
(`domain/system.py:74`); the aggregate is `PreflightReport` (`:88`). It writes
`preflight.json` and `machine_env.json` when `--out` is supplied.
