# System Diagnostics

System diagnostics capture machine and runtime facts that make benchmark and
profiling artifacts reproducible.

## GPU And Torch Readiness

```powershell
itse system gpu-check --device auto --json
```

The report includes detected GPU adapters, the recommended backend, and a torch
runtime probe for `auto`, `cpu`, `cuda`, or `xpu`. Torch is optional; when it is
missing the report marks torch unavailable rather than failing import-time.

## Machine Reports

```powershell
itse system report --out out/system/machine_env.json --device auto
```

Machine reports include OS, Python, CPU, RAM, detected GPUs, torch runtime,
selected package versions, and best-effort git provenance.

## Preflight

```powershell
itse system preflight --prepared examples/generated/OPCUA_SYNTH --detector forecast-ridge --out out/preflight --strict
```

Preflight validates supplied prepared datasets, detector lookup and parameters,
torch runtime readiness for torch-backed detectors, and output writability. It
writes `preflight.json` and `machine_env.json` when `--out` is supplied.
