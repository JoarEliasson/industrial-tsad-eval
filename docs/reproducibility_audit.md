# Reproducibility Audit

The audit workflow is a clean-repo readiness check. It verifies that the package
imports, the CLI loads, architecture tests pass, the OPC-UA smoke fixture can be
generated, and thesis-smoke reproduction produces benchmark, evidence, XAI, and
RQ3 artifacts.

Run:

```powershell
itse audit run --out out/audit
```

For a deterministic CI-style pass without optional local resources:

```powershell
itse audit run --out out/audit --audit-id smoke --skip-optional
```

The audit writes:

```text
out/audit/<audit_id>/
  audit_summary.json
  audit_summary.md
  logs/
  workspace/
  reproduction/smoke-audit/
```

Statuses mean:

- `pass`: required or optional check succeeded.
- `fail`: required check failed and the audit exits nonzero.
- `warn`: optional check ran but produced a non-blocking problem.
- `skipped`: optional local resource was unavailable.

Optional probes include a tiny torch detector smoke, a live llama.cpp RQ3 smoke
when `http://127.0.0.1:8080/v1` is reachable, and thesis-full local prepared
dataset readiness when `prepared/TEP`, `prepared/SWaT`, `prepared/HAI`, and
`prepared/HAI-CPPS` exist.

The standard quality gates remain:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```
