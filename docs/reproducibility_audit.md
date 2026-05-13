# Reproducibility Audit

The audit workflow verifies that a clean checkout can run the committed
architecture end to end. It checks package imports, CLI loading, architecture
tests, OPC-UA fixture generation, thesis-smoke reproduction, synthetic raw data
setup, evidence/XAI outputs, assistant replay artifacts, and optional local
resources. The use case is `RunReproducibilityAudit`
(`src/industrial_tsad_eval/application/audit.py:88`); its config dataclass is
`ReproducibilityAuditConfig` (`:79`); per-check results use `AuditCheck`
(`domain/audit.py:12`) and aggregate as `AuditRunResult` (`:44`).

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
  synthetic-full-reproduction/thesis-full-smoke/
```

Statuses mean:

- `pass`: required or optional check succeeded.
- `fail`: required check failed and the audit exits nonzero.
- `warn`: optional check ran but produced a non-blocking problem.
- `skipped`: optional local resource was unavailable.

Required checks include a synthetic thesis-full setup rehearsal:

```text
examples make-thesis-raw-fixtures
data acquire --method manual
data validate
prepared prepare
prepared validate
reproduce run --run-id thesis-full-smoke
```

The generated raw fixtures cover TEP, SWaT, HAI, and HAI-CPPS adapter shapes.
They do not replace real datasets; they verify that the acquisition,
preparation, benchmark, evidence, XAI, and assistant replay orchestration surfaces still fit
together.

Optional probes include a tiny torch detector smoke, profiling extras
availability, a live llama.cpp assistant replay smoke when `http://127.0.0.1:8080/v1` is
reachable, and thesis-full local prepared dataset setup when `prepared/TEP`,
`prepared/SWaT`, `prepared/HAI`, and `prepared/HAI_CPPS` exist. The llama.cpp
probe checks structured planner/referee JSON output; a reachable server that
does not produce schema-valid assistant replay artifacts is reported as `warn`.

`audit_summary.json` and `audit_summary.md` include `setup_recommendations`
(`AuditSetupRecommendation`,
`src/industrial_tsad_eval/domain/audit.py:29`) for skipped optional resources.
Each recommendation contains commands and success criteria for the next local
setup step.

The standard quality gates remain:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
```
