# Evidence Bundles

Evidence Bundle v1 is the explanation boundary artifact for one oracle or
operational event. It can be built from detector-native explanation artifacts
when available, or from the deterministic robust baseline.

```powershell
itse evidence generate --prepared examples/generated/OPCUA_SYNTH --scores out/scores --out out/evidence
itse evidence validate --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence
```

Operational evidence consumes event matches from an evaluation directory:

```powershell
itse evidence generate --prepared examples/generated/OPCUA_SYNTH --scores out/scores --eval out/eval --event-source operational --out out/evidence-operational
```

By default, `explanation_source=auto` uses native explanation artifacts from
`scores/explanations/` when a detector produced them, then falls back to robust
train/validation z-score deviation. Use `explanation_source=native` when a run
must fail if native artifacts are missing, or `explanation_source=robust` for
the detector-agnostic baseline.

DRA writes residual-gradient saliency, InterFusion writes Monte Carlo
reconstruction/imputation attribution, and DRCAD writes counterfactual
reconstruction deltas. Forecast Ridge remains on the robust baseline.

## Layout

```text
<evidence_out>/
  manifest.json
  index.jsonl
  bundles/<safe_run_id>/<safe_event_id>/evidence.json
```

Each bundle records event identity, source, matched GT event when available,
top variables, top time windows, score context, local rankings, and provenance.
The provenance records whether native or robust evidence was used and which
explainer method produced the rankings.
