# Evidence Bundles

Evidence Bundle v1 is the detector-agnostic explanation artifact for one
oracle or operational event.

```powershell
itse evidence generate --prepared examples/generated/OPCUA_SYNTH --scores out/scores --out out/evidence
itse evidence validate --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence
```

Operational evidence consumes event matches from an evaluation directory:

```powershell
itse evidence generate --prepared examples/generated/OPCUA_SYNTH --scores out/scores --eval out/eval --event-source operational --out out/evidence-operational
```

The generator ranks variables by robust train/validation z-score deviation
inside the event window. It does not call detector-native explainers yet.

## Layout

```text
<evidence_out>/
  manifest.json
  index.jsonl
  bundles/<safe_run_id>/<safe_event_id>/evidence.json
```

Each bundle records event identity, source, matched GT event when available,
top variables, top time windows, score context, local rankings, and provenance.

