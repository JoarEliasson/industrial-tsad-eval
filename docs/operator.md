# Operator Cards

Operator cards are deterministic, evidence-grounded summaries for industrial
anomaly events. They consume Evidence Bundle v1 artifacts and optional local
Markdown playbooks. They do not call LLMs, remote services, PDF parsers, or
arbitrary scripts.

## Commands

```powershell
itse operator retrieve --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence --query "what should the operator check"
itse operator card generate --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence --out out/operator-cards
itse operator card validate --prepared examples/generated/OPCUA_SYNTH --evidence out/evidence --cards out/operator-cards
```

Optional playbooks are local Markdown files:

```powershell
itse operator retrieve --prepared prepared/SWaT --evidence out/evidence --query "preserve artifacts" --playbooks docs/playbooks
```

## Artifact Layout

```text
<operator_out>/
  manifest.json
  index.jsonl
  retrieval/retrieval_result.json
  cards/<safe_event_id>/
    operator_card.json
    operator_card.md
```

## Card Contract

`operator_card.json` uses `operator-card-v1` and includes:

- status: `answered` or `abstained`
- dataset, run, event, event source, matched GT id when available
- situation summary
- evidence highlights
- checks
- recommended actions
- escalation criteria
- citations
- retrieval diagnostics and provenance

Answered cards require citations. Abstained cards require an abstention reason.

## Retrieval Behavior

Retrieval chunks Evidence Bundle v1 into deterministic roles: overview, top
variables, time windows, score context, local rankings, and provenance. Local
Markdown playbooks are added as playbook chunks when supplied.

Ranking uses lexical overlap, event/dataset filters, and small role bonuses from
detected intent. Supported intents are general, checks, recommended actions,
likely causes, and escalation criteria.

## Abstention

The card generator abstains when no matching evidence exists or when retrieval
returns no relevant chunks. Abstention cards are written and validated like
answered cards, but contain no operator actions.
