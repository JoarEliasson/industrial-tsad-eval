"""Thesis-style assistant replay services."""
# ruff: noqa: E501

from __future__ import annotations

import csv
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.operator import RetrieveOperatorEvidence
from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.domain.assistant_replay import (
    AssistantCase,
    AssistantReplayConfig,
    AssistantReplayRunResult,
    AssistantRunMetrics,
    ClaimEvaluation,
    DraftClaim,
    DraftResponse,
    ReplaySuiteManifest,
    aggregate_assistant_metrics,
)
from industrial_tsad_eval.domain.errors import AssistantReplayError
from industrial_tsad_eval.domain.llm import LLMMessage, LLMStructuredRequest
from industrial_tsad_eval.domain.operator import OperatorRetrievalResult
from industrial_tsad_eval.domain.progress import CompositeProgressSink, ProgressEvent, ProgressSink
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.assistant_replay_repository import (
    LocalAssistantReplayMetricsRepository,
    LocalAssistantRunRepository,
    LocalReplaySuiteRepository,
)
from industrial_tsad_eval.infrastructure.evidence_repository import LocalEvidenceRepository
from industrial_tsad_eval.infrastructure.json_utils import read_json
from industrial_tsad_eval.infrastructure.operator_repository import LocalOperatorCardRepository
from industrial_tsad_eval.infrastructure.progress import LocalProgressSink
from industrial_tsad_eval.plugins.providers import LLMProviderRegistry

CLAIM_RE = re.compile(r"(?:^|\n)\s*(?:[-*]|\d+[.)])?\s*([^\n]+)")
CITATION_RE = re.compile(r"\[(C\d+)\]")
TOKEN_RE = re.compile(r"[A-Za-z0-9_:/.-]+")
CHECK_TARGET_RE = re.compile(
    r"^\s*(?:check|inspect|verify|review|monitor)\s+([A-Za-z0-9_:/.-]+)",
    re.IGNORECASE,
)
TAG_ROLE_PRIORITY = {
    "top_variables": 0,
    "overview": 1,
    "local_rankings": 2,
    "time_windows": 3,
    "score_context": 4,
    "provenance": 5,
}
PLANNER_SYSTEM_PROMPT = """You are the Planner for an operator-facing industrial assistant.

Return JSON matching DraftResponse only.

Rules:
- Use only evidence_summary and retrieval_hits; no outside knowledge or fabricated thresholds.
- Be conservative: leave unsupported sections empty.
- symptom_summary: zero or one neutral evidence-grounded sentence, never an instruction. If evidence only ranks variables or artifacts, use an empty string.
- checks: short inspection steps tied to named evidence targets. If evidence only names salient variables, write "Check <target>."
- likely_causes: optional evidence-bound causes, not actions.
- recommended_actions and escalation_criteria: include only when a retrieved technical document, operating guide, playbook, or evidence text directly states the action/trigger.
- Each list item must be short, atomic, and contain one claim or instruction.
- Do not mention current, baseline, expected, status, threshold, compare, restart, notify, timing, or roles unless those exact terms are in the evidence.
- Return no markdown and no commentary.
"""
REFEREE_SYSTEM_PROMPT = """You are the Referee for an operator-facing industrial assistant.

Verify one claim using only the cited evidence.
Return JSON matching ClaimEvaluation only.

Rules:
- Use only the provided cited evidence chunks.
- No outside knowledge or unstated assumptions.
- If support is incomplete or inferential, use entailment_label="insufficient" and is_supported=false.
- entailment_label must be one of: entails, insufficient, contradicts.
- final_disposition must be one of: keep, rewrite, remove.
- keep only if the claim is directly supported as written.
- A conservative checks claim of the form "Check <tag>" is directly supported when the cited evidence or supporting_facts lists that exact tag as a top variable or matched evidence target.
- Broader recommended actions, escalation criteria, causes, or status claims still require direct textual support in cited evidence.
- rewrite only if a narrower statement is directly supported and remains valid for the claim's section semantics.
- remove if unsupported or contradicted.
- rewritten_statement must be null unless final_disposition is rewrite.
- Any rewrite must be strictly more conservative than the original claim.
- A rewritten checks item must remain a concrete inspection or verification step.
- A rewritten recommended_actions item must remain a concrete action step.
- A rewritten escalation_criteria item must remain a concrete trigger or condition.
- Never rewrite an action into a mere observation.
- Keep entailment_reasoning brief and evidence-focused.
- Return no markdown and no commentary.
"""


@dataclass(frozen=True)
class AssistantReplayPreflightResult:
    """assistant replay preflight status."""

    ok: bool
    checks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {"ok": self.ok, "checks": [dict(check) for check in self.checks]}


class PreflightAssistantReplay:
    """Validate assistant replay configuration, provider readiness, and prepared inputs."""

    def __init__(self, *, config: AssistantReplayConfig, provider_registry: LLMProviderRegistry):
        self.config = config
        self.provider_registry = provider_registry

    def run(self) -> AssistantReplayPreflightResult:
        """Run assistant replay preflight checks."""
        checks: list[dict[str, Any]] = []
        prepared_report = ValidatePreparedDataset(self.config.prepared).run()
        checks.append(
            {
                "name": "prepared",
                "status": "pass" if prepared_report.ok else "fail",
                "message": "Prepared dataset validation passed."
                if prepared_report.ok
                else "Prepared dataset validation failed.",
                "details": prepared_report.to_dict(),
            }
        )
        try:
            plugin = self.provider_registry.get(self.config.provider.name)
            provider = plugin.create(self.config.provider)
            health = provider.healthcheck()
            checks.append(
                {
                    "name": "provider",
                    "status": "pass" if health.ok else "warn",
                    "message": health.message,
                    "details": health.to_dict(),
                }
            )
        except Exception as exc:
            checks.append(
                {
                    "name": "provider",
                    "status": "fail",
                    "message": f"{type(exc).__name__}: {exc}",
                    "details": {"provider": self.config.provider.name},
                }
            )
        ok = not any(check["status"] == "fail" for check in checks)
        return AssistantReplayPreflightResult(ok=ok, checks=checks)


class BuildReplaySuite:
    """Build a deterministic replay suite from Evidence Bundle v1 artifacts."""

    def __init__(self, *, config: AssistantReplayConfig, evidence: str | Path, out: str | Path):
        self.config = config
        self.evidence = Path(evidence)
        self.out = Path(out)

    def run(self) -> ReplaySuiteManifest:
        """Select evidence-backed assistant cases and write suite artifacts."""
        repository = LocalEvidenceRepository(self.evidence)
        rows = sorted(
            repository.index_rows(),
            key=lambda row: (row.dataset, row.run_id, row.event_id),
        )
        selected: list[AssistantCase] = []
        counts: dict[str, int] = {}
        for row in rows:
            count = counts.get(row.dataset, 0)
            if count >= self.config.cases_per_dataset:
                continue
            counts[row.dataset] = count + 1
            top_variables = list(row.top_variables)
            case_id = _safe_id(f"{row.dataset}__{row.event_id}")
            query = self.config.query_template.format(
                dataset=row.dataset,
                run_id=row.run_id,
                event_id=row.event_id,
                top_variables=", ".join(top_variables) or "ranked variables",
            )
            selected.append(
                AssistantCase(
                    case_id=case_id,
                    dataset=row.dataset,
                    run_id=row.run_id,
                    event_id=row.event_id,
                    query=query,
                    top_variables=top_variables,
                    evidence_relative_path=row.relative_path,
                    expected_retrieval_event_ids=[row.event_id],
                    expected_abstain=False,
                    minimum_supported_claims=self.config.minimum_supported_claims,
                )
            )
        if not selected:
            raise AssistantReplayError(
                f"No evidence rows were available for replay: {self.evidence}"
            )
        manifest = ReplaySuiteManifest(suite_id=self.config.suite_id, cases=selected)
        LocalReplaySuiteRepository(self.out).write_suite(manifest)
        return manifest


class RunAssistantCase:
    """Run one provider-backed assistant case and write assistant replay artifacts."""

    def __init__(
        self,
        *,
        config: AssistantReplayConfig,
        evidence: str | Path,
        out: str | Path,
        case: AssistantCase,
        provider_registry: LLMProviderRegistry,
    ):
        self.config = config
        self.evidence = Path(evidence)
        self.out = Path(out)
        self.case = case
        self.provider_registry = provider_registry

    def run(self) -> AssistantRunMetrics:
        """Run retrieval, generation, deterministic referee checks, and artifact writing."""
        playbooks = Path(self.config.playbooks) if self.config.playbooks is not None else None
        retrieval = RetrieveOperatorEvidence(
            prepared=self.config.prepared,
            evidence=self.evidence,
            query=self.case.query,
            dataset=self.case.dataset,
            event_id=self.case.event_id,
            playbooks=playbooks,
            top_k=self.config.top_k,
        ).run()
        planner_max_tokens = _planner_max_tokens(self.config)
        planner_payload, budget = _planner_payload_for_case(
            self.case,
            retrieval,
            _effective_prompt_budget_chars(self.config, planner_max_tokens),
            max_hit_chars=_planner_hit_max_chars(self.config),
            max_query_chars=_planner_query_max_chars(self.config),
        )
        planner_request = LLMStructuredRequest(
            messages=[
                LLMMessage(role="system", content=PLANNER_SYSTEM_PROMPT),
                LLMMessage(role="user", content=json.dumps(planner_payload, sort_keys=True)),
            ],
            schema_name="DraftResponse",
            json_schema=DraftResponse.model_json_schema(),
            max_tokens=planner_max_tokens,
            metadata={
                "case_id": self.case.case_id,
                "suite_id": self.config.suite_id,
                "prompt_budget": budget,
            },
        )
        provider = self.provider_registry.get(self.config.provider.name).create(
            self.config.provider
        )
        planner_response = provider.generate_json(planner_request)
        draft = DraftResponse.model_validate(planner_response.payload)
        claims, citation_diagnostics = _claims_from_draft_with_diagnostics(
            draft,
            retrieval,
        )
        claim_evaluations: list[ClaimEvaluation] = []
        referee_requests: list[dict[str, Any]] = []
        referee_responses: list[dict[str, Any]] = []
        for claim in claims:
            referee_request = _referee_request_for_claim(
                case=self.case,
                claim=claim,
                retrieval=retrieval,
                config=self.config,
            )
            referee_response = provider.generate_json(referee_request)
            claim_evaluations.append(ClaimEvaluation.model_validate(referee_response.payload))
            referee_requests.append(referee_request.to_dict())
            referee_responses.append(referee_response.to_dict())

        evaluation = EvaluateStructuredAssistantClaims(
            case=self.case,
            retrieval=retrieval,
            claims=claims,
            claim_evaluations=claim_evaluations,
            budget_truncation_count=budget["truncation_count"],
            evidence_pack_overflow_count=budget["overflow_count"],
        ).run()
        repository = LocalAssistantRunRepository(self.out)
        rendered_response = _render_draft_response(draft, claims)
        planner_output = {
            "format_version": "assistant-planner-output-v1",
            "draft_response": draft.model_dump(),
            "claims": evaluation["claims"],
            "citation_selection": citation_diagnostics,
        }
        referee_output = {
            "format_version": "assistant-referee-output-v1",
            "claim_evaluations": evaluation["claim_evaluations"],
            "abstain_flag": evaluation["metrics"].abstain_flag,
            "verified_response_safe": evaluation["metrics"].verified_response_safe,
        }
        run_log = {
            "format_version": "assistant-run-log-v1",
            "run_id": self.case.case_id,
            "case_id": self.case.case_id,
            "provider": provider.name,
            "model": self.config.provider.model,
            "retrieval_hit_count": len(retrieval.hits),
            "planner_claim_count": evaluation["metrics"].total_claims,
            "citation_selection_count": len(citation_diagnostics),
            "abstain_flag": evaluation["metrics"].abstain_flag,
        }
        repository.write_case_run(
            case=self.case,
            retrieval=retrieval.to_dict(),
            provider_request={
                "planner": planner_request.to_dict(),
                "referee": referee_requests,
            },
            provider_response={
                "planner": planner_response.to_dict(),
                "referee": referee_responses,
            },
            planner_output=planner_output,
            referee_output=referee_output,
            run_log=run_log,
            rendered_response=rendered_response,
        )
        return evaluation["metrics"]


class EvaluateStructuredAssistantClaims:
    """Compute assistant replay metrics from planner claims and structured referee decisions."""

    def __init__(
        self,
        *,
        case: AssistantCase,
        retrieval: OperatorRetrievalResult,
        claims: list[DraftClaim],
        claim_evaluations: list[ClaimEvaluation],
        budget_truncation_count: int,
        evidence_pack_overflow_count: int,
    ):
        self.case = case
        self.retrieval = retrieval
        self.claims = claims
        self.claim_evaluations = claim_evaluations
        self.budget_truncation_count = budget_truncation_count
        self.evidence_pack_overflow_count = evidence_pack_overflow_count

    def run(self) -> dict[str, Any]:
        """Return claim rows, referee rows, and per-run metrics."""
        hit_ids = {hit.citation_id for hit in self.retrieval.hits}
        hit_by_id = {hit.citation_id: hit for hit in self.retrieval.hits}
        claim_rows: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        supported = 0
        citation_compliant = 0
        final_document_claims = 0
        final_evidence_claims = 0

        for claim, evaluation in zip(self.claims, self.claim_evaluations, strict=False):
            cited = [citation for citation in claim.cited_evidence_ids if citation in hit_ids]
            valid_citations = bool(cited) and set(claim.cited_evidence_ids).issubset(hit_ids)
            kept = evaluation.final_disposition in {"keep", "rewrite"}
            supported_claim = bool(evaluation.is_supported and kept and valid_citations)
            supported += int(supported_claim)
            citation_compliant += int(valid_citations)
            source_types = {
                hit_by_id[citation].source_type for citation in cited if citation in hit_by_id
            }
            final_document_claims += int(supported_claim and "playbook" in source_types)
            final_evidence_claims += int(supported_claim and "evidence_bundle" in source_types)
            claim_rows.append(
                {
                    "claim_id": claim.claim_id,
                    "section": claim.section,
                    "statement": claim.statement,
                    "cited_evidence_ids": claim.cited_evidence_ids,
                }
            )
            evaluations.append(
                {
                    "claim_id": claim.claim_id,
                    "statement": claim.statement,
                    "citation_compliant": valid_citations,
                    "supported_evidence_ids": cited if supported_claim else [],
                    **evaluation.model_dump(),
                }
            )

        total_claims = len(self.claims)
        abstain_flag = total_claims == 0
        safe = abstain_flag or supported == total_claims
        expected_events = set(self.case.expected_retrieval_event_ids)
        retrieved_events = {
            str(hit.event_id) for hit in self.retrieval.hits if hit.event_id is not None
        }
        retrieval_met = expected_events.issubset(retrieved_events) if expected_events else None
        abstain_matched = abstain_flag == self.case.expected_abstain
        minimum_met = supported >= self.case.minimum_supported_claims
        metrics = AssistantRunMetrics(
            run_id=self.case.case_id,
            case_id=self.case.case_id,
            total_claims=total_claims,
            supported_claims=supported,
            citation_compliant_claims=citation_compliant,
            retrieved_document_hit_count=sum(
                1 for hit in self.retrieval.hits if hit.source_type == "playbook"
            ),
            final_supported_document_claims=final_document_claims,
            final_supported_evidence_bundle_claims=final_evidence_claims,
            abstain_flag=abstain_flag,
            verified_response_safe=safe,
            retrieval_expectations_met=retrieval_met,
            abstain_expectation_matched=abstain_matched,
            minimum_supported_claim_met=minimum_met,
            evidence_pack_overflow_count=self.evidence_pack_overflow_count,
            budget_truncation_count=self.budget_truncation_count,
            propositional_alignment_proxy=_ratio(supported, total_claims),
            citation_compliance_proxy=_ratio(citation_compliant, total_claims),
            verified_response_safety_proxy=1.0 if safe else 0.0,
        )
        return {"claims": claim_rows, "claim_evaluations": evaluations, "metrics": metrics}


class EvaluateAssistantClaims:
    """Compute deterministic claim/citation metrics for one assistant response."""

    def __init__(
        self,
        *,
        case: AssistantCase,
        retrieval: OperatorRetrievalResult,
        response_text: str,
        budget_truncation_count: int,
        evidence_pack_overflow_count: int,
    ):
        self.case = case
        self.retrieval = retrieval
        self.response_text = response_text
        self.budget_truncation_count = budget_truncation_count
        self.evidence_pack_overflow_count = evidence_pack_overflow_count

    def run(self) -> dict[str, Any]:
        """Return claim rows, referee rows, and per-run metrics."""
        claims = _extract_claims(self.response_text)
        hit_ids = {hit.citation_id for hit in self.retrieval.hits}
        hit_by_id = {hit.citation_id: hit for hit in self.retrieval.hits}
        claim_rows: list[dict[str, Any]] = []
        evaluations: list[dict[str, Any]] = []
        supported = 0
        citation_compliant = 0
        final_document_claims = 0
        final_evidence_claims = 0

        for index, claim in enumerate(claims, start=1):
            cited = CITATION_RE.findall(claim)
            valid = bool(cited) and set(cited).issubset(hit_ids)
            supported += int(valid)
            citation_compliant += int(valid)
            source_types = {
                hit_by_id[citation].source_type for citation in cited if citation in hit_by_id
            }
            final_document_claims += int(valid and "playbook" in source_types)
            final_evidence_claims += int(valid and "evidence_bundle" in source_types)
            claim_id = f"claim-{index}"
            claim_rows.append(
                {
                    "claim_id": claim_id,
                    "statement": claim,
                    "cited_evidence_ids": cited,
                }
            )
            evaluations.append(
                {
                    "claim_id": claim_id,
                    "statement": claim,
                    "is_supported": valid,
                    "citation_compliant": valid,
                    "supported_evidence_ids": cited if valid else [],
                    "final_disposition": "keep" if valid else "remove",
                }
            )

        total_claims = len(claims)
        abstain_flag = _abstained(self.response_text, total_claims)
        safe = abstain_flag or supported == total_claims
        expected_events = set(self.case.expected_retrieval_event_ids)
        retrieved_events = {
            str(hit.event_id) for hit in self.retrieval.hits if hit.event_id is not None
        }
        retrieval_met = expected_events.issubset(retrieved_events) if expected_events else None
        abstain_matched = abstain_flag == self.case.expected_abstain
        minimum_met = supported >= self.case.minimum_supported_claims
        metrics = AssistantRunMetrics(
            run_id=self.case.case_id,
            case_id=self.case.case_id,
            total_claims=total_claims,
            supported_claims=supported,
            citation_compliant_claims=citation_compliant,
            retrieved_document_hit_count=sum(
                1 for hit in self.retrieval.hits if hit.source_type == "playbook"
            ),
            final_supported_document_claims=final_document_claims,
            final_supported_evidence_bundle_claims=final_evidence_claims,
            abstain_flag=abstain_flag,
            verified_response_safe=safe,
            retrieval_expectations_met=retrieval_met,
            abstain_expectation_matched=abstain_matched,
            minimum_supported_claim_met=minimum_met,
            evidence_pack_overflow_count=self.evidence_pack_overflow_count,
            budget_truncation_count=self.budget_truncation_count,
            propositional_alignment_proxy=_ratio(supported, total_claims),
            citation_compliance_proxy=_ratio(citation_compliant, total_claims),
            verified_response_safety_proxy=1.0 if safe else 0.0,
        )
        return {"claims": claim_rows, "claim_evaluations": evaluations, "metrics": metrics}


class RunAssistantReplaySuite:
    """Run a full assistant replay suite over evidence artifacts."""

    def __init__(
        self,
        *,
        config: AssistantReplayConfig,
        evidence: str | Path,
        out: str | Path,
        provider_registry: LLMProviderRegistry,
        benchmark: str | Path | None = None,
        progress_sink: ProgressSink | None = None,
    ):
        self.config = config
        self.evidence = Path(evidence)
        self.out = Path(out)
        self.provider_registry = provider_registry
        self.benchmark = Path(benchmark) if benchmark is not None else None
        self.progress_sink = progress_sink

    def run(self) -> AssistantReplayRunResult:
        """Build and execute a replay suite, then write aggregate metrics."""
        if (self.out / "assistant_summary.json").exists():
            raise AssistantReplayError(f"assistant replay run already exists: {self.out}")
        preflight = PreflightAssistantReplay(
            config=self.config, provider_registry=self.provider_registry
        ).run()
        if not preflight.ok:
            raise AssistantReplayError(f"assistant replay preflight failed: {preflight.to_dict()}")
        writer = LocalArtifactWriter(self.out)
        writer.write_json("preflight.json", preflight.to_dict())
        writer.write_json(
            "resolved_config.json",
            {
                "assistant": self.config.to_dict(),
                "benchmark": str(self.benchmark) if self.benchmark is not None else None,
                "evidence": str(self.evidence),
            },
        )
        manifest = BuildReplaySuite(config=self.config, evidence=self.evidence, out=self.out).run()
        progress = CompositeProgressSink(
            [LocalProgressSink(self.out, self.config.suite_id), self.progress_sink]
        )
        per_run: list[AssistantRunMetrics] = []
        for ordinal, case in enumerate(manifest.cases, start=1):
            progress.emit(
                ProgressEvent(
                    run_id=self.config.suite_id,
                    stage="assistant_case",
                    item_id=case.case_id,
                    status="planned",
                    ordinal=ordinal,
                    total=len(manifest.cases),
                    path=str(self.out / "runs" / case.case_id),
                )
            )
        for ordinal, case in enumerate(manifest.cases, start=1):
            started = time.perf_counter()
            progress.emit(
                ProgressEvent(
                    run_id=self.config.suite_id,
                    stage="assistant_case",
                    item_id=case.case_id,
                    status="running",
                    ordinal=ordinal,
                    total=len(manifest.cases),
                    path=str(self.out / "runs" / case.case_id),
                )
            )
            try:
                case_metrics = RunAssistantCase(
                    config=self.config,
                    evidence=self.evidence,
                    out=self.out,
                    case=case,
                    provider_registry=self.provider_registry,
                ).run()
                per_run.append(case_metrics)
                progress.emit(
                    ProgressEvent(
                        run_id=self.config.suite_id,
                        stage="assistant_case",
                        item_id=case.case_id,
                        status="completed" if case_metrics.verified_response_safe else "failed",
                        ordinal=ordinal,
                        total=len(manifest.cases),
                        path=str(self.out / "runs" / case.case_id),
                        duration_s=round(time.perf_counter() - started, 6),
                        metrics={
                            "supported_claims": case_metrics.supported_claims,
                            "total_claims": case_metrics.total_claims,
                            "safe": case_metrics.verified_response_safe,
                        },
                    )
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                _write_failed_case_artifacts(self.out, case, error)
                case_metrics = _failed_case_metric(case, error)
                per_run.append(case_metrics)
                progress.emit(
                    ProgressEvent(
                        run_id=self.config.suite_id,
                        stage="assistant_case",
                        item_id=case.case_id,
                        status="failed",
                        ordinal=ordinal,
                        total=len(manifest.cases),
                        path=str(self.out / "runs" / case.case_id),
                        duration_s=round(time.perf_counter() - started, 6),
                        metrics={
                            "supported_claims": case_metrics.supported_claims,
                            "total_claims": case_metrics.total_claims,
                            "safe": case_metrics.verified_response_safe,
                        },
                        error=error,
                    )
                )
        aggregate_metrics = aggregate_assistant_metrics(self.config.suite_id, per_run)
        LocalAssistantReplayMetricsRepository(self.out).write_summary(aggregate_metrics)
        if self.config.include_operator_cards:
            _write_operator_cards(self.config, self.evidence, self.out)
        return AssistantReplayRunResult(
            suite_id=self.config.suite_id,
            run_dir=str(self.out),
            ok=all(item.verified_response_safe for item in per_run),
            metrics=aggregate_metrics,
        )


class SummarizeAssistantReplay:
    """Read and summarize an assistant replay run directory."""

    def __init__(self, run: str | Path):
        self.run = Path(run)

    def run_summary(self) -> dict[str, Any]:
        """Return the JSON summary payload."""
        return LocalAssistantReplayMetricsRepository(self.run).read_summary()


def summary_csv_from_runs(rows: list[dict[str, Any]]) -> str:
    """Render combined assistant replay summary rows as CSV."""
    if not rows:
        return ""
    fieldnames = sorted({key for row in rows for key in row})
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return handle.getvalue()


def _planner_payload_for_case(
    case: AssistantCase,
    retrieval: OperatorRetrievalResult,
    prompt_budget_chars: int,
    *,
    max_hit_chars: int | None = None,
    max_query_chars: int | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    query = case.query.strip()
    query_truncated = 0
    if max_query_chars is not None and len(query) > max_query_chars:
        query = query[:max_query_chars].rstrip() + "..."
        query_truncated = 1
    compact_case = {
        "case_id": case.case_id,
        "dataset": case.dataset,
        "run_id": case.run_id,
        "event_id": case.event_id,
        "top_variables": case.top_variables,
    }
    evidence_summary = {
        "dataset": case.dataset,
        "run_id": case.run_id,
        "event_id": case.event_id,
        "top_variables": case.top_variables,
        "retrieval_hit_count": 0,
    }
    used_chars = len(
        json.dumps(
            {
                "task": query,
                "case": compact_case,
                "evidence_summary": evidence_summary,
                "retrieval_hits": [],
            },
            sort_keys=True,
        )
    )
    truncation_count = 0
    overflow_count = 0
    hits: list[dict[str, Any]] = []
    for hit in retrieval.hits:
        text = hit.text.strip()
        allowance = prompt_budget_chars - used_chars - 256
        if allowance <= 0:
            overflow_count += 1
            continue
        text_limit = allowance
        if max_hit_chars is not None:
            text_limit = min(text_limit, max_hit_chars)
        if len(text) > text_limit:
            text = text[: max(text_limit, 0)].rstrip() + "..."
            truncation_count += 1
        payload = {
            "citation_id": hit.citation_id,
            "source_type": hit.source_type,
            "title": hit.title,
            "role": hit.role,
            "rank": hit.rank,
            "dataset": hit.dataset,
            "event_id": hit.event_id,
            "run_id": hit.run_id,
            "text": text,
        }
        hits.append(payload)
        used_chars += len(json.dumps(payload, sort_keys=True))
    evidence_summary["retrieval_hit_count"] = len(hits)
    return (
        {
            "task": query,
            "case": compact_case,
            "evidence_summary": evidence_summary,
            "retrieval_hits": hits,
        },
        {
            "effective_prompt_budget_chars": prompt_budget_chars,
            "truncation_count": truncation_count,
            "overflow_count": overflow_count,
            "query_truncated": query_truncated,
        },
    )


def _planner_max_tokens(config: AssistantReplayConfig) -> int:
    extra = config.provider.extra
    default_max = 256 if config.provider.name == "llama-cpp" else config.provider.max_tokens
    configured = int(extra.get("planner_max_tokens", default_max))
    return max(64, min(config.provider.max_tokens, configured))


def _referee_max_tokens(config: AssistantReplayConfig) -> int:
    extra = config.provider.extra
    default_max = 128 if config.provider.name == "llama-cpp" else 512
    configured = int(extra.get("referee_max_tokens", default_max))
    return max(64, min(config.provider.max_tokens, configured))


def _planner_hit_max_chars(config: AssistantReplayConfig) -> int | None:
    value = config.provider.extra.get("hit_max_chars")
    if value is None and config.provider.name == "llama-cpp":
        return 300
    if value is None:
        return None
    return max(120, int(value))


def _planner_query_max_chars(config: AssistantReplayConfig) -> int | None:
    value = config.provider.extra.get("planner_query_max_chars")
    if value is None and config.provider.name == "llama-cpp":
        return 360
    if value is None:
        return None
    return max(120, int(value))


def _effective_prompt_budget_chars(
    config: AssistantReplayConfig,
    max_tokens: int,
) -> int:
    requested = config.prompt_budget_chars
    extra = config.provider.extra
    context_window = int(extra.get("context_window_tokens", 4096))
    if config.provider.name != "llama-cpp" and "context_window_tokens" not in extra:
        return requested
    reserve_tokens = int(extra.get("prompt_context_reserve_tokens", 768))
    chars_per_token = float(extra.get("prompt_chars_per_token", 1.4))
    available_tokens = context_window - max_tokens - reserve_tokens
    context_budget = max(2000, int(max(available_tokens, 1) * chars_per_token))
    return min(requested, context_budget)


def _claims_from_draft(
    draft: DraftResponse,
    retrieval: OperatorRetrievalResult,
) -> list[DraftClaim]:
    """Extract claims and assign evidence citations."""
    claims, _diagnostics = _claims_from_draft_with_diagnostics(draft, retrieval)
    return claims


def _claims_from_draft_with_diagnostics(
    draft: DraftResponse,
    retrieval: OperatorRetrievalResult,
) -> tuple[list[DraftClaim], list[dict[str, Any]]]:
    rows: list[tuple[str, str]] = []
    if draft.symptom_summary.strip():
        rows.append(("symptom_summary", draft.symptom_summary.strip()))
    for section, items in (
        ("likely_causes", draft.likely_causes),
        ("checks", draft.checks),
        ("recommended_actions", draft.recommended_actions),
        ("escalation_criteria", draft.escalation_criteria),
    ):
        rows.extend((section, item.strip()) for item in items if item.strip())

    claims: list[DraftClaim] = []
    diagnostics: list[dict[str, Any]] = []
    hit_ids = {hit.citation_id for hit in retrieval.hits}
    for index, (section, statement) in enumerate(rows, start=1):
        explicit = [citation for citation in CITATION_RE.findall(statement) if citation in hit_ids]
        clean_statement = CITATION_RE.sub("", statement).strip()
        if explicit:
            citations = explicit
            diagnostic = {
                "claim_id": f"claim-{index}",
                "mode": "explicit",
                "statement": clean_statement,
                "selected_citations": citations,
                "matched_tags": sorted(_tag_tokens(clean_statement)),
                "candidates": [],
            }
        else:
            citations, diagnostic = _select_citations_with_diagnostics(
                clean_statement,
                retrieval,
                claim_id=f"claim-{index}",
            )
        claim = DraftClaim(
            claim_id=f"claim-{index}",
            section=section,
            statement=clean_statement,
            cited_evidence_ids=citations,
        )
        claims.append(claim)
        diagnostics.append({**diagnostic, "section": section})
    return claims, diagnostics


def _select_citations(statement: str, retrieval: OperatorRetrievalResult) -> list[str]:
    """Select citations for a claim statement."""
    citations, _diagnostics = _select_citations_with_diagnostics(statement, retrieval)
    return citations


def _select_citations_with_diagnostics(
    statement: str,
    retrieval: OperatorRetrievalResult,
    *,
    claim_id: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    statement_tokens = _tokens(statement)
    statement_tags = _tag_tokens(statement)
    candidates: list[dict[str, Any]] = []
    for hit in retrieval.hits:
        hit_text = " ".join(
            [hit.title, hit.role, hit.text, json.dumps(hit.metadata, sort_keys=True)]
        )
        hit_tokens = _tokens(hit_text)
        hit_tags = _tag_tokens(hit_text)
        matched_tags = sorted(statement_tags & hit_tags)
        overlap = len(statement_tokens & hit_tokens)
        role_priority = TAG_ROLE_PRIORITY.get(hit.role, 20)
        direct_tag_match = bool(matched_tags)
        candidates.append(
            {
                "citation_id": hit.citation_id,
                "role": hit.role,
                "rank": hit.rank,
                "retrieval_score": hit.score,
                "matched_tags": matched_tags,
                "token_overlap": overlap,
                "direct_tag_match": direct_tag_match,
                "role_priority": role_priority,
                "sort_key": (
                    1 if direct_tag_match else 0,
                    -role_priority,
                    overlap,
                    float(hit.score),
                    -hit.rank,
                ),
            }
        )

    direct = [candidate for candidate in candidates if candidate["direct_tag_match"]]
    selected_pool = direct or candidates
    selected_pool.sort(key=lambda item: item["sort_key"], reverse=True)
    max_citations = 3 if direct else 1
    selected = [str(item["citation_id"]) for item in selected_pool[:max_citations]]
    diagnostics = {
        "claim_id": claim_id,
        "mode": "direct_tag_match" if direct else "token_overlap",
        "statement": statement,
        "check_target": _check_target(statement),
        "matched_tags": sorted(statement_tags),
        "selected_citations": selected,
        "candidates": [
            {key: value for key, value in candidate.items() if key != "sort_key"}
            for candidate in sorted(candidates, key=lambda item: item["sort_key"], reverse=True)
        ],
    }
    return selected, diagnostics


def _referee_request_for_claim(
    *,
    case: AssistantCase,
    claim: DraftClaim,
    retrieval: OperatorRetrievalResult,
    config: AssistantReplayConfig,
) -> LLMStructuredRequest:
    hit_by_id = {hit.citation_id: hit for hit in retrieval.hits}
    cited_hits = [
        hit_by_id[citation].to_dict()
        for citation in claim.cited_evidence_ids
        if citation in hit_by_id
    ]
    supporting_facts = _supporting_facts_for_claim(claim, cited_hits)
    payload = {
        "case": case.to_dict(),
        "claim": claim.model_dump(),
        "cited_evidence": cited_hits,
        "supporting_facts": supporting_facts,
    }
    return LLMStructuredRequest(
        messages=[
            LLMMessage(role="system", content=REFEREE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=json.dumps(payload, sort_keys=True)),
        ],
        schema_name="ClaimEvaluation",
        json_schema=ClaimEvaluation.model_json_schema(),
        max_tokens=_referee_max_tokens(config),
        metadata={
            "case_id": case.case_id,
            "claim_id": claim.claim_id,
            "suite_id": config.suite_id,
        },
    )


def _supporting_facts_for_claim(
    claim: DraftClaim,
    cited_hits: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_tags = _tag_tokens(claim.statement)
    matches: list[dict[str, Any]] = []
    for hit in cited_hits:
        hit_text = " ".join(
            [
                str(hit.get("title", "")),
                str(hit.get("role", "")),
                str(hit.get("text", "")),
                json.dumps(hit.get("metadata", {}), sort_keys=True),
            ]
        )
        matched_tags = sorted(claim_tags & _tag_tokens(hit_text))
        if matched_tags:
            matches.append(
                {
                    "citation_id": str(hit.get("citation_id", "")),
                    "role": str(hit.get("role", "")),
                    "matched_tags": matched_tags,
                    "source_type": str(hit.get("source_type", "")),
                }
            )
    return {
        "bounded_variable_inspection": _is_bounded_variable_inspection(claim),
        "check_target": _check_target(claim.statement),
        "claim_tags": sorted(claim_tags),
        "matched_citations": matches,
        "exact_top_variable_citations": [
            match["citation_id"] for match in matches if match["role"] == "top_variables"
        ],
    }


def _render_draft_response(draft: DraftResponse, claims: list[DraftClaim]) -> str:
    by_section: dict[str, list[DraftClaim]] = {}
    for claim in claims:
        by_section.setdefault(claim.section, []).append(claim)
    lines = ["# Assistant Draft", ""]
    for section in (
        "symptom_summary",
        "likely_causes",
        "checks",
        "recommended_actions",
        "escalation_criteria",
    ):
        items = by_section.get(section, [])
        if not items:
            continue
        lines.extend([f"## {section.replace('_', ' ').title()}", ""])
        for claim in items:
            citations = " ".join(f"[{citation}]" for citation in claim.cited_evidence_ids)
            lines.append(f"- {claim.statement} {citations}".rstrip())
        lines.append("")
    if len(lines) == 2 and not draft.model_dump(exclude_defaults=True):
        lines.append("I must abstain because no sufficient cited evidence was provided.")
    return "\n".join(lines).rstrip() + "\n"


def _prompt_for_case(
    case: AssistantCase,
    retrieval: OperatorRetrievalResult,
    prompt_budget_chars: int,
) -> tuple[str, dict[str, int]]:
    sections = [
        f"Query: {case.query}",
        "",
        "Retrieved evidence:",
    ]
    used_chars = sum(len(section) for section in sections)
    truncation_count = 0
    overflow_count = 0
    for hit in retrieval.hits:
        header = f"[{hit.citation_id}] {hit.source_type}/{hit.role}: {hit.title}"
        text = hit.text.strip()
        allowance = prompt_budget_chars - used_chars - len(header) - 16
        if allowance <= 0:
            overflow_count += 1
            continue
        if len(text) > allowance:
            text = text[: max(allowance, 0)].rstrip() + "..."
            truncation_count += 1
        block = f"{header}\n{text}"
        sections.append(block)
        used_chars += len(block)
    sections.extend(
        [
            "",
            "Return concise bullet points. Every non-abstention bullet must cite [C#].",
        ]
    )
    return "\n".join(sections), {
        "truncation_count": truncation_count,
        "overflow_count": overflow_count,
    }


def _extract_claims(text: str) -> list[str]:
    claims: list[str] = []
    for match in CLAIM_RE.finditer(text):
        claim = match.group(1).strip()
        if not claim or claim.lower().startswith("i must abstain"):
            continue
        if len(_tokens(claim)) >= 3:
            claims.append(claim)
    return claims


def _abstained(text: str, total_claims: int) -> bool:
    lowered = text.lower()
    return "abstain" in lowered and total_claims == 0


def _tokens(text: str) -> set[str]:
    return {
        normalized for token in TOKEN_RE.findall(text) if (normalized := _normalize_token(token))
    }


def _tag_tokens(text: str) -> set[str]:
    return {token for token in _tokens(text) if "/" in token or token.startswith("plant")}


def _normalize_token(token: str) -> str:
    normalized = token.strip().strip(".,;:!?)]}\"'`").strip("([{\"'`").lower()
    return normalized


def _check_target(statement: str) -> str | None:
    match = CHECK_TARGET_RE.match(statement)
    if match is None:
        return None
    return _normalize_token(match.group(1))


def _is_bounded_variable_inspection(claim: DraftClaim) -> bool:
    return claim.section == "checks" and _check_target(claim.statement) is not None


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _failed_case_metric(case: AssistantCase, error: str) -> AssistantRunMetrics:
    return AssistantRunMetrics(
        run_id=case.case_id,
        case_id=case.case_id,
        total_claims=0,
        supported_claims=0,
        citation_compliant_claims=0,
        retrieved_document_hit_count=0,
        final_supported_document_claims=0,
        final_supported_evidence_bundle_claims=0,
        abstain_flag=True,
        verified_response_safe=False,
        retrieval_expectations_met=False,
        abstain_expectation_matched=case.expected_abstain,
        minimum_supported_claim_met=False,
        evidence_pack_overflow_count=0,
        budget_truncation_count=0,
        propositional_alignment_proxy=0.0,
        citation_compliance_proxy=0.0,
        verified_response_safety_proxy=0.0,
    )


def _write_failed_case_artifacts(out: Path, case: AssistantCase, error: str) -> None:
    repository = LocalAssistantRunRepository(out)
    repository.write_case_run(
        case=case,
        retrieval={
            "format_version": "assistant-retrieval-unavailable-v1",
            "hits": [],
            "error": error,
        },
        provider_request={
            "format_version": "assistant-provider-request-unavailable-v1",
            "error": error,
        },
        provider_response={
            "format_version": "assistant-provider-response-error-v1",
            "error": error,
        },
        planner_output={
            "format_version": "assistant-planner-output-v1",
            "status": "failed",
            "error": error,
        },
        referee_output={
            "format_version": "assistant-referee-output-v1",
            "status": "failed",
            "error": error,
        },
        run_log={
            "format_version": "assistant-run-log-v1",
            "run_id": case.case_id,
            "case_id": case.case_id,
            "status": "failed",
            "error": error,
        },
        rendered_response=f"# Assistant Replay Failure\n\n{error}\n",
    )


def _write_operator_cards(config: AssistantReplayConfig, evidence: Path, out: Path) -> None:
    cards_root = out / "operator_cards"
    # The deterministic card writer remains a rendering artifact, not the assistant replay metric source.
    from industrial_tsad_eval.application.operator import GenerateOperatorCards

    GenerateOperatorCards(
        prepared=config.prepared,
        evidence=evidence,
        out=cards_root,
        playbooks=Path(config.playbooks) if config.playbooks is not None else None,
        max_cards=config.cases_per_dataset,
    ).run()
    manifest = read_json(cards_root / "manifest.json")
    LocalOperatorCardRepository(cards_root).manifest()
    LocalArtifactWriter(out).write_json("operator_cards_manifest.json", manifest)


def _safe_id(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )
    return cleaned.strip("._-") or "item"
