"""Thesis-style RQ3 assistant replay services."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.operator import RetrieveOperatorEvidence
from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.domain.errors import RQ3RunError
from industrial_tsad_eval.domain.llm import LLMMessage, LLMRequest
from industrial_tsad_eval.domain.operator import OperatorRetrievalResult
from industrial_tsad_eval.domain.rq3 import (
    AssistantCase,
    AssistantRunMetrics,
    ReplaySuiteManifest,
    RQ3Config,
    RQ3RunResult,
    aggregate_rq3_metrics,
)
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.evidence_repository import LocalEvidenceRepository
from industrial_tsad_eval.infrastructure.json_utils import read_json
from industrial_tsad_eval.infrastructure.operator_repository import LocalOperatorCardRepository
from industrial_tsad_eval.infrastructure.rq3_repository import (
    LocalAssistantRunRepository,
    LocalReplaySuiteRepository,
    LocalRQ3MetricsRepository,
)
from industrial_tsad_eval.plugins.providers import LLMProviderRegistry

CLAIM_RE = re.compile(r"(?:^|\n)\s*(?:[-*]|\d+[.)])?\s*([^\n]+)")
CITATION_RE = re.compile(r"\[(C\d+)\]")
TOKEN_RE = re.compile(r"[A-Za-z0-9_:/.-]+")


@dataclass(frozen=True)
class RQ3PreflightResult:
    """RQ3 preflight status."""

    ok: bool
    checks: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {"ok": self.ok, "checks": [dict(check) for check in self.checks]}


class PreflightRQ3:
    """Validate RQ3 configuration, provider readiness, and prepared inputs."""

    def __init__(self, *, config: RQ3Config, provider_registry: LLMProviderRegistry):
        self.config = config
        self.provider_registry = provider_registry

    def run(self) -> RQ3PreflightResult:
        """Run RQ3 preflight checks."""
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
        return RQ3PreflightResult(ok=ok, checks=checks)


class BuildReplaySuite:
    """Build a deterministic replay suite from Evidence Bundle v1 artifacts."""

    def __init__(self, *, config: RQ3Config, evidence: str | Path, out: str | Path):
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
            raise RQ3RunError(f"No evidence rows were available for replay: {self.evidence}")
        manifest = ReplaySuiteManifest(suite_id=self.config.suite_id, cases=selected)
        LocalReplaySuiteRepository(self.out).write_suite(manifest)
        return manifest


class RunAssistantCase:
    """Run one provider-backed assistant case and write RQ3 artifacts."""

    def __init__(
        self,
        *,
        config: RQ3Config,
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
        prompt, budget = _prompt_for_case(self.case, retrieval, self.config.prompt_budget_chars)
        request = LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You are an industrial anomaly-detection assistant. Use only cited "
                        "retrieved evidence. Every factual claim must cite one or more [C#] ids. "
                        "Abstain when evidence is insufficient."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ],
            metadata={"case_id": self.case.case_id, "suite_id": self.config.suite_id},
        )
        provider = self.provider_registry.get(self.config.provider.name).create(
            self.config.provider
        )
        response = provider.generate(request)
        evaluation = EvaluateAssistantClaims(
            case=self.case,
            retrieval=retrieval,
            response_text=response.text,
            budget_truncation_count=budget["truncation_count"],
            evidence_pack_overflow_count=budget["overflow_count"],
        ).run()
        repository = LocalAssistantRunRepository(self.out)
        planner_output = {
            "format_version": "rq3-planner-output-v1",
            "response_text": response.text,
            "claims": evaluation["claims"],
        }
        referee_output = {
            "format_version": "rq3-referee-output-v1",
            "claim_evaluations": evaluation["claim_evaluations"],
            "abstain_flag": evaluation["metrics"].abstain_flag,
            "verified_response_safe": evaluation["metrics"].verified_response_safe,
        }
        run_log = {
            "format_version": "rq3-run-log-v1",
            "run_id": self.case.case_id,
            "case_id": self.case.case_id,
            "provider": response.provider,
            "model": response.model,
            "retrieval_hit_count": len(retrieval.hits),
            "planner_claim_count": evaluation["metrics"].total_claims,
            "abstain_flag": evaluation["metrics"].abstain_flag,
        }
        repository.write_case_run(
            case=self.case,
            retrieval=retrieval.to_dict(),
            provider_request=request.to_dict(),
            provider_response=response.to_dict(),
            planner_output=planner_output,
            referee_output=referee_output,
            run_log=run_log,
            rendered_response=response.text,
        )
        return evaluation["metrics"]


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


class RunReplaySuite:
    """Run a full RQ3 replay suite over evidence artifacts."""

    def __init__(
        self,
        *,
        config: RQ3Config,
        evidence: str | Path,
        out: str | Path,
        provider_registry: LLMProviderRegistry,
        benchmark: str | Path | None = None,
    ):
        self.config = config
        self.evidence = Path(evidence)
        self.out = Path(out)
        self.provider_registry = provider_registry
        self.benchmark = Path(benchmark) if benchmark is not None else None

    def run(self) -> RQ3RunResult:
        """Build and execute a replay suite, then write aggregate metrics."""
        if (self.out / "rq3_summary.json").exists():
            raise RQ3RunError(f"RQ3 run already exists: {self.out}")
        preflight = PreflightRQ3(config=self.config, provider_registry=self.provider_registry).run()
        if not preflight.ok:
            raise RQ3RunError(f"RQ3 preflight failed: {preflight.to_dict()}")
        writer = LocalArtifactWriter(self.out)
        writer.write_json("preflight.json", preflight.to_dict())
        writer.write_json(
            "resolved_config.json",
            {
                "rq3": self.config.to_dict(),
                "benchmark": str(self.benchmark) if self.benchmark is not None else None,
                "evidence": str(self.evidence),
            },
        )
        manifest = BuildReplaySuite(config=self.config, evidence=self.evidence, out=self.out).run()
        per_run: list[AssistantRunMetrics] = []
        for case in manifest.cases:
            try:
                per_run.append(
                    RunAssistantCase(
                        config=self.config,
                        evidence=self.evidence,
                        out=self.out,
                        case=case,
                        provider_registry=self.provider_registry,
                    ).run()
                )
            except Exception as exc:
                per_run.append(_failed_case_metric(case, f"{type(exc).__name__}: {exc}"))
        metrics = aggregate_rq3_metrics(self.config.suite_id, per_run)
        LocalRQ3MetricsRepository(self.out).write_summary(metrics)
        if self.config.include_operator_cards:
            _write_operator_cards(self.config, self.evidence, self.out)
        return RQ3RunResult(
            suite_id=self.config.suite_id,
            run_dir=str(self.out),
            ok=all(item.verified_response_safe for item in per_run),
            metrics=metrics,
        )


class SummarizeRQ3:
    """Read and summarize an RQ3 run directory."""

    def __init__(self, run: str | Path):
        self.run = Path(run)

    def run_summary(self) -> dict[str, Any]:
        """Return the JSON summary payload."""
        return LocalRQ3MetricsRepository(self.run).read_summary()


def summary_csv_from_runs(rows: list[dict[str, Any]]) -> str:
    """Render combined RQ3 summary rows as CSV."""
    if not rows:
        return ""
    fieldnames = sorted({key for row in rows for key in row})
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return handle.getvalue()


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
    return {token.lower() for token in TOKEN_RE.findall(text) if token.strip()}


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


def _write_operator_cards(config: RQ3Config, evidence: Path, out: Path) -> None:
    cards_root = out / "operator_cards"
    # The deterministic card writer remains a rendering artifact, not the RQ3 metric source.
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
