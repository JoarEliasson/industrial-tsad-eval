"""RQ3 assistant-replay contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from industrial_tsad_eval.domain.errors import BenchmarkConfigError
from industrial_tsad_eval.domain.llm import LLMProviderConfig


@dataclass(frozen=True)
class RQ3Config:
    """Configuration for thesis-style assistant replay suites."""

    suite_id: str
    prepared: str
    provider: LLMProviderConfig
    query_template: str = (
        "For dataset {dataset}, event {event_id}: summarize likely causes, first "
        "operator checks, and immediate actions using only cited retrieved evidence. "
        "Prioritize {top_variables}."
    )
    cases_per_dataset: int = 4
    top_k: int = 8
    minimum_supported_claims: int = 1
    prompt_budget_chars: int = 12000
    playbooks: str | None = None
    include_operator_cards: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> RQ3Config:
        """Build an RQ3 config from TOML-compatible data."""
        rq3 = _required_mapping(payload, "rq3")
        provider_payload = _required_mapping(rq3, "provider")
        suite_id = _string(rq3, "suite_id", "rq3-smoke")
        prepared = _required_string(rq3, "prepared", "rq3.prepared")
        cases_per_dataset = int(rq3.get("cases_per_dataset", 4))
        top_k = int(rq3.get("top_k", 8))
        minimum_supported_claims = int(rq3.get("minimum_supported_claims", 1))
        prompt_budget_chars = int(rq3.get("prompt_budget_chars", 12000))
        if cases_per_dataset <= 0:
            raise BenchmarkConfigError("rq3.cases_per_dataset must be greater than 0.")
        if top_k <= 0:
            raise BenchmarkConfigError("rq3.top_k must be greater than 0.")
        if minimum_supported_claims < 0:
            raise BenchmarkConfigError("rq3.minimum_supported_claims cannot be negative.")
        if prompt_budget_chars <= 0:
            raise BenchmarkConfigError("rq3.prompt_budget_chars must be greater than 0.")
        return cls(
            suite_id=suite_id,
            prepared=prepared,
            provider=_provider_config(provider_payload),
            query_template=_string(rq3, "query_template", cls.query_template),
            cases_per_dataset=cases_per_dataset,
            top_k=top_k,
            minimum_supported_claims=minimum_supported_claims,
            prompt_budget_chars=prompt_budget_chars,
            playbooks=_optional_string(rq3.get("playbooks")),
            include_operator_cards=bool(rq3.get("include_operator_cards", False)),
        )

    def with_prepared(self, prepared: str) -> RQ3Config:
        """Return a copy targeting a concrete prepared dataset."""
        return RQ3Config(
            suite_id=self.suite_id,
            prepared=prepared,
            provider=self.provider,
            query_template=self.query_template,
            cases_per_dataset=self.cases_per_dataset,
            top_k=self.top_k,
            minimum_supported_claims=self.minimum_supported_claims,
            prompt_budget_chars=self.prompt_budget_chars,
            playbooks=self.playbooks,
            include_operator_cards=self.include_operator_cards,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "suite_id": self.suite_id,
            "prepared": self.prepared,
            "query_template": self.query_template,
            "cases_per_dataset": self.cases_per_dataset,
            "top_k": self.top_k,
            "minimum_supported_claims": self.minimum_supported_claims,
            "prompt_budget_chars": self.prompt_budget_chars,
            "playbooks": self.playbooks,
            "include_operator_cards": self.include_operator_cards,
            "provider": self.provider.to_dict(),
        }


@dataclass(frozen=True)
class AssistantCase:
    """One event-level assistant replay case."""

    case_id: str
    dataset: str
    run_id: str
    event_id: str
    query: str
    top_variables: list[str]
    evidence_relative_path: str
    expected_retrieval_event_ids: list[str] = field(default_factory=list)
    expected_abstain: bool = False
    minimum_supported_claims: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AssistantCase:
        """Parse one assistant case."""
        return cls(
            case_id=str(payload["case_id"]),
            dataset=str(payload["dataset"]),
            run_id=str(payload["run_id"]),
            event_id=str(payload["event_id"]),
            query=str(payload["query"]),
            top_variables=[str(item) for item in payload.get("top_variables", [])],
            evidence_relative_path=str(payload["evidence_relative_path"]),
            expected_retrieval_event_ids=[
                str(item) for item in payload.get("expected_retrieval_event_ids", [])
            ],
            expected_abstain=bool(payload.get("expected_abstain", False)),
            minimum_supported_claims=int(payload.get("minimum_supported_claims", 1)),
        )


@dataclass(frozen=True)
class ReplaySuiteManifest:
    """Manifest for a deterministic set of RQ3 replay cases."""

    suite_id: str
    cases: list[AssistantCase]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "format_version": "rq3-replay-suite-v1",
            "suite_id": self.suite_id,
            "case_count": len(self.cases),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class AssistantRunMetrics:
    """Thesis-compatible proxy metrics for one assistant replay run."""

    run_id: str
    case_id: str
    total_claims: int
    supported_claims: int
    citation_compliant_claims: int
    retrieved_document_hit_count: int
    final_supported_document_claims: int
    final_supported_evidence_bundle_claims: int
    abstain_flag: bool
    verified_response_safe: bool
    retrieval_expectations_met: bool | None
    abstain_expectation_matched: bool | None
    minimum_supported_claim_met: bool | None
    evidence_pack_overflow_count: int
    budget_truncation_count: int
    propositional_alignment_proxy: float
    citation_compliance_proxy: float
    verified_response_safety_proxy: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class RQ3AggregateMetrics:
    """Aggregate RQ3 metrics for a replay suite."""

    suite_id: str
    runs_evaluated: int
    total_claims: int
    supported_claims: int
    citation_compliant_claims: int
    retrieved_document_hit_count: int
    final_supported_document_claims: int
    final_supported_evidence_bundle_claims: int
    safe_runs: int
    abstained_runs: int
    retrieval_expectation_matches: int
    abstain_expectation_matches: int
    minimum_supported_claim_matches: int
    evidence_pack_overflow_count: int
    budget_truncation_count: int
    propositional_alignment_proxy: float
    citation_compliance_proxy: float
    verified_response_safety_proxy: float
    abstain_rate: float
    retrieval_expectation_hit_rate: float
    abstain_expectation_match_rate: float
    minimum_supported_claim_compliance: float
    document_grounded_run_rate: float
    document_grounding_coverage_proxy: float
    per_run: list[AssistantRunMetrics]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            **asdict(self),
            "per_run": [metric.to_dict() for metric in self.per_run],
        }


@dataclass(frozen=True)
class RQ3RunResult:
    """Application-level result for an RQ3 replay suite."""

    suite_id: str
    run_dir: str
    ok: bool
    metrics: RQ3AggregateMetrics

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "suite_id": self.suite_id,
            "run_dir": self.run_dir,
            "ok": self.ok,
            "metrics": self.metrics.to_dict(),
        }


def aggregate_rq3_metrics(suite_id: str, per_run: list[AssistantRunMetrics]) -> RQ3AggregateMetrics:
    """Aggregate thesis-compatible RQ3 metrics."""
    runs_evaluated = len(per_run)
    total_claims = sum(item.total_claims for item in per_run)
    supported_claims = sum(item.supported_claims for item in per_run)
    citation_compliant_claims = sum(item.citation_compliant_claims for item in per_run)
    retrieved_document_hit_count = sum(item.retrieved_document_hit_count for item in per_run)
    final_supported_document_claims = sum(item.final_supported_document_claims for item in per_run)
    final_supported_evidence_bundle_claims = sum(
        item.final_supported_evidence_bundle_claims for item in per_run
    )
    safe_runs = sum(1 for item in per_run if item.verified_response_safe)
    abstained_runs = sum(1 for item in per_run if item.abstain_flag)
    retrieval_expected = [item for item in per_run if item.retrieval_expectations_met is not None]
    abstain_expected = [item for item in per_run if item.abstain_expectation_matched is not None]
    minimum_expected = [item for item in per_run if item.minimum_supported_claim_met is not None]
    return RQ3AggregateMetrics(
        suite_id=suite_id,
        runs_evaluated=runs_evaluated,
        total_claims=total_claims,
        supported_claims=supported_claims,
        citation_compliant_claims=citation_compliant_claims,
        retrieved_document_hit_count=retrieved_document_hit_count,
        final_supported_document_claims=final_supported_document_claims,
        final_supported_evidence_bundle_claims=final_supported_evidence_bundle_claims,
        safe_runs=safe_runs,
        abstained_runs=abstained_runs,
        retrieval_expectation_matches=sum(
            1 for item in retrieval_expected if item.retrieval_expectations_met
        ),
        abstain_expectation_matches=sum(
            1 for item in abstain_expected if item.abstain_expectation_matched
        ),
        minimum_supported_claim_matches=sum(
            1 for item in minimum_expected if item.minimum_supported_claim_met
        ),
        evidence_pack_overflow_count=sum(item.evidence_pack_overflow_count for item in per_run),
        budget_truncation_count=sum(item.budget_truncation_count for item in per_run),
        propositional_alignment_proxy=_ratio(supported_claims, total_claims),
        citation_compliance_proxy=_ratio(citation_compliant_claims, total_claims),
        verified_response_safety_proxy=_ratio(safe_runs, runs_evaluated),
        abstain_rate=_ratio(abstained_runs, runs_evaluated),
        retrieval_expectation_hit_rate=_ratio(
            sum(1 for item in retrieval_expected if item.retrieval_expectations_met),
            len(retrieval_expected),
        ),
        abstain_expectation_match_rate=_ratio(
            sum(1 for item in abstain_expected if item.abstain_expectation_matched),
            len(abstain_expected),
        ),
        minimum_supported_claim_compliance=_ratio(
            sum(1 for item in minimum_expected if item.minimum_supported_claim_met),
            len(minimum_expected),
        ),
        document_grounded_run_rate=_ratio(
            sum(1 for item in per_run if item.final_supported_document_claims > 0),
            runs_evaluated,
        ),
        document_grounding_coverage_proxy=_ratio(final_supported_document_claims, supported_claims),
        per_run=per_run,
    )


def _provider_config(payload: dict[str, Any]) -> LLMProviderConfig:
    name = _required_string(payload, "name", "rq3.provider.name")
    model = _string(payload, "model", f"{name}-default")
    extra = payload.get("extra", {})
    if not isinstance(extra, dict):
        raise BenchmarkConfigError("rq3.provider.extra must be a table/object.")
    return LLMProviderConfig(
        name=name,
        model=model,
        base_url=_optional_string(payload.get("base_url")),
        api_key_env=_optional_string(payload.get("api_key_env")),
        timeout_s=float(payload.get("timeout_s", 60.0)),
        temperature=float(payload.get("temperature", 0.0)),
        top_p=float(payload.get("top_p", 1.0)),
        max_tokens=int(payload.get("max_tokens", 512)),
        seed=int(payload["seed"]) if payload.get("seed") is not None else None,
        extra=dict(extra),
    )


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise BenchmarkConfigError(f"{key} must be a table.")
    return value


def _required_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkConfigError(f"{label} must be a non-empty string.")
    return value.strip()


def _string(payload: dict[str, Any], key: str, default: str) -> str:
    value = payload.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkConfigError(f"{key} must be a non-empty string.")
    return value.strip()


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0
