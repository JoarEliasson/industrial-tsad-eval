"""Deterministic operator-assistant application services."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.evidence import ValidateEvidence
from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.domain.errors import OperatorAssistantError
from industrial_tsad_eval.domain.evidence import EvidenceBundle
from industrial_tsad_eval.domain.operator import (
    OPERATOR_CARD_MANIFEST_VERSION,
    OPERATOR_CARD_VERSION,
    OperatorCard,
    OperatorCardIndexRow,
    OperatorEvidenceHit,
    OperatorIntent,
    OperatorQuery,
    OperatorRetrievalResult,
)
from industrial_tsad_eval.domain.validation import ValidationReport
from industrial_tsad_eval.infrastructure.evidence_repository import LocalEvidenceRepository
from industrial_tsad_eval.infrastructure.operator_repository import LocalOperatorCardRepository
from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository

TOKEN_RE = re.compile(r"[A-Za-z0-9_:/.-]+")
DEFAULT_CARD_QUERY = "What should the operator inspect and preserve for this event?"
INTENT_KEYWORDS: dict[OperatorIntent, tuple[str, ...]] = {
    "checks": ("check", "checks", "inspect", "verify", "review", "compare"),
    "recommended_actions": ("action", "actions", "respond", "preserve", "mitigate", "do"),
    "likely_causes": ("cause", "causes", "why", "reason", "root"),
    "escalation_criteria": ("escalate", "escalation", "handoff", "call"),
    "general": (),
}
ROLE_INTENT_BONUS: dict[OperatorIntent, tuple[str, ...]] = {
    "checks": ("top_variables", "time_windows", "score_context", "playbook"),
    "recommended_actions": ("score_context", "provenance", "playbook", "top_variables"),
    "likely_causes": ("top_variables", "time_windows", "overview"),
    "escalation_criteria": ("score_context", "time_windows", "playbook"),
    "general": ("overview", "top_variables", "time_windows", "score_context"),
}


@dataclass(frozen=True)
class GenerateOperatorCardsResult:
    """Summary returned after writing operator cards."""

    dataset: str
    card_count: int
    out_dir: str
    statuses: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "dataset": self.dataset,
            "card_count": self.card_count,
            "out_dir": self.out_dir,
            "statuses": dict(self.statuses),
        }


@dataclass(frozen=True)
class _CandidateChunk:
    source_id: str
    source_type: str
    title: str
    role: str
    text: str
    dataset: str | None
    event_id: str | None
    run_id: str | None
    relative_path: str | None
    metadata: dict[str, Any]


class RetrieveOperatorEvidence:
    """Retrieve deterministic evidence hits for an operator query."""

    def __init__(
        self,
        *,
        prepared: str | Path,
        evidence: str | Path,
        query: str,
        dataset: str | None = None,
        event_id: str | None = None,
        playbooks: str | Path | None = None,
        top_k: int = 8,
    ):
        self.prepared = Path(prepared)
        self.evidence = Path(evidence)
        self.query = query
        self.dataset = dataset
        self.event_id = event_id
        self.playbooks = Path(playbooks) if playbooks is not None else None
        self.top_k = top_k

    def run(self) -> OperatorRetrievalResult:
        """Return ranked evidence hits without writing artifacts."""
        if not self.query.strip():
            raise ValueError("query must be non-empty.")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than 0.")
        _validate_inputs(self.prepared, self.evidence)

        tokens = _tokens(self.query)
        intent = _detect_intent(tokens)
        query = OperatorQuery(
            query=self.query,
            top_k=self.top_k,
            dataset=self.dataset,
            event_id=self.event_id,
            detected_intent=intent,
        )
        chunks = _load_chunks(
            evidence=self.evidence,
            dataset=self.dataset,
            event_id=self.event_id,
            playbooks=self.playbooks,
        )
        ranked = _rank_chunks(chunks, tokens, intent, self.dataset, self.event_id)
        hits = [
            OperatorEvidenceHit(
                source_id=chunk.source_id,
                source_type="playbook" if chunk.source_type == "playbook" else "evidence_bundle",
                title=chunk.title,
                role=chunk.role,
                rank=rank,
                score=round(score, 6),
                text=chunk.text,
                citation_id=f"C{rank}",
                dataset=chunk.dataset,
                event_id=chunk.event_id,
                run_id=chunk.run_id,
                relative_path=chunk.relative_path,
                metadata=chunk.metadata,
            )
            for rank, (score, chunk) in enumerate(ranked[: self.top_k], start=1)
        ]
        return OperatorRetrievalResult(
            query=query,
            hits=hits,
            diagnostics={
                "candidate_count": len(chunks),
                "ranked_candidate_count": len(ranked),
                "returned_count": len(hits),
                "normalized_query_tokens": sorted(tokens),
                "detected_intent": intent,
            },
        )


class GenerateOperatorCards:
    """Generate deterministic operator cards from evidence bundles."""

    def __init__(
        self,
        *,
        prepared: str | Path,
        evidence: str | Path,
        out: str | Path,
        query: str | None = None,
        dataset: str | None = None,
        event_id: str | None = None,
        playbooks: str | Path | None = None,
        max_cards: int = 25,
    ):
        self.prepared = Path(prepared)
        self.evidence = Path(evidence)
        self.out = Path(out)
        self.query = query
        self.dataset = dataset
        self.event_id = event_id
        self.playbooks = Path(playbooks) if playbooks is not None else None
        self.max_cards = max_cards

    def run(self) -> GenerateOperatorCardsResult:
        """Generate cards and write card artifacts."""
        if self.max_cards <= 0:
            raise ValueError("max_cards must be greater than 0.")
        _validate_inputs(self.prepared, self.evidence)
        prepared_repository = LocalPreparedDatasetRepository(self.prepared)
        evidence_repository = LocalEvidenceRepository(self.evidence)
        bundles = _matching_bundles(
            evidence_repository,
            dataset=self.dataset,
            event_id=self.event_id,
        )
        cards: list[OperatorCard] = []
        retrieval_results: list[dict[str, Any]] = []

        if not bundles:
            card = _abstention_card(
                dataset=self.dataset or prepared_repository.dataset_name,
                event_id=self.event_id or "no_matching_evidence",
                query=self.query or DEFAULT_CARD_QUERY,
                reason="No matching Evidence Bundle v1 artifacts were available.",
                diagnostics={"bundle_count": 0},
            )
            cards.append(card)
            retrieval_results.append(
                OperatorRetrievalResult(
                    query=OperatorQuery(
                        query=card.query,
                        dataset=card.dataset,
                        event_id=card.event_id,
                    ),
                    hits=[],
                    diagnostics={"candidate_count": 0, "returned_count": 0},
                ).to_dict()
            )
        else:
            for bundle in bundles[: self.max_cards]:
                query_text = self.query or DEFAULT_CARD_QUERY
                retrieval = RetrieveOperatorEvidence(
                    prepared=self.prepared,
                    evidence=self.evidence,
                    query=query_text,
                    dataset=bundle.dataset,
                    event_id=bundle.event_id,
                    playbooks=self.playbooks,
                    top_k=8,
                ).run()
                cards.append(_card_from_bundle(bundle, query_text, retrieval))
                retrieval_results.append(retrieval.to_dict())

        rows = LocalOperatorCardRepository(self.out).write_card_set(
            dataset=prepared_repository.dataset_name,
            cards=cards,
            retrieval_payload={
                "format_version": "operator-card-retrieval-set-v1",
                "result_count": len(retrieval_results),
                "results": retrieval_results,
            },
        )
        statuses = {
            "answered": sum(1 for row in rows if row.status == "answered"),
            "abstained": sum(1 for row in rows if row.status == "abstained"),
        }
        return GenerateOperatorCardsResult(
            dataset=prepared_repository.dataset_name,
            card_count=len(rows),
            out_dir=str(self.out),
            statuses=statuses,
        )


class ValidateOperatorCards:
    """Validate Operator Card v1 artifacts against prepared and evidence inputs."""

    def __init__(self, *, prepared: str | Path, evidence: str | Path, cards: str | Path):
        self.prepared = Path(prepared)
        self.evidence = Path(evidence)
        self.cards = Path(cards)

    def run(self) -> ValidationReport:
        """Validate card manifest, index rows, cards, Markdown files, and citations."""
        errors: list[str] = []
        warnings: list[str] = []
        prepared_report = ValidatePreparedDataset(self.prepared).run()
        if not prepared_report.ok:
            errors.extend(f"Prepared dataset: {error}" for error in prepared_report.errors)
            return ValidationReport("operator-cards", str(self.cards), errors, warnings)
        evidence_report = ValidateEvidence(self.prepared, self.evidence).run()
        if not evidence_report.ok:
            errors.extend(f"Evidence: {error}" for error in evidence_report.errors)
            return ValidationReport("operator-cards", str(self.cards), errors, warnings)

        repository = LocalOperatorCardRepository(self.cards)
        known_events = {row.event_id for row in LocalEvidenceRepository(self.evidence).index_rows()}
        try:
            manifest = repository.manifest()
            if manifest.get("format_version") != OPERATOR_CARD_MANIFEST_VERSION:
                errors.append("manifest.json: unsupported format_version.")
            rows = repository.index_rows()
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            return ValidationReport("operator-cards", str(self.cards), errors, warnings)

        for row in rows:
            markdown_path = self.cards / row.relative_markdown_path
            json_path = self.cards / row.relative_json_path
            if not markdown_path.exists():
                errors.append(f"Missing Markdown card: {row.relative_markdown_path}")
            if not json_path.exists():
                errors.append(f"Missing JSON card: {row.relative_json_path}")
                continue
            try:
                card = repository.read_card(row)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                errors.append(f"{row.relative_json_path}: {type(exc).__name__}: {exc}")
                continue
            _validate_card(card, row, known_events, errors, warnings)

        expected_count = manifest.get("card_count")
        if isinstance(expected_count, int) and expected_count != len(rows):
            errors.append(
                f"manifest.json: card_count={expected_count} but index has {len(rows)} rows."
            )
        return ValidationReport("operator-cards", str(self.cards), errors, warnings)


def _validate_inputs(prepared: Path, evidence: Path) -> None:
    prepared_report = ValidatePreparedDataset(prepared).run()
    if not prepared_report.ok:
        raise OperatorAssistantError(
            f"Prepared dataset validation failed: {prepared_report.errors}"
        )
    evidence_report = ValidateEvidence(prepared, evidence).run()
    if not evidence_report.ok:
        raise OperatorAssistantError(f"Evidence validation failed: {evidence_report.errors}")


def _matching_bundles(
    repository: LocalEvidenceRepository,
    *,
    dataset: str | None,
    event_id: str | None,
) -> list[EvidenceBundle]:
    bundles = []
    for row in repository.index_rows():
        if dataset and row.dataset.lower() != dataset.lower():
            continue
        if event_id and row.event_id != event_id:
            continue
        bundles.append(repository.read_bundle(row))
    return sorted(bundles, key=lambda bundle: (bundle.dataset, bundle.run_id, bundle.event_id))


def _load_chunks(
    *,
    evidence: Path,
    dataset: str | None,
    event_id: str | None,
    playbooks: Path | None,
) -> list[_CandidateChunk]:
    evidence_repository = LocalEvidenceRepository(evidence)
    chunks: list[_CandidateChunk] = []
    for row in evidence_repository.index_rows():
        if dataset and row.dataset.lower() != dataset.lower():
            continue
        if event_id and row.event_id != event_id:
            continue
        bundle = evidence_repository.read_bundle(row)
        chunks.extend(_bundle_chunks(bundle, row.relative_path))
    if playbooks is not None:
        chunks.extend(_playbook_chunks(playbooks, dataset=dataset, event_id=event_id))
    return chunks


def _bundle_chunks(bundle: EvidenceBundle, relative_path: str) -> list[_CandidateChunk]:
    event_title = f"Evidence Bundle: {bundle.event_id}"
    top_variables = ", ".join(variable.variable for variable in bundle.top_variables[:5]) or "none"
    score_context = json.dumps(bundle.score_context, sort_keys=True)
    windows = (
        "; ".join(
            (f"{window.start_ts_ns}->{window.end_ts_ns} importance={window.importance:.3f}")
            for window in bundle.top_time_windows[:5]
        )
        or "none"
    )
    local_rankings = json.dumps(bundle.local_rankings[:8], sort_keys=True)
    provenance = json.dumps(bundle.provenance, sort_keys=True)
    sections = [
        (
            "overview",
            (
                f"Dataset {bundle.dataset}; event {bundle.event_id}; run {bundle.run_id}; "
                f"source {bundle.event_source}; matched GT {bundle.matched_gt_event_id or 'none'}; "
                f"bounds {bundle.event_start_ts_ns}->{bundle.event_end_ts_ns}; "
                f"top variables {top_variables}."
            ),
        ),
        ("top_variables", f"Top ranked variables for event {bundle.event_id}: {top_variables}."),
        ("time_windows", f"Top anomalous time windows for event {bundle.event_id}: {windows}."),
        ("score_context", f"Score context for event {bundle.event_id}: {score_context}."),
        (
            "local_rankings",
            f"Local variable rankings for event {bundle.event_id}: {local_rankings}.",
        ),
        ("provenance", f"Evidence provenance for event {bundle.event_id}: {provenance}."),
    ]
    return [
        _CandidateChunk(
            source_id=f"evidence::{bundle.event_id}::{role}",
            source_type="evidence_bundle",
            title=event_title,
            role=role,
            text=text,
            dataset=bundle.dataset,
            event_id=bundle.event_id,
            run_id=bundle.run_id,
            relative_path=relative_path,
            metadata={
                "event_source": bundle.event_source,
                "matched_gt_event_id": bundle.matched_gt_event_id,
                "top_variables": [variable.variable for variable in bundle.top_variables],
            },
        )
        for role, text in sections
        if text
    ]


def _playbook_chunks(
    playbooks: Path,
    *,
    dataset: str | None,
    event_id: str | None,
) -> list[_CandidateChunk]:
    if not playbooks.exists():
        raise OperatorAssistantError(f"Playbook directory does not exist: {playbooks}")
    chunks: list[_CandidateChunk] = []
    for path in sorted(playbooks.rglob("*.md")):
        relative = path.relative_to(playbooks)
        text = path.read_text(encoding="utf-8")
        title = _markdown_title(path, text)
        chunks.append(
            _CandidateChunk(
                source_id=f"playbook::{relative.as_posix()}",
                source_type="playbook",
                title=title,
                role="playbook",
                text=text.strip(),
                dataset=dataset,
                event_id=event_id,
                run_id=None,
                relative_path=relative.as_posix(),
                metadata={"path": relative.as_posix()},
            )
        )
    return chunks


def _rank_chunks(
    chunks: list[_CandidateChunk],
    query_tokens: set[str],
    intent: OperatorIntent,
    dataset: str | None,
    event_id: str | None,
) -> list[tuple[float, _CandidateChunk]]:
    ranked: list[tuple[float, _CandidateChunk]] = []
    for chunk in chunks:
        chunk_tokens = _tokens(f"{chunk.title} {chunk.role} {chunk.text}")
        overlap = len(query_tokens & chunk_tokens)
        normalized_overlap = overlap / max(len(query_tokens), 1)
        selector_bonus = 0.0
        if event_id and chunk.event_id == event_id:
            selector_bonus += 1.0
        if dataset and chunk.dataset and chunk.dataset.lower() == dataset.lower():
            selector_bonus += 0.25
        role_bonus = 0.25 if chunk.role in ROLE_INTENT_BONUS[intent] else 0.0
        playbook_bonus = 0.1 if chunk.source_type == "playbook" and overlap > 0 else 0.0
        score = normalized_overlap + selector_bonus + role_bonus + playbook_bonus
        if score > 0:
            ranked.append((score, chunk))
    return sorted(
        ranked,
        key=lambda item: (-item[0], item[1].source_type, item[1].event_id or "", item[1].role),
    )


def _card_from_bundle(
    bundle: EvidenceBundle,
    query: str,
    retrieval: OperatorRetrievalResult,
) -> OperatorCard:
    if not retrieval.hits:
        return _abstention_card(
            dataset=bundle.dataset,
            run_id=bundle.run_id,
            event_id=bundle.event_id,
            event_source=bundle.event_source,
            matched_gt_event_id=bundle.matched_gt_event_id,
            query=query,
            reason="No relevant evidence chunks were available for this event.",
            diagnostics=retrieval.diagnostics,
        )

    citations = [_citation(hit) for hit in retrieval.hits]
    citation_by_role = _citation_by_role(retrieval.hits)
    overview_cite = citation_by_role.get("overview", citations[0]["citation_id"])
    variables_cite = citation_by_role.get("top_variables", overview_cite)
    windows_cite = citation_by_role.get("time_windows", overview_cite)
    score_cite = citation_by_role.get("score_context", overview_cite)
    provenance_cite = citation_by_role.get("provenance", overview_cite)
    top_variable = bundle.top_variables[0].variable if bundle.top_variables else "the top variable"
    top_window = bundle.top_time_windows[0] if bundle.top_time_windows else None
    max_score = bundle.score_context.get("max_score")
    max_score_text = (
        f" with max score {float(max_score):.3f}" if isinstance(max_score, int | float) else ""
    )

    evidence_highlights = [
        f"{top_variable} is the highest-ranked explanatory variable [{variables_cite}].",
        (
            f"The event spans {bundle.event_start_ts_ns} to "
            f"{bundle.event_end_ts_ns} ns [{overview_cite}]."
        ),
        f"Score context was computed{max_score_text} [{score_cite}].",
    ]
    if top_window is not None:
        evidence_highlights.append(
            f"The strongest local window is {top_window.start_ts_ns} to "
            f"{top_window.end_ts_ns} ns [{windows_cite}]."
        )
    checks = [
        (
            f"Inspect {top_variable} around the event window before changing process "
            f"state [{variables_cite}]."
        ),
        f"Compare the event window with adjacent normal trend context [{windows_cite}].",
        f"Review score artifacts and event evidence before acknowledging the alert [{score_cite}].",
    ]
    recommended_actions = [
        (
            "Preserve score, evidence, and card artifacts for post-incident review "
            f"[{provenance_cite}]."
        ),
        f"Correlate top-ranked tags with local historian or HMI readings [{variables_cite}].",
    ]
    escalation = [
        (
            f"Escalate if the same ranked tags remain abnormal in subsequent windows or "
            f"if process constraints are affected [{windows_cite}]."
        )
    ]
    return OperatorCard(
        dataset=bundle.dataset,
        run_id=bundle.run_id,
        event_id=bundle.event_id,
        event_source=bundle.event_source,
        matched_gt_event_id=bundle.matched_gt_event_id,
        query=query,
        status="answered",
        situation_summary=(
            f"Evidence for event {bundle.event_id} in run {bundle.run_id} identifies "
            f"{top_variable} as the leading explanatory tag [{overview_cite}]."
        ),
        evidence_highlights=evidence_highlights,
        checks=checks,
        recommended_actions=recommended_actions,
        escalation_criteria=escalation,
        citations=citations,
        diagnostics={
            **retrieval.diagnostics,
            "top_hit_roles": [hit.role for hit in retrieval.hits[:5]],
        },
        provenance={
            "generator": "deterministic-operator-card-v1",
            "evidence_format": bundle.format_version,
        },
    )


def _abstention_card(
    *,
    dataset: str,
    event_id: str,
    query: str,
    reason: str,
    run_id: str | None = None,
    event_source: str | None = None,
    matched_gt_event_id: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> OperatorCard:
    return OperatorCard(
        dataset=dataset,
        run_id=run_id,
        event_id=event_id,
        event_source=event_source,
        matched_gt_event_id=matched_gt_event_id,
        query=query,
        status="abstained",
        abstain_reason=reason,
        situation_summary="The assistant abstained because it could not ground a card in evidence.",
        evidence_highlights=[],
        checks=[],
        recommended_actions=[],
        escalation_criteria=[],
        citations=[],
        diagnostics=dict(diagnostics or {}),
        provenance={"generator": "deterministic-operator-card-v1"},
    )


def _validate_card(
    card: OperatorCard,
    row: OperatorCardIndexRow,
    known_events: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if card.format_version != OPERATOR_CARD_VERSION:
        errors.append(f"{row.relative_json_path}: unsupported format_version.")
    if card.event_id != row.event_id or card.status != row.status:
        errors.append(f"{row.relative_json_path}: index identity does not match card.")
    if row.citation_count != len(card.citations):
        errors.append(f"{row.relative_json_path}: citation_count does not match card.")
    if card.status == "abstained" and not card.abstain_reason:
        errors.append(f"{row.relative_json_path}: abstained card requires abstain_reason.")
    if card.status == "answered" and not card.citations:
        errors.append(f"{row.relative_json_path}: answered card requires citations.")
    if card.status == "answered" and card.event_id not in known_events:
        errors.append(f"{row.relative_json_path}: answered card event_id is not in evidence index.")
    citation_ids = [str(citation.get("citation_id", "")) for citation in card.citations]
    if len(citation_ids) != len(set(citation_ids)):
        errors.append(f"{row.relative_json_path}: duplicate citation ids.")
    text = "\n".join(
        [
            card.situation_summary,
            *card.evidence_highlights,
            *card.checks,
            *card.recommended_actions,
            *card.escalation_criteria,
        ]
    )
    for citation_id in citation_ids:
        if citation_id and f"[{citation_id}]" not in text:
            warnings.append(f"{row.relative_json_path}: citation {citation_id} is not referenced.")
    for citation in card.citations:
        source_type = citation.get("source_type")
        citation_event_id = citation.get("event_id")
        if source_type == "evidence_bundle" and citation_event_id not in known_events:
            errors.append(f"{row.relative_json_path}: citation references unknown evidence event.")


def _citation(hit: OperatorEvidenceHit) -> dict[str, Any]:
    return {
        "citation_id": hit.citation_id,
        "source_id": hit.source_id,
        "source_type": hit.source_type,
        "title": hit.title,
        "role": hit.role,
        "dataset": hit.dataset,
        "event_id": hit.event_id,
        "run_id": hit.run_id,
        "relative_path": hit.relative_path,
        "score": hit.score,
    }


def _citation_by_role(hits: list[OperatorEvidenceHit]) -> dict[str, str]:
    by_role: dict[str, str] = {}
    for hit in hits:
        by_role.setdefault(hit.role, hit.citation_id)
    return by_role


def _detect_intent(tokens: set[str]) -> OperatorIntent:
    scores = {
        intent: len(tokens.intersection(keywords))
        for intent, keywords in INTENT_KEYWORDS.items()
        if intent != "general"
    }
    if not scores:
        return "general"
    best_intent, best_score = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[0]
    return best_intent if best_score > 0 else "general"


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text) if token.strip()}


def _markdown_title(path: Path, text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or path.stem
    return path.stem.replace("_", " ")
