"""Appendix-compatible exports for thesis-style reproduction runs."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.assistant_replay import (
    PLANNER_SYSTEM_PROMPT,
    REFEREE_SYSTEM_PROMPT,
)
from industrial_tsad_eval.domain.assistant_replay import (
    AssistantCase,
    ClaimEvaluation,
    DraftClaim,
    DraftResponse,
)
from industrial_tsad_eval.domain.operator import OperatorEvidenceHit, OperatorQuery
from industrial_tsad_eval.domain.reproduction import ReproductionConfig
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.prepared_repository import LocalPreparedDatasetRepository


def write_thesis_draft_exports(
    *,
    run_root: Path,
    config: ReproductionConfig,
    reproduction_toml: str,
) -> None:
    """Write thesis-draft-compatible export filenames beside clean summaries."""
    writer = LocalArtifactWriter(run_root / "summaries")
    detection_text = _read_text(run_root / "benchmark" / "summary.csv")
    writer.write_text("detection_tables.csv", detection_text)
    writer.write_text(
        "explanation_results.csv", _csv(_collect_bundle_metric_rows(run_root / "xai"))
    )
    writer.write_text(
        "explanation_results_split_summary.csv",
        _csv(_collect_xai_summary_rows(run_root / "xai")),
    )
    writer.write_text(
        "assistant_faithfulness_logs.csv",
        _csv(_assistant_faithfulness_rows(run_root / "assistant" / "assistant_summary.csv")),
    )
    writer.write_text("profiling_logs.csv", _csv(_collect_profile_rows(run_root / "profiles")))
    writer.write_text("hyperparameters.toml", reproduction_toml)
    writer.write_json("dataset_splits.json", _dataset_splits(config))
    writer.write_json("scoring_config.json", _scoring_config(config))
    for dataset in config.benchmark.datasets:
        policy = config.benchmark.evaluation.policy_for(dataset.id, "naive")
        writer.write_json(f"scoring_config.per_dataset/{dataset.id}.json", policy.to_dict())
    writer.write_text("planner_prompt.txt", PLANNER_SYSTEM_PROMPT)
    writer.write_text("referee_prompt.txt", REFEREE_SYSTEM_PROMPT)
    writer.write_json(
        "assistant/schemas/draft_response.schema.json", DraftResponse.model_json_schema()
    )
    writer.write_json("assistant/schemas/draft_claim.schema.json", DraftClaim.model_json_schema())
    writer.write_json(
        "assistant/schemas/claim_evaluation.schema.json",
        ClaimEvaluation.model_json_schema(),
    )
    writer.write_json("assistant/schemas/assistant_case.schema.json", _schema(AssistantCase))
    writer.write_json("assistant/schemas/operator_query.schema.json", _schema(OperatorQuery))
    writer.write_json(
        "assistant/schemas/operator_evidence_hit.schema.json",
        _schema(OperatorEvidenceHit),
    )


def _dataset_splits(config: ReproductionConfig) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for dataset in config.benchmark.datasets:
        try:
            splits[dataset.id] = LocalPreparedDatasetRepository(dataset.prepared).splits()
        except Exception as exc:
            splits[dataset.id] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    return {
        "format_version": "dataset-splits-export-v1",
        "datasets": splits,
    }


def _scoring_config(config: ReproductionConfig) -> dict[str, Any]:
    return {
        "format_version": "scoring-config-export-v1",
        "global_policy": config.benchmark.evaluation.policy.to_dict(),
        "dataset_policies": {
            dataset.id: config.benchmark.evaluation.policy_for(dataset.id, "naive").to_dict()
            for dataset in config.benchmark.datasets
        },
    }


def _collect_bundle_metric_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("bundle_metrics.csv")):
        for row in _csv_rows(path):
            rows.append({**_xai_path_context(root, path), **row})
    return rows


def _collect_xai_summary_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("summary.csv")):
        for row in _csv_rows(path):
            rows.append({**_xai_path_context(root, path), **row})
    return rows


def _xai_path_context(root: Path, path: Path) -> dict[str, str]:
    relative = path.relative_to(root)
    if len(relative.parts) >= 3:
        return {"experiment_id": relative.parts[0], "evidence_source": relative.parts[1]}
    if len(relative.parts) >= 2:
        return {"experiment_id": relative.parts[0], "evidence_source": "oracle"}
    return {"experiment_id": "", "evidence_source": ""}


def _assistant_faithfulness_rows(path: Path) -> list[dict[str, Any]]:
    rows = _csv_rows(path)
    if not rows:
        return []
    output = [_aggregate_assistant_rows("overall", rows)]
    for dataset in sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")}):
        output.append(
            _aggregate_assistant_rows(
                dataset,
                [row for row in rows if str(row.get("dataset", "")) == dataset],
            )
        )
    return output


def _aggregate_assistant_rows(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    count_keys = {"runs_evaluated", "supported_claims", "citation_compliant_claims"}
    mean_keys = {
        "propositional_alignment_proxy",
        "citation_compliance_proxy",
        "verified_response_safety_proxy",
        "abstain_rate",
        "retrieval_expectation_hit_rate",
        "document_grounding_coverage_proxy",
    }
    output: dict[str, Any] = {"group": label, "row_count": len(rows)}
    for key in sorted(count_keys):
        output[key] = sum(_float(row.get(key)) for row in rows)
    for key in sorted(mean_keys):
        values = [_float(row.get(key)) for row in rows if str(row.get(key, "")) != ""]
        output[key] = sum(values) / len(values) if values else ""
    return output


def _collect_profile_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("stages.csv")):
        profile_id = path.parent.name
        for row in _csv_rows(path):
            rows.append({"profile_id": profile_id, **row})
    return rows


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    text = _read_text(path)
    if not text.strip():
        return []
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


def _csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = sorted({key for row in rows for key in row})
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _schema(model: Any) -> dict[str, Any]:
    schema = getattr(model, "model_json_schema", None)
    if callable(schema):
        return dict(schema())
    return {"title": getattr(model, "__name__", str(model)), "type": "object"}
