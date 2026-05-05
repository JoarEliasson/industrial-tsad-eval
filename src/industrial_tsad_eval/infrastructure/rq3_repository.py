"""Filesystem repositories for RQ3 replay artifacts."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from industrial_tsad_eval.domain.rq3 import (
    AssistantCase,
    AssistantRunMetrics,
    ReplaySuiteManifest,
    RQ3AggregateMetrics,
)
from industrial_tsad_eval.infrastructure.json_utils import read_json, write_json, write_jsonl


class LocalReplaySuiteRepository:
    """Read and write RQ3 replay suite case artifacts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_suite(self, manifest: ReplaySuiteManifest) -> None:
        """Write replay suite manifest and case specs."""
        for case in manifest.cases:
            write_json(self.root / "cases" / _safe_id(case.case_id) / "case.json", case.to_dict())
        write_json(self.root / "suites" / "suite_manifest.json", manifest.to_dict())
        write_jsonl(
            self.root / "cases" / "index.jsonl",
            [
                {
                    "case_id": case.case_id,
                    "dataset": case.dataset,
                    "event_id": case.event_id,
                    "relative_path": f"cases/{_safe_id(case.case_id)}/case.json",
                }
                for case in manifest.cases
            ],
        )

    def read_suite(self) -> ReplaySuiteManifest:
        """Read a replay suite manifest."""
        payload = read_json(self.root / "suites" / "suite_manifest.json")
        cases = [AssistantCase.from_dict(item) for item in _list_of_dicts(payload.get("cases", []))]
        return ReplaySuiteManifest(suite_id=str(payload["suite_id"]), cases=cases)


class LocalAssistantRunRepository:
    """Write per-case RQ3 run artifacts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_case_run(
        self,
        *,
        case: AssistantCase,
        retrieval: dict[str, Any],
        provider_request: dict[str, Any],
        provider_response: dict[str, Any],
        planner_output: dict[str, Any],
        referee_output: dict[str, Any],
        run_log: dict[str, Any],
        rendered_response: str,
    ) -> None:
        """Write all artifacts for one assistant replay case."""
        case_root = self.root / "runs" / _safe_id(case.case_id)
        write_json(case_root / "case_spec.json", case.to_dict())
        write_json(case_root / "retrieval_result.json", retrieval)
        write_json(case_root / "provider_request.json", provider_request)
        write_json(case_root / "provider_response.json", provider_response)
        write_json(case_root / "planner_output.json", planner_output)
        write_json(case_root / "referee_output.json", referee_output)
        write_json(case_root / "run_log.json", run_log)
        response_path = case_root / "rendered_response.md"
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(rendered_response, encoding="utf-8")


class LocalRQ3MetricsRepository:
    """Read and write aggregate RQ3 metrics."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_summary(self, metrics: RQ3AggregateMetrics) -> None:
        """Write JSON and CSV summary artifacts."""
        write_json(self.root / "rq3_summary.json", metrics.to_dict())
        (self.root / "rq3_summary.csv").parent.mkdir(parents=True, exist_ok=True)
        (self.root / "rq3_summary.csv").write_text(_metrics_csv(metrics.per_run), encoding="utf-8")

    def read_summary(self) -> dict[str, Any]:
        """Read the JSON summary artifact."""
        return read_json(self.root / "rq3_summary.json")


def _metrics_csv(rows: list[AssistantRunMetrics]) -> str:
    if not rows:
        return ""
    fieldnames = sorted({key for row in rows for key in row.to_dict()})
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row.to_dict())
    return handle.getvalue()


def _safe_id(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )
    return cleaned.strip("._-") or "item"


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]
