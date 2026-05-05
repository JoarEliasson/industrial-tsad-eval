"""Filesystem repository for Evidence Bundle v1 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from industrial_tsad_eval.domain.evidence import EvidenceBundle, EvidenceIndexRow
from industrial_tsad_eval.infrastructure.json_utils import read_json, write_json, write_jsonl


class LocalEvidenceRepository:
    """Read and write Evidence Bundle v1 artifacts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_bundle_set(
        self,
        *,
        dataset: str,
        event_source: str,
        bundles: list[EvidenceBundle],
    ) -> list[EvidenceIndexRow]:
        """Write evidence bundles, index rows, and manifest."""
        rows: list[EvidenceIndexRow] = []
        for bundle in bundles:
            relative_path = _bundle_relative_path(bundle.run_id, bundle.event_id)
            write_json(self.root / relative_path, bundle.to_dict())
            rows.append(
                EvidenceIndexRow(
                    dataset=bundle.dataset,
                    run_id=bundle.run_id,
                    event_id=bundle.event_id,
                    event_source=bundle.event_source,
                    relative_path=relative_path,
                    matched_gt_event_id=bundle.matched_gt_event_id,
                    is_matched_pred_event=bundle.is_matched_pred_event,
                    top_variables=[item.variable for item in bundle.top_variables],
                )
            )

        write_jsonl(self.root / "index.jsonl", [row.to_dict() for row in rows])
        write_json(
            self.root / "manifest.json",
            {
                "format_version": "evidence-manifest-v1",
                "dataset": dataset,
                "event_source": event_source,
                "bundle_count": len(rows),
                "index_path": "index.jsonl",
                "bundle_root": "bundles",
            },
        )
        return rows

    def manifest(self) -> dict[str, Any]:
        """Read the evidence manifest."""
        return read_json(self.root / "manifest.json")

    def index_rows(self) -> list[EvidenceIndexRow]:
        """Read index rows from `index.jsonl`."""
        index_path = self.root / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Evidence index not found: {index_path}")
        rows: list[EvidenceIndexRow] = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError(f"Evidence index row must be an object: {index_path}")
                    rows.append(EvidenceIndexRow.from_dict(payload))
        return rows

    def read_bundle(self, row: EvidenceIndexRow) -> EvidenceBundle:
        """Read one evidence bundle from an index row."""
        return self.read_bundle_at(row.relative_path)

    def read_bundle_at(self, relative_path: str) -> EvidenceBundle:
        """Read one evidence bundle by relative path."""
        payload = read_json(self.root / relative_path)
        return EvidenceBundle.from_dict(payload)

    def discover(self) -> list[EvidenceBundle]:
        """Read all bundles in index order."""
        return [self.read_bundle(row) for row in self.index_rows()]


def _bundle_relative_path(run_id: str, event_id: str) -> str:
    return f"bundles/{_safe_id(run_id)}/{_safe_id(event_id)}/evidence.json"


def _safe_id(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in value
    )
    return cleaned.strip("._-") or "item"
