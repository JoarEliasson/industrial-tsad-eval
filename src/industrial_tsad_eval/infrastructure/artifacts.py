"""Local filesystem artifact writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from industrial_tsad_eval.infrastructure.json_utils import write_json


class LocalArtifactWriter:
    """Write application artifacts below a local root."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> None:
        """Write a JSON artifact below the writer root."""
        write_json(self.root / relative_path, payload)

    def write_text(self, relative_path: str, payload: str) -> None:
        """Write a text artifact below the writer root."""
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
