"""Artifact writer port."""

from __future__ import annotations

from typing import Any, Protocol


class ArtifactWriter(Protocol):
    """Write JSON-compatible application artifacts."""

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> None:
        """Write a JSON artifact below the writer root."""
