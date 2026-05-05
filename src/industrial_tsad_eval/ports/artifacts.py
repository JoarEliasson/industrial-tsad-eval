"""Artifact writer ports."""

from __future__ import annotations

from typing import Any, Protocol


class ArtifactWriter(Protocol):
    """Write named artifacts for an application use case."""

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> None:
        """Write a JSON artifact."""
