from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from industrial_tsad_eval.interfaces.cli.main import app

runner = CliRunner()


def test_cli_reproducibility_audit_smoke(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "audit",
            "run",
            "--out",
            str(tmp_path / "audit"),
            "--audit-id",
            "test-audit",
            "--skip-optional",
        ],
    )

    assert result.exit_code == 0, result.output
    audit_root = tmp_path / "audit" / "test-audit"
    assert (audit_root / "audit_summary.json").exists()
    assert (audit_root / "audit_summary.md").exists()
    assert (audit_root / "reproduction" / "smoke-audit" / "summary.json").exists()
    payload = json.loads((audit_root / "audit_summary.json").read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert {check["name"] for check in payload["checks"]} >= {
        "package-import",
        "cli-help",
        "architecture-tests",
        "reproduction-run",
        "reproduction-artifacts",
        "assistant-preflight",
        "synthetic-thesis-setup",
    }
    synthetic_root = audit_root / "synthetic-full-reproduction" / "thesis-full-smoke"
    assert (synthetic_root / "summary.json").exists()
    assert (synthetic_root / "summaries" / "assistant_summary.csv").exists()
    assert "setup_recommendations" in payload


def test_audit_optional_recommendations_are_actionable(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "audit",
            "run",
            "--out",
            str(tmp_path / "audit"),
            "--audit-id",
            "optional-audit",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(
        (tmp_path / "audit" / "optional-audit" / "audit_summary.json").read_text(encoding="utf-8")
    )
    recommendations = payload["setup_recommendations"]
    assert isinstance(recommendations, list)
    for recommendation in recommendations:
        assert recommendation["commands"]
        assert recommendation["success_criteria"]
