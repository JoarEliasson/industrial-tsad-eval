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
        "rq3-preflight",
    }
