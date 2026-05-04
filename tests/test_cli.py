from __future__ import annotations

from typer.testing import CliRunner

from industrial_tsad_eval.interfaces.cli.main import app


def test_cli_vertical_smoke(tmp_path):
    runner = CliRunner()
    examples = tmp_path / "examples"
    scores = tmp_path / "scores"
    eval_out = tmp_path / "eval"

    result = runner.invoke(app, ["examples", "make-opcua-fixture", "--out", str(examples)])
    assert result.exit_code == 0, result.output

    prepared = examples / "OPCUA_SYNTH"
    result = runner.invoke(app, ["prepared", "validate", "--prepared", str(prepared)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "score",
            "run",
            "--prepared",
            str(prepared),
            "--out",
            str(scores),
            "--window",
            "32",
            "--stride",
            "4",
            "--lags",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        ["scores", "validate", "--prepared", str(prepared), "--scores", str(scores)],
    )
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--prepared",
            str(prepared),
            "--scores",
            str(scores),
            "--out",
            str(eval_out),
            "--threshold",
            "0.1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (eval_out / "metrics.json").exists()
