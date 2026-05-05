from __future__ import annotations

import numpy as np
import pandas as pd
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


def test_cli_dataset_adapter_commands(tmp_path):
    runner = CliRunner()
    raw = tmp_path / "raw_swat"
    _make_cli_swat_raw(raw)

    result = runner.invoke(app, ["prepared", "adapters"])
    assert result.exit_code == 0, result.output
    assert "swat" in result.output
    assert "hai-cpps" in result.output

    result = runner.invoke(app, ["prepared", "describe", "--dataset", "swat"])
    assert result.exit_code == 0, result.output
    assert "SWaT" in result.output

    out = tmp_path / "prepared"
    result = runner.invoke(
        app,
        [
            "prepared",
            "prepare",
            "--dataset",
            "swat",
            "--raw",
            str(raw),
            "--out",
            str(out),
            "--extra-json",
            '{"remove_startup": false}',
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "SWaT" / "meta" / "manifest.json").exists()


def _make_cli_swat_raw(root):
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=12, freq="1s"),
            "FIT101": np.linspace(1.0, 2.0, 12),
            "MV101": np.zeros(12, dtype=np.int64),
            "Normal/Attack": ["Normal"] * 12,
        }
    ).to_csv(root / "SWaT_Normal.csv", index=False)
    labels = ["Normal"] * 12
    labels[6:8] = ["Attack", "Attack"]
    pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-02", periods=12, freq="1s"),
            "FIT101": np.linspace(1.5, 2.5, 12),
            "MV101": np.ones(12, dtype=np.int64),
            "Normal/Attack": labels,
        }
    ).to_csv(root / "SWaT_Attack.csv", index=False)
