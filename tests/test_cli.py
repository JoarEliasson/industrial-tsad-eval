from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from industrial_tsad_eval.interfaces.cli.main import app
from tests.conftest import write_swat_raw

runner = CliRunner()


def test_cli_core_vertical_slice(tmp_path: Path):
    examples = tmp_path / "examples"
    scores = tmp_path / "scores"
    eval_out = tmp_path / "eval"

    result = runner.invoke(app, ["examples", "make-opcua-fixture", "--out", str(examples)])
    assert result.exit_code == 0, result.output
    prepared = examples / "OPCUA_SYNTH"

    assert runner.invoke(app, ["prepared", "validate", "--prepared", str(prepared)]).exit_code == 0
    score_result = runner.invoke(
        app,
        [
            "score",
            "run",
            "--prepared",
            str(prepared),
            "--detector",
            "forecast-ridge",
            "--out",
            str(scores),
            "--parameters-json",
            '{"window": 24, "stride": 4, "lags": 1}',
        ],
    )
    assert score_result.exit_code == 0, score_result.output
    assert (
        runner.invoke(
            app, ["scores", "validate", "--prepared", str(prepared), "--scores", str(scores)]
        ).exit_code
        == 0
    )
    eval_result = runner.invoke(
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
        ],
    )
    assert eval_result.exit_code == 0, eval_result.output


def test_cli_lists_detectors():
    result = runner.invoke(app, ["score", "detectors"])

    assert result.exit_code == 0, result.output
    assert "forecast-ridge" in result.output
    assert "forecast-lstm" in result.output


def test_cli_scores_with_torch_detector(tmp_path: Path):
    pytest.importorskip("torch")
    examples = tmp_path / "examples"
    scores = tmp_path / "scores"
    runner.invoke(app, ["examples", "make-opcua-fixture", "--out", str(examples)])

    result = runner.invoke(
        app,
        [
            "score",
            "run",
            "--prepared",
            str(examples / "OPCUA_SYNTH"),
            "--detector",
            "forecast-lstm",
            "--out",
            str(scores),
            "--parameters-json",
            (
                '{"window": 16, "train_stride": 8, "score_stride": 8, '
                '"max_train_windows": 24, "epochs": 1, "batch_size": 8, '
                '"device": "cpu", "hidden_size": 8}'
            ),
        ],
    )

    assert result.exit_code == 0, result.output


def test_cli_prepared_adapters_describe_and_prepare(tmp_path: Path):
    raw = write_swat_raw(tmp_path / "raw")

    assert runner.invoke(app, ["prepared", "adapters"]).exit_code == 0
    assert runner.invoke(app, ["prepared", "describe", "--dataset", "swat"]).exit_code == 0
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
            str(tmp_path / "prepared"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "prepared" / "SWaT" / "meta" / "manifest.json").exists()


def test_cli_data_sources_describe_acquire_and_validate(tmp_path: Path):
    raw = write_swat_raw(tmp_path / "raw")

    assert runner.invoke(app, ["data", "sources"]).exit_code == 0
    assert runner.invoke(app, ["data", "describe", "--source", "swat"]).exit_code == 0
    acquire = runner.invoke(
        app,
        [
            "data",
            "acquire",
            "--source",
            "swat",
            "--method",
            "manual",
            "--manual",
            str(raw),
            "--out",
            str(tmp_path / "raw-cache"),
        ],
    )
    assert acquire.exit_code == 0, acquire.output
    raw_cache = tmp_path / "raw-cache" / "SWaT"
    assert (raw_cache / "raw_provenance.json").exists()
    validate = runner.invoke(
        app,
        ["data", "validate", "--source", "swat", "--raw", str(raw_cache)],
    )
    assert validate.exit_code == 0, validate.output


def test_cli_benchmark_commands(tmp_path: Path):
    examples = tmp_path / "examples"
    runner.invoke(app, ["examples", "make-opcua-fixture", "--out", str(examples)])
    config = tmp_path / "benchmark.toml"
    init = runner.invoke(app, ["bench", "init-config", "--out", str(config)])
    assert init.exit_code == 0, init.output
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            "examples/generated/OPCUA_SYNTH",
            str((examples / "OPCUA_SYNTH").resolve()).replace("\\", "\\\\"),
        ),
        encoding="utf-8",
    )

    assert runner.invoke(app, ["bench", "plan", "--config", str(config)]).exit_code == 0
    run = runner.invoke(
        app,
        [
            "bench",
            "run",
            "--config",
            str(config),
            "--out",
            str(tmp_path / "bench-runs"),
            "--run-id",
            "cli-run",
        ],
    )
    assert run.exit_code == 0, run.output
    assert (
        runner.invoke(
            app, ["bench", "summarize", "--run", str(tmp_path / "bench-runs" / "cli-run")]
        ).exit_code
        == 0
    )


def test_cli_system_and_profile_commands(tmp_path: Path):
    examples = tmp_path / "examples"
    runner.invoke(app, ["examples", "make-opcua-fixture", "--out", str(examples)])
    prepared = examples / "OPCUA_SYNTH"

    gpu = runner.invoke(app, ["system", "gpu-check", "--json"])
    assert gpu.exit_code == 0, gpu.output

    report_path = tmp_path / "machine_env.json"
    report = runner.invoke(app, ["system", "report", "--out", str(report_path)])
    assert report.exit_code == 0, report.output
    assert report_path.exists()

    preflight = runner.invoke(
        app,
        [
            "system",
            "preflight",
            "--prepared",
            str(prepared),
            "--detector",
            "forecast-ridge",
            "--out",
            str(tmp_path / "preflight"),
        ],
    )
    assert preflight.exit_code == 0, preflight.output

    profile = runner.invoke(
        app,
        [
            "profile",
            "run",
            "--prepared",
            str(prepared),
            "--detector",
            "forecast-ridge",
            "--out",
            str(tmp_path / "profiles"),
            "--profile-id",
            "cli-profile",
            "--parameters-json",
            '{"window": 24, "stride": 4, "lags": 1}',
        ],
    )
    assert profile.exit_code == 0, profile.output
    assert (tmp_path / "profiles" / "cli-profile" / "summary.json").exists()


def test_cli_evidence_and_xai_commands(tmp_path: Path):
    examples = tmp_path / "examples"
    scores = tmp_path / "scores"
    eval_out = tmp_path / "eval"
    evidence = tmp_path / "evidence"
    gt_map = tmp_path / "gt_map.json"
    xai_out = tmp_path / "xai"
    runner.invoke(app, ["examples", "make-opcua-fixture", "--out", str(examples)])
    prepared = examples / "OPCUA_SYNTH"

    score = runner.invoke(
        app,
        [
            "score",
            "run",
            "--prepared",
            str(prepared),
            "--detector",
            "forecast-ridge",
            "--out",
            str(scores),
            "--parameters-json",
            '{"window": 24, "stride": 4, "lags": 1}',
        ],
    )
    assert score.exit_code == 0, score.output
    eval_result = runner.invoke(
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
        ],
    )
    assert eval_result.exit_code == 0, eval_result.output

    generated = runner.invoke(
        app,
        [
            "evidence",
            "generate",
            "--prepared",
            str(prepared),
            "--scores",
            str(scores),
            "--out",
            str(evidence),
        ],
    )
    assert generated.exit_code == 0, generated.output
    assert (
        runner.invoke(
            app, ["evidence", "validate", "--prepared", str(prepared), "--evidence", str(evidence)]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app, ["xai", "gt-map", "build", "--prepared", str(prepared), "--out", str(gt_map)]
        ).exit_code
        == 0
    )
    assert runner.invoke(app, ["xai", "gt-map", "validate", "--gt-map", str(gt_map)]).exit_code == 0
    xai = runner.invoke(
        app,
        [
            "xai",
            "eval",
            "--prepared",
            str(prepared),
            "--evidence",
            str(evidence),
            "--gt-map",
            str(gt_map),
            "--out",
            str(xai_out),
            "--ks",
            "1,3,5",
        ],
    )
    assert xai.exit_code == 0, xai.output
    assert (xai_out / "metrics.json").exists()
