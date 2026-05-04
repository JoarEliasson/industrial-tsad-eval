"""Command-line entrypoint for Industrial TSAD Eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.errors import IndustrialTSADError
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture
from industrial_tsad_eval.plugins.registry import default_detector_registry

console = Console()
app = typer.Typer(
    help="Industrial time-series anomaly-detection evaluation toolkit.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
prepared_app = typer.Typer(help="Prepared Format v1 commands.", no_args_is_help=True)
score_app = typer.Typer(help="Detector scoring commands.", no_args_is_help=True)
scores_app = typer.Typer(help="Score Contract v1 commands.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation commands.", no_args_is_help=True)
examples_app = typer.Typer(help="Example fixture commands.", no_args_is_help=True)

app.add_typer(prepared_app, name="prepared")
app.add_typer(score_app, name="score")
app.add_typer(scores_app, name="scores")
app.add_typer(eval_app, name="eval")
app.add_typer(examples_app, name="examples")


@prepared_app.command("validate")
def validate_prepared_cmd(
    prepared: Path = typer.Option(..., "--prepared", exists=True, file_okay=False),
) -> None:
    """Validate a Prepared Format v1 dataset."""
    report = ValidatePreparedDataset(prepared).run()
    _print_validation_report("Prepared Dataset", report.to_dict())
    if not report.ok:
        raise typer.Exit(2)


@score_app.command("run")
def score_run_cmd(
    prepared: Path = typer.Option(..., "--prepared", exists=True, file_okay=False),
    detector: str = typer.Option("forecast-ridge", "--detector"),
    out: Path = typer.Option(..., "--out"),
    protocol: str = typer.Option("naive", "--protocol"),
    window: int = typer.Option(128, "--window"),
    stride: int = typer.Option(16, "--stride"),
    alpha: float = typer.Option(1.0, "--alpha"),
    lags: int = typer.Option(1, "--lags"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize"),
    seed: int = typer.Option(1337, "--seed"),
) -> None:
    """Train a detector plugin and write Score Contract v1 artifacts."""
    parameters: dict[str, Any] = {
        "window": window,
        "stride": stride,
        "alpha": alpha,
        "lags": lags,
        "standardize": standardize,
        "seed": seed,
    }
    try:
        result = ScoreRuns(
            detector_registry=default_detector_registry(),
            prepared=prepared,
            scores=out,
            detector_name=detector,
            protocol=protocol,
            detector_parameters=parameters,
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print(f"[green]Scored {len(result.runs_scored)} runs[/green]")
    console.print_json(json.dumps(result.__dict__, sort_keys=True))


@scores_app.command("validate")
def validate_scores_cmd(
    prepared: Path = typer.Option(..., "--prepared", exists=True, file_okay=False),
    scores: Path = typer.Option(..., "--scores", exists=True, file_okay=False),
) -> None:
    """Validate Score Contract v1 artifacts against a prepared dataset."""
    report = ValidateScores(prepared, scores).run()
    _print_validation_report("Scores", report.to_dict())
    if not report.ok:
        raise typer.Exit(2)


@eval_app.command("run")
def eval_run_cmd(
    prepared: Path = typer.Option(..., "--prepared", exists=True, file_okay=False),
    scores: Path = typer.Option(..., "--scores", exists=True, file_okay=False),
    out: Path = typer.Option(..., "--out"),
    protocol: str = typer.Option("naive", "--protocol"),
    threshold: float | None = typer.Option(None, "--threshold"),
    threshold_quantile: float = typer.Option(0.995, "--threshold-quantile"),
) -> None:
    """Evaluate score artifacts and write metrics."""
    try:
        from industrial_tsad_eval.domain.policy import EvalPolicy

        result = EvaluateScores(
            prepared=prepared,
            scores=scores,
            out=out,
            protocol=protocol,
            threshold=threshold,
            policy=EvalPolicy(protocol=protocol, threshold_quantile=threshold_quantile),
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print(f"[green]Evaluation written to {result.out_dir}[/green]")
    console.print_json(json.dumps(result.metrics, sort_keys=True))


@examples_app.command("make-opcua-fixture")
def make_fixture_cmd(
    out: Path = typer.Option(..., "--out"),
) -> None:
    """Generate a small OPC-UA-like Prepared Format v1 fixture."""
    dataset_root = make_opcua_fixture(out)
    console.print(f"[green]Prepared fixture written to {dataset_root}[/green]")


def _print_validation_report(title: str, payload: dict[str, Any]) -> None:
    table = Table(title=title, show_lines=True)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Path", str(payload["path"]))
    table.add_row("OK", str(payload["ok"]))
    table.add_row("Errors", str(len(payload["errors"])))
    table.add_row("Warnings", str(len(payload["warnings"])))
    console.print(table)
    for label in ("errors", "warnings"):
        values = payload[label]
        if values:
            console.print(f"[bold]{label.title()}[/bold]")
            for value in values:
                console.print(f"- {value}")


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
