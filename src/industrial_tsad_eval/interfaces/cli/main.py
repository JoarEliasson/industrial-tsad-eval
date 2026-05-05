"""Command-line entrypoint for Industrial TSAD Eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.preparation import PrepareDataset
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig
from industrial_tsad_eval.domain.errors import IndustrialTSADError
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture
from industrial_tsad_eval.plugins.registry import (
    default_dataset_adapter_registry,
    default_detector_registry,
)

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


@prepared_app.command("adapters")
def list_prepared_adapters_cmd() -> None:
    """List built-in dataset adapter plugins."""
    registry = default_dataset_adapter_registry()
    table = Table(title="Dataset Adapters")
    table.add_column("Name")
    table.add_column("Dataset")
    for name in registry.names():
        plugin = registry.get_dataset_adapter(name)
        table.add_row(plugin.name, plugin.dataset_name)
    console.print(table)


@prepared_app.command("describe")
def describe_prepared_adapter_cmd(
    dataset: str = typer.Option(..., "--dataset"),
) -> None:
    """Describe the raw layout expected by one adapter."""
    try:
        plugin = default_dataset_adapter_registry().get_dataset_adapter(dataset)
    except IndustrialTSADError as exc:
        _fail(str(exc))
    console.print(f"[bold]{plugin.dataset_name}[/bold] ({plugin.name})")
    console.print(plugin.describe_expected_raw_layout())


@prepared_app.command("prepare")
def prepare_dataset_cmd(
    dataset: str = typer.Option(..., "--dataset"),
    raw: Path = typer.Option(..., "--raw", exists=True, file_okay=False),
    out: Path = typer.Option(..., "--out"),
    overwrite: bool = typer.Option(False, "--overwrite"),
    extra_json: str = typer.Option("{}", "--extra-json"),
    base_epoch_iso: str = typer.Option("2020-01-01T00:00:00Z", "--base-epoch-iso"),
    default_period_ms: int = typer.Option(100, "--default-period-ms"),
    strict: bool = typer.Option(True, "--strict/--no-strict"),
) -> None:
    """Prepare a local raw dataset into Prepared Format v1."""
    try:
        extra = _parse_json_object(extra_json)
        result = PrepareDataset(
            adapter_registry=default_dataset_adapter_registry(),
            dataset=dataset,
            raw=raw,
            out=out,
            overwrite=overwrite,
            config=DatasetAdapterConfig(
                base_epoch_iso=base_epoch_iso,
                default_period_ms=default_period_ms,
                strict=strict,
                extra=extra,
            ),
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print(f"[green]Prepared {result.dataset_name} with {result.run_count} runs[/green]")
    console.print_json(json.dumps(result.to_dict(), sort_keys=True))


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


def _fail(message: str) -> NoReturn:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)


def _parse_json_object(payload: str) -> dict[str, Any]:
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("--extra-json must be a JSON object.")
    return parsed


if __name__ == "__main__":
    app()
