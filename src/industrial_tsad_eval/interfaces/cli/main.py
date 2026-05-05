"""Typer command-line interface for Industrial TSAD Eval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from industrial_tsad_eval.application.benchmark import RunBenchmark
from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.preparation import PrepareDataset
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig
from industrial_tsad_eval.domain.errors import IndustrialTSADError
from industrial_tsad_eval.domain.policy import EvalPolicy
from industrial_tsad_eval.infrastructure.benchmark_config import (
    load_benchmark_config,
    write_default_benchmark_config,
)
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture
from industrial_tsad_eval.plugins.registry import (
    default_dataset_adapter_registry,
    default_detector_registry,
)

console = Console()

app = typer.Typer(help="Industrial time-series anomaly detection evaluation toolkit.")
prepared_app = typer.Typer(help="Prepared Format workflows.")
score_app = typer.Typer(help="Detector scoring workflows.")
scores_app = typer.Typer(help="Score Contract workflows.")
eval_app = typer.Typer(help="Evaluation workflows.")
examples_app = typer.Typer(help="Synthetic example fixture workflows.")
bench_app = typer.Typer(help="Benchmark orchestration workflows.")

app.add_typer(prepared_app, name="prepared")
app.add_typer(score_app, name="score")
app.add_typer(scores_app, name="scores")
app.add_typer(eval_app, name="eval")
app.add_typer(examples_app, name="examples")
app.add_typer(bench_app, name="bench")


@prepared_app.command("validate")
def validate_prepared(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
) -> None:
    """Validate a Prepared Format v1 dataset."""
    _emit_validation(ValidatePreparedDataset(prepared).run().to_dict())


@prepared_app.command("adapters")
def list_adapters() -> None:
    """List registered dataset adapter plugins."""
    registry = default_dataset_adapter_registry()
    table = Table("Name", "Dataset", "Expected Raw Layout")
    for name in registry.names():
        plugin = registry.get_dataset_adapter(name)
        table.add_row(plugin.name, plugin.dataset_name, plugin.describe_expected_raw_layout())
    console.print(table)


@prepared_app.command("describe")
def describe_adapter(
    dataset: str = typer.Option(..., "--dataset", help="Dataset adapter name."),
) -> None:
    """Describe the local raw layout expected by a dataset adapter."""
    try:
        plugin = default_dataset_adapter_registry().get_dataset_adapter(dataset)
    except IndustrialTSADError as exc:
        _fail(str(exc))
    console.print_json(
        data={
            "name": plugin.name,
            "dataset_name": plugin.dataset_name,
            "expected_raw_layout": plugin.describe_expected_raw_layout(),
        }
    )


@prepared_app.command("prepare")
def prepare_dataset(
    dataset: str = typer.Option(..., "--dataset", help="Dataset adapter name."),
    raw: Path = typer.Option(..., "--raw", file_okay=False, help="Local raw dataset directory."),
    out: Path = typer.Option(
        ..., "--out", file_okay=False, help="Output directory for prepared datasets."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Replace an existing prepared output."
    ),
    extra_json: str | None = typer.Option(
        None, "--extra-json", help="Adapter-specific JSON options."
    ),
    base_epoch_iso: str = typer.Option(
        "2020-01-01T00:00:00Z",
        "--base-epoch-iso",
        help="Base timestamp for raw data without timestamps.",
    ),
    default_period_ms: int = typer.Option(
        100,
        "--default-period-ms",
        min=1,
        help="Default sample period when raw data lacks timestamps.",
    ),
    strict: bool = typer.Option(
        True, "--strict/--no-strict", help="Enable strict adapter behavior."
    ),
) -> None:
    """Prepare local raw data through a registered dataset adapter."""
    try:
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
                extra=_json_object(extra_json),
            ),
        ).run()
    except (IndustrialTSADError, ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())


@score_app.command("run")
def score_run(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    detector: str = typer.Option(..., "--detector", help="Detector plugin name."),
    out: Path = typer.Option(
        ..., "--out", file_okay=False, help="Output score artifact directory."
    ),
    protocol: str = typer.Option("naive", "--protocol", help="Split protocol to score."),
    parameters_json: str | None = typer.Option(
        None,
        "--parameters-json",
        help="Detector parameter JSON object.",
    ),
) -> None:
    """Train a detector and write Score Contract v1 artifacts."""
    try:
        result = ScoreRuns(
            detector_registry=default_detector_registry(),
            prepared=prepared,
            scores=out,
            detector_name=detector,
            protocol=protocol,
            detector_parameters=_json_object(parameters_json),
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.__dict__)


@scores_app.command("validate")
def validate_scores(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    scores: Path = typer.Option(..., "--scores", file_okay=False, help="Score artifact directory."),
) -> None:
    """Validate Score Contract v1 artifacts."""
    _emit_validation(ValidateScores(prepared, scores).run().to_dict())


@eval_app.command("run")
def eval_run(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    scores: Path = typer.Option(..., "--scores", file_okay=False, help="Score artifact directory."),
    out: Path = typer.Option(
        ..., "--out", file_okay=False, help="Output evaluation artifact directory."
    ),
    protocol: str = typer.Option("naive", "--protocol", help="Split protocol to evaluate."),
    threshold: float | None = typer.Option(None, "--threshold", help="Manual score threshold."),
    threshold_quantile: float = typer.Option(
        0.995,
        "--threshold-quantile",
        min=0.0,
        max=1.0,
        help="Calibration quantile when no manual threshold is provided.",
    ),
) -> None:
    """Evaluate scores against prepared event labels."""
    try:
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
    console.print_json(data=result.__dict__)


@examples_app.command("make-opcua-fixture")
def make_fixture(
    out: Path = typer.Option(..., "--out", file_okay=False, help="Output directory."),
) -> None:
    """Generate a small OPC-UA-like Prepared Format fixture."""
    try:
        path = make_opcua_fixture(out)
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data={"prepared": str(path)})


@bench_app.command("init-config")
def bench_init_config(
    out: Path = typer.Option(..., "--out", dir_okay=False, help="TOML file to create."),
) -> None:
    """Write a starter benchmark TOML config."""
    try:
        path = write_default_benchmark_config(out)
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data={"config": str(path)})


@bench_app.command("plan")
def bench_plan(
    config: Path = typer.Option(..., "--config", dir_okay=False, help="Benchmark TOML config."),
) -> None:
    """Print the resolved benchmark matrix without running it."""
    try:
        loaded = load_benchmark_config(config)
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(
        data={
            "benchmark": loaded.name,
            "experiment_count": len(loaded.experiments()),
            "experiments": [
                {
                    "experiment_id": experiment.experiment_id,
                    "dataset": experiment.dataset.id,
                    "detector": experiment.detector.id,
                    "protocol": experiment.protocol,
                }
                for experiment in loaded.experiments()
            ],
        }
    )


@bench_app.command("run")
def bench_run(
    config: Path = typer.Option(..., "--config", dir_okay=False, help="Benchmark TOML config."),
    out: Path = typer.Option(
        ..., "--out", file_okay=False, help="Benchmark runs output directory."
    ),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id override."),
) -> None:
    """Run the benchmark matrix and write structured run artifacts."""
    try:
        loaded = load_benchmark_config(config)
        result = RunBenchmark(
            config=loaded,
            detector_registry=default_detector_registry(),
            out=out,
            run_id=run_id,
            source_config=config,
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())
    if not result.ok:
        raise typer.Exit(1)


@bench_app.command("summarize")
def bench_summarize(
    run: Path = typer.Option(..., "--run", file_okay=False, help="Benchmark run directory."),
) -> None:
    """Print benchmark summary rows from a completed run directory."""
    summary_path = run / "summary.json"
    if not summary_path.exists():
        _fail(f"Missing benchmark summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    experiments = payload.get("experiments", [])
    console.print_json(
        data={"run": str(run), "experiments": experiments if isinstance(experiments, list) else []}
    )


def _emit_validation(report: dict[str, Any]) -> None:
    console.print_json(data=report)
    if not bool(report.get("ok")):
        raise typer.Exit(1)


def _json_object(value: str | None) -> dict[str, Any]:
    if value is None or not value.strip():
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("JSON value must be an object.")
    return payload


def _fail(message: str) -> NoReturn:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)
