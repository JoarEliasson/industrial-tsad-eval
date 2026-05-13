"""Typer command-line interface for Industrial TSAD Eval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from industrial_tsad_eval.application.acquisition import (
    AcquireDatasetSource,
    DescribeDatasetSource,
    ListDatasetSources,
    ValidateRawAcquisition,
)
from industrial_tsad_eval.application.assistant_replay import (
    PreflightAssistantReplay,
    RunAssistantReplaySuite,
    SummarizeAssistantReplay,
)
from industrial_tsad_eval.application.audit import (
    ReproducibilityAuditConfig,
    RunReproducibilityAudit,
)
from industrial_tsad_eval.application.benchmark import RunBenchmark
from industrial_tsad_eval.application.evaluation import EvaluateScores
from industrial_tsad_eval.application.evidence import (
    BuildGroundTruthTagMap,
    GenerateEvidence,
    ValidateEvidence,
    ValidateGroundTruthTagMap,
)
from industrial_tsad_eval.application.operator import (
    GenerateOperatorCards,
    RetrieveOperatorEvidence,
    ValidateOperatorCards,
)
from industrial_tsad_eval.application.preflight import PreflightInput, RunPreflight
from industrial_tsad_eval.application.preparation import PrepareDataset
from industrial_tsad_eval.application.profiling import (
    ProfileScoreEvaluate,
    ProfileScoreEvaluateConfig,
)
from industrial_tsad_eval.application.reproduction import (
    DiagnoseThesisReproduction,
    PlanThesisReproduction,
    PreflightThesisReproduction,
    RunThesisReproduction,
    SummarizeThesisReproduction,
)
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.application.xai import EvaluateEvidence, EvaluateEvidenceConfig
from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig
from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig
from industrial_tsad_eval.domain.errors import IndustrialTSADError
from industrial_tsad_eval.domain.policy import EvalPolicy
from industrial_tsad_eval.infrastructure.benchmark_config import (
    load_benchmark_config,
    write_default_benchmark_config,
)
from industrial_tsad_eval.infrastructure.examples import (
    make_opcua_fixture,
    make_thesis_raw_fixtures,
)
from industrial_tsad_eval.infrastructure.json_utils import write_json
from industrial_tsad_eval.infrastructure.progress import read_run_progress
from industrial_tsad_eval.infrastructure.reproduction_config import (
    load_assistant_config,
    load_reproduction_config,
    write_default_reproduction_config,
    write_provider_config_template,
)
from industrial_tsad_eval.infrastructure.system import (
    capture_machine_environment,
    detect_system_gpus,
    probe_torch_runtime,
    recommend_backend_for_runtime,
)
from industrial_tsad_eval.interfaces.cli.progress import cli_progress
from industrial_tsad_eval.plugins.providers import default_llm_provider_registry
from industrial_tsad_eval.plugins.registry import (
    default_dataset_adapter_registry,
    default_dataset_source_registry,
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
system_app = typer.Typer(help="System and accelerator diagnostics.")
profile_app = typer.Typer(help="Runtime profiling workflows.")
evidence_app = typer.Typer(help="Evidence Bundle workflows.")
xai_app = typer.Typer(help="Explanation-quality evaluation workflows.")
xai_gt_map_app = typer.Typer(help="Ground-truth tag-map workflows.")
data_app = typer.Typer(help="Raw dataset acquisition workflows.")
operator_app = typer.Typer(help="Deterministic operator-assistant workflows.")
operator_card_app = typer.Typer(help="Operator card workflows.")
assistant_app = typer.Typer(help="Thesis assistant replay workflows.")
reproduce_app = typer.Typer(help="Thesis-style reproduction workflows.")
audit_app = typer.Typer(help="Clean-repo reproducibility audit workflows.")

app.add_typer(prepared_app, name="prepared")
app.add_typer(score_app, name="score")
app.add_typer(scores_app, name="scores")
app.add_typer(eval_app, name="eval")
app.add_typer(examples_app, name="examples")
app.add_typer(bench_app, name="bench")
app.add_typer(system_app, name="system")
app.add_typer(profile_app, name="profile")
app.add_typer(evidence_app, name="evidence")
app.add_typer(xai_app, name="xai")
app.add_typer(data_app, name="data")
app.add_typer(operator_app, name="operator")
app.add_typer(assistant_app, name="assistant")
app.add_typer(reproduce_app, name="reproduce")
app.add_typer(audit_app, name="audit")
xai_app.add_typer(xai_gt_map_app, name="gt-map")
operator_app.add_typer(operator_card_app, name="card")


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


@data_app.command("sources")
def data_sources() -> None:
    """List registered raw dataset source plugins."""
    rows = ListDatasetSources(source_registry=default_dataset_source_registry()).run()
    table = Table("Name", "Dataset", "Methods", "Description")
    for row in rows:
        table.add_row(row.name, row.dataset_name, ", ".join(row.supported_methods), row.description)
    console.print(table)


@data_app.command("describe")
def data_describe(
    source: str = typer.Option(..., "--source", help="Dataset source name."),
) -> None:
    """Describe a raw dataset source plugin."""
    try:
        description = DescribeDatasetSource(
            source_registry=default_dataset_source_registry(),
            source=source,
        ).run()
    except IndustrialTSADError as exc:
        _fail(str(exc))
    console.print_json(data=description.to_dict())


@data_app.command("acquire")
def data_acquire(
    source: str = typer.Option(..., "--source", help="Dataset source name."),
    out: Path = typer.Option(..., "--out", file_okay=False, help="Raw data cache directory."),
    method: str = typer.Option("manual", "--method", help="Acquisition method."),
    manual: Path | None = typer.Option(
        None, "--manual", exists=False, help="Manual local raw path."
    ),
    ref: str | None = typer.Option(None, "--ref", help="Online source reference."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing raw output."),
    extra_json: str | None = typer.Option(
        None, "--extra-json", help="Source-specific JSON options."
    ),
) -> None:
    """Acquire raw dataset files into a local raw-data cache."""
    try:
        result = AcquireDatasetSource(
            source_registry=default_dataset_source_registry(),
            source=source,
            out=out,
            config=DatasetSourceConfig(
                method=method,
                manual_path=str(manual) if manual is not None else None,
                ref=ref,
                overwrite=overwrite,
                extra=_json_object(extra_json),
            ),
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())


@data_app.command("validate")
def data_validate(
    source: str = typer.Option(..., "--source", help="Dataset source name."),
    raw: Path = typer.Option(..., "--raw", file_okay=False, help="Raw acquisition directory."),
) -> None:
    """Validate raw acquisition provenance and file inventory."""
    _emit_validation(
        ValidateRawAcquisition(
            source_registry=default_dataset_source_registry(),
            source=source,
            raw=raw,
        )
        .run()
        .to_dict()
    )


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


@score_app.command("detectors")
def list_detectors() -> None:
    """List registered detector plugins."""
    registry = default_detector_registry()
    table = Table("Name")
    for name in registry.names():
        table.add_row(name)
    console.print(table)


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


@examples_app.command("make-thesis-raw-fixtures")
def make_thesis_raw_fixtures_command(
    out: Path = typer.Option(..., "--out", file_okay=False, help="Output directory."),
) -> None:
    """Generate tiny raw fixtures for thesis-style dataset adapters."""
    try:
        paths = make_thesis_raw_fixtures(out)
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data={"raw_fixtures": paths})


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
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable live progress UI."),
) -> None:
    """Run the benchmark matrix and write structured run artifacts."""
    try:
        loaded = load_benchmark_config(config)
        with cli_progress(not no_progress) as progress:
            result = RunBenchmark(
                config=loaded,
                detector_registry=default_detector_registry(),
                out=out,
                run_id=run_id,
                source_config=config,
                progress_sink=progress,
            ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())
    if not result.ok:
        raise typer.Exit(1)


@bench_app.command("status")
def bench_status(
    run: Path = typer.Option(..., "--run", file_okay=False, help="Benchmark run directory."),
    watch: bool = typer.Option(False, "--watch", help="Refresh until the run completes."),
    interval_s: float = typer.Option(10.0, "--interval-s", help="Watch refresh interval."),
) -> None:
    """Print benchmark run progress status."""
    _status_loop(run, watch=watch, interval_s=interval_s)


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


@system_app.command("gpu-check")
def system_gpu_check(
    device: str = typer.Option("auto", "--device", help="Requested device: auto, cpu, cuda, xpu."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of tables."),
    strict: bool = typer.Option(
        False, "--strict", help="Exit nonzero when recommended backend is not ready."
    ),
) -> None:
    """Inspect GPU adapters and torch runtime readiness."""
    try:
        gpus = detect_system_gpus()
        runtime = probe_torch_runtime(device)
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    recommended = recommend_backend_for_runtime(gpus, runtime)
    recommended_ready = _backend_ready(recommended, runtime.to_dict())
    payload = {
        "system_gpus": [gpu.to_dict() for gpu in gpus],
        "recommended_backend": recommended,
        "recommended_backend_ready": recommended_ready,
        "runtime": runtime.to_dict(),
    }
    if json_out:
        console.print_json(data=payload)
    else:
        table = Table("GPU", "Vendor", "Source")
        for gpu in gpus:
            table.add_row(gpu.name, gpu.vendor, gpu.source)
        if not gpus:
            table.add_row("No GPU detected", "-", "-")
        console.print(table)
        console.print_json(data=payload["runtime"])
    if strict and not recommended_ready:
        raise typer.Exit(1)


@system_app.command("report")
def system_report(
    out: Path = typer.Option(..., "--out", dir_okay=False, help="Machine report JSON file."),
    device: str = typer.Option("auto", "--device", help="Requested device: auto, cpu, cuda, xpu."),
    full_package_dump: bool = typer.Option(
        False, "--full-package-dump", help="Include all packages."
    ),
) -> None:
    """Write a machine environment report."""
    try:
        report = capture_machine_environment(
            device_request=device,
            full_package_dump=full_package_dump,
        )
        write_json(out, report.to_dict())
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data={"out": str(out), "machine_profile": report.machine_profile})


@system_app.command("preflight")
def system_preflight(
    prepared: Path | None = typer.Option(
        None, "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    detector: str | None = typer.Option(None, "--detector", help="Detector plugin name."),
    parameters_json: str | None = typer.Option(
        None, "--parameters-json", help="Detector parameter JSON object."
    ),
    device: str = typer.Option("auto", "--device", help="Requested device: auto, cpu, cuda, xpu."),
    out: Path | None = typer.Option(
        None, "--out", file_okay=False, help="Optional report output directory."
    ),
    strict: bool = typer.Option(False, "--strict", help="Exit nonzero when preflight fails."),
) -> None:
    """Run dataset, detector, runtime, and output preflight checks."""
    try:
        report = RunPreflight(
            detector_registry=default_detector_registry(),
            config=PreflightInput(
                prepared=prepared,
                detector_name=detector,
                detector_parameters=_json_object(parameters_json),
                device=device,
                out=out,
                strict=False,
            ),
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=report.to_dict())
    if strict and not report.ok:
        raise typer.Exit(1)


@profile_app.command("run")
def profile_run(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    detector: str = typer.Option(..., "--detector", help="Detector plugin name."),
    out: Path = typer.Option(..., "--out", file_okay=False, help="Profile output root."),
    protocol: str = typer.Option("naive", "--protocol", help="Split protocol."),
    parameters_json: str | None = typer.Option(
        None, "--parameters-json", help="Detector parameter JSON object."
    ),
    device: str = typer.Option("auto", "--device", help="Requested device: auto, cpu, cuda, xpu."),
    profile_id: str | None = typer.Option(None, "--profile-id", help="Optional profile id."),
) -> None:
    """Profile validate, score, validate scores, and evaluate stages."""
    try:
        result = ProfileScoreEvaluate(
            detector_registry=default_detector_registry(),
            config=ProfileScoreEvaluateConfig(
                prepared=prepared,
                detector_name=detector,
                out=out,
                protocol=protocol,
                detector_parameters=_json_object(parameters_json),
                device=device,
                profile_id=profile_id,
            ),
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())


@evidence_app.command("generate")
def evidence_generate(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    scores: Path = typer.Option(..., "--scores", file_okay=False, help="Score artifact root."),
    out: Path = typer.Option(..., "--out", file_okay=False, help="Evidence output root."),
    eval_dir: Path | None = typer.Option(
        None, "--eval", file_okay=False, help="Evaluation artifact root."
    ),
    event_source: str = typer.Option(
        "oracle", "--event-source", help="Event source: oracle or operational."
    ),
    protocol: str = typer.Option("naive", "--protocol", help="Split protocol."),
    top_k: int = typer.Option(5, "--top-k", min=1, help="Number of top variables."),
    max_events: int = typer.Option(100, "--max-events", min=1, help="Maximum events."),
    explanation_source: str = typer.Option(
        "auto", "--explanation-source", help="Explanation source: auto, native, or robust."
    ),
    native_missing_policy: str = typer.Option(
        "skip_bundle",
        "--native-missing-policy",
        help="Policy when native rows do not overlap an event: skip_bundle, fallback_robust, fail.",
    ),
) -> None:
    """Generate detector-agnostic Evidence Bundle v1 artifacts."""
    try:
        result = GenerateEvidence(
            prepared=prepared,
            scores=scores,
            out=out,
            eval_dir=eval_dir,
            event_source=event_source,
            protocol=protocol,
            top_k=top_k,
            max_events=max_events,
            explanation_source=explanation_source,
            native_missing_policy=native_missing_policy,
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())


@evidence_app.command("validate")
def evidence_validate(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    evidence: Path = typer.Option(..., "--evidence", file_okay=False, help="Evidence root."),
) -> None:
    """Validate Evidence Bundle v1 artifacts."""
    _emit_validation(ValidateEvidence(prepared, evidence).run().to_dict())


@operator_app.command("retrieve")
def operator_retrieve(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    evidence: Path = typer.Option(..., "--evidence", file_okay=False, help="Evidence root."),
    query: str = typer.Option(..., "--query", help="Operator query."),
    event_id: str | None = typer.Option(None, "--event-id", help="Optional event id filter."),
    dataset: str | None = typer.Option(None, "--dataset", help="Optional dataset filter."),
    playbooks: Path | None = typer.Option(
        None, "--playbooks", file_okay=False, help="Optional local Markdown playbooks."
    ),
    top_k: int = typer.Option(8, "--top-k", min=1, help="Number of retrieval hits."),
) -> None:
    """Retrieve deterministic evidence chunks for an operator query."""
    try:
        result = RetrieveOperatorEvidence(
            prepared=prepared,
            evidence=evidence,
            query=query,
            event_id=event_id,
            dataset=dataset,
            playbooks=playbooks,
            top_k=top_k,
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())


@operator_card_app.command("generate")
def operator_card_generate(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    evidence: Path = typer.Option(..., "--evidence", file_okay=False, help="Evidence root."),
    out: Path = typer.Option(..., "--out", file_okay=False, help="Operator card output root."),
    query: str | None = typer.Option(None, "--query", help="Optional operator query."),
    event_id: str | None = typer.Option(None, "--event-id", help="Optional event id filter."),
    dataset: str | None = typer.Option(None, "--dataset", help="Optional dataset filter."),
    playbooks: Path | None = typer.Option(
        None, "--playbooks", file_okay=False, help="Optional local Markdown playbooks."
    ),
    max_cards: int = typer.Option(25, "--max-cards", min=1, help="Maximum cards to generate."),
) -> None:
    """Generate deterministic operator cards from evidence bundles."""
    try:
        result = GenerateOperatorCards(
            prepared=prepared,
            evidence=evidence,
            out=out,
            query=query,
            event_id=event_id,
            dataset=dataset,
            playbooks=playbooks,
            max_cards=max_cards,
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())


@operator_card_app.command("validate")
def operator_card_validate(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    evidence: Path = typer.Option(..., "--evidence", file_okay=False, help="Evidence root."),
    cards: Path = typer.Option(..., "--cards", file_okay=False, help="Operator card root."),
) -> None:
    """Validate deterministic operator-card artifacts."""
    _emit_validation(
        ValidateOperatorCards(prepared=prepared, evidence=evidence, cards=cards).run().to_dict()
    )


@xai_gt_map_app.command("build")
def xai_gt_map_build(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    out: Path = typer.Option(..., "--out", dir_okay=False, help="GT tag-map JSON file."),
) -> None:
    """Build a ground-truth tag map from prepared event metadata."""
    try:
        result = BuildGroundTruthTagMap(prepared=prepared, out=out).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())


@xai_gt_map_app.command("validate")
def xai_gt_map_validate(
    gt_map: Path = typer.Option(..., "--gt-map", dir_okay=False, help="GT tag-map JSON file."),
) -> None:
    """Validate a ground-truth tag map."""
    _emit_validation(ValidateGroundTruthTagMap(gt_map).run().to_dict())


@xai_app.command("eval")
def xai_eval(
    prepared: Path = typer.Option(
        ..., "--prepared", file_okay=False, help="Prepared dataset root."
    ),
    evidence: Path = typer.Option(..., "--evidence", file_okay=False, help="Evidence root."),
    gt_map: Path = typer.Option(..., "--gt-map", dir_okay=False, help="GT tag-map JSON file."),
    out: Path = typer.Option(..., "--out", file_okay=False, help="XAI evaluation output root."),
    ks: str = typer.Option("1,3,5", "--ks", help="Comma-separated K values."),
    protocol: str = typer.Option("naive", "--protocol", help="Split protocol."),
) -> None:
    """Evaluate evidence bundles with deterministic XAI metrics."""
    try:
        result = EvaluateEvidence(
            EvaluateEvidenceConfig(
                prepared=prepared,
                evidence=evidence,
                gt_map=gt_map,
                out=out,
                ks=_parse_ks(ks),
                protocol=protocol,
            )
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())


@assistant_app.command("providers")
def assistant_providers() -> None:
    """List registered assistant replay LLM provider plugins."""
    registry = default_llm_provider_registry()
    table = Table("Name", "Family", "Default Model", "Base URL", "API Key", "Description")
    for name in registry.names():
        description = registry.get(name).describe()
        table.add_row(
            description.name,
            description.family,
            description.default_model,
            description.default_base_url or "-",
            "required" if description.requires_api_key else "not required",
            description.description,
        )
    console.print(table)


@assistant_app.command("provider-template")
def assistant_provider_template(
    out: Path = typer.Option(..., "--out", dir_okay=False, help="Provider TOML snippet file."),
) -> None:
    """Write provider configuration examples."""
    try:
        path = write_provider_config_template(out)
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data={"out": str(path)})


@assistant_app.command("preflight")
def assistant_preflight(
    config: Path = typer.Option(
        ...,
        "--config",
        dir_okay=False,
        help="assistant replay TOML config.",
    ),
) -> None:
    """Preflight an assistant replay config and provider."""
    try:
        loaded = load_assistant_config(config)
        result = PreflightAssistantReplay(
            config=loaded,
            provider_registry=default_llm_provider_registry(),
        ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())
    if not result.ok:
        raise typer.Exit(1)


@assistant_app.command("run")
def assistant_run(
    config: Path = typer.Option(
        ...,
        "--config",
        dir_okay=False,
        help="assistant replay TOML config.",
    ),
    benchmark: Path = typer.Option(..., "--benchmark", file_okay=False, help="Benchmark run dir."),
    evidence: Path = typer.Option(..., "--evidence", file_okay=False, help="Evidence Bundle dir."),
    out: Path = typer.Option(..., "--out", file_okay=False, help="assistant replay output dir."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable live progress UI."),
) -> None:
    """Run a thesis-style assistant replay suite."""
    try:
        loaded = load_assistant_config(config)
        with cli_progress(not no_progress) as progress:
            result = RunAssistantReplaySuite(
                config=loaded,
                evidence=evidence,
                out=out,
                provider_registry=default_llm_provider_registry(),
                benchmark=benchmark,
                progress_sink=progress,
            ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())
    if not result.ok:
        raise typer.Exit(1)


@assistant_app.command("summarize")
def assistant_summarize(
    run: Path = typer.Option(..., "--run", file_okay=False, help="assistant replay run directory."),
) -> None:
    """Print an assistant replay summary."""
    try:
        payload = SummarizeAssistantReplay(run).run_summary()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=payload)


@reproduce_app.command("init-config")
def reproduce_init_config(
    out: Path = typer.Option(..., "--out", dir_okay=False, help="Reproduction TOML file."),
    profile: str = typer.Option(
        "thesis-smoke",
        "--profile",
        help="Config profile: thesis-smoke, thesis-verification, or thesis-full.",
    ),
) -> None:
    """Write a starter thesis reproduction config."""
    try:
        path = write_default_reproduction_config(out, profile=profile)
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data={"config": str(path)})


@reproduce_app.command("plan")
def reproduce_plan(
    config: Path = typer.Option(..., "--config", dir_okay=False, help="Reproduction TOML config."),
) -> None:
    """Print the resolved reproduction plan."""
    try:
        loaded = load_reproduction_config(config)
        plan = PlanThesisReproduction(loaded).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=plan.to_dict())


@reproduce_app.command("preflight")
def reproduce_preflight(
    config: Path = typer.Option(..., "--config", dir_okay=False, help="Reproduction TOML config."),
    out: Path = typer.Option(..., "--out", file_okay=False, help="Preflight output directory."),
) -> None:
    """Run thesis reproduction preflight checks."""
    try:
        loaded = load_reproduction_config(config)
        result = PreflightThesisReproduction(
            config=loaded,
            provider_registry=default_llm_provider_registry(),
            detector_registry=default_detector_registry(),
        ).run()
        write_json(out / "preflight.json", result)
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result)
    if not bool(result.get("ok")):
        raise typer.Exit(1)


@reproduce_app.command("run")
def reproduce_run(
    config: Path = typer.Option(..., "--config", dir_okay=False, help="Reproduction TOML config."),
    out: Path = typer.Option(..., "--out", file_okay=False, help="Reproduction output root."),
    run_id: str | None = typer.Option(None, "--run-id", help="Optional run id override."),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable live progress UI."),
) -> None:
    """Run a thesis-style reproduction workflow."""
    try:
        loaded = load_reproduction_config(config)
        with cli_progress(not no_progress) as progress:
            result = RunThesisReproduction(
                config=loaded,
                detector_registry=default_detector_registry(),
                provider_registry=default_llm_provider_registry(),
                out=out,
                run_id=run_id,
                source_config=config,
                progress_sink=progress,
            ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())
    if not result.ok:
        raise typer.Exit(1)


@reproduce_app.command("status")
def reproduce_status(
    run: Path = typer.Option(..., "--run", file_okay=False, help="Reproduction run directory."),
    watch: bool = typer.Option(False, "--watch", help="Refresh until the run completes."),
    interval_s: float = typer.Option(10.0, "--interval-s", help="Watch refresh interval."),
) -> None:
    """Print thesis-style reproduction progress status."""
    _status_loop(run, watch=watch, interval_s=interval_s)


@reproduce_app.command("summarize")
def reproduce_summarize(
    run: Path = typer.Option(..., "--run", file_okay=False, help="Reproduction run directory."),
) -> None:
    """Print a thesis reproduction summary."""
    try:
        payload = SummarizeThesisReproduction(run).run_summary()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=payload)


@reproduce_app.command("diagnose")
def reproduce_diagnose(
    run: Path = typer.Option(..., "--run", file_okay=False, help="Reproduction run directory."),
) -> None:
    """Group failures and slow-stage diagnostics for a reproduction run."""
    try:
        payload = DiagnoseThesisReproduction(run).run_diagnostics()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=payload)


@audit_app.command("run")
def audit_run(
    out: Path = typer.Option(
        Path("out/audit"), "--out", file_okay=False, help="Audit output root."
    ),
    audit_id: str | None = typer.Option(None, "--audit-id", help="Optional audit id."),
    include_optional: bool = typer.Option(
        True,
        "--include-optional/--skip-optional",
        help="Include torch, llama.cpp, and thesis-full optional probes.",
    ),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable live progress UI."),
) -> None:
    """Run clean-repo reproducibility audit checks."""
    try:
        with cli_progress(not no_progress) as progress:
            result = RunReproducibilityAudit(
                detector_registry=default_detector_registry(),
                dataset_adapter_registry=default_dataset_adapter_registry(),
                dataset_source_registry=default_dataset_source_registry(),
                provider_registry=default_llm_provider_registry(),
                config=ReproducibilityAuditConfig(
                    out=out,
                    audit_id=audit_id,
                    include_optional=include_optional,
                ),
                progress_sink=progress,
            ).run()
    except (IndustrialTSADError, ValueError, RuntimeError, FileNotFoundError) as exc:
        _fail(str(exc))
    console.print_json(data=result.to_dict())
    if not result.ok:
        raise typer.Exit(1)


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


def _parse_ks(value: str) -> list[int]:
    ks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ks:
        raise ValueError("--ks must include at least one integer.")
    if any(k <= 0 for k in ks):
        raise ValueError("--ks values must be positive.")
    return ks


def _status_loop(run: Path, *, watch: bool, interval_s: float) -> None:
    while True:
        payload = read_run_progress(run)
        if watch:
            console.clear()
        _print_status(run, payload)
        manifest_status = _manifest_status(run)
        if not watch or manifest_status in {"completed", "failed"}:
            return
        time.sleep(interval_s)


def _print_status(run: Path, payload: dict[str, Any]) -> None:
    counts = payload.get("counts", {})
    table = Table("Status", "Count")
    if isinstance(counts, dict) and counts:
        for status, count in sorted(counts.items()):
            table.add_row(str(status), str(count))
    else:
        table.add_row("unknown", "0")
    console.print(f"[bold]Run:[/bold] {run}")
    console.print(table)
    latest = payload.get("latest_event")
    if isinstance(latest, dict):
        console.print_json(
            data={
                "latest_event": {
                    "stage": latest.get("stage"),
                    "item_id": latest.get("item_id"),
                    "status": latest.get("status"),
                    "path": latest.get("path"),
                    "error": latest.get("error"),
                }
            }
        )
    failure = payload.get("latest_failure")
    if isinstance(failure, dict):
        console.print_json(
            data={
                "latest_failure": {
                    "stage": failure.get("stage"),
                    "item_id": failure.get("item_id"),
                    "path": failure.get("path"),
                    "error": failure.get("error"),
                }
            }
        )


def _manifest_status(run: Path) -> str:
    manifest_path = run / "run_manifest.json"
    if not manifest_path.exists():
        return "unknown"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(payload.get("status", "unknown"))


def _fail(message: str) -> NoReturn:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(1)


def _backend_ready(backend: str, runtime: dict[str, Any]) -> bool:
    if backend == "cpu":
        return True
    if backend == "cuda":
        return bool(runtime.get("cuda_available"))
    if backend == "xpu":
        return bool(runtime.get("xpu_available"))
    return False
