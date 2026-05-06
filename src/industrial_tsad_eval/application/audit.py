"""Clean-repo reproducibility audit workflow."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.acquisition import (
    AcquireDatasetSource,
    ValidateRawAcquisition,
)
from industrial_tsad_eval.application.assistant_replay import (
    PreflightAssistantReplay,
    RunAssistantReplaySuite,
)
from industrial_tsad_eval.application.preparation import PrepareDataset
from industrial_tsad_eval.application.reproduction import (
    PreflightThesisReproduction,
    RunThesisReproduction,
)
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig
from industrial_tsad_eval.domain.assistant_replay import (
    THESIS_ASSISTANT_QUERY_TEMPLATE,
    AssistantReplayConfig,
)
from industrial_tsad_eval.domain.audit import (
    AuditCheck,
    AuditRunResult,
    AuditSetupRecommendation,
    AuditStatus,
)
from industrial_tsad_eval.domain.benchmark import (
    BenchmarkConfig,
    BenchmarkDatasetConfig,
    BenchmarkDetectorConfig,
    BenchmarkEvaluationConfig,
)
from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig
from industrial_tsad_eval.domain.errors import ReproductionError
from industrial_tsad_eval.domain.llm import LLMProviderConfig
from industrial_tsad_eval.domain.reproduction import ReproductionConfig
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.examples import (
    make_opcua_fixture,
    make_thesis_raw_fixtures,
)
from industrial_tsad_eval.infrastructure.json_utils import read_json
from industrial_tsad_eval.infrastructure.reproduction_config import (
    load_reproduction_config,
    render_reproduction_config_toml,
    write_default_reproduction_config,
)
from industrial_tsad_eval.plugins.providers import LLMProviderRegistry
from industrial_tsad_eval.plugins.registry import (
    DatasetAdapterRegistry,
    DatasetSourceRegistry,
    DetectorRegistry,
)


@dataclass(frozen=True)
class ReproducibilityAuditConfig:
    """Configuration for a reproducibility audit run."""

    out: str | Path = "out/audit"
    audit_id: str | None = None
    include_optional: bool = True
    python_executable: str = sys.executable


class RunReproducibilityAudit:
    """Run architecture, smoke-reproduction, and optional setup checks."""

    def __init__(
        self,
        *,
        detector_registry: DetectorRegistry,
        dataset_adapter_registry: DatasetAdapterRegistry,
        dataset_source_registry: DatasetSourceRegistry,
        provider_registry: LLMProviderRegistry,
        config: ReproducibilityAuditConfig,
    ):
        self.detector_registry = detector_registry
        self.dataset_adapter_registry = dataset_adapter_registry
        self.dataset_source_registry = dataset_source_registry
        self.provider_registry = provider_registry
        self.config = config
        self.audit_id = config.audit_id or _default_audit_id()
        self.audit_root = Path(config.out) / self.audit_id
        self.workspace = self.audit_root / "workspace"
        self.logs_root = self.audit_root / "logs"

    def run(self) -> AuditRunResult:
        """Execute the audit and write summary artifacts."""
        if self.audit_root.exists():
            raise ReproductionError(f"Audit output already exists: {self.audit_root}")
        self.audit_root.mkdir(parents=True, exist_ok=False)
        checks = [
            self._subprocess_check(
                "package-import",
                [
                    self.config.python_executable,
                    "-c",
                    "import industrial_tsad_eval; print(industrial_tsad_eval.__version__)",
                ],
                required=True,
            ),
            self._subprocess_check(
                "cli-help",
                [
                    self.config.python_executable,
                    "-c",
                    (
                        "from typer.testing import CliRunner; "
                        "from industrial_tsad_eval.interfaces.cli.main import app; "
                        "r=CliRunner().invoke(app, ['--help']); print(r.output); "
                        "raise SystemExit(r.exit_code)"
                    ),
                ],
                required=True,
            ),
            self._registry_check(),
            self._subprocess_check(
                "architecture-tests",
                [self.config.python_executable, "-m", "pytest", "tests/test_architecture.py"],
                required=True,
            ),
        ]
        checks.extend(self._smoke_reproduction_checks())
        checks.append(self._synthetic_thesis_setup_check())
        if self.config.include_optional:
            checks.append(self._torch_smoke_check())
            checks.append(self._profile_extras_check())
            checks.append(self._llama_cpp_smoke_check())
            checks.append(self._thesis_full_local_preflight_check())

        ok = not any(check.required and check.status == "fail" for check in checks)
        setup_recommendations = _setup_recommendations(checks)
        result = AuditRunResult(
            audit_id=self.audit_id,
            audit_dir=str(self.audit_root),
            ok=ok,
            checks=checks,
            setup_recommendations=setup_recommendations,
        )
        writer = LocalArtifactWriter(self.audit_root)
        writer.write_json("audit_summary.json", result.to_dict())
        writer.write_text("audit_summary.md", _render_audit_markdown(result))
        return result

    def _subprocess_check(
        self,
        name: str,
        command: list[str],
        *,
        required: bool,
    ) -> AuditCheck:
        start = time.perf_counter()
        env = os.environ.copy()
        src_path = str(Path.cwd() / "src")
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = (
            src_path
            if not env.get("PYTHONPATH")
            else src_path + os.pathsep + str(env["PYTHONPATH"])
        )
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        duration = time.perf_counter() - start
        stdout_path = self.logs_root / f"{name}.stdout.txt"
        stderr_path = self.logs_root / f"{name}.stderr.txt"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        status: AuditStatus = "pass" if completed.returncode == 0 else "fail"
        return AuditCheck(
            name=name,
            status=status,
            required=required,
            message="Command completed." if status == "pass" else "Command failed.",
            duration_s=duration,
            details={"command": command, "returncode": completed.returncode},
            log_paths=[str(stdout_path), str(stderr_path)],
        )

    def _registry_check(self) -> AuditCheck:
        return _timed_check(
            "registry-discovery",
            required=True,
            action=lambda: {
                "detectors": self.detector_registry.names(),
                "providers": self.provider_registry.names(),
            },
        )

    def _smoke_reproduction_checks(self) -> list[AuditCheck]:
        checks: list[AuditCheck] = []
        fixture_root = self.workspace / "examples" / "generated"
        config_path = self.workspace / "thesis_smoke.toml"
        reproduction_out = self.audit_root / "reproduction"
        run_root = reproduction_out / "smoke-audit"
        checks.append(
            _timed_check(
                "opcua-fixture",
                required=True,
                action=lambda: self._make_and_validate_fixture(fixture_root),
            )
        )
        checks.append(
            _timed_check(
                "reproduction-init-config",
                required=True,
                action=lambda: {"config": str(write_default_reproduction_config(config_path))},
            )
        )

        loaded = load_reproduction_config(config_path)
        checks.append(
            _timed_check(
                "reproduction-preflight",
                required=True,
                action=lambda: PreflightThesisReproduction(
                    config=loaded,
                    provider_registry=self.provider_registry,
                ).run(),
                ok_predicate=lambda payload: bool(payload.get("ok")),
            )
        )
        checks.append(
            _timed_check(
                "reproduction-run",
                required=True,
                action=lambda: (
                    RunThesisReproduction(
                        config=loaded,
                        detector_registry=self.detector_registry,
                        provider_registry=self.provider_registry,
                        out=reproduction_out,
                        run_id="smoke-audit",
                        source_config=config_path,
                    )
                    .run()
                    .to_dict()
                ),
                ok_predicate=lambda payload: bool(payload.get("ok")),
            )
        )
        checks.append(
            _timed_check(
                "reproduction-artifacts",
                required=True,
                action=lambda: _artifact_report(
                    run_root,
                    [
                        "benchmark/summary.json",
                        "summaries/detection_summary.csv",
                        "summaries/xai_summary.csv",
                        "summaries/assistant_summary.csv",
                        "summaries/thesis_crosswalk.md",
                        "assistant/assistant_summary.json",
                    ],
                ),
                ok_predicate=lambda payload: bool(payload.get("all_present")),
            )
        )
        checks.append(
            _timed_check(
                "assistant-preflight",
                required=True,
                action=lambda: (
                    PreflightAssistantReplay(
                        config=loaded.assistant,
                        provider_registry=self.provider_registry,
                    )
                    .run()
                    .to_dict()
                ),
                ok_predicate=lambda payload: bool(payload.get("ok")),
            )
        )
        return checks

    def _make_and_validate_fixture(self, fixture_root: Path) -> dict[str, Any]:
        prepared = make_opcua_fixture(fixture_root)
        report = ValidatePreparedDataset(prepared).run()
        if not report.ok:
            raise ReproductionError(f"Generated fixture validation failed: {report.errors}")
        return {"prepared": str(prepared), "validation": report.to_dict()}

    def _synthetic_thesis_setup_check(self) -> AuditCheck:
        return _timed_check(
            "synthetic-thesis-setup",
            required=True,
            action=self._run_synthetic_thesis_setup,
            ok_predicate=lambda payload: bool(payload.get("ok")),
        )

    def _run_synthetic_thesis_setup(self) -> dict[str, Any]:
        raw_fixtures = make_thesis_raw_fixtures(self.workspace / "thesis_raw")
        prepared_roots: dict[str, str] = {}
        rows: list[dict[str, Any]] = []
        for source in ("tep", "swat", "hai", "hai-cpps"):
            acquired = AcquireDatasetSource(
                source_registry=self.dataset_source_registry,
                source=source,
                out=self.workspace / "raw-cache",
                config=DatasetSourceConfig(
                    method="manual",
                    manual_path=raw_fixtures[source],
                ),
            ).run()
            raw_report = ValidateRawAcquisition(
                source_registry=self.dataset_source_registry,
                source=source,
                raw=acquired.raw_path,
            ).run()
            prepared = PrepareDataset(
                adapter_registry=self.dataset_adapter_registry,
                dataset=source,
                raw=acquired.raw_path,
                out=self.workspace / "prepared",
                config=DatasetAdapterConfig(),
            ).run()
            prepared_report = ValidatePreparedDataset(prepared.prepared_path).run()
            if not raw_report.ok or not prepared_report.ok:
                raise ReproductionError(f"Synthetic setup failed for source {source!r}.")
            prepared_roots[source] = prepared.prepared_path
            rows.append(
                {
                    "source": source,
                    "raw": acquired.raw_path,
                    "prepared": prepared.prepared_path,
                    "events": prepared.event_count,
                    "runs": prepared.run_count,
                }
            )

        config = _synthetic_full_smoke_config(prepared_roots)
        config_path = self.workspace / "thesis_full_smoke.toml"
        config_path.write_text(render_reproduction_config_toml(config), encoding="utf-8")
        preflight = PreflightThesisReproduction(
            config=config,
            provider_registry=self.provider_registry,
        ).run()
        if not preflight["ok"]:
            raise ReproductionError("Synthetic thesis-full-smoke preflight failed.")
        reproduction = RunThesisReproduction(
            config=config,
            detector_registry=self.detector_registry,
            provider_registry=self.provider_registry,
            out=self.audit_root / "synthetic-full-reproduction",
            run_id="thesis-full-smoke",
            source_config=config_path,
        ).run()
        run_root = Path(reproduction.run_dir)
        artifacts = _artifact_report(
            run_root,
            [
                "benchmark/summary.json",
                "summaries/detection_summary.csv",
                "summaries/xai_summary.csv",
                "summaries/assistant_summary.csv",
                "summaries/thesis_crosswalk.md",
                "assistant/assistant_summary.json",
            ],
        )
        return {
            "ok": reproduction.ok and artifacts["all_present"],
            "raw_fixtures": raw_fixtures,
            "prepared": rows,
            "config": str(config_path),
            "preflight": preflight,
            "reproduction": reproduction.to_dict(),
            "artifact_report": artifacts,
        }

    def _torch_smoke_check(self) -> AuditCheck:
        if importlib.util.find_spec("torch") is None:
            return AuditCheck(
                name="torch-smoke",
                status="skipped",
                required=False,
                message="Torch is not installed; optional detector smoke skipped.",
            )
        return _timed_check(
            "torch-smoke",
            required=False,
            action=self._run_torch_smoke,
            warn_on_failure=True,
        )

    def _run_torch_smoke(self) -> dict[str, Any]:
        prepared = self.workspace / "examples" / "generated" / "OPCUA_SYNTH"
        scores = self.audit_root / "optional" / "torch_scores"
        result = ScoreRuns(
            detector_registry=self.detector_registry,
            prepared=prepared,
            scores=scores,
            detector_name="forecast-lstm",
            detector_parameters={
                "window": 16,
                "train_stride": 8,
                "score_stride": 8,
                "max_train_windows": 24,
                "epochs": 1,
                "batch_size": 8,
                "device": "cpu",
                "hidden_size": 8,
            },
        ).run()
        validation = ValidateScores(prepared, scores).run()
        if not validation.ok:
            raise ReproductionError(f"Torch score validation failed: {validation.errors}")
        return {"scores": str(scores), "runs_scored": result.runs_scored}

    def _profile_extras_check(self) -> AuditCheck:
        missing = [
            module for module in ("psutil", "pynvml") if importlib.util.find_spec(module) is None
        ]
        if missing:
            return AuditCheck(
                name="profile-extras",
                status="skipped",
                required=False,
                message="Optional profiling dependencies are not fully installed.",
                details={"missing": missing},
            )
        return AuditCheck(
            name="profile-extras",
            status="pass",
            required=False,
            message="Optional profiling dependencies are installed.",
            details={"modules": ["psutil", "pynvml"]},
        )

    def _llama_cpp_smoke_check(self) -> AuditCheck:
        plugin = self.provider_registry.get("llama-cpp")
        config = plugin.default_config()
        provider = plugin.create(config)
        health = provider.healthcheck()
        if not health.ok:
            return AuditCheck(
                name="llama-cpp-smoke",
                status="skipped",
                required=False,
                message="llama.cpp OpenAI-compatible endpoint was not reachable.",
                details=health.to_dict(),
            )
        return _timed_check(
            "llama-cpp-smoke",
            required=False,
            action=lambda: self._run_llama_cpp_assistant_smoke(config),
            ok_predicate=lambda payload: bool(payload.get("ok")),
            warn_on_failure=True,
        )

    def _run_llama_cpp_assistant_smoke(self, provider_config: Any) -> dict[str, Any]:
        loaded = load_reproduction_config(self.workspace / "thesis_smoke.toml")
        config = AssistantReplayConfig(
            suite_id="llama-cpp-smoke",
            prepared=loaded.assistant.prepared,
            provider=provider_config,
            query_template=loaded.assistant.query_template,
            cases_per_dataset=1,
            top_k=loaded.assistant.top_k,
            minimum_supported_claims=0,
            prompt_budget_chars=loaded.assistant.prompt_budget_chars,
            playbooks=loaded.assistant.playbooks,
        )
        evidence = next((self.audit_root / "reproduction" / "smoke-audit" / "evidence").glob("*"))
        assistant_out = self.audit_root / "optional" / "llama_cpp_assistant"
        result = RunAssistantReplaySuite(
            config=config,
            evidence=evidence,
            out=assistant_out,
            provider_registry=self.provider_registry,
            benchmark=self.audit_root / "reproduction" / "smoke-audit" / "benchmark",
        ).run()
        return {
            **result.to_dict(),
            "failed_case_logs": _failed_assistant_case_logs(assistant_out),
        }

    def _thesis_full_local_preflight_check(self) -> AuditCheck:
        prepared_roots = [Path("prepared") / name for name in ("TEP", "SWaT", "HAI", "HAI_CPPS")]
        missing = [str(path) for path in prepared_roots if not path.exists()]
        if missing:
            return AuditCheck(
                name="thesis-full-local-preflight",
                status="skipped",
                required=False,
                message="Local thesis-full Prepared Format roots were not all present.",
                details={"missing": missing},
            )
        return AuditCheck(
            name="thesis-full-local-preflight",
            status="warn",
            required=False,
            message="Local roots exist; run thesis-full reproduction manually with llama.cpp.",
            details={"prepared_roots": [str(path) for path in prepared_roots]},
        )


def _synthetic_full_smoke_config(prepared_roots: dict[str, str]) -> ReproductionConfig:
    benchmark = BenchmarkConfig(
        name="thesis-full-smoke",
        protocols=["naive", "all_in_one", "zero_shot"],
        datasets=[
            BenchmarkDatasetConfig(id="TEP", prepared=prepared_roots["tep"]),
            BenchmarkDatasetConfig(id="SWaT", prepared=prepared_roots["swat"]),
            BenchmarkDatasetConfig(id="HAI", prepared=prepared_roots["hai"]),
            BenchmarkDatasetConfig(id="HAI_CPPS", prepared=prepared_roots["hai-cpps"]),
        ],
        detectors=[
            BenchmarkDetectorConfig(
                id="forecast-ridge-smoke",
                name="forecast-ridge",
                parameters={
                    "window": 24,
                    "stride": 4,
                    "lags": 1,
                    "alpha": 1.0,
                    "standardize": True,
                    "seed": 1337,
                },
            )
        ],
        evaluation=BenchmarkEvaluationConfig(threshold_quantile=0.995),
    )
    assistant = AssistantReplayConfig(
        suite_id="thesis-full-smoke-assistant",
        prepared=prepared_roots["tep"],
        provider=LLMProviderConfig(name="fake", model="fake-assistant"),
        cases_per_dataset=1,
        top_k=6,
        minimum_supported_claims=1,
        prompt_budget_chars=8000,
        query_template=THESIS_ASSISTANT_QUERY_TEMPLATE,
    )
    return ReproductionConfig(
        name="thesis-full-smoke",
        benchmark=benchmark,
        assistant=assistant,
        run_evidence=True,
        run_xai=True,
        run_profiles=False,
        run_assistant=True,
        xai_ks=[1, 3, 5],
    )


def _timed_check(
    name: str,
    *,
    required: bool,
    action: Callable[[], dict[str, Any]],
    ok_predicate: Callable[[dict[str, Any]], bool] | None = None,
    warn_on_failure: bool = False,
) -> AuditCheck:
    start = time.perf_counter()
    try:
        payload = action()
        ok = ok_predicate(payload) if ok_predicate is not None else True
        status: AuditStatus = "pass" if ok else ("warn" if warn_on_failure else "fail")
        message = "Check passed." if ok else "Check completed with non-passing result."
        return AuditCheck(
            name=name,
            status=status,
            required=required,
            message=message,
            duration_s=time.perf_counter() - start,
            details=payload,
        )
    except Exception as exc:
        return AuditCheck(
            name=name,
            status="warn" if warn_on_failure else "fail",
            required=required,
            message=f"{type(exc).__name__}: {exc}",
            duration_s=time.perf_counter() - start,
            details={},
        )


def _artifact_report(root: Path, relative_paths: list[str]) -> dict[str, Any]:
    rows = [
        {"path": relative_path, "exists": (root / relative_path).exists()}
        for relative_path in relative_paths
    ]
    return {
        "root": str(root),
        "all_present": all(row["exists"] for row in rows),
        "artifacts": rows,
    }


def _failed_assistant_case_logs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_log in sorted((root / "runs").glob("*/run_log.json")):
        payload = read_json(run_log)
        if payload.get("status") == "failed":
            rows.append({"path": str(run_log), "error": payload.get("error")})
    return rows


def _setup_recommendations(checks: list[AuditCheck]) -> list[AuditSetupRecommendation]:
    by_name = {check.name: check for check in checks}
    recommendations: list[AuditSetupRecommendation] = []
    torch = by_name.get("torch-smoke")
    if torch is not None and torch.status in {"skipped", "warn", "fail"}:
        recommendations.append(
            AuditSetupRecommendation(
                resource="torch",
                status=torch.status,
                reason=torch.message,
                commands=[
                    'python -m pip install -e ".[torch]"',
                    "itse score detectors",
                    (
                        "itse score run --prepared examples/generated/OPCUA_SYNTH "
                        "--detector forecast-lstm --out out/lstm-smoke "
                        '--parameters-json "{\\"window\\": 16, \\"train_stride\\": 8, '
                        '\\"score_stride\\": 8, \\"epochs\\": 1, \\"device\\": \\"cpu\\"}"'
                    ),
                ],
                success_criteria="The torch-smoke audit check passes or writes valid scores.",
            )
        )
    profile = by_name.get("profile-extras")
    if profile is not None and profile.status in {"skipped", "warn", "fail"}:
        recommendations.append(
            AuditSetupRecommendation(
                resource="profiling-extras",
                status=profile.status,
                reason=profile.message,
                commands=[
                    'python -m pip install -e ".[profile]"',
                    (
                        "itse profile run --prepared examples/generated/OPCUA_SYNTH "
                        "--detector forecast-ridge --out out/profiles --profile-id smoke"
                    ),
                ],
                success_criteria="The profile run writes summary.json and stages.csv.",
            )
        )
    llama = by_name.get("llama-cpp-smoke")
    if llama is not None and llama.status in {"skipped", "warn", "fail"}:
        recommendations.append(
            AuditSetupRecommendation(
                resource="llama-cpp",
                status=llama.status,
                reason=llama.message,
                commands=[
                    ("llama-server -m C:\\path\\to\\model.gguf --host 127.0.0.1 --port 8080"),
                    "itse assistant providers",
                    "itse assistant preflight --config config/thesis_full.toml",
                ],
                success_criteria="The llama-cpp provider healthcheck reports ready.",
            )
        )
    thesis = by_name.get("thesis-full-local-preflight")
    if thesis is not None and thesis.status in {"skipped", "warn", "fail"}:
        recommendations.append(
            AuditSetupRecommendation(
                resource="real-thesis-datasets",
                status=thesis.status,
                reason=thesis.message,
                commands=[
                    (
                        "itse data acquire --source tep --method manual "
                        "--manual data/downloads/TEP --out data/raw"
                    ),
                    "itse prepared prepare --dataset tep --raw data/raw/TEP --out prepared",
                    (
                        "Repeat the acquire/prepare flow for swat, hai, and hai-cpps, "
                        "then run: itse reproduce preflight --config config/thesis_full.toml "
                        "--out out/preflight"
                    ),
                ],
                success_criteria=(
                    "Prepared roots for TEP, SWaT, HAI, and HAI_CPPS validate locally."
                ),
            )
        )
    return recommendations


def _render_audit_markdown(result: AuditRunResult) -> str:
    lines = [
        f"# Reproducibility Audit: {result.audit_id}",
        "",
        f"- Status: {'pass' if result.ok else 'fail'}",
        f"- Audit directory: `{result.audit_dir}`",
        "",
        "| Check | Required | Status | Message |",
        "| --- | --- | --- | --- |",
    ]
    for check in result.checks:
        lines.append(
            f"| `{check.name}` | {check.required} | {check.status} | "
            f"{check.message.replace('|', '/')} |"
        )
    if result.setup_recommendations:
        lines.extend(["", "## Setup Recommendations", ""])
        for recommendation in result.setup_recommendations:
            lines.extend(
                [
                    f"### {recommendation.resource}",
                    "",
                    f"- Status: {recommendation.status}",
                    f"- Reason: {recommendation.reason}",
                    f"- Success: {recommendation.success_criteria}",
                    "",
                    "```powershell",
                    *recommendation.commands,
                    "```",
                    "",
                ]
            )
    lines.extend(
        [
            "",
            "Required checks must pass for the audit to be considered green. Optional",
            "torch, llama.cpp, and thesis-full checks may be skipped when local resources",
            "are unavailable.",
            "",
        ]
    )
    return "\n".join(lines)


def _default_audit_id() -> str:
    return "audit-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
