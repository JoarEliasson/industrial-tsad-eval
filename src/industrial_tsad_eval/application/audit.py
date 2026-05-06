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

from industrial_tsad_eval.application.reproduction import (
    PreflightThesisReproduction,
    RunThesisReproduction,
)
from industrial_tsad_eval.application.rq3 import PreflightRQ3, RunReplaySuite
from industrial_tsad_eval.application.scoring import ScoreRuns
from industrial_tsad_eval.application.validation import ValidatePreparedDataset, ValidateScores
from industrial_tsad_eval.domain.audit import AuditCheck, AuditRunResult, AuditStatus
from industrial_tsad_eval.domain.errors import ReproductionError
from industrial_tsad_eval.domain.rq3 import RQ3Config
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.examples import make_opcua_fixture
from industrial_tsad_eval.infrastructure.reproduction_config import (
    load_reproduction_config,
    write_default_reproduction_config,
)
from industrial_tsad_eval.plugins.providers import LLMProviderRegistry
from industrial_tsad_eval.plugins.registry import DetectorRegistry


@dataclass(frozen=True)
class ReproducibilityAuditConfig:
    """Configuration for a reproducibility audit run."""

    out: str | Path = "out/audit"
    audit_id: str | None = None
    include_optional: bool = True
    python_executable: str = sys.executable


class RunReproducibilityAudit:
    """Run architecture, smoke-reproduction, and optional readiness checks."""

    def __init__(
        self,
        *,
        detector_registry: DetectorRegistry,
        provider_registry: LLMProviderRegistry,
        config: ReproducibilityAuditConfig,
    ):
        self.detector_registry = detector_registry
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
        if self.config.include_optional:
            checks.append(self._torch_smoke_check())
            checks.append(self._llama_cpp_smoke_check())
            checks.append(self._thesis_full_local_preflight_check())

        ok = not any(check.required and check.status == "fail" for check in checks)
        result = AuditRunResult(
            audit_id=self.audit_id,
            audit_dir=str(self.audit_root),
            ok=ok,
            checks=checks,
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
                        "summaries/rq3_summary.csv",
                        "summaries/thesis_crosswalk.md",
                        "rq3/rq3_summary.json",
                    ],
                ),
                ok_predicate=lambda payload: bool(payload.get("all_present")),
            )
        )
        checks.append(
            _timed_check(
                "rq3-preflight",
                required=True,
                action=lambda: (
                    PreflightRQ3(
                        config=loaded.rq3,
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
            action=lambda: self._run_llama_cpp_rq3_smoke(config),
            warn_on_failure=True,
        )

    def _run_llama_cpp_rq3_smoke(self, provider_config: Any) -> dict[str, Any]:
        loaded = load_reproduction_config(self.workspace / "thesis_smoke.toml")
        config = RQ3Config(
            suite_id="llama-cpp-smoke",
            prepared=loaded.rq3.prepared,
            provider=provider_config,
            query_template=loaded.rq3.query_template,
            cases_per_dataset=1,
            top_k=loaded.rq3.top_k,
            minimum_supported_claims=0,
            prompt_budget_chars=loaded.rq3.prompt_budget_chars,
            playbooks=loaded.rq3.playbooks,
        )
        evidence = next((self.audit_root / "reproduction" / "smoke-audit" / "evidence").glob("*"))
        result = RunReplaySuite(
            config=config,
            evidence=evidence,
            out=self.audit_root / "optional" / "llama_cpp_rq3",
            provider_registry=self.provider_registry,
            benchmark=self.audit_root / "reproduction" / "smoke-audit" / "benchmark",
        ).run()
        return result.to_dict()

    def _thesis_full_local_preflight_check(self) -> AuditCheck:
        prepared_roots = [Path("prepared") / name for name in ("TEP", "SWaT", "HAI", "HAI-CPPS")]
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
