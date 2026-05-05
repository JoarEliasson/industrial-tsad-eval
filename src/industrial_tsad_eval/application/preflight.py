"""System preflight application service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.domain.errors import PreflightError
from industrial_tsad_eval.domain.system import CheckStatus, PreflightCheck, PreflightReport
from industrial_tsad_eval.infrastructure.artifacts import LocalArtifactWriter
from industrial_tsad_eval.infrastructure.system import (
    capture_machine_environment,
    probe_torch_runtime,
)
from industrial_tsad_eval.plugins.registry import DetectorRegistry
from industrial_tsad_eval.ports.detectors import DetectorRunConfig


@dataclass(frozen=True)
class PreflightInput:
    """Input options for preflight execution."""

    prepared: str | Path | None = None
    detector_name: str | None = None
    detector_parameters: dict[str, Any] | None = None
    device: str = "auto"
    out: str | Path | None = None
    strict: bool = False
    full_package_dump: bool = False


class RunPreflight:
    """Run machine, dataset, detector, and output preflight checks."""

    def __init__(self, *, detector_registry: DetectorRegistry, config: PreflightInput):
        self.detector_registry = detector_registry
        self.config = config

    def run(self) -> PreflightReport:
        """Execute preflight checks and write optional artifacts."""
        machine = capture_machine_environment(
            device_request=self.config.device,
            full_package_dump=self.config.full_package_dump,
        )
        checks = [
            self._check_machine_runtime(machine.torch_runtime.ready),
            *self._prepared_checks(),
            *self._detector_checks(),
            *self._output_checks(),
        ]
        report = PreflightReport(status=_aggregate_status(checks), checks=checks, machine=machine)
        if self.config.out is not None:
            writer = LocalArtifactWriter(self.config.out)
            writer.write_json("preflight.json", report.to_dict())
            writer.write_json("machine_env.json", machine.to_dict())
        if self.config.strict and not report.ok:
            raise PreflightError("Strict preflight failed.")
        return report

    def _check_machine_runtime(self, torch_ready: bool) -> PreflightCheck:
        if torch_ready:
            return PreflightCheck("machine-runtime", "pass", "Machine runtime probe completed.")
        return PreflightCheck(
            "machine-runtime",
            "warn",
            "Torch runtime is unavailable or requested accelerator is not ready.",
        )

    def _prepared_checks(self) -> list[PreflightCheck]:
        if self.config.prepared is None:
            return [PreflightCheck("prepared-dataset", "warn", "No prepared dataset supplied.")]
        report = ValidatePreparedDataset(self.config.prepared).run()
        if report.ok:
            return [
                PreflightCheck(
                    "prepared-dataset",
                    "pass",
                    "Prepared dataset validation passed.",
                    {"path": str(self.config.prepared), "warnings": report.warnings},
                )
            ]
        return [
            PreflightCheck(
                "prepared-dataset",
                "fail",
                "Prepared dataset validation failed.",
                {"path": str(self.config.prepared), "errors": report.errors},
            )
        ]

    def _detector_checks(self) -> list[PreflightCheck]:
        if self.config.detector_name is None:
            return [PreflightCheck("detector", "warn", "No detector supplied.")]

        parameters = _parameters_with_device(self.config.detector_parameters, self.config.device)
        try:
            plugin = self.detector_registry.get(self.config.detector_name)
            plugin.create(DetectorRunConfig(parameters=parameters))
        except Exception as exc:
            return [
                PreflightCheck(
                    "detector",
                    "fail",
                    f"Detector preflight failed: {type(exc).__name__}: {exc}",
                    {"detector": self.config.detector_name},
                )
            ]

        checks = [
            PreflightCheck(
                "detector",
                "pass",
                "Detector lookup and parameter validation passed.",
                {"detector": self.config.detector_name},
            )
        ]
        if _is_torch_plugin(plugin):
            runtime = probe_torch_runtime(self.config.device)
            status: CheckStatus = "pass" if runtime.ready else "fail"
            checks.append(
                PreflightCheck(
                    "torch-runtime",
                    status,
                    "Torch runtime is ready." if runtime.ready else "Torch runtime is not ready.",
                    runtime.to_dict(),
                )
            )
        return checks

    def _output_checks(self) -> list[PreflightCheck]:
        if self.config.out is None:
            return [PreflightCheck("output", "warn", "No output directory supplied.")]
        output = Path(self.config.out)
        try:
            output.mkdir(parents=True, exist_ok=True)
            probe = output / ".preflight_write_check"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            return [
                PreflightCheck(
                    "output",
                    "fail",
                    f"Output directory is not writable: {exc}",
                    {"path": str(output)},
                )
            ]
        return [
            PreflightCheck("output", "pass", "Output directory is writable.", {"path": str(output)})
        ]


def _aggregate_status(checks: list[PreflightCheck]) -> CheckStatus:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def _parameters_with_device(
    parameters: dict[str, Any] | None,
    device: str,
) -> dict[str, Any]:
    merged = dict(parameters or {})
    merged.setdefault("device", device)
    return merged


def _is_torch_plugin(plugin: object) -> bool:
    return plugin.__class__.__module__.endswith("torch_detectors")
