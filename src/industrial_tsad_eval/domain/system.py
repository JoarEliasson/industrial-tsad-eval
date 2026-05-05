"""System diagnostics and preflight contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DeviceRequest = Literal["auto", "cpu", "cuda", "xpu"]
ResolvedDevice = Literal["cpu", "cuda", "xpu"]
CheckStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class SystemGpu:
    """Detected GPU adapter record."""

    name: str
    vendor: str
    source: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class TorchRuntimeStatus:
    """Torch runtime and accelerator readiness summary."""

    requested_device: DeviceRequest
    resolved_device: ResolvedDevice
    ready: bool
    torch_available: bool
    torch_version: str | None
    torch_import_error: str | None
    cuda_available: bool
    xpu_available: bool
    device_name: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class MachineEnvironment:
    """Reproducibility snapshot for a local machine."""

    captured_at_utc: str
    hostname: str | None
    os: str
    python_version: str
    platform_machine: str
    cpu: str
    ram_total_gb: float | None
    gpu_adapters: list[SystemGpu]
    torch_runtime: TorchRuntimeStatus
    recommended_backend: ResolvedDevice
    machine_profile: str
    resolved_batch_size: int
    git_commit: str | None
    git_dirty: bool | None
    packages: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        payload = asdict(self)
        payload["gpu_adapters"] = [gpu.to_dict() for gpu in self.gpu_adapters]
        payload["torch_runtime"] = self.torch_runtime.to_dict()
        return payload


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight check result."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)


@dataclass(frozen=True)
class PreflightReport:
    """Aggregated preflight report."""

    status: CheckStatus
    checks: list[PreflightCheck]
    machine: MachineEnvironment

    @property
    def ok(self) -> bool:
        """Return true when no failed checks were recorded."""
        return self.status != "fail"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "status": self.status,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "machine": self.machine.to_dict(),
        }
