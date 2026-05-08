"""Lazy system, environment, and accelerator probes."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from industrial_tsad_eval.domain.system import (
    DeviceRequest,
    MachineEnvironment,
    ResolvedDevice,
    SystemGpu,
    TorchRuntimeStatus,
)

KEY_PACKAGES = [
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "torch",
    "typer",
    "rich",
]

MACHINE_PROFILES = {
    "small": {"max_ram_gb": 16.0, "batch_size": 16},
    "medium": {"max_ram_gb": 64.0, "batch_size": 64},
    "large": {"max_ram_gb": float("inf"), "batch_size": 128},
}

_VALID_DEVICE_REQUESTS = {"auto", "cpu", "cuda", "xpu"}


def normalize_device_request(requested_device: str) -> DeviceRequest:
    """Validate and normalize a runtime device request."""
    normalized = requested_device.strip().lower()
    if normalized not in _VALID_DEVICE_REQUESTS:
        valid = ", ".join(sorted(_VALID_DEVICE_REQUESTS))
        raise ValueError(
            f"Unsupported device request {requested_device!r}. Expected one of: {valid}."
        )
    return cast(DeviceRequest, normalized)


def classify_gpu_name(name: str) -> str:
    """Classify a GPU adapter name into a vendor bucket."""
    lowered = name.lower()
    if "nvidia" in lowered:
        return "nvidia"
    if re.search(r"intel.*arc", lowered) or re.search(r"intel.*max", lowered):
        return "intel"
    if "amd" in lowered or "radeon" in lowered:
        return "amd"
    return "unknown"


def recommend_backend_for_gpus(gpus: list[SystemGpu]) -> ResolvedDevice:
    """Recommend a runtime backend from detected GPU adapters."""
    if any(gpu.vendor == "nvidia" for gpu in gpus):
        return "cuda"
    if any(gpu.vendor == "intel" for gpu in gpus):
        return "xpu"
    return "cpu"


def recommend_backend_for_runtime(
    gpus: list[SystemGpu],
    runtime: TorchRuntimeStatus,
) -> ResolvedDevice:
    """Recommend a backend using torch readiness when OS GPU probes are sparse."""
    if runtime.ready and runtime.resolved_device in {"cuda", "xpu"}:
        return runtime.resolved_device
    return recommend_backend_for_gpus(gpus)


def detect_system_gpus() -> list[SystemGpu]:
    """Detect display adapters with safe platform-specific fallbacks."""
    system_name = platform.system().lower()
    if system_name == "windows":
        return _detect_windows_gpus()
    if system_name == "linux":
        return _detect_linux_gpus()
    if system_name == "darwin":
        return _detect_macos_gpus()
    return []


def probe_torch_runtime(requested_device: str = "auto") -> TorchRuntimeStatus:
    """Probe optional torch runtime and accelerator readiness."""
    normalized = normalize_device_request(requested_device)
    try:
        torch_module = importlib.import_module("torch")
    except ImportError as exc:
        resolved = "cpu" if normalized == "auto" else normalized
        return TorchRuntimeStatus(
            requested_device=normalized,
            resolved_device=resolved,
            ready=False,
            torch_available=False,
            torch_version=None,
            torch_import_error=f"{type(exc).__name__}: {exc}",
            cuda_available=False,
            xpu_available=False,
            device_name=None,
        )

    cuda_available = _cuda_available(torch_module)
    xpu_available = _xpu_available(torch_module)
    resolved = _resolve_backend(normalized, cuda_available, xpu_available)
    ready = (
        resolved == "cpu"
        or (resolved == "cuda" and cuda_available)
        or (resolved == "xpu" and xpu_available)
    )
    return TorchRuntimeStatus(
        requested_device=normalized,
        resolved_device=resolved,
        ready=ready,
        torch_available=True,
        torch_version=getattr(torch_module, "__version__", None),
        torch_import_error=None,
        cuda_available=cuda_available,
        xpu_available=xpu_available,
        device_name=_device_name(torch_module, resolved),
    )


def capture_machine_environment(
    *,
    device_request: str = "auto",
    full_package_dump: bool = False,
    cwd: str | Path | None = None,
) -> MachineEnvironment:
    """Capture a reproducibility-oriented machine environment report."""
    runtime = probe_torch_runtime(device_request)
    gpus = detect_system_gpus()
    ram_gb = _ram_total_gb()
    profile = classify_machine_profile(ram_gb)
    git_commit, git_dirty = _git_info(Path(cwd) if cwd is not None else Path.cwd())
    packages = _all_packages() if full_package_dump else _key_packages()
    return MachineEnvironment(
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        hostname=platform.node() or None,
        os=platform.platform(),
        python_version=platform.python_version(),
        platform_machine=platform.machine(),
        cpu=platform.processor() or platform.machine() or "unknown",
        ram_total_gb=ram_gb,
        gpu_adapters=gpus,
        torch_runtime=runtime,
        recommended_backend=recommend_backend_for_runtime(gpus, runtime),
        machine_profile=profile,
        resolved_batch_size=resolve_batch_size(profile),
        git_commit=git_commit,
        git_dirty=git_dirty,
        packages=packages,
    )


def classify_machine_profile(ram_gb: float | None) -> str:
    """Classify a machine profile from total RAM."""
    if ram_gb is None:
        return "medium"
    for profile_name, payload in MACHINE_PROFILES.items():
        if ram_gb < float(payload["max_ram_gb"]):
            return profile_name
    return "large"


def resolve_batch_size(profile: str) -> int:
    """Resolve a default batch size from a machine profile."""
    return int(MACHINE_PROFILES[profile]["batch_size"])


def module_available(module_name: str) -> bool:
    """Return whether a Python module is importable without importing it."""
    return importlib.util.find_spec(module_name) is not None


def _resolve_backend(
    requested: DeviceRequest,
    cuda_available: bool,
    xpu_available: bool,
) -> ResolvedDevice:
    if requested == "auto":
        if cuda_available:
            return "cuda"
        if xpu_available:
            return "xpu"
        return "cpu"
    return requested


def _cuda_available(torch_module: Any) -> bool:
    try:
        return bool(torch_module.cuda.is_available()) if hasattr(torch_module, "cuda") else False
    except Exception:
        return False


def _xpu_available(torch_module: Any) -> bool:
    try:
        return bool(torch_module.xpu.is_available()) if hasattr(torch_module, "xpu") else False
    except Exception:
        return False


def _device_name(torch_module: Any, device: ResolvedDevice) -> str | None:
    if device == "cuda":
        try:
            return str(torch_module.cuda.get_device_name(0))
        except Exception:
            return None
    if device == "xpu" and hasattr(torch_module, "xpu"):
        try:
            return str(torch_module.xpu.get_device_name(0))
        except Exception:
            return None
    return platform.processor() or platform.machine() or None


def _detect_windows_gpus() -> list[SystemGpu]:
    commands = [
        (
            "powershell-get-ciminstance",
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
        ),
        ("wmic-video-controller", ["cmd", "/c", "wmic path win32_VideoController get Name"]),
    ]
    for source, command in commands:
        names = _run_detection_command(command)
        if names:
            return [
                SystemGpu(name=name, vendor=classify_gpu_name(name), source=source)
                for name in names
            ]
    return []


def _detect_linux_gpus() -> list[SystemGpu]:
    names = _run_detection_command(["sh", "-lc", "lspci | grep -Ei 'vga|3d|display'"])
    return [SystemGpu(name=name, vendor=classify_gpu_name(name), source="lspci") for name in names]


def _detect_macos_gpus() -> list[SystemGpu]:
    names = _run_detection_command(["system_profiler", "SPDisplaysDataType"])
    return [
        SystemGpu(name=name, vendor=classify_gpu_name(name), source="system_profiler")
        for name in names
    ]


def _run_detection_command(command: list[str]) -> list[str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0 and not completed.stdout:
        return []
    return _parse_gpu_names(completed.stdout)


def _parse_gpu_names(raw_output: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for raw_line in raw_output.splitlines():
        line = raw_line.strip()
        if not line or line.lower() == "name":
            continue
        if ":" in line and not re.search(r"(nvidia|intel|amd|radeon|arc)", line, re.IGNORECASE):
            continue
        if line not in seen:
            seen.add(line)
            names.append(line)
    return names


def _ram_total_gb() -> float | None:
    try:
        psutil_module = importlib.import_module("psutil")
        return round(float(psutil_module.virtual_memory().total) / (1024**3), 1)
    except Exception:
        pass

    if platform.system().lower() == "windows":
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return round(int(completed.stdout.strip()) / (1024**3), 1)
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired, ValueError):
            pass
    return None


def _git_info(cwd: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "-c", f"safe.directory={cwd}", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(
            ["git", "-c", f"safe.directory={cwd}", "status", "--porcelain"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit.stdout.strip(), dirty
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None, None


def _key_packages() -> dict[str, str]:
    result: dict[str, str] = {}
    for package_name in KEY_PACKAGES:
        try:
            result[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def _all_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        try:
            name = distribution.metadata["Name"]
        except KeyError:
            continue
        if name:
            packages[str(name)] = distribution.version
    return packages
