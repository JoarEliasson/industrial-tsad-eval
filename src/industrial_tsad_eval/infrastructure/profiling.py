"""Runtime stage monitoring and profiling report helpers."""

from __future__ import annotations

import csv
import importlib
import statistics
import threading
import time
import tracemalloc
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path
from typing import Any

from industrial_tsad_eval.domain.profiling import StageSample


class StageMonitor:
    """Context manager that records timing and best-effort memory metrics."""

    def __init__(
        self,
        stage: str,
        *,
        meta: dict[str, Any] | None = None,
        enable_vram: bool = False,
        sample_interval_ms: int = 10,
    ):
        self.stage = stage
        self.meta = dict(meta or {})
        self.enable_vram = enable_vram
        self.sample_interval_ms = sample_interval_ms
        self.sample: StageSample | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._rss_peak: int | None = None
        self._vram_peak: int | None = None
        self._tracemalloc_owner = False
        self._warnings: list[str] = []

    def __enter__(self) -> StageMonitor:
        """Start monitoring."""
        self.start_ts_ns = time.perf_counter_ns()
        self.rss_before = _rss_bytes()
        self._rss_peak = self.rss_before
        self.vram_before = _vram_bytes() if self.enable_vram else None
        self._vram_peak = self.vram_before
        self.torch_before = _torch_memory_bytes()

        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._tracemalloc_owner = True

        self._thread = threading.Thread(target=self._poll_memory, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Stop monitoring and build the sample."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.end_ts_ns = time.perf_counter_ns()
        rss_after = _rss_bytes()
        vram_after = _vram_bytes() if self.enable_vram else None
        torch_after = _torch_memory_bytes()
        python_current, python_peak = _traced_memory()
        if self._tracemalloc_owner:
            tracemalloc.stop()

        rss_peak = _max_optional(self._rss_peak, rss_after)
        vram_peak = _max_optional(self._vram_peak, vram_after)
        torch_peak = _max_optional(self.torch_before, torch_after)
        self.sample = StageSample(
            stage=self.stage,
            start_ts_ns=self.start_ts_ns,
            end_ts_ns=self.end_ts_ns,
            duration_ms=(self.end_ts_ns - self.start_ts_ns) / 1_000_000.0,
            rss_before_bytes=self.rss_before,
            rss_after_bytes=rss_after,
            rss_peak_bytes=rss_peak,
            python_current_bytes=python_current,
            python_peak_bytes=python_peak,
            torch_before_bytes=self.torch_before,
            torch_after_bytes=torch_after,
            torch_peak_bytes=torch_peak,
            vram_before_bytes=self.vram_before,
            vram_after_bytes=vram_after,
            vram_peak_bytes=vram_peak,
            warnings=list(self._warnings),
            meta=dict(self.meta),
        )

    def _poll_memory(self) -> None:
        interval_seconds = max(self.sample_interval_ms, 1) / 1000.0
        while not self._stop_event.is_set():
            self._rss_peak = _max_optional(self._rss_peak, _rss_bytes())
            if self.enable_vram:
                self._vram_peak = _max_optional(self._vram_peak, _vram_bytes())
            time.sleep(interval_seconds)


def percentile(values: list[float], p: float) -> float:
    """Return a simple linear percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (index - low))


def summarize_samples(samples: list[StageSample]) -> dict[str, Any]:
    """Summarize stage samples by stage name."""
    by_stage: dict[str, list[StageSample]] = {}
    for sample in samples:
        by_stage.setdefault(sample.stage, []).append(sample)

    stages: dict[str, Any] = {}
    for stage, stage_samples in sorted(by_stage.items()):
        durations = [sample.duration_ms for sample in stage_samples]
        stages[stage] = {
            "duration_ms": {
                "mean": statistics.mean(durations),
                "p50": percentile(durations, 0.5),
                "p95": percentile(durations, 0.95),
                "max": max(durations),
            },
            "rss_peak_bytes": _max_values(sample.rss_peak_bytes for sample in stage_samples),
            "python_peak_bytes": _max_values(sample.python_peak_bytes for sample in stage_samples),
            "torch_peak_bytes": _max_values(sample.torch_peak_bytes for sample in stage_samples),
            "vram_peak_bytes": _max_values(sample.vram_peak_bytes for sample in stage_samples),
        }
    return {
        "stage_count": len(samples),
        "stages": stages,
        "peak_memory": {
            "rss_peak_bytes": _max_values(sample.rss_peak_bytes for sample in samples),
            "python_peak_bytes": _max_values(sample.python_peak_bytes for sample in samples),
            "torch_peak_bytes": _max_values(sample.torch_peak_bytes for sample in samples),
            "vram_peak_bytes": _max_values(sample.vram_peak_bytes for sample in samples),
        },
    }


def write_stage_csv(path: Path, samples: list[StageSample]) -> None:
    """Write stage samples as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stage",
        "start_ts_ns",
        "end_ts_ns",
        "duration_ms",
        "rss_before_bytes",
        "rss_after_bytes",
        "rss_peak_bytes",
        "python_current_bytes",
        "python_peak_bytes",
        "torch_before_bytes",
        "torch_after_bytes",
        "torch_peak_bytes",
        "vram_before_bytes",
        "vram_after_bytes",
        "vram_peak_bytes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_dict())


def render_budget_markdown(summary: dict[str, Any]) -> str:
    """Render a compact human-readable budget report."""
    lines = ["# Profiling Budget Check", ""]
    stages = summary.get("stages", {})
    if not isinstance(stages, dict):
        return "# Profiling Budget Check\n\nNo stages recorded.\n"
    for stage, payload in stages.items():
        duration = dict(payload.get("duration_ms", {}))
        lines.extend(
            [
                f"## {stage}",
                f"- Mean latency: {_fmt(duration.get('mean'))} ms",
                f"- P95 latency: {_fmt(duration.get('p95'))} ms",
                f"- Max latency: {_fmt(duration.get('max'))} ms",
                f"- Peak RSS: {_mb(payload.get('rss_peak_bytes'))}",
                f"- Peak Python allocation: {_mb(payload.get('python_peak_bytes'))}",
                f"- Peak torch memory: {_mb(payload.get('torch_peak_bytes'))}",
                f"- Peak VRAM: {_mb(payload.get('vram_peak_bytes'))}",
                "",
            ]
        )
    return "\n".join(lines)


def _rss_bytes() -> int | None:
    try:
        psutil_module = importlib.import_module("psutil")
        return int(psutil_module.Process().memory_info().rss)
    except Exception:
        return None


def _vram_bytes() -> int | None:
    try:
        pynvml = importlib.import_module("pynvml")
        with suppress(Exception):
            pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(info.used)
    except Exception:
        return None


def _torch_memory_bytes() -> int | None:
    try:
        torch_module = importlib.import_module("torch")
    except Exception:
        return None
    try:
        if hasattr(torch_module, "cuda") and torch_module.cuda.is_available():
            return int(torch_module.cuda.memory_allocated(0))
        if (
            hasattr(torch_module, "xpu")
            and torch_module.xpu.is_available()
            and hasattr(torch_module.xpu, "memory_allocated")
        ):
            return int(torch_module.xpu.memory_allocated(0))
    except Exception:
        return None
    return None


def _traced_memory() -> tuple[int | None, int | None]:
    if not tracemalloc.is_tracing():
        return None, None
    current, peak = tracemalloc.get_traced_memory()
    return int(current), int(peak)


def _max_optional(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _max_values(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _fmt(value: int | float | None) -> str:
    return f"{float(value):.2f}" if value is not None else "N/A"


def _mb(value: int | float | None) -> str:
    return f"{float(value) / (1024 * 1024):.2f} MB" if value is not None else "N/A"
