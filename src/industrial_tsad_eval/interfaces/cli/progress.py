"""Rich progress rendering for CLI commands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from industrial_tsad_eval.domain.progress import NullProgressSink, ProgressEvent, ProgressSink


class RichProgressSink:
    """Render progress events as stage-level Rich tasks."""

    def __init__(self, progress: Progress):
        self.progress = progress
        self.tasks: dict[str, TaskID] = {}
        self.finished_items: set[str] = set()

    def emit(self, event: ProgressEvent) -> None:
        """Update live progress from one event."""
        task_id = self._task(event)
        item_key = event.key
        description = f"{event.stage}: {event.item_id}"
        if event.status == "running":
            self.progress.update(task_id, description=description)
            return
        finished_statuses = {"completed", "failed", "skipped", "warn"}
        if event.status in finished_statuses and item_key not in self.finished_items:
            self.finished_items.add(item_key)
            self.progress.advance(task_id)
        if event.status in {"failed", "warn"}:
            self.progress.update(task_id, description=f"{description} [{event.status}]")

    def _task(self, event: ProgressEvent) -> TaskID:
        key = event.stage
        if key not in self.tasks:
            self.tasks[key] = self.progress.add_task(
                event.stage,
                total=event.total,
            )
        task_id = self.tasks[key]
        if event.total is not None:
            self.progress.update(task_id, total=event.total)
        return task_id


@contextmanager
def cli_progress(enabled: bool) -> Iterator[ProgressSink]:
    """Yield a CLI progress sink or a null sink."""
    if not enabled:
        yield NullProgressSink()
        return
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )
    with progress:
        yield RichProgressSink(progress)
