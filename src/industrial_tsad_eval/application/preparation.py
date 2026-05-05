"""Dataset preparation use case."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.domain.datasets import DatasetAdapterConfig, DatasetAdapterResult
from industrial_tsad_eval.domain.errors import ContractValidationError, PreparationError
from industrial_tsad_eval.plugins.registry import DatasetAdapterRegistry


class PrepareDataset:
    """Prepare a raw dataset through an adapter and validate the result."""

    def __init__(
        self,
        *,
        adapter_registry: DatasetAdapterRegistry,
        dataset: str,
        raw: str | Path,
        out: str | Path,
        config: DatasetAdapterConfig | None = None,
        overwrite: bool = False,
    ):
        self.adapter_registry = adapter_registry
        self.dataset = dataset
        self.raw = Path(raw)
        self.out = Path(out)
        self.config = config or DatasetAdapterConfig()
        self.overwrite = overwrite

    def run(self) -> DatasetAdapterResult:
        """Run raw-to-prepared conversion using a staging directory."""
        plugin = self.adapter_registry.get_dataset_adapter(self.dataset)
        target = self.out / plugin.dataset_name
        if target.exists() and not self.overwrite:
            raise PreparationError(
                f"Prepared output already exists: {target}. Use overwrite=True to replace it."
            )

        staging = self._staging_path(plugin.dataset_name)
        try:
            result = plugin.prepare(raw=self.raw, prepared=staging, config=self.config)
            report = ValidatePreparedDataset(staging).run()
            if not report.ok:
                errors = "; ".join(report.errors)
                raise ContractValidationError(
                    f"Adapter {plugin.name!r} produced an invalid prepared dataset: {errors}"
                )
            if target.exists():
                _remove_existing_target(target, self.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(target))
            warnings = list(result.warnings) + list(report.warnings)
            return DatasetAdapterResult(
                dataset_name=plugin.dataset_name,
                prepared_path=str(target),
                run_count=result.run_count,
                event_count=result.event_count,
                warnings=warnings,
            )
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _staging_path(self, dataset_name: str) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        staging_parent = self.out / ".staging"
        staging_parent.mkdir(parents=True, exist_ok=True)
        while True:
            candidate = staging_parent / f"{dataset_name}-{uuid4().hex}"
            if not candidate.exists():
                return candidate


def _remove_existing_target(target: Path, root: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if root_resolved not in target_resolved.parents:
        raise PreparationError(f"Refusing to remove path outside output root: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
