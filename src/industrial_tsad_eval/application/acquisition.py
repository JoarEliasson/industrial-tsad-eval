"""Raw dataset acquisition use cases."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig, DatasetSourceResult
from industrial_tsad_eval.domain.errors import AcquisitionError
from industrial_tsad_eval.domain.validation import ValidationReport
from industrial_tsad_eval.infrastructure.acquisition import (
    PROVENANCE_FILE,
    file_inventory,
    remove_existing_target,
    write_raw_provenance,
)
from industrial_tsad_eval.infrastructure.json_utils import read_json
from industrial_tsad_eval.plugins.registry import DatasetSourceRegistry
from industrial_tsad_eval.ports.dataset_sources import DatasetSourcePlugin


@dataclass(frozen=True)
class DatasetSourceDescription:
    """User-facing dataset source description."""

    name: str
    dataset_name: str
    supported_methods: list[str]
    description: str

    def to_dict(self) -> dict[str, object]:
        """Serialize the description to JSON-compatible data."""
        return {
            "name": self.name,
            "dataset_name": self.dataset_name,
            "supported_methods": list(self.supported_methods),
            "description": self.description,
        }


class ListDatasetSources:
    """List registered raw dataset source plugins."""

    def __init__(self, *, source_registry: DatasetSourceRegistry):
        self.source_registry = source_registry

    def run(self) -> list[DatasetSourceDescription]:
        """Return descriptions for all registered sources."""
        return [
            _description(self.source_registry.get_dataset_source(name))
            for name in self.source_registry.names()
        ]


class DescribeDatasetSource:
    """Describe a registered raw dataset source plugin."""

    def __init__(self, *, source_registry: DatasetSourceRegistry, source: str):
        self.source_registry = source_registry
        self.source = source

    def run(self) -> DatasetSourceDescription:
        """Return a structured source description."""
        return _description(self.source_registry.get_dataset_source(self.source))


class AcquireDatasetSource:
    """Acquire raw dataset files through a source plugin."""

    def __init__(
        self,
        *,
        source_registry: DatasetSourceRegistry,
        source: str,
        out: str | Path,
        config: DatasetSourceConfig,
    ):
        self.source_registry = source_registry
        self.source = source
        self.out = Path(out)
        self.config = config

    def run(self) -> DatasetSourceResult:
        """Acquire raw files using a staging directory before promotion."""
        plugin = self.source_registry.get_dataset_source(self.source)
        _validate_method(plugin.supported_methods(), self.config.method, plugin.name)
        target = self.out / plugin.dataset_name
        if target.exists() and not self.config.overwrite:
            raise AcquisitionError(
                f"Raw output already exists: {target}. Use overwrite=True to replace it."
            )

        staging = self._staging_path(plugin.dataset_name)
        try:
            plugin_result = plugin.acquire(target=staging, config=self.config)
            warnings = list(plugin_result.warnings)
            inventory = file_inventory(staging)
            if not inventory:
                raise AcquisitionError(f"Source {plugin.name!r} did not produce any raw files.")
            write_raw_provenance(
                raw=staging,
                source_name=plugin.name,
                dataset_name=plugin.dataset_name,
                method=self.config.method,
                manual_path=self.config.manual_path,
                ref=self.config.ref,
                extra=self.config.extra,
                warnings=warnings,
            )
            if target.exists():
                remove_existing_target(target, self.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(target))
            return DatasetSourceResult(
                source_name=plugin.name,
                dataset_name=plugin.dataset_name,
                method=self.config.method,
                raw_path=str(target),
                file_count=len(inventory),
                provenance_path=str(target / PROVENANCE_FILE),
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


class ValidateRawAcquisition:
    """Validate raw acquisition provenance and file inventory."""

    def __init__(self, *, source_registry: DatasetSourceRegistry, source: str, raw: str | Path):
        self.source_registry = source_registry
        self.source = source
        self.raw = Path(raw)

    def run(self) -> ValidationReport:
        """Return a validation report for raw acquisition artifacts."""
        errors: list[str] = []
        warnings: list[str] = []
        try:
            plugin = self.source_registry.get_dataset_source(self.source)
        except Exception as exc:
            return ValidationReport("raw-acquisition", str(self.raw), [str(exc)], warnings)

        provenance_path = self.raw / PROVENANCE_FILE
        if not self.raw.exists():
            errors.append(f"Raw path does not exist: {self.raw}")
            return ValidationReport(plugin.dataset_name, str(self.raw), errors, warnings)
        if not provenance_path.exists():
            errors.append(f"Missing required file: {PROVENANCE_FILE}")
            return ValidationReport(plugin.dataset_name, str(self.raw), errors, warnings)

        try:
            provenance = read_json(provenance_path)
        except Exception as exc:
            errors.append(f"Invalid raw provenance JSON: {exc}")
            return ValidationReport(plugin.dataset_name, str(self.raw), errors, warnings)

        if provenance.get("contract_version") != "raw-provenance-v1":
            errors.append("raw_provenance.json has unsupported contract_version")
        if provenance.get("source_name") != plugin.name:
            errors.append("raw_provenance.json source_name does not match requested source")
        if provenance.get("dataset_name") != plugin.dataset_name:
            errors.append("raw_provenance.json dataset_name does not match requested source")
        if provenance.get("method") not in plugin.supported_methods():
            errors.append("raw_provenance.json method is not supported by this source")
        _validate_inventory(self.raw, provenance, errors, warnings)
        return ValidationReport(plugin.dataset_name, str(self.raw), errors, warnings)


def _description(plugin: DatasetSourcePlugin) -> DatasetSourceDescription:
    return DatasetSourceDescription(
        name=plugin.name,
        dataset_name=plugin.dataset_name,
        supported_methods=list(plugin.supported_methods()),
        description=plugin.describe(),
    )


def _validate_method(methods: list[str], method: str, source_name: str) -> None:
    if method not in methods:
        available = ", ".join(methods)
        raise AcquisitionError(
            f"Source {source_name!r} does not support method {method!r}. "
            f"Available methods: {available}."
        )


def _validate_inventory(
    raw: Path,
    provenance: dict[str, object],
    errors: list[str],
    warnings: list[str],
) -> None:
    files = provenance.get("files")
    if not isinstance(files, list):
        errors.append("raw_provenance.json files must be a list")
        return
    file_count = provenance.get("file_count")
    if isinstance(file_count, int) and file_count != len(files):
        errors.append("raw_provenance.json file_count does not match files length")
    for index, row in enumerate(files, start=1):
        if not isinstance(row, dict):
            errors.append(f"raw_provenance.json files[{index}] must be an object")
            continue
        relative = row.get("path")
        if not isinstance(relative, str) or not relative:
            errors.append(f"raw_provenance.json files[{index}] missing path")
            continue
        path = raw / Path(*relative.split("/"))
        if not path.exists():
            errors.append(f"raw_provenance.json references missing file: {relative}")
        elif path.stat().st_size != row.get("size_bytes"):
            warnings.append(f"raw_provenance.json size mismatch for file: {relative}")
