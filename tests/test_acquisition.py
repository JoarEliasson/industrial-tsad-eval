from __future__ import annotations

import importlib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from industrial_tsad_eval.application.acquisition import (
    AcquireDatasetSource,
    DescribeDatasetSource,
    ListDatasetSources,
    ValidateRawAcquisition,
)
from industrial_tsad_eval.application.preparation import PrepareDataset
from industrial_tsad_eval.application.validation import ValidatePreparedDataset
from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig
from industrial_tsad_eval.domain.errors import (
    AcquisitionError,
    OptionalDependencyError,
    PluginNotFoundError,
)
from industrial_tsad_eval.infrastructure.acquisition import safe_unpack_archive
from industrial_tsad_eval.infrastructure.examples import make_thesis_raw_fixtures
from industrial_tsad_eval.infrastructure.json_utils import read_json
from industrial_tsad_eval.plugins.registry import (
    default_dataset_adapter_registry,
    default_dataset_source_registry,
)
from industrial_tsad_eval.plugins.sources.swat import SWaTDatasetSourcePlugin
from tests.conftest import write_hai_cpps_raw, write_hai_raw, write_swat_raw, write_tep_raw


def test_dataset_source_registry_lists_and_rejects_unknown():
    registry = default_dataset_source_registry()

    descriptions = ListDatasetSources(source_registry=registry).run()

    assert [description.name for description in descriptions] == ["hai", "hai-cpps", "swat", "tep"]
    assert DescribeDatasetSource(source_registry=registry, source="tep").run().dataset_name == "TEP"
    with pytest.raises(PluginNotFoundError):
        registry.get_dataset_source("missing")


def test_manual_acquisition_writes_provenance_inventory_and_validates(tmp_path: Path):
    raw_input = write_swat_raw(tmp_path / "input")

    result = AcquireDatasetSource(
        source_registry=default_dataset_source_registry(),
        source="swat",
        out=tmp_path / "raw-cache",
        config=DatasetSourceConfig(method="manual", manual_path=str(raw_input)),
    ).run()

    assert result.dataset_name == "SWaT"
    assert result.file_count == 2
    assert (Path(result.raw_path) / "SWaT_Dataset_Normal.csv").exists()
    provenance = read_json(Path(result.provenance_path))
    assert provenance["contract_version"] == "raw-provenance-v1"
    assert provenance["file_count"] == 2
    assert all("sha256" in row for row in provenance["files"])
    assert (
        ValidateRawAcquisition(
            source_registry=default_dataset_source_registry(),
            source="swat",
            raw=result.raw_path,
        )
        .run()
        .ok
    )


def test_manual_acquisition_unpacks_safe_archives(tmp_path: Path):
    raw_input = write_swat_raw(tmp_path / "input")
    archive = tmp_path / "swat.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in raw_input.iterdir():
            handle.write(path, arcname=path.name)

    result = AcquireDatasetSource(
        source_registry=default_dataset_source_registry(),
        source="swat",
        out=tmp_path / "raw-cache",
        config=DatasetSourceConfig(method="manual", manual_path=str(archive)),
    ).run()

    assert (Path(result.raw_path) / "SWaT_Dataset_Attack.csv").exists()


def test_archive_path_traversal_is_rejected(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../evil.txt", "nope")

    with pytest.raises(AcquisitionError):
        safe_unpack_archive(archive, tmp_path / "out")


def test_acquisition_refuses_existing_output_unless_overwrite(tmp_path: Path):
    raw_input = write_swat_raw(tmp_path / "input")
    service = AcquireDatasetSource(
        source_registry=default_dataset_source_registry(),
        source="swat",
        out=tmp_path / "raw-cache",
        config=DatasetSourceConfig(method="manual", manual_path=str(raw_input)),
    )
    service.run()

    with pytest.raises(AcquisitionError):
        service.run()

    overwritten = AcquireDatasetSource(
        source_registry=default_dataset_source_registry(),
        source="swat",
        out=tmp_path / "raw-cache",
        config=DatasetSourceConfig(
            method="manual",
            manual_path=str(raw_input),
            overwrite=True,
        ),
    ).run()
    assert Path(overwritten.raw_path).exists()


def test_kaggle_method_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name == "kagglehub":
            raise ImportError("missing")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(OptionalDependencyError):
        SWaTDatasetSourcePlugin().acquire(
            target=tmp_path / "target",
            config=DatasetSourceConfig(method="kaggle", ref="owner/dataset"),
        )


@pytest.mark.parametrize(
    ("source", "writer", "dataset_name"),
    [
        ("tep", write_tep_raw, "TEP"),
        ("swat", write_swat_raw, "SWaT"),
        ("hai", write_hai_raw, "HAI"),
        ("hai-cpps", write_hai_cpps_raw, "HAI_CPPS"),
    ],
)
def test_manual_acquire_then_prepare_for_all_adapters(
    tmp_path: Path,
    source: str,
    writer,
    dataset_name: str,
):
    raw_input = writer(tmp_path / "input")
    acquired = AcquireDatasetSource(
        source_registry=default_dataset_source_registry(),
        source=source,
        out=tmp_path / "raw-cache",
        config=DatasetSourceConfig(method="manual", manual_path=str(raw_input)),
    ).run()

    prepared = PrepareDataset(
        adapter_registry=default_dataset_adapter_registry(),
        dataset=source,
        raw=acquired.raw_path,
        out=tmp_path / "prepared",
    ).run()

    assert acquired.dataset_name == dataset_name
    assert prepared.dataset_name == dataset_name
    assert ValidatePreparedDataset(prepared.prepared_path).run().ok


def test_generated_thesis_raw_fixtures_acquire_prepare_and_validate(tmp_path: Path):
    raw_fixtures = make_thesis_raw_fixtures(tmp_path / "raw-fixtures")
    expected_sources = {"tep", "swat", "hai", "hai-cpps"}

    assert set(raw_fixtures) == expected_sources
    for source in sorted(expected_sources):
        acquired = AcquireDatasetSource(
            source_registry=default_dataset_source_registry(),
            source=source,
            out=tmp_path / "raw-cache",
            config=DatasetSourceConfig(method="manual", manual_path=raw_fixtures[source]),
        ).run()
        raw_report = ValidateRawAcquisition(
            source_registry=default_dataset_source_registry(),
            source=source,
            raw=acquired.raw_path,
        ).run()
        prepared = PrepareDataset(
            adapter_registry=default_dataset_adapter_registry(),
            dataset=source,
            raw=acquired.raw_path,
            out=tmp_path / "prepared",
        ).run()

        assert raw_report.ok
        assert prepared.run_count >= 1
        assert ValidatePreparedDataset(prepared.prepared_path).run().ok
