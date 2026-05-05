"""Shared helpers for raw dataset source plugins."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from industrial_tsad_eval.domain.acquisition import DatasetSourceConfig, DatasetSourceResult
from industrial_tsad_eval.domain.errors import AcquisitionError, OptionalDependencyError
from industrial_tsad_eval.infrastructure.acquisition import (
    file_inventory,
    import_manual_source,
    is_supported_archive,
    safe_unpack_archive,
)


def acquire_manual(
    *,
    source_name: str,
    dataset_name: str,
    target: Path,
    config: DatasetSourceConfig,
) -> DatasetSourceResult:
    """Import user-provided raw files from a local path."""
    if not config.manual_path:
        raise AcquisitionError("Manual acquisition requires --manual <path>.")
    import_manual_source(Path(config.manual_path), target)
    return _result(source_name, dataset_name, config.method, target)


def acquire_kaggle(
    *,
    source_name: str,
    dataset_name: str,
    target: Path,
    config: DatasetSourceConfig,
) -> DatasetSourceResult:
    """Download a Kaggle dataset via the optional kagglehub package."""
    ref = config.ref or _string_extra(config, "dataset")
    if not ref:
        raise AcquisitionError("Kaggle acquisition requires --ref <owner/dataset>.")
    try:
        kagglehub = importlib.import_module("kagglehub")
    except ImportError as exc:
        raise OptionalDependencyError(
            "Kaggle acquisition requires the optional acquisition extra: "
            "pip install industrial-tsad-eval[acquisition]."
        ) from exc
    downloaded = Path(str(kagglehub.dataset_download(ref)))
    import_manual_source(downloaded, target)
    return _result(source_name, dataset_name, config.method, target)


def acquire_http(
    *,
    source_name: str,
    dataset_name: str,
    target: Path,
    config: DatasetSourceConfig,
) -> DatasetSourceResult:
    """Download a single HTTP(S) resource and unpack it when it is an archive."""
    url = config.ref or _string_extra(config, "url")
    if not url:
        raise AcquisitionError("HTTP acquisition requires --ref <https-url>.")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise AcquisitionError("HTTP acquisition supports only http:// and https:// URLs.")
    target.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.unquote(parsed.path)).name or "downloaded.raw"
    downloaded = target / filename
    with urllib.request.urlopen(url, timeout=60) as response, downloaded.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if is_supported_archive(downloaded):
        safe_unpack_archive(downloaded, target)
        downloaded.unlink()
    return _result(source_name, dataset_name, config.method, target)


def acquire_git(
    *,
    source_name: str,
    dataset_name: str,
    target: Path,
    config: DatasetSourceConfig,
) -> DatasetSourceResult:
    """Clone a git repository into the raw target."""
    ref = config.ref or _string_extra(config, "url")
    if not ref:
        raise AcquisitionError("Git acquisition requires --ref <repository-url>.")
    target.mkdir(parents=True, exist_ok=True)
    clone_target = target / "repository"
    command = ["git", "clone", "--depth", "1", ref, str(clone_target)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AcquisitionError("Git acquisition requires the git executable on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        raise AcquisitionError(f"Git acquisition failed: {stderr}") from exc
    return _result(source_name, dataset_name, config.method, target)


def _result(
    source_name: str,
    dataset_name: str,
    method: str,
    target: Path,
    warnings: list[str] | None = None,
) -> DatasetSourceResult:
    inventory = file_inventory(target)
    return DatasetSourceResult(
        source_name=source_name,
        dataset_name=dataset_name,
        method=method,
        raw_path=str(target),
        file_count=len(inventory),
        provenance_path=str(target / "raw_provenance.json"),
        warnings=list(warnings or []),
    )


def _string_extra(config: DatasetSourceConfig, key: str) -> str | None:
    value = config.extra.get(key)
    return value if isinstance(value, str) and value else None
