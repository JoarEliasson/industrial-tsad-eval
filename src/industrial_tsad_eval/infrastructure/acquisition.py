"""Filesystem helpers for safe raw dataset acquisition."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from industrial_tsad_eval.domain.errors import AcquisitionError
from industrial_tsad_eval.infrastructure.json_utils import write_json

PROVENANCE_FILE = "raw_provenance.json"
ARCHIVE_SUFFIXES = {
    ".zip",
    ".tar",
    ".tgz",
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
}


def import_manual_source(source: Path, target: Path) -> None:
    """Import a local directory, archive, or single file into a raw target."""
    if not source.exists():
        raise AcquisitionError(f"Manual raw source does not exist: {source}")
    target.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        copy_directory_contents(source, target)
        return
    if is_supported_archive(source):
        safe_unpack_archive(source, target)
        return
    shutil.copy2(source, target / source.name)


def copy_directory_contents(source: Path, target: Path) -> None:
    """Copy a directory tree into a target without following unsafe output paths."""
    source_resolved = source.resolve()
    target_resolved = target.resolve()
    if source_resolved == target_resolved:
        raise AcquisitionError("Source and target raw directories must be different.")
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        destination = target / relative
        _ensure_within_root(destination, target)
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def safe_unpack_archive(archive: Path, target: Path) -> None:
    """Unpack a zip or tar archive with path-traversal protection."""
    target.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for zip_member in handle.infolist():
                destination = target / zip_member.filename
                _ensure_within_root(destination, target)
            handle.extractall(target)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            for tar_member in members:
                if tar_member.issym() or tar_member.islnk():
                    raise AcquisitionError(
                        f"Refusing to unpack archive link member: {tar_member.name}"
                    )
                destination = target / tar_member.name
                _ensure_within_root(destination, target)
            handle.extractall(target, members=members)
        return

    raise AcquisitionError(f"Unsupported archive type: {archive}")


def is_supported_archive(path: Path) -> bool:
    """Return true when a path has a supported archive suffix."""
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def file_inventory(root: Path) -> list[dict[str, Any]]:
    """Return a stable SHA256 inventory for files below a raw root."""
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == PROVENANCE_FILE:
            continue
        rows.append(
            {
                "path": _as_posix(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    """Compute SHA256 for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_raw_provenance(
    *,
    raw: Path,
    source_name: str,
    dataset_name: str,
    method: str,
    manual_path: str | None,
    ref: str | None,
    extra: dict[str, Any],
    warnings: Iterable[str],
) -> Path:
    """Write raw acquisition provenance and return its path."""
    inventory = file_inventory(raw)
    path = raw / PROVENANCE_FILE
    write_json(
        path,
        {
            "contract_version": "raw-provenance-v1",
            "source_name": source_name,
            "dataset_name": dataset_name,
            "method": method,
            "manual_path": manual_path,
            "ref": ref,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "file_count": len(inventory),
            "files": inventory,
            "warnings": list(warnings),
            "extra": dict(extra),
        },
    )
    return path


def remove_existing_target(target: Path, root: Path) -> None:
    """Remove an existing output after verifying it lives below the output root."""
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if target_resolved != root_resolved and root_resolved not in target_resolved.parents:
        raise AcquisitionError(f"Refusing to remove path outside output root: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def _ensure_within_root(path: Path, root: Path) -> None:
    root_resolved = root.resolve()
    resolved = path.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise AcquisitionError(f"Refusing path outside raw root: {path}")


def _as_posix(path: Path) -> str:
    return "/".join(path.parts)
