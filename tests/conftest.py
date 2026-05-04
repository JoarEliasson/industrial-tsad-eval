from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class AccessibleTmpPathFactory:
    """Tmp factory that avoids locked global temp roots on Windows."""

    def __init__(self, root: Path):
        self.root = root
        self.counter = 0
        self.root.mkdir(parents=True, exist_ok=True)

    def mktemp(self, basename: str, numbered: bool = True) -> Path:
        safe_name = "".join(
            char if char.isalnum() or char in {"-", "_"} else "_" for char in basename
        )
        while True:
            suffix = ""
            if numbered:
                self.counter += 1
                suffix = str(self.counter)
            path = self.root / f"{safe_name}{suffix}"
            if not path.exists():
                os.mkdir(path, 0o777)
                return path


@pytest.fixture(scope="session")
def tmp_path_factory() -> AccessibleTmpPathFactory:
    root = PROJECT_ROOT / ".pytest_tmp" / "session"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    return AccessibleTmpPathFactory(root)


@pytest.fixture
def tmp_path(request: pytest.FixtureRequest) -> Path:
    root = PROJECT_ROOT / ".pytest_tmp" / "functions"
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(request.node.nodeid.encode("utf-8")).hexdigest()[:10]
    path = root / digest
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path, 0o777)
    return path
