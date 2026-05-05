from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "industrial_tsad_eval"


def test_cli_dependencies_do_not_leak_outside_cli_package():
    offenders: list[str] = []
    for path in _python_files():
        if "\\interfaces\\cli\\" in str(path) or "/interfaces/cli/" in str(path):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            if names & {"typer", "rich"}:
                offenders.append(str(path.relative_to(SRC_ROOT)))

    assert offenders == []


def test_new_package_does_not_import_old_tsad_package():
    offenders = [
        str(path.relative_to(SRC_ROOT))
        for path in _python_files()
        if "from tsad" in path.read_text(encoding="utf-8")
        or "import tsad" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_adapters_do_not_directly_delete_output_trees():
    adapters_root = SRC_ROOT / "plugins" / "datasets"
    offenders = [
        str(path.relative_to(SRC_ROOT))
        for path in adapters_root.glob("*.py")
        if "rmtree" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_torch_imports_stay_inside_torch_plugin_modules():
    offenders: list[str] = []
    for path in _python_files():
        relative = str(path.relative_to(SRC_ROOT)).replace("\\", "/")
        if relative.startswith("plugins/torch_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            if "torch" in names:
                offenders.append(relative)

    assert offenders == []


def test_optional_profile_dependencies_are_lazy_imports():
    offenders: list[str] = []
    optional_modules = {"psutil", "pynvml"}
    for path in _python_files():
        relative = str(path.relative_to(SRC_ROOT)).replace("\\", "/")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            if names & optional_modules:
                offenders.append(relative)

    assert offenders == []


def _python_files() -> list[Path]:
    return sorted(path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts)
