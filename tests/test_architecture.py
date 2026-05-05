from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src" / "industrial_tsad_eval"


def _python_files() -> list[Path]:
    return sorted(SRC_ROOT.rglob("*.py"))


def test_cli_libraries_are_limited_to_cli_interface():
    offenders: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC_ROOT)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            is_cli_file = relative.parts[:2] == ("interfaces", "cli")
            if module.split(".")[0] in {"typer", "rich"} and not is_cli_file:
                offenders.append(str(relative))
    assert offenders == []


def test_new_package_does_not_import_legacy_tsad_package():
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            if module == "tsad" or module.startswith("tsad."):
                offenders.append(str(path.relative_to(SRC_ROOT)))
    assert offenders == []


def test_dataset_adapters_do_not_delete_output_directly():
    offenders: list[str] = []
    adapters_root = SRC_ROOT / "plugins" / "datasets"
    for path in sorted(adapters_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_rmtree_call(node.func):
                offenders.append(str(path.relative_to(SRC_ROOT)))
    assert offenders == []


def test_detector_name_branching_is_not_used_for_dispatch():
    offenders: list[str] = []
    for path in _python_files():
        relative = path.relative_to(SRC_ROOT)
        if str(relative).startswith("plugins/registry"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _mentions_detector_name(node.test):
                offenders.append(str(relative))
    assert offenders == []


def _mentions_detector_name(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in {"detector", "detector_name"}:
            return True
    return False


def _is_rmtree_call(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "rmtree"
    if isinstance(node, ast.Name):
        return node.id == "rmtree"
    return False
