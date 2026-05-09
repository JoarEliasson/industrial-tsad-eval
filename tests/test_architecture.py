from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "industrial_tsad_eval"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_src_layout_root_is_not_a_python_package():
    assert not (PROJECT_ROOT / "src" / "__init__.py").exists()


def test_public_repo_has_no_personal_review_meta_wording():
    forbidden = ["inter" + "view"]
    roots = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "examples",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
    ]
    offenders: list[str] = []
    for path in _text_files(roots):
        text = path.read_text(encoding="utf-8").lower()
        if any(term in text for term in forbidden):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_public_repo_uses_neutral_assistant_replay_language():
    forbidden = ["rq" + "3", "rq" + "4"]
    roots = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "examples",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "tests",
    ]
    offenders: list[str] = []
    for path in _text_files(roots):
        relative = str(path.relative_to(PROJECT_ROOT)).lower()
        text = path.read_text(encoding="utf-8").lower()
        if any(term in relative or term in text for term in forbidden):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


def test_temporary_thesis_draft_pdf_is_ignored():
    ignore_patterns = {
        line.strip()
        for line in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "latest_draft.pdf" in ignore_patterns


def test_tracked_files_do_not_reference_old_repo_paths():
    forbidden = ["tsad-" + "toolkit", "pycharmprojects/tsad-" + "toolkit"]
    offenders: list[str] = []
    for path in _text_files(
        [
            PROJECT_ROOT / "README.md",
            PROJECT_ROOT / "docs",
            PROJECT_ROOT / "examples",
            PROJECT_ROOT / "src",
            PROJECT_ROOT / "tests",
        ]
    ):
        text = path.read_text(encoding="utf-8").lower()
        if any(term in text for term in forbidden):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == []


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


def test_optional_acquisition_dependencies_are_lazy_imports():
    offenders: list[str] = []
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
            if "kagglehub" in names:
                offenders.append(relative)

    assert offenders == []


def test_operator_assistant_has_no_provider_or_network_dependencies():
    offenders: list[str] = []
    forbidden = {"openai", "requests", "httpx", "urllib"}
    operator_paths = [
        SRC_ROOT / "domain" / "operator.py",
        SRC_ROOT / "application" / "operator.py",
        SRC_ROOT / "infrastructure" / "operator_repository.py",
    ]
    for path in operator_paths:
        relative = str(path.relative_to(SRC_ROOT)).replace("\\", "/")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            if names & forbidden:
                offenders.append(relative)

    assert offenders == []


def test_provider_sdk_imports_are_not_eager_dependencies():
    offenders: list[str] = []
    forbidden = {"openai", "anthropic", "google", "xai"}
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
            if names & forbidden:
                offenders.append(relative)

    assert offenders == []


def test_dataset_sources_do_not_directly_delete_output_trees():
    sources_root = SRC_ROOT / "plugins" / "sources"
    offenders = [
        str(path.relative_to(SRC_ROOT))
        for path in sources_root.glob("*.py")
        if "rmtree" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def _python_files() -> list[Path]:
    return sorted(path for path in SRC_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _text_files(roots: list[Path]) -> list[Path]:
    suffixes = {".md", ".py", ".toml"}
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix in suffixes:
            files.append(root)
            continue
        if root.is_dir():
            files.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts
            )
    return sorted(files)
