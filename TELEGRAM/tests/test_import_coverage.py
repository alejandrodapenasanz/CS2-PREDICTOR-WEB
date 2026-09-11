"""Keep Telegram's stdlib-only runtime and declared quality tools verifiable offline."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys


def test_runtime_imports_are_stdlib_or_local_and_boot() -> None:
    """Audit every runtime import and import each local module without sending."""

    root = Path(__file__).resolve().parents[1]
    runtime_files = [*root.joinpath("src").rglob("*.py"), *root.joinpath("scripts").glob("*.py")]
    external: set[str] = set()
    for path in runtime_files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                external.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                external.add(node.module.split(".")[0])
    assert external - sys.stdlib_module_names - {"cs2_telegram", "__future__"} == set()
    for path in root.joinpath("src", "cs2_telegram").glob("*.py"):
        module = "cs2_telegram" if path.stem == "__init__" else f"cs2_telegram.{path.stem}"
        importlib.import_module(module)
    requirements = root.joinpath("requirements.txt").read_text(encoding="utf-8")
    assert all(package in requirements for package in ("pytest", "mypy", "ruff", "tzdata"))
