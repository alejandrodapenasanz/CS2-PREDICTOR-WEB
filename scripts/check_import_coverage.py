#!/usr/bin/env python3
"""Fail when project Python imports lack a direct requirements declaration.

The check is intentionally static and network-free. It scans Python ASTs, ignores
standard-library and repository-local modules, maps well-known import names to
their distribution names, and compares them with one component's requirements.

Example:
    python scripts/check_import_coverage.py --requirements CS2/requirements.txt \
        --source CS2/MODEL --source CS2/BBDD --source CS2/PIPELINE \
        --source CS2/TESTS --exclude CS2/PIPELINE/start.py
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Sequence


IMPORT_DISTRIBUTIONS = {
    "bs4": "beautifulsoup4",
    "curl_cffi": "curl-cffi",
    "flask": "flask",
    "flasgger": "flasgger",
    "itemadapter": "itemadapter",
    "joblib": "joblib",
    "lightgbm": "lightgbm",
    "matplotlib": "matplotlib",
    "nodriver": "nodriver",
    "numpy": "numpy",
    "optuna": "optuna",
    "packaging": "packaging",
    "pandas": "pandas",
    "parsel": "parsel",
    "pyarrow": "pyarrow",
    "pytest": "pytest",
    "requests": "requests",
    "scipy": "scipy",
    "scrapling": "scrapling",
    "scrapy": "scrapy",
    "shap": "shap",
    "sklearn": "scikit-learn",
    "unidecode": "unidecode",
    "urllib3": "urllib3",
    "xgboost": "xgboost",
    "catboost": "catboost",
    "cloudscraper": "cloudscraper",
    "yaml": "pyyaml",
}


def canonical_name(value: str) -> str:
    """Return the PEP 503 normalized project name."""

    return re.sub(r"[-_.]+", "-", value).lower()


def requirement_names(path: Path) -> set[str]:
    """Read direct distribution names from a requirements input file."""

    names: set[str] = set()
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match is None:
            raise ValueError(f"Unsupported requirement on {path}:{number}: {line}")
        names.add(canonical_name(match.group(1)))
    return names


def python_files(sources: Sequence[Path], excluded: set[Path]) -> list[Path]:
    """Expand source files/directories into deterministic Python paths."""

    files: set[Path] = set()
    for source in sources:
        candidates = [source] if source.is_file() else source.rglob("*.py")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in excluded:
                continue
            if any(
                part.startswith(".venv")
                or part in {"site-packages", "__pycache__", ".pytest_cache"}
                for part in candidate.parts
            ):
                continue
            files.add(candidate)
    return sorted(files)


def local_module_names(files: Sequence[Path]) -> set[str]:
    """Return module names implemented by the selected repository sources."""

    names = {path.stem for path in files}
    for path in files:
        for parent in path.parents:
            if (parent / "__init__.py").is_file():
                names.add(parent.name)
    return names


def imported_modules(files: Sequence[Path]) -> dict[str, set[Path]]:
    """Map absolute top-level imports to the files that use them."""

    imports: dict[str, set[Path]] = {}
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".", 1)[0]]
            for name in names:
                imports.setdefault(name, set()).add(path)
    return imports


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line contract without touching project files."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[], type=Path)
    parser.add_argument("--allow-import", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Check import coverage and return a CI-friendly status code."""

    args = build_parser().parse_args(argv)
    files = python_files(args.source, {path.resolve() for path in args.exclude})
    local = local_module_names(files) | set(args.allow_import)
    declared = requirement_names(args.requirements)
    imports = imported_modules(files)
    missing: dict[str, tuple[str, set[Path]]] = {}
    unknown: dict[str, set[Path]] = {}
    for module, used_by in imports.items():
        if module in sys.stdlib_module_names or module in local:
            continue
        distribution = IMPORT_DISTRIBUTIONS.get(module)
        if distribution is None:
            unknown[module] = used_by
        elif canonical_name(distribution) not in declared:
            missing[module] = (canonical_name(distribution), used_by)

    if unknown or missing:
        for module, used_by in sorted(unknown.items()):
            print(
                f"UNKNOWN import mapping: {module} ({', '.join(map(str, sorted(used_by)))})",
                file=sys.stderr,
            )
        for module, (distribution, used_by) in sorted(missing.items()):
            print(
                f"MISSING dependency: import {module!r} requires {distribution!r} "
                f"({', '.join(map(str, sorted(used_by)))})",
                file=sys.stderr,
            )
        return 1
    print(
        f"import_coverage=ok files={len(files)} external_imports="
        f"{sum(1 for name in imports if name not in sys.stdlib_module_names and name not in local)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
