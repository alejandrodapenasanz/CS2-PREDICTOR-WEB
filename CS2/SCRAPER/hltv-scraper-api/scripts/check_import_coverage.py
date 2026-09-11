"""Ensure every direct third-party import has a declared direct dependency."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterable
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
SOURCE_PATHS = (
    PROJECT_ROOT / "app.py",
    PROJECT_ROOT / "config.py",
    PROJECT_ROOT / "routes",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "hltv_scraper",
    PROJECT_ROOT / "tests",
)
EXCLUDED_PARTS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv.previous",
    "__pycache__",
}
LOCAL_MODULES = {"app", "config", "hltv_scraper", "routes", "scripts", "tests"}
IMPORT_TO_DISTRIBUTION = {
    "cloudscraper": "cloudscraper",
    "flasgger": "flasgger",
    "flask": "flask",
    "itemadapter": "itemadapter",
    "nodriver": "nodriver",
    "packaging": "packaging",
    "parsel": "parsel",
    "pytest": "pytest",
    "requests": "requests",
    "scrapy": "scrapy",
    "scrapy_impersonate": "scrapy-impersonate",
}
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")


def normalize_distribution(name: str) -> str:
    """Return the canonical comparison form used for project names."""

    return re.sub(r"[-_.]+", "-", name).lower()


def iter_python_files(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield checked Python files in stable order."""

    files: set[Path] = set()
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.add(path)
        elif path.is_dir():
            files.update(
                candidate for candidate in path.rglob("*.py") if not EXCLUDED_PARTS.intersection(candidate.parts)
            )
    yield from sorted(files)


def imported_top_level_modules(paths: Iterable[Path]) -> set[str]:
    """Collect absolute top-level imports from Python syntax trees."""

    modules: set[str] = set()
    for path in iter_python_files(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.partition(".")[0])
    return modules


def declared_distributions(requirements_path: Path) -> set[str]:
    """Read direct requirement names without importing packaging itself."""

    declared: set[str] = set()
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith(("-", "http://", "https://")):
            continue
        match = REQUIREMENT_NAME.match(line)
        if match:
            declared.add(normalize_distribution(match.group(1)))
    return declared


def uncovered_imports_from_declared(declared: set[str]) -> dict[str, str]:
    """Map uncovered import names using an already parsed dependency set."""

    imported = imported_top_level_modules(SOURCE_PATHS)
    third_party = imported - sys.stdlib_module_names - LOCAL_MODULES
    return {
        module: normalize_distribution(IMPORT_TO_DISTRIBUTION.get(module, module))
        for module in sorted(third_party)
        if normalize_distribution(IMPORT_TO_DISTRIBUTION.get(module, module)) not in declared
    }


def uncovered_imports(requirements_path: Path = DEFAULT_REQUIREMENTS) -> dict[str, str]:
    """Map uncovered imports against a direct-requirements file."""

    return uncovered_imports_from_declared(declared_distributions(requirements_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help="Direct requirements file to audit.",
    )
    args = parser.parse_args(argv)

    missing = uncovered_imports(args.requirements)
    if missing:
        print("Direct dependency coverage failed:", file=sys.stderr)
        for module, distribution in missing.items():
            print(f"- import {module!r} requires {distribution!r}", file=sys.stderr)
        return 1

    print("Direct dependency coverage: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
