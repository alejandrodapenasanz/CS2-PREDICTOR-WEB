"""Comprueba que cada import externo directo esté declarado.

Recibe opcionalmente ``--requirements`` y ``--root``. Se ejecuta desde
``TENNIS/`` con ``python scripts/check_import_coverage.py``. No importa el
código del proyecto, no accede a red y no modifica artefactos ni SQLite.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Sequence
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ("src", "scripts")
LOCAL_IMPORT_ROOTS = {"src", "scripts"}
IMPORT_TO_DISTRIBUTION = {
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
}
REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)")


def canonical_name(value: str) -> str:
    """Normaliza nombres según la equivalencia de distribuciones Python."""

    return re.sub(r"[-_.]+", "-", value).lower()


def parse_direct_requirements(path: Path) -> set[str]:
    """Devuelve los nombres canónicos declarados en un requirements directo."""

    requirements: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = REQUIREMENT_NAME.match(line)
        if match is None:
            raise ValueError(f"Requirement no soportado en {path}:{line_number}: {line}")
        requirements.add(canonical_name(match.group(1)))
    if not requirements:
        raise ValueError(f"No hay dependencias directas en {path}.")
    return requirements


def _python_files(root: Path) -> Iterable[Path]:
    """Enumera fuentes productivas y scripts, excluyendo cachés derivados."""

    for directory_name in SOURCE_DIRECTORIES:
        directory = root / directory_name
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def discover_third_party_imports(root: Path) -> dict[str, set[str]]:
    """Relaciona cada raíz externa importada con los archivos que la usan."""

    discovered: dict[str, set[str]] = {}
    for path in _python_files(root):
        relative_path = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.partition(".")[0])
        for imported_root in imported_roots:
            if imported_root in sys.stdlib_module_names or imported_root in LOCAL_IMPORT_ROOTS:
                continue
            discovered.setdefault(imported_root, set()).add(relative_path)
    return discovered


def missing_import_requirements(
    declared: set[str], imported: dict[str, set[str]]
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Devuelve imports sin distribución directa, con evidencia de archivos."""

    missing: dict[str, tuple[str, tuple[str, ...]]] = {}
    for imported_root, paths in sorted(imported.items()):
        distribution = canonical_name(IMPORT_TO_DISTRIBUTION.get(imported_root, imported_root))
        if distribution not in declared:
            missing[imported_root] = (distribution, tuple(sorted(paths)))
    return missing


def build_parser() -> argparse.ArgumentParser:
    """Construye la CLI del auditor de cobertura de imports."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--requirements", type=Path, default=PROJECT_ROOT / "requirements.txt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el chequeo y devuelve uno si falta una dependencia directa."""

    args = build_parser().parse_args(argv)
    declared = parse_direct_requirements(args.requirements)
    imported = discover_third_party_imports(args.root)
    missing = missing_import_requirements(declared, imported)
    if missing:
        print("Imports externos sin dependencia directa:", file=sys.stderr)
        for imported_root, (distribution, paths) in missing.items():
            print(
                f"- import {imported_root!r} requiere {distribution!r}: {', '.join(paths)}",
                file=sys.stderr,
            )
        return 1
    print(
        f"Cobertura de imports OK: {len(imported)} raíces externas, "
        f"{len(declared)} dependencias directas."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
