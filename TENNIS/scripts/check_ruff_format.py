"""Aplica Ruff format con una línea base exacta y monotónica.

No recibe datos operativos. Se ejecuta desde ``TENNIS/`` con
``python scripts/check_ruff_format.py``. Falla ante deuda nueva y también si
una entrada de la línea base ya puede eliminarse.
"""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = PROJECT_ROOT / "quality" / "ruff_format_baseline.txt"
SOURCE_DIRECTORIES = ("src", "scripts", "tests")
WOULD_REFORMAT = re.compile(r"^Would reformat: (.+)$", re.MULTILINE)
UNFORMATTED_CONCISE = re.compile(
    r"^(.+?):\d+:\d+: unformatted: File would be reformatted$", re.MULTILINE
)


def _normalise_reported_path(raw_path: str) -> str:
    """Convierte una ruta emitida por Ruff a una ruta POSIX relativa."""

    path = Path(raw_path.strip())
    if path.is_absolute():
        path = path.relative_to(PROJECT_ROOT)
    return path.as_posix()


def load_baseline() -> set[str]:
    """Carga y valida la lista exacta de archivos heredados sin formato."""

    entries = [
        line.strip()
        for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(entries) != len(set(entries)):
        raise ValueError(f"Hay entradas duplicadas en {BASELINE_PATH}.")
    for entry in entries:
        path = PROJECT_ROOT / entry
        if not path.is_file() or path.suffix != ".py":
            raise ValueError(f"Entrada inválida en la línea base de formato: {entry}")
    return set(entries)


def _python_files() -> list[str]:
    """Enumera todas las fuentes sujetas a la puerta de formato."""

    return [
        path.relative_to(PROJECT_ROOT).as_posix()
        for directory_name in SOURCE_DIRECTORIES
        for path in sorted((PROJECT_ROOT / directory_name).rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def main() -> int:
    """Acepta solo la deuda conocida; cualquier crecimiento o stale entry falla."""

    baseline = load_baseline()
    files = _python_files()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--no-cache",
            "--check",
            "--output-format=concise",
            *files,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    report = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    reported_paths = WOULD_REFORMAT.findall(report) + UNFORMATTED_CONCISE.findall(report)
    unformatted = {_normalise_reported_path(path) for path in reported_paths}
    if completed.returncode not in {0, 1} or (completed.returncode == 1 and not unformatted):
        print(report, file=sys.stderr)
        return completed.returncode
    unexpected = sorted(unformatted - baseline)
    stale = sorted(baseline - unformatted)
    if unexpected or stale:
        if unexpected:
            print("Deuda nueva de Ruff format:", *unexpected, sep="\n- ", file=sys.stderr)
        if stale:
            print(
                "Entradas ya formateadas que deben retirarse de la línea base:",
                *stale,
                sep="\n- ",
                file=sys.stderr,
            )
        return 1
    print(
        f"Ruff format OK: {len(files) - len(baseline)} archivos protegidos; "
        f"{len(baseline)} pendientes heredados sin crecimiento."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
