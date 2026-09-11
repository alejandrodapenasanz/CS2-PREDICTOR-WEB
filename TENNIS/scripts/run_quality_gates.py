"""Ejecuta todas las puertas locales de TENNIS en orden fail-fast.

Requiere el entorno de Python 3.13 instalado desde ``requirements.txt`` o su
lock. No recibe datos operativos. Ejecución desde ``TENNIS/``:
``python scripts/run_quality_gates.py``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(label: str, command: list[str]) -> None:
    """Ejecuta una puerta y propaga inmediatamente su código de error."""

    print(f"[TENNIS quality] {label}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    """Ejecuta lint, formato, tipos, tests, imports y smoke determinista."""

    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"TENNIS requiere Python 3.13; activo: {sys.version.split()[0]}")
    python = sys.executable
    with TemporaryDirectory(
        prefix=".tennis-quality-",
        dir=PROJECT_ROOT,
        ignore_cleanup_errors=True,
    ) as temporary:
        mypy_cache = str(Path(temporary) / "cache")
        pytest_temp = str(Path(temporary) / "pytest")
        gates = (
            (
                "ruff lint",
                [python, "-m", "ruff", "check", "--no-cache", "src", "scripts", "tests"],
            ),
            ("ruff format", [python, "scripts/check_ruff_format.py"]),
            ("mypy", [python, "-m", "mypy", "--cache-dir", mypy_cache]),
            (
                "pytest",
                [
                    python,
                    "-m",
                    "pytest",
                    "tests",
                    "-q",
                    f"--basetemp={pytest_temp}",
                ],
            ),
            ("cobertura de imports", [python, "scripts/check_import_coverage.py"]),
            ("smoke de arranque", [python, "scripts/smoke_project.py"]),
        )
        for label, command in gates:
            _run(label, command)
    print("[TENNIS quality] Todas las puertas están en verde.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
