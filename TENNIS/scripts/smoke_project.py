"""Verifica imports de runtime y boot determinista de entrypoints reales.

No recibe datos ni ejecuta pipelines: importa las dependencias directas y
arranca con ``--help`` cada script invocado por ``run_tennis.ps1``. Ejecución:
``python scripts/smoke_project.py``.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_IMPORTS = (
    "bs4",
    "curl_cffi",
    "joblib",
    "lightgbm",
    "lxml.etree",
    "matplotlib",
    "mypy.main",
    "numpy",
    "pandas",
    "pyarrow",
    "pyarrow.parquet",
    "requests",
    "sklearn",
    "scrapling.fetchers",
    "unidecode",
    "urllib3",
)
LAUNCHER_ENTRYPOINTS = (
    "scripts/update_tennis_abstract.py",
    "scripts/update_tennisratio.py",
    "scripts/update_sources.py",
    "scripts/audit_identities.py",
    "scripts/build_elo.py",
    "scripts/build_features.py",
    "scripts/retrain_models.py",
    "scripts/daily_predictions.py",
)
DIAGNOSTIC_ENTRYPOINTS = ("scripts/audit_tennis_abstract.py",)


def _boot_entrypoint(relative_path: str) -> str:
    """Arranca dos veces un entrypoint y devuelve su salida estable."""

    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "42",
        }
    )
    outputs: list[str] = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, relative_path, "--help"],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"No arranca {relative_path} ({completed.returncode}):\n{completed.stderr}"
            )
        outputs.append(completed.stdout + completed.stderr)
    if not outputs[0].strip() or outputs[0] != outputs[1]:
        raise RuntimeError(f"Boot no determinista o vacío: {relative_path}")
    return outputs[0]


def main() -> int:
    """Ejecuta el smoke sin red, escrituras operativas ni carga de SQLite."""

    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"TENNIS requiere Python 3.13; activo: {sys.version.split()[0]}")
    for module_name in RUNTIME_IMPORTS:
        importlib.import_module(module_name)
    entrypoints = LAUNCHER_ENTRYPOINTS + DIAGNOSTIC_ENTRYPOINTS
    for relative_path in entrypoints:
        _boot_entrypoint(relative_path)
    print(
        f"Smoke OK: {len(RUNTIME_IMPORTS)} imports y {len(entrypoints)} entrypoints deterministas."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
