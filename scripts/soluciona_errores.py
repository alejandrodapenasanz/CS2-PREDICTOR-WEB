"""Lanzador del mantenimiento CS2; no contiene SQL ni lógica del scraper."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reintenta incidencias HLTV; no entrena ni modifica el ledger.")
    parser.add_argument("--solo-diagnostico", action="store_true", help="Solo lectura, sin red ni cambios en la BBDD.")
    parser.add_argument(
        "--limite", type=int, default=0, help="Máximo de entidades; 0 (defecto) revisa todas las incidencias elegibles."
    )
    args = parser.parse_args(argv)
    if args.limite < 0:
        parser.error("--limite debe ser 0 (todas) o un entero positivo")
    python = ROOT / "VAULT" / "CS2" / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.is_file():
        print("Falta el entorno CS2 en VAULT. Ejecuta start.ps1 para prepararlo; no se instala nada a escondidas.")
        return 1
    command = [str(python), str(ROOT / "CS2/PIPELINE/repair_fetch_errors.py"), "--limit", str(args.limite)]
    if args.solo_diagnostico:
        command.append("--diagnose")
    try:
        return subprocess.run(command, cwd=ROOT / "CS2", check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
