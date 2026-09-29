"""Menú local; las operaciones pertenecen a los entrypoints de cada componente."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main() -> int:
    while True:
        print("\nCS2-Predictor\n1. Solucionar incidencias de descarga de CS2\n0. Salir", flush=True)
        try:
            choice = input("Opción: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            return 0
        if choice == "0":
            return 0
        if choice != "1":
            print("Elige 1 o 0.")
            continue
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "soluciona_errores.py")], cwd=ROOT, check=False
            )
            print(
                {0: "Finalizado.", 2: "Quedan pendientes; consulta el resumen.", 130: "Cancelado."}.get(
                    result.returncode, "La operación falló; consulta el error anterior."
                )
            )
        except KeyboardInterrupt:
            print("\nOperación interrumpida.")


if __name__ == "__main__":
    raise SystemExit(main())
