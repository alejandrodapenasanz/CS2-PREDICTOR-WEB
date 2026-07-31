"""Comprueba si el pin de Scrapling coincide con su última release estable.

Qué hace:
    Consulta una vez la API oficial ``releases/latest`` de GitHub mediante el
    módulo ``src.scrapling_release``. No sigue la rama ``main``, no instala ni
    actualiza dependencias y no escribe archivos.

Qué recibe:
    No recibe argumentos funcionales. ``--help`` muestra este contrato.

Cómo se ejecuta:
    Desde ``TENNIS/``:
    ``python scripts/check_scrapling_release.py``

Códigos de salida:
    ``0`` si el pin ``0.4.12`` coincide con la última release estable.
    ``1`` si existe una release estable posterior.
    ``2`` si la comprobación no puede completarse con seguridad.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scrapling_release import (  # noqa: E402
    ScraplingReleaseCheckError,
    check_latest_scrapling_release,
)


def build_parser() -> argparse.ArgumentParser:
    """Construye el CLI sin opciones que puedan modificar el proyecto."""

    return argparse.ArgumentParser(
        description=(
            "Compara el pin local de Scrapling con la última release estable "
            "publicada por la API oficial de GitHub."
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la comprobación y devuelve un código apto para automatización."""

    build_parser().parse_args(argv)
    try:
        result = check_latest_scrapling_release()
    except ScraplingReleaseCheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if result.update_available:
        print(
            "NUEVA RELEASE ESTABLE DE SCRAPLING: "
            f"pin={result.pinned_version}, "
            f"GitHub={result.latest_version} "
            f"({result.release_url}).",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: el pin de Scrapling "
        f"{result.pinned_version} coincide con la última release estable."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
