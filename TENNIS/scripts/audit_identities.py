"""Genera la cuarentena reproducible de identidades Sackmann conflictivas.

Qué hace:
    Verifica el snapshot Sackmann, recorre sus partidos y pone en cuarentena
    cualquier ``(gender, player_id)`` con apariciones anteriores al décimo
    cumpleaños declarado. No divide personas ni inventa identificadores.

Qué recibe:
    ``--chunksize`` controla únicamente el tamaño de lectura de los CSV.

Cómo se ejecuta, desde ``TENNIS/``:
    ``python scripts/audit_identities.py``
    ``python scripts/audit_identities.py --chunksize 200000``
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.identity_integrity import (  # noqa: E402
    MIN_PLAUSIBLE_MATCH_AGE_YEARS,
    publish_identity_quarantine,
)


def build_parser() -> argparse.ArgumentParser:
    """Construye la CLI de auditoría de identidades."""

    parser = argparse.ArgumentParser(
        description="Detecta IDs Sackmann temporalmente incompatibles."
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Filas por chunk al recorrer los CSV verificados.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Publica la cuarentena y muestra todas las claves detectadas."""

    args = build_parser().parse_args(argv)
    report = publish_identity_quarantine(chunksize=args.chunksize)
    print(
        f"Commit: {report.source_commit}\n"
        f"Umbral conservador: < {MIN_PLAUSIBLE_MATCH_AGE_YEARS} años\n"
        f"Identidades en cuarentena: {len(report.conflicts)}\n"
        f"CSV: {report.csv_path}\n"
        f"Manifiesto: {report.manifest_path}"
    )
    if report.conflicts:
        print(
            "gender\tplayer_id\tplayer_name\tbirth_date\t"
            "first_match\tlast_match\tearliest_age\tappearances"
        )
        for conflict in report.conflicts:
            print(
                f"{conflict.gender}\t{conflict.player_id}\t"
                f"{conflict.player_name or '-'}\t"
                f"{conflict.birth_date}\t{conflict.first_match_date}\t"
                f"{conflict.last_match_date}\t"
                f"{conflict.earliest_age_years:.3f}\t"
                f"{conflict.appearances}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
