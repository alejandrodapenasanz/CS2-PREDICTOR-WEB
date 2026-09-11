"""Reconcilia resultados Tennis Explorer ya guardados con partidos oficiales.

QuÃ© hace:
    Lee snapshots crudos append-only anteriores al corte, cruza identidades por
    la puerta causal de operaciones y aÃ±ade observaciones/settlements seguros.
    Nunca actualiza ni elimina el triplete sagrado.

QuÃ© recibe:
    ``--before-date YYYY-MM-DD`` es un corte civil exclusivo opcional.

CÃ³mo se ejecuta, desde ``TENNIS/``:
    ``python scripts/reconcile_results.py``
    ``python scripts/reconcile_results.py --before-date 2026-08-31``
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    OPERATIONS_DATABASE_PATH,
    PLAYER_MAPPING_DATABASE_PATH,
)
from src.operations import reconcile_stored_tennis_explorer_results  # noqa: E402


def _parse_date(value: str) -> date:
    """Valida una fecha ISO exacta para argparse."""

    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD.") from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("Use exactamente YYYY-MM-DD.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Construye la CLI estrecha de conciliaciÃ³n append-only."""

    parser = argparse.ArgumentParser(
        description="Mapea resultados Explorer almacenados a partidos oficiales."
    )
    parser.add_argument(
        "--before-date",
        type=_parse_date,
        default=None,
        help="Corte exclusivo; por defecto se usa hoy.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=OPERATIONS_DATABASE_PATH,
        help="SQLite operativa.",
    )
    parser.add_argument(
        "--mapping-database",
        type=Path,
        default=PLAYER_MAPPING_DATABASE_PATH,
        help="SQLite auditable de identidades Tennis Explorer.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el replay sancionado y muestra cobertura por fecha."""

    args = build_parser().parse_args(argv)
    cutoff = args.before_date or date.today()
    try:
        reports = reconcile_stored_tennis_explorer_results(
            cutoff,
            database_path=args.database,
            mapping_database_path=args.mapping_database,
        )
    except Exception as exc:
        print(f"ERROR [{type(exc).__name__}]: {exc}", file=sys.stderr)
        return 1
    if not reports:
        print("No hay snapshots almacenados pendientes de reconciliar.")
        return 0
    for report in reports:
        result = report.reconciliation
        snapshot = report.snapshot
        print(
            f"{report.result_date.isoformat()}: "
            f"mapped={snapshot.identity_mapped_count}, "
            f"paired={snapshot.paired_inferred_count}, "
            f"settled={result.settlements_inserted}, "
            f"pending={snapshot.unmatched_official_count}, "
            f"ambiguous={snapshot.ambiguous_pair_rows}, "
            f"reused={result.reused}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
