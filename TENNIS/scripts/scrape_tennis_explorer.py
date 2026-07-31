"""Obtiene y resume la cartelera diaria de singles de Tennis Explorer.

Qué hace:
    Carga desde caché o descarga responsablemente una única página diaria,
    valida su estructura y muestra partidos, slugs, cuotas y recuentos por
    género y nivel.

Qué recibe:
    ``--date YYYY-MM-DD`` selecciona la fecha. Si se omite, se usa el día
    local actual. No existe una opción para redescargar una fecha cacheada.

Cómo se ejecuta:
    Desde ``TENNIS/``:
    ``python scripts/scrape_tennis_explorer.py``
    ``python scripts/scrape_tennis_explorer.py --date 2026-07-30``
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennis_explorer.client import get_daily_matches  # noqa: E402
from src.tennis_explorer.types import TennisExplorerError  # noqa: E402


DISPLAY_COLUMNS = (
    "tournament",
    "tour_level",
    "gender",
    "surface",
    "player_1_name",
    "player_1_slug",
    "player_1_odds",
    "player_2_name",
    "player_2_slug",
    "player_2_odds",
    "status",
)


def _parse_iso_date(value: str) -> date:
    """Convierte una fecha ISO del CLI o produce un error comprensible."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "La fecha debe usar el formato YYYY-MM-DD."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos del scraper diario."""

    parser = argparse.ArgumentParser(
        description=(
            "Obtiene singles ATP, WTA, Challenger e ITF de Tennis Explorer "
            "con caché diaria y sin reintentos."
        )
    )
    parser.add_argument(
        "--date",
        type=_parse_iso_date,
        default=None,
        help="Fecha YYYY-MM-DD; por defecto se usa hoy.",
    )
    return parser


def _print_matches(frame: pd.DataFrame) -> None:
    """Muestra una muestra compacta con las identidades y cuotas requeridas."""

    if frame.empty:
        print("\nNo hay partidos de singles dentro del alcance.")
        return
    sample = frame.loc[:, DISPLAY_COLUMNS].head(10).copy()
    print("\nPrimeras filas")
    print(sample.to_string(index=False))


def _print_level_summary(frame: pd.DataFrame) -> None:
    """Muestra el número de partidos por género y nivel diario."""

    if frame.empty:
        return
    summary = (
        frame.groupby(
            ["gender", "tour_level"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("matches")
        .reset_index()
        .sort_values(["gender", "tour_level"], kind="stable")
    )
    print("\nPartidos por género y nivel")
    print(summary.to_string(index=False))


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta la captura diaria y devuelve un código distinto de cero al parar."""

    args = build_parser().parse_args(argv)
    try:
        frame = get_daily_matches(match_date=args.date)
    except TennisExplorerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    resolved_date = (
        args.date.isoformat()
        if args.date is not None
        else (
            frame["match_date"].iloc[0].date().isoformat()
            if len(frame)
            else date.today().isoformat()
        )
    )
    print(f"\nFecha: {resolved_date}")
    print(f"Partidos dentro del alcance: {len(frame)}")
    _print_matches(frame)
    _print_level_summary(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
