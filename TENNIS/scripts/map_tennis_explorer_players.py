"""Mapea la cartelera diaria de Tennis Explorer a IDs Sackmann.

Qué hace:
    Obtiene una cartelera ya cacheada o mediante el cliente responsable de la
    fase 4, resuelve cada slug dentro del género correcto, persiste asociaciones
    nuevas y muestra cobertura y pendientes sin eliminar partidos no mapeados.

Qué recibe:
    ``--date YYYY-MM-DD`` selecciona la fecha; si se omite usa hoy. Las opciones
    ``--raw-dir``, ``--database``, ``--overrides`` y ``--unresolved`` permiten
    inyectar rutas explícitas para tests o diagnósticos.

Cómo se ejecuta:
    Desde ``TENNIS/``:
    ``python scripts/map_tennis_explorer_players.py``
    ``python scripts/map_tennis_explorer_players.py --date 2026-07-30``
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

from src.config import (  # noqa: E402
    PLAYER_MAPPING_DATABASE_PATH,
    PLAYER_OVERRIDES_PATH,
    RAW_DATA_DIR,
    UNRESOLVED_PLAYERS_PATH,
)
from src.player_mapping import (  # noqa: E402
    PlayerMappingError,
    load_unresolved_players,
    resolve_scraped_matches,
    summarize_mapping_coverage,
)
from src.tennis_explorer import (  # noqa: E402
    TennisExplorerError,
    get_daily_matches,
)


DISPLAY_COLUMNS = (
    "tour_level",
    "gender",
    "player_1_name",
    "player_1_slug",
    "player_1_id",
    "player_1_mapping_method",
    "player_2_name",
    "player_2_slug",
    "player_2_id",
    "player_2_mapping_method",
    "mapping_status",
)

UNRESOLVED_DISPLAY_COLUMNS = (
    "gender",
    "slug",
    "visible_name",
    "reason",
    "candidate_count",
    "candidate_player_ids",
    "last_tour_level",
    "last_tournament",
)


def _parse_iso_date(value: str) -> date:
    """Convierte una fecha ISO del CLI o devuelve un error legible."""

    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "La fecha debe usar el formato YYYY-MM-DD."
        ) from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError(
            "La fecha debe usar exactamente el formato YYYY-MM-DD."
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser del punto de entrada de la fase 5."""

    parser = argparse.ArgumentParser(
        description=(
            "Resuelve slugs Tennis Explorer a IDs Sackmann con caché, "
            "overrides y degradación elegante."
        )
    )
    parser.add_argument(
        "--date",
        type=_parse_iso_date,
        default=None,
        help="Fecha YYYY-MM-DD; por defecto se usa hoy.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DATA_DIR,
        help="Raíz local de los CSV Sackmann.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PLAYER_MAPPING_DATABASE_PATH,
        help="SQLite persistente del mapa de identidades.",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=PLAYER_OVERRIDES_PATH,
        help="CSV de overrides manuales.",
    )
    parser.add_argument(
        "--unresolved",
        type=Path,
        default=UNRESOLVED_PLAYERS_PATH,
        help="CSV agregado de no resueltos.",
    )
    return parser


def _print_sample(mapped: pd.DataFrame) -> None:
    """Muestra las primeras identidades, métodos y estados del lote."""

    if mapped.empty:
        print("\nNo hay partidos de singles dentro del alcance.")
        return
    print("\nPrimeras filas mapeadas")
    print(mapped.loc[:, DISPLAY_COLUMNS].head(10).to_string(index=False))


def _print_coverage(mapped: pd.DataFrame) -> None:
    """Muestra cobertura automática y total por género y nivel."""

    summary = summarize_mapping_coverage(mapped)
    if summary.empty:
        return
    printable = summary.copy()
    for column in (
        "automatic_player_slot_pct",
        "mapped_match_pct",
        "automatic_match_pct",
    ):
        printable[column] = printable[column].map(lambda value: f"{value:.2f}")
    print("\nCobertura por género y nivel (%)")
    print(printable.to_string(index=False))


def _observed_on_date(
    unresolved: pd.DataFrame,
    selected_date: date,
) -> pd.DataFrame:
    """Filtra pendientes cuya lista auditada incluye la fecha solicitada."""

    if unresolved.empty:
        return unresolved
    expected = selected_date.isoformat()
    mask = unresolved["observed_dates"].fillna("").str.split("|").map(
        lambda values: expected in values
    )
    return unresolved.loc[mask].copy()


def _print_unresolved(
    unresolved: pd.DataFrame,
    selected_date: date,
) -> None:
    """Lista únicamente las identidades no resueltas vistas en este lote."""

    observed = _observed_on_date(unresolved, selected_date)
    print(
        "\nIdentidades no resueltas observadas en "
        f"{selected_date.isoformat()}: {len(observed)}"
    )
    if observed.empty:
        return
    print(
        observed.loc[:, UNRESOLVED_DISPLAY_COLUMNS]
        .sort_values(["gender", "last_tour_level", "visible_name"])
        .to_string(index=False)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el scraper cacheado y el resolvedor; devuelve cero si terminan."""

    args = build_parser().parse_args(argv)
    selected_date = args.date or date.today()
    try:
        scraped = get_daily_matches(match_date=selected_date)
        mapped = resolve_scraped_matches(
            scraped,
            as_of_date=selected_date,
            raw_dir=args.raw_dir,
            database_path=args.database,
            overrides_path=args.overrides,
            unresolved_path=args.unresolved,
        )
        unresolved = load_unresolved_players(args.unresolved)
    except (PlayerMappingError, TennisExplorerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nFecha: {selected_date.isoformat()}")
    print(f"Partidos conservados: {len(mapped)}")
    print(
        "Partidos completamente mapeados: "
        f"{int(mapped['mapping_status'].eq('mapped').sum())}"
    )
    _print_sample(mapped)
    _print_coverage(mapped)
    _print_unresolved(unresolved, selected_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
