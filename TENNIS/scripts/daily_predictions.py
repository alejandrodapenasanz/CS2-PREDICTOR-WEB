"""Ejecuta y publica las predicciones de la cartelera diaria.

Qué hace:
    Encadena scraper cacheado, mapping de slugs, features estrictamente as-of,
    LightGBM del género correcto, calibración Platt, comparación con mercado y
    publicación CSV y registro append-only en ``BBDD/tennis.sqlite3``.
    Después observa una sola jornada anterior pendiente y concilia resultados
    por slug estable. Conserva sin probabilidad los partidos no elegibles.

Qué recibe:
    ``--date YYYY-MM-DD`` es opcional; si se omite usa la fecha local actual.
    ``--retrained`` lo usa el lanzador para auditar que antes ejecutó el ciclo
    completo de actualización y reentreno.

Cómo se ejecuta:
    Desde ``TENNIS/``:
    ``python scripts/daily_predictions.py``
    ``python scripts/daily_predictions.py --date 2026-07-30``
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

from src.config import OPERATIONS_DATABASE_PATH  # noqa: E402
from src.operations import run_operational_daily_pipeline  # noqa: E402


CONSOLE_MAX_ROWS = 50
CONSOLE_COLUMNS = (
    "tournament",
    "tour_level",
    "surface",
    "player_a_name",
    "player_b_name",
    "model_probability_a",
    "model_probability_b",
    "market_probability_a",
    "market_probability_b",
    "edge_a",
    "confidence",
)


def _parse_iso_date(value: str) -> date:
    """Convierte una fecha ISO exacta o devuelve un error de argumentos."""

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
    """Construye el parser estable del punto de entrada diario."""

    parser = argparse.ArgumentParser(
        description=(
            "Predice la cartelera de Tennis Explorer con modelos calibrados "
            "y publica un CSV auditable."
        )
    )
    parser.add_argument(
        "--date",
        type=_parse_iso_date,
        default=None,
        help="Fecha YYYY-MM-DD; por defecto se usa hoy.",
    )
    parser.add_argument(
        "--retrained",
        action="store_true",
        help="Anota que run_tennis.ps1 completó antes el ciclo de reentreno.",
    )
    return parser


def _percentage(value: object) -> str:
    """Formatea probabilidades/edges y conserva los nulos de forma visible."""

    if value is None or bool(pd.isna(value)):
        return "-"
    return f"{float(value):+.1%}"


def _print_prediction_table(frame: pd.DataFrame) -> None:
    """Muestra hasta 50 predicciones, priorizando el edge absoluto disponible."""

    predicted = frame.loc[frame["model_probability_a"].notna()].copy()
    if predicted.empty:
        print("\nNo hay partidos prospectivos con ambos jugadores mapeados.")
        return
    predicted["_edge_sort"] = predicted["edge_a"].abs().fillna(-1.0)
    predicted.sort_values(
        ["_edge_sort", "tournament"],
        ascending=[False, True],
        kind="stable",
        inplace=True,
    )
    shown = predicted.head(CONSOLE_MAX_ROWS).loc[:, CONSOLE_COLUMNS].copy()
    for column in (
        "model_probability_a",
        "model_probability_b",
        "market_probability_a",
        "market_probability_b",
        "edge_a",
    ):
        shown[column] = shown[column].map(_percentage)
    shown["surface"] = shown["surface"].fillna("Unknown")
    print(
        "\nPredicciones calibradas "
        f"(mostrando {len(shown)} de {len(predicted)})"
    )
    print(shown.to_string(index=False))


def _print_summary(frame: pd.DataFrame) -> None:
    """Resume cobertura de predicción y confianza por nivel."""

    scheduled = int(frame["status"].eq("scheduled").sum())
    predicted = int(frame["model_probability_a"].notna().sum())
    print(f"\nPartidos totales: {len(frame)}")
    print(f"Programados en el snapshot: {scheduled}")
    print(f"Con predicción calibrada: {predicted}")
    print(f"Sin predicción: {len(frame) - predicted}")
    summary = (
        frame.groupby(
            ["gender", "tour_level", "confidence"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("matches")
        .reset_index()
        .sort_values(
            ["gender", "tour_level", "confidence"],
            kind="stable",
        )
    )
    if not summary.empty:
        print("\nCobertura y confianza")
        print(summary.to_string(index=False))


def main(argv: Sequence[str] | None = None) -> int:
    """Ejecuta el pipeline y devuelve un código de proceso auditable."""

    args = build_parser().parse_args(argv)
    try:
        operational = run_operational_daily_pipeline(
            args.date,
            retrained=args.retrained,
        )
    except Exception as exc:
        print(
            f"ERROR [{type(exc).__name__}]: {exc}",
            file=sys.stderr,
        )
        return 1

    run = operational.daily_run
    print(f"\nFecha de cartelera: {run.match_date.isoformat()}")
    print(
        "Predicción generada: "
        f"{run.prediction_as_of_utc.isoformat()}"
    )
    _print_prediction_table(run.predictions)
    _print_summary(run.predictions)
    if run.output_path is not None:
        print(f"\nCSV publicado: {run.output_path}")
    prediction = operational.prediction_registration
    statistics = operational.statistics_registration
    print(f"Base operativa: {OPERATIONS_DATABASE_PATH}")
    print(
        "Registro: "
        f"{prediction.predictions_inserted} filas, "
        f"{prediction.official_predictions_selected} nuevas oficiales, "
        f"{statistics.statistics_inserted} estadísticas prepartido"
    )
    if operational.result_date is None:
        print("Conciliación: no hay una jornada anterior pendiente.")
    elif operational.reconciliation_warning is not None:
        print(
            "ADVERTENCIA de conciliación "
            f"[{operational.result_date.isoformat()}]: "
            f"{operational.reconciliation_warning}",
            file=sys.stderr,
        )
    else:
        reconciliation = operational.observation_reconciliation
        assert reconciliation is not None
        print(
            f"Conciliación {operational.result_date.isoformat()}: "
            f"{reconciliation.observations_inserted} observaciones nuevas, "
            f"{reconciliation.settlements_inserted} resultados cerrados, "
            f"{reconciliation.conflicts_inserted} conflictos"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
