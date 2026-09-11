"""Ejecuta y publica las predicciones de la cartelera diaria.

Qué hace:
    Encadena scraper cacheado, mapping de slugs, features estrictamente as-of,
    LightGBM del género correcto, calibración Platt, comparación con mercado y
    publicación CSV y registro append-only en ``BBDD/tennis.sqlite3``.
    Después observa una sola jornada anterior pendiente y concilia resultados
    por identidad estable. Antes reprocesa de forma idempotente los snapshots
    ya guardados para recuperar cruces seguros entre fuentes. Conserva sin
    probabilidad los partidos no elegibles.

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
from datetime import UTC, date, datetime
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OPERATIONS_DATABASE_PATH  # noqa: E402
from src.daily_pipeline.format_report import match_format_summary  # noqa: E402
from src.freshness import (  # noqa: E402
    build_freshness_report,
    freshness_summary_lines,
    write_freshness_report,
)
from src.operations import (  # noqa: E402
    reconcile_stored_tennis_explorer_results,
    run_operational_daily_pipeline,
)
from src.operations.result_sources import (  # noqa: E402
    TennisRatioResultSnapshot,
)


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
        raise argparse.ArgumentTypeError("La fecha debe usar el formato YYYY-MM-DD.") from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("La fecha debe usar exactamente el formato YYYY-MM-DD.")
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
    parser.add_argument(
        "--tennisratio-attempt-status",
        choices=("success", "failed", "not_attempted"),
        default="not_attempted",
        help="Estado del intento de actualización efectuado por run_tennis.ps1.",
    )
    parser.add_argument(
        "--tennisratio-attempted-at-utc",
        default="",
        help="Instante ISO UTC del intento de actualización de TennisRatio.",
    )
    return parser


def _parse_optional_utc(value: str) -> datetime | None:
    """Normaliza un instante opcional del launcher a UTC."""

    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("El instante de actualización debe incluir zona horaria.")
    return parsed.astimezone(UTC)


def _percentage(value: object) -> str:
    """Formatea probabilidades/edges y conserva los nulos de forma visible."""

    if value is None or bool(pd.isna(value)):
        return "-"
    return f"{float(str(value)):+.1%}"


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
    print(f"\nPredicciones calibradas (mostrando {len(shown)} de {len(predicted)})")
    print(shown.to_string(index=False))


def _print_summary(frame: pd.DataFrame) -> None:
    """Resume cobertura de predicción y confianza por nivel."""

    scheduled = int(frame["status"].eq("scheduled").sum())
    predicted = int(frame["model_probability_a"].notna().sum())
    print(f"\nPartidos totales: {len(frame)}")
    print(f"Programados en el snapshot: {scheduled}")
    print(f"Con predicción calibrada: {predicted}")
    print(f"Sin predicción: {len(frame) - predicted}")
    for line in match_format_summary(frame):
        print(line)
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
        selected_date = args.date or date.today()
        stored_reconciliations = reconcile_stored_tennis_explorer_results(
            selected_date,
        )
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
    attempted_at = _parse_optional_utc(args.tennisratio_attempted_at_utc)
    source_families = set(run.predictions.get("source_family", pd.Series(dtype="string")).dropna())
    freshness = build_freshness_report(
        observed_at=run.prediction_as_of_utc,
        pipeline_last_run_at=run.prediction_as_of_utc,
        attempts={
            "tennisratio": {
                "status": args.tennisratio_attempt_status,
                "attempted_at_utc": attempted_at.isoformat() if attempted_at else None,
            }
        },
        fallback_sources=(
            frozenset({"tennis_explorer"}) if "tennis_explorer" in source_families else frozenset()
        ),
    )
    write_freshness_report(freshness)
    print(f"\nFecha de cartelera: {run.match_date.isoformat()}")
    print(f"Predicción generada: {run.prediction_as_of_utc.isoformat()}")
    _print_prediction_table(run.predictions)
    _print_summary(run.predictions)
    print("\nFrescura de fuentes")
    for line in freshness_summary_lines(freshness):
        is_warning = any(token in line for token in ("stale", "failed", "fallback"))
        print(f"{'ADVERTENCIA' if is_warning else 'OK'}: {line}")
    if run.output_path is not None:
        print(f"\nCSV publicado: {run.output_path}")
    prediction = operational.prediction_registration
    statistics = operational.statistics_registration
    print(f"Base operativa: {OPERATIONS_DATABASE_PATH}")
    for stored in stored_reconciliations:
        stored_reconciliation = stored.reconciliation
        print(
            f"Reconciliacion almacenada {stored.result_date.isoformat()}: "
            f"{stored_reconciliation.observations_inserted} observaciones nuevas, "
            f"{stored_reconciliation.settlements_inserted} resultados cerrados, "
            f"{stored.snapshot.identity_mapped_count} cruces entre fuentes, "
            f"{stored.snapshot.unmatched_official_count} pendientes."
        )
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
            f"{reconciliation.conflicts_inserted} conflictos, "
            f"{reconciliation.queued_rows} pendientes/revisión"
        )
        if isinstance(
            operational.result_snapshot,
            TennisRatioResultSnapshot,
        ):
            snapshot = operational.result_snapshot
            print(
                "Cobertura TennisRatio: "
                f"{snapshot.matched_count}/{snapshot.official_pending_count} "
                "predicciones oficiales enlazadas; "
                f"{snapshot.unmatched_count} siguen pendientes."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
