"""Download Tennis Abstract histories with Scrapling and scoped browser recovery.

Usage: python scripts/update_tennis_abstract.py [--date YYYY-MM-DD | --full-inventory | --status]
By default selects today's agenda first plus up to fifty pending inspected profiles.
The inspected Elo-report links resolve profiles; they are not all downloaded.
Persists source evidence in a dedicated SQLite, never model/operational data.
--full-inventory is an explicit, resumable full-catalogue acquisition.
--status reads committed progress without network access or database writes.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
import json
import logging
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.responsible_http import ResponsibleHttpError  # noqa: E402
from src.tennis_abstract_daily import acquisition_status, update_daily  # noqa: E402
from src.tennis_abstract_elo import TennisAbstractEloError  # noqa: E402
from src.tennis_abstract_selection import load_daily_selection  # noqa: E402
from src.tennisratio import TennisRatioError  # noqa: E402


def progress(message: str) -> None:
    """Expose progress during a first inventory-wide refresh, not only at the end."""

    print(f"[Tennis Abstract / Scrapling] {message}", file=sys.stderr, flush=True)


def main() -> int:
    """Return success only for a complete daily acquisition, without hiding gaps."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Estado de solo lectura; sin red.")
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument(
        "--extra-profiles",
        type=int,
        default=None,
        help="Cupo diario de pendientes adicionales tras la cartelera: 50 por defecto; 0 lo desactiva.",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--date", type=date.fromisoformat, help="Fecha de cartelera; defecto: hoy UTC."
    )
    scope.add_argument("--full-inventory", action="store_true", help="Barrido completo explícito.")
    parser.add_argument(
        "--browser-recovery",
        action="store_true",
        help="Reintento interactivo de fallo LOCAL; nunca ignora Retry-After del servidor.",
    )
    args = parser.parse_args()
    if args.extra_profiles is not None and args.extra_profiles < 0:
        parser.error("--extra-profiles no puede ser negativo.")
    if args.full_inventory and args.extra_profiles is not None:
        parser.error("--extra-profiles no se combina con --full-inventory.")
    if args.status and (
        args.browser_recovery
        or args.max_profiles is not None
        or args.date
        or args.extra_profiles is not None
    ):
        parser.error("--status no se combina con opciones de descarga.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.status:
            report = acquisition_status()
        else:
            selection = (
                None
                if args.full_inventory
                else load_daily_selection(args.date or datetime.now(UTC).date())
            )
            if selection is not None and not selection.players:
                report = {
                    "status": "agenda_unavailable"
                    if not selection.report["agenda_rows"]
                    else "selection_unresolved",
                    "agenda_selection": selection.report,
                    "production_changed": False,
                    "message": "Sin perfiles seleccionables. Actualice la cartelera o revise identidades; no se ejecuta el catálogo completo.",
                }
            else:
                progress(
                    "Adquisicion HTTP con Scrapling; Chrome visible solo si un challenge "
                    "requiere recuperacion (respetando Retry-After)."
                )
                report = update_daily(
                    players=None if selection is None else list(selection.players),
                    selection_context=None if selection is None else selection.report,
                    extra_profiles=0
                    if args.full_inventory
                    else (50 if args.extra_profiles is None else args.extra_profiles),
                    max_profiles=args.max_profiles,
                    browser_recovery=args.browser_recovery,
                    progress=progress,
                )
                progress(
                    f"RESUMEN: nuevos cartelera={report.get('refreshed_agenda', 0)}, "
                    f"nuevos adicionales={report.get('refreshed_extra', 0)}; "
                    f"pendientes cartelera={report.get('pending_agenda', 0)}, "
                    f"adicionales={report.get('pending_extra', 0)}."
                )
    except (
        ResponsibleHttpError,
        TennisAbstractEloError,
        TennisRatioError,
        ValueError,
        RuntimeError,
        OSError,
        sqlite3.Error,
    ) as exc:
        report = {"status": "failed", "failure": str(exc), "production_changed": False}
    print(json.dumps(report, sort_keys=True, ensure_ascii=False), flush=True)
    return (
        0
        if (args.status and report["status"] != "failed")
        or report["status"] in {"completed", "already_refreshed"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
