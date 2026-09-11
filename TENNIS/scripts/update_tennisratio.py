"""Run the public TennisRatio UTC-daily sidecar refresh.

The command accepts optional raw/SQLite/Sackmann paths plus bounded acquisition
flags and prints one machine-readable JSON report.  Run it from ``TENNIS`` as
``python scripts/update_tennisratio.py``; it never accepts a private/API URL.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennisratio import (  # noqa: E402
    TennisRatioError,
    remap_tennisratio_identities,
    refresh_tennisratio,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the non-interactive updater command-line contract."""

    parser = argparse.ArgumentParser(
        description=(
            "Actualiza agendas y perfiles públicos de TennisRatio una vez por día UTC, "
            "sin usar /api/."
        )
    )
    parser.add_argument("--force", action="store_true", help="Publica otro batch validado hoy.")
    parser.add_argument(
        "--remap-only",
        action="store_true",
        help="ReevalÃºa identidades tras actualizar Sackmann, sin ninguna red.",
    )
    parser.add_argument(
        "--full-inventory",
        action="store_true",
        default=None,
        help="Procesa todo sitemap; por defecto solo ocurre en la primera sincronización.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=1.0,
        help="Pausa adicional entre perfiles (default: 1.0; nunca reduce el límite común).",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        help="Límite diagnóstico para perfiles de inventario; no limita los visibles.",
    )
    parser.add_argument("--database-path", type=Path, help="SQLite lateral alternativo.")
    parser.add_argument("--raw-dir", type=Path, help="Raíz alternativa de snapshots.")
    parser.add_argument(
        "--sackmann-raw-dir",
        type=Path,
        help="Raíz que contiene atp/atp_players.csv y wta/wta_players.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the updater and emit one JSON object suitable for orchestration."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.remap_only and (
        arguments.force or arguments.full_inventory or arguments.max_profiles is not None
    ):
        parser.error("--remap-only no admite --force, --full-inventory ni --max-profiles.")
    try:
        if arguments.remap_only:
            report = remap_tennisratio_identities(
                raw_dir=arguments.raw_dir,
                database_path=arguments.database_path,
                sackmann_raw_dir=arguments.sackmann_raw_dir,
            )
        else:
            report = refresh_tennisratio(
                raw_dir=arguments.raw_dir,
                database_path=arguments.database_path,
                force=arguments.force,
                full_inventory=arguments.full_inventory,
                request_delay_seconds=arguments.request_delay_seconds,
                max_profiles=arguments.max_profiles,
                sackmann_raw_dir=arguments.sackmann_raw_dir,
            )
    except (TennisRatioError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    payload = asdict(report)
    for key in ("utc_date", "retrieved_at_utc", "observed_at_utc", "published_at_utc"):
        value = payload.get(key)
        if hasattr(value, "isoformat"):
            payload[key] = value.isoformat().replace("+00:00", "Z")
    for key in ("manifest_path", "database_path"):
        if key in payload:
            payload[key] = str(payload[key])
    payload["changed"] = report.changed
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
