"""Publica el estado de frescura de las fuentes de CS2.

El contrato es lateral a la BBDD: nunca modifica ``prediction_ledger``. La
edad se incluye para el resumen de la ejecución, pero el dashboard vuelve a
calcularla desde ``last_success_at_utc`` para seguir detectando obsolescencia
aunque el pipeline no vuelva a ejecutarse.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Final


CONFIG_PATH: Final[Path] = Path(__file__).with_name("freshness.config.json")
SCHEMA_VERSION: Final[str] = "data-freshness-v1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _parse_utc(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z") if value else None


def _status(
    *,
    observed_at: datetime,
    last_success: datetime | None,
    threshold_hours: float,
    attempt_status: str,
    attempted_at: datetime | None,
    fallback_used: bool,
) -> tuple[str, str, float | None]:
    age_hours = max(0.0, (observed_at - last_success).total_seconds() / 3600.0) if last_success is not None else None
    stale = age_hours is None or age_hours > threshold_hours
    failed_after_success = attempt_status == "failed" and (
        last_success is None or attempted_at is None or attempted_at >= last_success
    )
    if failed_after_success:
        cause = "source_update_failed"
    elif fallback_used:
        cause = "fallback_used"
    elif stale:
        cause = "source_not_refreshed"
    else:
        cause = "fresh"
    return ("stale" if stale else "fresh", cause, age_hours)


def build_hltv_report(
    run_dir: Path,
    *,
    observed_at: datetime | None = None,
    attempt_status: str = "not_attempted",
    attempted_at: datetime | None = None,
    fallback_used: bool = False,
    attempt_error: str | None = None,
) -> dict[str, Any]:
    """Construye el contrato de HLTV desde el último run publicado."""

    now = (observed_at or datetime.now(UTC)).astimezone(UTC)
    manifest = _read_json(Path(run_dir) / "manifest.json")
    config = _read_json(CONFIG_PATH).get("sources", {}).get("hltv", {})
    threshold_hours = float(config.get("threshold_hours", 24.0))
    last_success = _parse_utc(manifest.get("finished_at") or manifest.get("started_at"))
    normalized_attempt = attempt_status if attempt_status in {"success", "failed", "not_attempted"} else "not_attempted"
    state, cause, age_hours = _status(
        observed_at=now,
        last_success=last_success,
        threshold_hours=threshold_hours,
        attempt_status=normalized_attempt,
        attempted_at=attempted_at,
        fallback_used=fallback_used,
    )
    source = {
        "source_id": "hltv",
        "label": str(config.get("label") or "HLTV"),
        "cadence": str(config.get("cadence") or "daily"),
        "threshold_hours": threshold_hours,
        "last_success_at_utc": _iso(last_success),
        "last_attempt_at_utc": _iso(attempted_at),
        "last_attempt_status": normalized_attempt,
        "attempt_error": attempt_error,
        "fallback_used": bool(fallback_used),
        "age_hours": round(age_hours, 3) if age_hours is not None else None,
        "status": state,
        "cause": cause,
        "run_id": manifest.get("run_id") or Path(run_dir).name,
        "timestamp_kind": "successful_pipeline_batch",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "component": "CS2",
        "generated_at_utc": _iso(now),
        "pipeline_last_run_at_utc": _iso(now),
        "sources": [source],
        "has_warning": state != "fresh" or fallback_used or normalized_attempt == "failed",
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Publica el contrato mediante reemplazo atómico."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publica la frescura de HLTV para CS2.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--attempt-status",
        choices=("success", "failed", "not_attempted"),
        default="not_attempted",
    )
    parser.add_argument("--attempted-at-utc", default="")
    parser.add_argument("--attempt-error", default="")
    parser.add_argument("--fallback-used", action="store_true")
    args = parser.parse_args()
    attempted_at = _parse_utc(args.attempted_at_utc)
    report = build_hltv_report(
        args.run_dir,
        attempt_status=args.attempt_status,
        attempted_at=attempted_at,
        fallback_used=args.fallback_used,
        attempt_error=args.attempt_error or None,
    )
    output = args.run_dir / "freshness.json"
    write_report(report, output)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
