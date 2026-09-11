"""Contrato observable de frescura para las fuentes de TENNIS.

La publicación es lateral y no escribe en ``predictions``, ``observations`` ni
``settlements``. Para Sackmann se mide la fecha máxima del dato fuente, no la
hora de una descarga que puede no contener información nueva.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
import json
from pathlib import Path
from typing import Any, Final, Mapping

from .config import (
    FEATURE_DATASET_MANIFEST_PATH,
    PROJECT_ROOT,
    SACKMANN_ACTIVE_MANIFEST_PATH,
    TENNIS_EXPLORER_DAILY_RAW_DIR,
)


CONFIG_PATH: Final[Path] = PROJECT_ROOT / "freshness.config.json"
OUTPUT_PATH: Final[Path] = PROJECT_ROOT / "freshness.json"
TENNISRATIO_LAST_GOOD_PATH: Final[Path] = (
    PROJECT_ROOT / "data" / "raw" / "tennisratio" / "last_good.json"
)
SCHEMA_VERSION: Final[str] = "data-freshness-v1"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
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


def _active_feature_manifest() -> dict[str, Any]:
    pointer = _read_json(FEATURE_DATASET_MANIFEST_PATH)
    active_run = pointer.get("active_run")
    if isinstance(active_run, str) and active_run.strip():
        target = FEATURE_DATASET_MANIFEST_PATH.parent / active_run / "manifest.json"
        active = _read_json(target)
        if active:
            return active
    return pointer


def _latest_sackmann_data() -> datetime | None:
    manifest = _active_feature_manifest()
    values: list[date] = []
    for group in (manifest.get("datasets"), manifest.get("rankings")):
        if not isinstance(group, list):
            continue
        for row in group:
            if not isinstance(row, dict):
                continue
            try:
                values.append(date.fromisoformat(str(row.get("max_date") or "")))
            except ValueError:
                continue
    if not values:
        return None
    return datetime.combine(max(values), datetime.min.time(), tzinfo=UTC)


def _latest_explorer_success() -> datetime | None:
    latest: datetime | None = None
    if not TENNIS_EXPLORER_DAILY_RAW_DIR.is_dir():
        return None
    for path in TENNIS_EXPLORER_DAILY_RAW_DIR.rglob("*.metadata.json"):
        metadata = _read_json(path)
        status_code = metadata.get("status_code")
        if status_code is not None:
            try:
                if int(status_code) >= 400:
                    continue
            except (TypeError, ValueError):
                continue
        retrieved = _parse_utc(metadata.get("retrieved_at_utc"))
        if retrieved is not None and (latest is None or retrieved > latest):
            latest = retrieved
    return latest


def source_threshold_hours(source_id: str) -> float:
    """Devuelve el umbral configurado, con defaults seguros por fuente."""

    defaults = {"tennisratio": 24.0, "tennis_explorer": 24.0, "sackmann": 1080.0}
    configured = _read_json(CONFIG_PATH).get("sources", {}).get(source_id, {})
    try:
        value = float(configured.get("threshold_hours", defaults[source_id]))
    except (KeyError, TypeError, ValueError):
        return defaults.get(source_id, 24.0)
    return value if value > 0.0 else defaults.get(source_id, 24.0)


def _source_row(
    source_id: str,
    *,
    last_success: datetime | None,
    observed_at: datetime,
    pipeline_last_run_at: datetime,
    attempt: Mapping[str, object] | None = None,
    fallback_used: bool = False,
    last_batch_at: datetime | None = None,
    timestamp_kind: str = "successful_pipeline_batch",
) -> dict[str, Any]:
    config = _read_json(CONFIG_PATH).get("sources", {}).get(source_id, {})
    threshold = source_threshold_hours(source_id)
    age = (
        max(0.0, (observed_at - last_success).total_seconds() / 3600.0)
        if last_success is not None
        else None
    )
    stale = age is None or age > threshold
    attempt = attempt or {}
    attempt_status = str(attempt.get("status") or "not_attempted")
    attempted_at = _parse_utc(attempt.get("attempted_at_utc"))
    failed_after_success = attempt_status == "failed" and (
        last_success is None or attempted_at is None or attempted_at >= last_success
    )
    pipeline_age = max(0.0, (observed_at - pipeline_last_run_at).total_seconds() / 3600.0)
    if failed_after_success:
        cause = "source_update_failed"
    elif fallback_used:
        cause = "fallback_used"
    elif stale and pipeline_age > threshold:
        cause = "pipeline_not_run"
    elif stale:
        cause = "source_not_refreshed"
    else:
        cause = "fresh"
    return {
        "source_id": source_id,
        "label": str(config.get("label") or source_id),
        "cadence": str(config.get("cadence") or "daily"),
        "threshold_hours": threshold,
        "last_success_at_utc": _iso(last_success),
        "last_batch_at_utc": _iso(last_batch_at),
        "last_attempt_at_utc": _iso(attempted_at),
        "last_attempt_status": attempt_status,
        "attempt_error": attempt.get("error"),
        "fallback_used": bool(fallback_used),
        "age_hours": round(age, 3) if age is not None else None,
        "status": "stale" if stale else "fresh",
        "cause": cause,
        "timestamp_kind": timestamp_kind,
    }


def build_freshness_report(
    *,
    observed_at: datetime | None = None,
    pipeline_last_run_at: datetime | None = None,
    attempts: Mapping[str, Mapping[str, object]] | None = None,
    fallback_sources: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any]:
    """Construye el estado de TennisRatio, Tennis Explorer y Sackmann."""

    now = (observed_at or datetime.now(UTC)).astimezone(UTC)
    pipeline_at = (pipeline_last_run_at or now).astimezone(UTC)
    attempt_map = attempts or {}
    ratio_manifest = _read_json(TENNISRATIO_LAST_GOOD_PATH)
    sackmann_manifest = _read_json(SACKMANN_ACTIVE_MANIFEST_PATH)
    sources = [
        _source_row(
            "tennisratio",
            last_success=_parse_utc(ratio_manifest.get("retrieved_at_utc")),
            observed_at=now,
            pipeline_last_run_at=pipeline_at,
            attempt=attempt_map.get("tennisratio"),
            fallback_used="tennisratio" in fallback_sources,
        ),
        _source_row(
            "tennis_explorer",
            last_success=_latest_explorer_success(),
            observed_at=now,
            pipeline_last_run_at=pipeline_at,
            attempt=attempt_map.get("tennis_explorer"),
            fallback_used="tennis_explorer" in fallback_sources,
        ),
        _source_row(
            "sackmann",
            last_success=_latest_sackmann_data(),
            observed_at=now,
            pipeline_last_run_at=pipeline_at,
            attempt=attempt_map.get("sackmann"),
            last_batch_at=_parse_utc(sackmann_manifest.get("generated_at_utc")),
            timestamp_kind="latest_source_data_date",
        ),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "component": "TENNIS",
        "generated_at_utc": _iso(now),
        "pipeline_last_run_at_utc": _iso(pipeline_at),
        "sources": sources,
        "has_warning": any(
            source["status"] != "fresh"
            or source["fallback_used"]
            or source["last_attempt_status"] == "failed"
            for source in sources
        ),
    }


def write_freshness_report(report: Mapping[str, Any], path: Path = OUTPUT_PATH) -> None:
    """Publica el sidecar mediante reemplazo atómico."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def freshness_summary_lines(report: Mapping[str, Any]) -> tuple[str, ...]:
    """Formatea el mismo contrato para el resumen de consola."""

    lines: list[str] = []
    for source in report.get("sources", []):
        if not isinstance(source, Mapping):
            continue
        age = source.get("age_hours")
        age_text = "desconocida" if age is None else f"{float(age):.1f} h"
        lines.append(
            "{label}: {status}, antigüedad {age}, causa={cause}, umbral={threshold:g} h".format(
                label=source.get("label") or source.get("source_id"),
                status=source.get("status") or "unknown",
                age=age_text,
                cause=source.get("cause") or "unknown",
                threshold=float(source.get("threshold_hours") or 0.0),
            )
        )
    return tuple(lines)
