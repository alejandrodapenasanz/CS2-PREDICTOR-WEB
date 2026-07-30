from __future__ import annotations

import argparse
import copy
import os
import gzip
import hashlib
import json
import math
import random
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import requests
from parsel import Selector

try:
    import cloudscraper
except ModuleNotFoundError:  # tests de parsers offline pueden correr sin venv del scraper
    cloudscraper = None


ROOT = Path(__file__).resolve().parents[1]
SCRAPER_PROJECT = ROOT / "SCRAPER" / "hltv-scraper-api"
SCRAPY_ROOT = SCRAPER_PROJECT / "hltv_scraper"
SCRAPY_EXE = SCRAPER_PROJECT / ".venv" / "Scripts" / "scrapy.exe"
PYTHON_EXE = SCRAPER_PROJECT / ".venv" / "Scripts" / "python.exe"
COMPARE_SCRIPT = SCRAPER_PROJECT / "scripts" / "collect_player_compare_stats.py"
CF_SESSION = SCRAPY_ROOT / "cf_session.json"

if str(SCRAPY_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRAPY_ROOT))
try:
    from hltv_scraper.spiders.parsers import ParsersFactory as PF
except Exception:  # pragma: no cover - fallback to Scrapy paths if parser import breaks
    PF = None

try:
    from PIPELINE.match_context import parse_match_context_meta
except Exception:  # pragma: no cover - direct script execution
    from match_context import parse_match_context_meta

DATA_ROOT = ROOT / "PIPELINE"
RUNS_DIR = DATA_ROOT / "runs"
MASTER_DIR = DATA_ROOT / "master"
MASTER_MATCHES = MASTER_DIR / "matches.json"
MASTER_MANIFEST = MASTER_DIR / "manifest.json"
MASTER_ROSTERS = MASTER_DIR / "roster_history.json"
BBDD_DB = ROOT / "BBDD" / "cs2.db"

BLOCK_HTTP_CODES = {403, 429, 500, 502, 503, 504, 522, 524}
CHALLENGE_MARKERS = (
    "Just a moment",
    "cf-chl",
    "cf_chl_",
    "__cf_chl",
    "challenge-platform",
    "Checking your browser",
    "Enable JavaScript and cookies",
    "Attention Required",
    "turnstile",
)
FETCH_MAX_ATTEMPTS = int(os.environ.get("HLTV_FETCH_MAX_ATTEMPTS", "8"))
FETCH_BASE_DELAY = float(os.environ.get("HLTV_FETCH_BASE_DELAY", "3.0"))
FETCH_MAX_DELAY = float(os.environ.get("HLTV_FETCH_MAX_DELAY", "240.0"))
FETCH_MIN_INTERVAL = float(os.environ.get("HLTV_FETCH_MIN_INTERVAL", "1.25"))
FETCH_CACHE_TTL = float(os.environ.get("HLTV_FETCH_CACHE_TTL", "180.0"))
FETCH_BLOCK_COOLDOWN_BASE = float(os.environ.get("HLTV_BLOCK_COOLDOWN_BASE", "60.0"))
FETCH_BLOCK_COOLDOWN_MAX = float(os.environ.get("HLTV_BLOCK_COOLDOWN_MAX", "900.0"))
FETCH_BLOCK_STREAK_THRESHOLD = int(os.environ.get("HLTV_BLOCK_STREAK_THRESHOLD", "3"))
FETCH_CIRCUIT_BREAKER_SLEEP = float(os.environ.get("HLTV_CIRCUIT_BREAKER_SLEEP", "300.0"))
FETCH_WARMUP_ENABLED = os.environ.get("HLTV_FETCH_WARMUP", "1").strip().lower() not in {"0", "false", "no"}
FETCH_URL_QUARANTINE_SECONDS = float(os.environ.get("HLTV_URL_QUARANTINE_SECONDS", "900.0"))
FETCH_MAX_HTTP_REQUESTS_PER_RUN = int(os.environ.get("HLTV_MAX_HTTP_REQUESTS_PER_RUN", "2500"))
PLAYER_CORE_STAT_LABELS = ("Rating 3.0", "KPR", "DPR", "APR", "KAST", "Impact", "ADR")

# Un partido ya fotografiado conserva su primera foto pre-match, pero sus datos
# dinámicos (cuotas, lineup anunciada y Analytics) se vuelven a consultar con
# cadencia limitada. Así se guarda el movimiento sin convertir cada arranque en
# un re-scrape completo.
PREMATCH_REFRESH_HOURS = float(os.environ.get("HLTV_PREMATCH_REFRESH_HOURS", "6"))
PREMATCH_NEAR_START_REFRESH_HOURS = float(os.environ.get("HLTV_PREMATCH_NEAR_START_REFRESH_HOURS", "2"))
PREMATCH_NEAR_START_WINDOW_HOURS = float(os.environ.get("HLTV_PREMATCH_NEAR_START_WINDOW_HOURS", "24"))
CLOSING_ODDS_MAX_AGE_HOURS = float(os.environ.get("HLTV_CLOSING_ODDS_MAX_AGE_HOURS", "6"))

# --- Scrapling (curl_cffi TLS impersonation + stealth browser) -------------
# Tier 1 = HTTP con fingerprint TLS/JA3 real (impersonate); Tier 2 = navegador
# stealth que resuelve el challenge de Cloudflare y acuña cf_clearance. Ambos
# son opcionales: si scrapling no esta instalado, el fetch cae a requests/
# cloudscraper igual que antes. Ver LAST change.md.
def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", ""}


SCRAPLING_ENABLED = _env_bool("HLTV_USE_SCRAPLING", "1")
SCRAPLING_STEALTH_ENABLED = _env_bool("HLTV_SOLVE_CLOUDFLARE", "1")
SCRAPLING_IMPERSONATE = os.environ.get("HLTV_IMPERSONATE", "chrome").strip() or "chrome"
SCRAPLING_PROXY = os.environ.get("HLTV_PROXY", "").strip() or None
SCRAPLING_STEALTH_TIMEOUT_MS = int(os.environ.get("HLTV_STEALTH_TIMEOUT_MS", "90000"))
SCRAPLING_STEALTH_HEADLESS = _env_bool("HLTV_STEALTH_HEADLESS", "1")
SCRAPLING_STEALTH_MAX_SOLVES = int(os.environ.get("HLTV_STEALTH_MAX_SOLVES_PER_RUN", "6"))
SCRAPLING_TIER1_ATTEMPTS = int(os.environ.get("HLTV_SCRAPLING_TIER1_ATTEMPTS", "3"))
CF_REFRESH_ENABLED = _env_bool("HLTV_AUTO_REFRESH_CF_ON_BLOCK", "1")
CF_REFRESH_TIMEOUT_SECONDS = int(os.environ.get("HLTV_CF_REFRESH_TIMEOUT_SECONDS", "240"))

# CA bundle para redes con inspeccion TLS (proxy corporativo con CA propia).
# En un PC sin restricciones no hace falta: certifi funciona por defecto.
_DEFAULT_CORP_BUNDLE = SCRAPER_PROJECT / "corp_ca_bundle.pem"
HLTV_CA_BUNDLE = (
    os.environ.get("HLTV_CA_BUNDLE")
    or os.environ.get("CURL_CA_BUNDLE")
    or os.environ.get("SSL_CERT_FILE")
    or os.environ.get("REQUESTS_CA_BUNDLE")
    or (str(_DEFAULT_CORP_BUNDLE) if _DEFAULT_CORP_BUNDLE.exists() else None)
)

try:  # scrapling es opcional; el scraper funciona sin el (requests/cloudscraper)
    from scrapling.fetchers import Fetcher as _ScraplingFetcher, StealthyFetcher as _ScraplingStealthy
except Exception:  # pragma: no cover - scrapling no instalado / Python no soportado
    _ScraplingFetcher = None
    _ScraplingStealthy = None

_CA_ENV_APPLIED = False
_STEALTH_SOLVES = 0

_HTTP_SESSION: requests.Session | None = None
_LAST_FETCH_AT = 0.0
_HTML_CACHE: dict[str, tuple[float, str]] = {}
_URL_QUARANTINE_UNTIL: dict[str, tuple[float, str]] = {}
_URL_FAILURES: dict[str, dict[str, Any]] = {}
_FETCH_DOMAIN_COOLDOWN_UNTIL = 0.0
_FETCH_BLOCK_STREAK = 0
_CF_SESSION_REFRESHED_THIS_RUN = False
_REQUESTS_PREFERRED_UNTIL = 0.0
_FETCH_STATS: dict[str, Any] = {
    "http_attempts": 0,
    "successes": 0,
    "cache_hits": 0,
    "blocks": 0,
    "errors": 0,
    "url_quarantined": 0,
    "url_quarantine_hits": 0,
    "cf_session_refresh_attempts": 0,
    "cf_session_refresh_successes": 0,
    "budget_exhausted": False,
    "cooldown_sleeps": 0,
    "cooldown_seconds": 0.0,
    "requests_successes": 0,
    "cloudscraper_successes": 0,
    "scrapling_successes": 0,
    "scrapling_stealth_successes": 0,
    "stealth_solves": 0,
    "ca_bundle": None,
    "freshness_skipped": 0,
}
VERBOSE = False


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%SZ")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def safe_console(message: str, *, file=None) -> None:
    file = file or sys.stdout
    encoding = getattr(file, "encoding", None) or "utf-8"
    print(message.encode(encoding, errors="replace").decode(encoding), file=file, flush=True)


def log(message: str, *, force: bool = False) -> None:
    if not force and not VERBOSE:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_console(f"[{ts}] [daily] {message}")


def fetch_diagnostics() -> dict[str, Any]:
    stats = dict(_FETCH_STATS)
    stats["block_streak"] = _FETCH_BLOCK_STREAK
    remaining = max(0.0, _FETCH_DOMAIN_COOLDOWN_UNTIL - time.monotonic())
    stats["domain_cooldown_remaining_seconds"] = round(remaining, 1)
    stats["cache_entries"] = len(_HTML_CACHE)
    stats["max_http_requests_per_run"] = FETCH_MAX_HTTP_REQUESTS_PER_RUN
    stats["url_quarantine_entries"] = len(_URL_QUARANTINE_UNTIL)
    stats["top_failed_urls"] = sorted(
        (
            {"url": url, **info}
            for url, info in _URL_FAILURES.items()
        ),
        key=lambda item: (int(item.get("count") or 0), str(item.get("last_at") or "")),
        reverse=True,
    )[:20]
    return stats


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(BBDD_DB)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


def db_entity_is_fresh(entity_type: str, entity_key: str, now: str | None = None) -> bool:
    """Consulta `fetch_state`; si la BBDD no existe, no bloquea el scrape."""
    if not BBDD_DB.exists() or not entity_key:
        return False
    now = now or now_utc()
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT next_eligible_at_utc, last_status FROM fetch_state WHERE entity_type=? AND entity_key=?",
            (entity_type, str(entity_key)),
        ).fetchone()
        conn.close()
    except sqlite3.DatabaseError:
        return False
    if not row or not row[0]:
        return False
    return str(row[1]) in {"ok", "partial", "blocked", "not_found", "error"} and str(row[0]) > now


def db_fresh_entity_keys(entity_type: str, entity_keys: set[str], now: str | None = None) -> set[str]:
    if not BBDD_DB.exists() or not entity_keys:
        return set()
    now = now or now_utc()
    placeholders = ",".join("?" for _ in entity_keys)
    try:
        conn = db_connect()
        rows = conn.execute(
            f"""
            SELECT entity_key
            FROM fetch_state
            WHERE entity_type=?
              AND entity_key IN ({placeholders})
              AND last_status IN ('ok','partial','blocked','not_found','error')
              AND next_eligible_at_utc IS NOT NULL
              AND next_eligible_at_utc > ?
            """,
            (entity_type, *sorted(entity_keys), now),
        ).fetchall()
        conn.close()
    except sqlite3.DatabaseError:
        return set()
    return {str(row[0]) for row in rows}


def db_latest_team_profile(team_id: str) -> dict[str, Any] | None:
    if not BBDD_DB.exists() or not team_id:
        return None
    try:
        conn = db_connect()
        row = conn.execute(
            """
            SELECT payload_json
            FROM raw_snapshots
            WHERE kind='team_profile' AND hltv_team_id=?
            ORDER BY captured_at_utc DESC, raw_snapshot_id DESC
            LIMIT 1
            """,
            (str(team_id),),
        ).fetchone()
        conn.close()
    except sqlite3.DatabaseError:
        return None
    if not row:
        return None
    try:
        payload = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("profile") else None


def db_latest_player_snapshots(player_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    if not BBDD_DB.exists() or not player_ids:
        return {}
    placeholders = ",".join("?" for _ in player_ids)
    try:
        conn = db_connect()
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            WITH latest_capture AS (
                SELECT hltv_player_id, MAX(captured_at_utc) AS captured_at_utc
                FROM player_stat_snapshots
                WHERE hltv_player_id IN ({placeholders})
                GROUP BY hltv_player_id
            )
            SELECT p.*
            FROM player_stat_snapshots p
            JOIN latest_capture l
              ON l.hltv_player_id=p.hltv_player_id
             AND l.captured_at_utc=p.captured_at_utc
            ORDER BY p.hltv_player_id, p.player_stat_snapshot_id DESC
            """,
            tuple(sorted(player_ids)),
        ).fetchall()
        conn.close()
    except sqlite3.DatabaseError:
        return {}

    stat_columns = {
        "rating": "Rating 3.0",
        "kpr": "KPR",
        "dpr": "DPR",
        "apr": "APR",
        "kast": "KAST",
        "impact": "Impact",
        "adr": "ADR",
        "round_swing": "Round Swing",
        "multi_kill_rating": "Multi-kill rating",
        "awp_kpr": "AWP KPR",
        "hs_pct": "HS %",
        "opening_kpr": "Opening KPR",
        "opening_dpr": "Opening DPR",
        "flash_assists": "Flash assists",
    }
    seen: set[tuple[str, str]] = set()
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        player_id = str(row["hltv_player_id"])
        time_filter = str(row["time_filter"] or row["season_year"] or "unknown")
        key = (player_id, time_filter)
        if key in seen:
            continue
        seen.add(key)
        stats = {
            label: row[column]
            for column, label in stat_columns.items()
            if row[column] is not None
        }
        out.setdefault(player_id, []).append(
            {
                "id": player_id,
                "name": row["player_name"],
                "slug": row["player_slug"],
                "link": row["player_link"],
                "maps": row["maps"],
                "time_filter": time_filter,
                "selected_from_time_filter": time_filter,
                "stats": stats,
                "source": "BBDD.player_stat_snapshots",
                "captured_at": row["captured_at_utc"],
            }
        )
    return out


def db_match_has_coverage(hltv_match_id: str, flag_column: str) -> bool:
    if not BBDD_DB.exists() or not hltv_match_id:
        return False
    allowed = {"has_box_score", "has_veto", "has_analytics", "has_context", "has_prematch_odds"}
    if flag_column not in allowed:
        return False
    try:
        conn = db_connect()
        row = conn.execute(
            f"SELECT {flag_column} FROM matches WHERE hltv_match_id=?",
            (str(hltv_match_id),),
        ).fetchone()
        conn.close()
    except sqlite3.DatabaseError:
        return False
    return bool(row and row[0])


def db_pending_match_ids(now: str | None = None) -> set[str] | None:
    """Lista de trabajo de resultados desde SQLite; None = fallback legacy."""
    if not BBDD_DB.exists():
        return None
    now = now or now_utc()
    try:
        conn = db_connect()
        rows = conn.execute(
            """
            SELECT hltv_match_id
            FROM matches
            WHERE hltv_match_id IS NOT NULL
              AND (
                    status = 'pending_result'
                 OR (status = 'scheduled' AND datetime_utc < ?)
              )
            """,
            (now,),
        ).fetchall()
        conn.close()
    except sqlite3.DatabaseError:
        return None
    return {str(row[0]) for row in rows if row[0]}


def db_pending_match_info(now: str | None = None) -> dict[str, dict[str, Any]] | None:
    if not BBDD_DB.exists():
        return None
    now = now or now_utc()
    try:
        conn = db_connect()
        rows = conn.execute(
            """
            SELECT m.hltv_match_id, m.datetime_utc, m.status,
                   t1.name AS team1_name, t2.name AS team2_name, e.name AS event_name
            FROM matches m
            LEFT JOIN teams t1 ON t1.team_id = m.team1_id
            LEFT JOIN teams t2 ON t2.team_id = m.team2_id
            LEFT JOIN events e ON e.event_id = m.event_id
            WHERE m.hltv_match_id IS NOT NULL
              AND (
                    m.status = 'pending_result'
                 OR (m.status = 'scheduled' AND m.datetime_utc < ?)
              )
            """,
            (now,),
        ).fetchall()
        conn.close()
    except sqlite3.DatabaseError:
        return None
    return {
        str(row[0]): {
            "datetime_utc": row[1],
            "status": row[2],
            "team1": row[3],
            "team2": row[4],
            "event": row[5],
        }
        for row in rows if row[0]
    }


def format_pending_missing(match_id: str, info: dict[str, dict[str, Any]] | None) -> str:
    item = (info or {}).get(str(match_id)) or {}
    label = f"{item.get('team1') or '?'} vs {item.get('team2') or '?'}"
    when = item.get("datetime_utc") or "unknown_time"
    status = item.get("status") or "unknown_status"
    return f"{match_id} {when} {status} {label}"


def db_match_already_photographed(hltv_match_id: str) -> bool:
    if not BBDD_DB.exists() or not hltv_match_id:
        return False
    try:
        conn = db_connect()
        row = conn.execute(
            """
            SELECT data_tier, status
            FROM matches
            WHERE hltv_match_id = ?
            """,
            (str(hltv_match_id),),
        ).fetchone()
        conn.close()
    except sqlite3.DatabaseError:
        return False
    if not row:
        return False
    return str(row[0]) in {"prematch_captured", "completed"} or str(row[1]) == "completed"


def db_fetch_next_eligible(entity_type: str, status: str, fetched_at: str) -> str:
    ttl_days = {
        "team_profile": int(os.environ.get("BBDD_TEAM_PROFILE_TTL_DAYS", "7")),
        "player_stats": int(os.environ.get("BBDD_PLAYER_STATS_TTL_DAYS", "3")),
        "ranking_hltv": int(os.environ.get("BBDD_RANKING_TTL_DAYS", "7")),
        "ranking_valve": int(os.environ.get("BBDD_RANKING_TTL_DAYS", "7")),
    }
    base = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    if status in {"blocked", "error", "partial"}:
        delta = timedelta(hours=1)
    elif entity_type in {"match_assets", "match_analytics", "match_detail"}:
        delta = timedelta(days=3650)
    else:
        delta = timedelta(days=ttl_days.get(entity_type, 1))
    return (base + delta).strftime("%Y-%m-%dT%H:%M:%SZ")


def db_mark_fetch_state(entity_type: str, entity_key: str, status: str, note: str | None = None) -> None:
    """Checkpoint de scraper: persiste OK/error/blocked por entidad."""
    if not BBDD_DB.exists() or not entity_key:
        return
    fetched_at = now_utc()
    try:
        conn = db_connect()
        conn.execute(
            """
            INSERT INTO fetch_state(entity_type, entity_key, last_fetched_at_utc, last_status,
                                    fetch_count, next_eligible_at_utc, note)
            VALUES (?,?,?,?,1,?,?)
            ON CONFLICT(entity_type, entity_key) DO UPDATE SET
                last_fetched_at_utc=excluded.last_fetched_at_utc,
                last_status=excluded.last_status,
                fetch_count=fetch_state.fetch_count + 1,
                next_eligible_at_utc=excluded.next_eligible_at_utc,
                note=excluded.note
            """,
            (
                entity_type,
                str(entity_key),
                fetched_at,
                status,
                db_fetch_next_eligible(entity_type, status, fetched_at),
                note,
            ),
        )
        conn.commit()
        conn.close()
    except sqlite3.DatabaseError:
        return


def note_freshness_skip(count: int = 1) -> None:
    _FETCH_STATS["freshness_skipped"] = int(_FETCH_STATS.get("freshness_skipped") or 0) + count


class FetchBudgetExceeded(RuntimeError):
    pass


class FetchSuppressedError(RuntimeError):
    pass


def ensure_fetch_budget(url: str) -> None:
    if FETCH_MAX_HTTP_REQUESTS_PER_RUN <= 0:
        return
    if int(_FETCH_STATS["http_attempts"]) >= FETCH_MAX_HTTP_REQUESTS_PER_RUN:
        _FETCH_STATS["budget_exhausted"] = True
        raise FetchBudgetExceeded(
            f"HLTV request budget exhausted ({FETCH_MAX_HTTP_REQUESTS_PER_RUN}) before {url}"
        )


def register_url_failure(url: str, reason: str) -> None:
    entry = _URL_FAILURES.setdefault(url, {"count": 0, "last_reason": None, "last_at": None})
    entry["count"] = int(entry.get("count") or 0) + 1
    entry["last_reason"] = reason[:300]
    entry["last_at"] = now_utc()


def quarantine_url(url: str, reason: str) -> None:
    if FETCH_URL_QUARANTINE_SECONDS <= 0:
        return
    register_url_failure(url, reason)
    until = time.monotonic() + FETCH_URL_QUARANTINE_SECONDS
    _URL_QUARANTINE_UNTIL[url] = (until, reason[:300])
    _FETCH_STATS["url_quarantined"] += 1
    log(f"URL quarantine {FETCH_URL_QUARANTINE_SECONDS:.0f}s: {url} ({reason[:120]})", force=True)


def check_url_quarantine(url: str) -> None:
    item = _URL_QUARANTINE_UNTIL.get(url)
    if not item:
        return
    until, reason = item
    remaining = until - time.monotonic()
    if remaining <= 0:
        _URL_QUARANTINE_UNTIL.pop(url, None)
        return
    _FETCH_STATS["url_quarantine_hits"] += 1
    raise FetchSuppressedError(f"URL in quarantine for {remaining:.1f}s after prior failure: {reason}")


def fetch_cache_get(url: str) -> str | None:
    if FETCH_CACHE_TTL <= 0:
        return None
    cached = _HTML_CACHE.get(url)
    if not cached:
        return None
    captured_at, html = cached
    if time.monotonic() - captured_at > FETCH_CACHE_TTL:
        _HTML_CACHE.pop(url, None)
        return None
    _FETCH_STATS["cache_hits"] += 1
    log(f"CACHE hit {len(html)} bytes: {url}")
    return html


def fetch_cache_put(url: str, html: str) -> None:
    if FETCH_CACHE_TTL <= 0:
        return
    _HTML_CACHE[url] = (time.monotonic(), html)


def wait_for_domain_cooldown() -> None:
    remaining = _FETCH_DOMAIN_COOLDOWN_UNTIL - time.monotonic()
    if remaining <= 0:
        return
    _FETCH_STATS["cooldown_sleeps"] += 1
    _FETCH_STATS["cooldown_seconds"] += remaining
    log(f"domain cooldown active; sleeping {remaining:.1f}s")
    time.sleep(remaining)


def register_fetch_success(source: str, url: str) -> None:
    global _FETCH_BLOCK_STREAK
    _FETCH_STATS["successes"] += 1
    key = f"{source}_successes"
    if key in _FETCH_STATS:
        _FETCH_STATS[key] += 1
    elif source == "cloudscraper":
        _FETCH_STATS["cloudscraper_successes"] += 1
    else:
        _FETCH_STATS["requests_successes"] += 1
    if _FETCH_BLOCK_STREAK:
        log(f"block streak reset after success ({source}): {url}")
    _FETCH_BLOCK_STREAK = 0


def register_fetch_block(source: str, url: str, reason: str) -> float:
    global _FETCH_BLOCK_STREAK, _FETCH_DOMAIN_COOLDOWN_UNTIL
    _FETCH_STATS["blocks"] += 1
    register_url_failure(url, f"{source} {reason}")
    _FETCH_BLOCK_STREAK += 1
    adaptive = min(
        FETCH_BLOCK_COOLDOWN_MAX,
        FETCH_BLOCK_COOLDOWN_BASE * (2 ** max(0, _FETCH_BLOCK_STREAK - 1)),
    ) + random.uniform(2.0, 8.0)
    if _FETCH_BLOCK_STREAK >= FETCH_BLOCK_STREAK_THRESHOLD:
        adaptive = max(adaptive, FETCH_CIRCUIT_BREAKER_SLEEP + random.uniform(10.0, 45.0))
    _FETCH_DOMAIN_COOLDOWN_UNTIL = max(_FETCH_DOMAIN_COOLDOWN_UNTIL, time.monotonic() + adaptive)
    log(
        f"BLOCK {source} {reason}; streak={_FETCH_BLOCK_STREAK}; "
        f"domain cooldown {adaptive:.1f}s: {url}",
        force=True,
    )
    return adaptive


def register_fetch_error(source: str, url: str, exc: Exception) -> None:
    _FETCH_STATS["errors"] += 1
    register_url_failure(url, f"{source} {type(exc).__name__}: {exc}")
    log(f"ERROR {source} {type(exc).__name__}: {exc}: {url}")


def register_fetch_soft_block(source: str, url: str, reason: str) -> None:
    _FETCH_STATS["blocks"] += 1
    register_url_failure(url, f"{source} {reason}")
    log(f"BLOCK {source} {reason}; falling back without domain cooldown: {url}", force=True)


def looks_like_cf_or_waf_problem(error: Exception | None) -> bool:
    text = str(error or "").lower()
    return any(marker in text for marker in ("cloudflare", "challenge", "403", "429", "bloqueo", "cf-chl"))


def maybe_refresh_cf_session_once(reason: str, url: str) -> bool:
    global _CF_SESSION_REFRESHED_THIS_RUN, _HTTP_SESSION, _FETCH_DOMAIN_COOLDOWN_UNTIL, _FETCH_BLOCK_STREAK, _REQUESTS_PREFERRED_UNTIL
    if not CF_REFRESH_ENABLED:
        return False
    if _CF_SESSION_REFRESHED_THIS_RUN:
        return False
    helper = SCRAPY_ROOT / "hltv_scraper" / "grab_cf.py"
    if not helper.exists():
        return False
    _CF_SESSION_REFRESHED_THIS_RUN = True
    _FETCH_STATS["cf_session_refresh_attempts"] += 1
    log(f"cf_session refresh triggered by {reason}: {url}", force=True)
    ok, logs = run_cmd([str(PYTHON_EXE), str(helper)], SCRAPER_PROJECT, timeout=CF_REFRESH_TIMEOUT_SECONDS, stream=True)
    if ok and CF_SESSION.exists():
        _FETCH_STATS["cf_session_refresh_successes"] += 1
        _HTTP_SESSION = None
        _FETCH_BLOCK_STREAK = 0
        _FETCH_DOMAIN_COOLDOWN_UNTIL = min(_FETCH_DOMAIN_COOLDOWN_UNTIL, time.monotonic() + 5.0)
        _REQUESTS_PREFERRED_UNTIL = time.monotonic() + float(os.environ.get("HLTV_PREFER_REQUESTS_AFTER_CF_SECONDS", "900"))
        log("cf_session refreshed; retrying blocked request", force=True)
        return True
    log(f"cf_session refresh did not produce a usable session: {logs[-500:]}", force=True)
    return False


def save_raw_html(run_dir: Path, kind: str, identifier: str, link: str, html: str) -> dict[str, Any]:
    digest = hashlib.sha256(html.encode("utf-8", errors="replace")).hexdigest()
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier)[:160] or digest[:12]
    output = run_dir / "raw_html" / kind / f"{safe_id}.html.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as fh:
        fh.write(html)
    meta = {
        "kind": kind,
        "identifier": identifier,
        "url": hltv_url(link),
        "captured_at": now_utc(),
        "sha256": digest,
        "bytes_utf8": len(html.encode("utf-8", errors="replace")),
        "source_file": str(output.relative_to(DATA_ROOT)),
    }
    write_json(output.with_suffix(output.suffix + ".json"), meta)
    return meta


def raw_html_safe_id(identifier: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier)[:160]


def load_persisted_raw_html(kind: str, identifier: str) -> str | None:
    safe_id = raw_html_safe_id(identifier)
    if not safe_id:
        return None
    pattern = f"*/raw_html/{kind}/{safe_id}.html.gz"
    for path in sorted(RUNS_DIR.glob(pattern), reverse=True):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                html = fh.read()
            if html and not is_cloudflare_challenge(html):
                log(f"RAW reuse {kind}/{safe_id}: {path}")
                return html
        except Exception:
            continue
    return None


def run_cmd(cmd: list[str], cwd: Path, timeout: int = 300, stream: bool = False) -> tuple[bool, str]:
    if stream:
        log("RUN " + " ".join(str(part) for part in cmd), force=True)
        try:
            process = subprocess.run(
                [str(part) for part in cmd],
                cwd=cwd,
                timeout=timeout,
            )
            return process.returncode == 0, ""
        except subprocess.TimeoutExpired as exc:
            return False, f"TIMEOUT after {timeout}s: {exc}"
    try:
        process = subprocess.run(
            [str(part) for part in cmd],
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return process.returncode == 0, process.stdout
    except subprocess.TimeoutExpired as exc:
        output = ""
        if exc.stdout:
            output += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", errors="replace")
        if exc.stderr:
            output += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", errors="replace")
        return False, output + f"\nTIMEOUT after {timeout}s: {exc}"


def run_spider(spider: str, output_path: Path, spider_args: list[str] | None = None, timeout: int = 600) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(SCRAPY_EXE), "crawl", spider]
    if spider_args:
        cmd.extend(spider_args)
    cmd.extend(["-L", "ERROR", "-O", str(output_path)])
    return run_cmd(cmd, SCRAPY_ROOT, timeout=timeout)


def match_id_from_link(link: str | None) -> str | None:
    if not link:
        return None
    match = re.search(r"/matches/(\d+)", link)
    return match.group(1) if match else None


def slug_from_link(link: str | None) -> str:
    if not link:
        return ""
    return link.rstrip("/").split("/")[-1]


def hltv_url(link: str) -> str:
    return link if link.startswith("http") else f"https://www.hltv.org{link}"


def match_label(match: dict[str, Any]) -> str:
    match_id = str(match.get("id") or match_id_from_link(match.get("link")) or "?")
    team1 = (match.get("team1") or {}).get("name") or (match.get("teamA") or {}).get("name")
    team2 = (match.get("team2") or {}).get("name") or (match.get("teamB") or {}).get("name")
    event = match.get("event") or match.get("event_name") or "sin evento"
    when = " ".join(str(match.get(key) or "").strip() for key in ("date", "hour")).strip()
    teams = f"{team1 or '?'} vs {team2 or '?'}"
    return f"{match_id} | {teams} | {when or 'sin hora'} | {event}"


def record_label(record: dict[str, Any]) -> str:
    match = (record.get("detail") or {}).get("match") or {}
    row = record.get("upcoming_row") or {}
    team1 = (
        (match.get("team1") or {}).get("name")
        or (row.get("team1") or {}).get("name")
        or record.get("team1_name")
    )
    team2 = (
        (match.get("team2") or {}).get("name")
        or (row.get("team2") or {}).get("name")
        or record.get("team2_name")
    )
    return match_label(
        {
            "id": record.get("id"),
            "link": record.get("link"),
            "team1": {"name": team1},
            "team2": {"name": team2},
            "date": record.get("date"),
            "hour": record.get("hour"),
            "event": record.get("event"),
        }
    )


def warm_up_hltv_session(run_dir: Path) -> dict[str, Any]:
    if not FETCH_WARMUP_ENABLED:
        return {"enabled": False, "reason": "disabled"}
    started = time.monotonic()
    checks: list[dict[str, Any]] = []
    for link in ["/"]:
        try:
            log(f"warmup: fetching {link}", force=VERBOSE)
            html = fetch_html(link, min_interval=max(FETCH_MIN_INTERVAL, 2.5), use_cache=True)
            meta = save_raw_html(run_dir, "warmup", "home", link, html)
            checks.append(
                {
                    "link": link,
                    "ok": True,
                    "bytes": len(html.encode("utf-8", errors="replace")),
                    "source_file": meta.get("source_file"),
                }
            )
        except Exception as exc:
            checks.append({"link": link, "ok": False, "error": str(exc)})
            log(f"warmup: ERROR {exc}", force=True)
    return {
        "enabled": True,
        "ok": all(item.get("ok") for item in checks),
        "checks": checks,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


class HltvFetchError(RuntimeError):
    def __init__(self, message: str, *, url: str, status_code: int | None = None, source: str = "requests"):
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.source = source


def http_session() -> requests.Session:
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        _HTTP_SESSION = requests.Session()
    return _HTTP_SESSION


def is_cloudflare_challenge(html: str) -> bool:
    head = html[:8000]
    return any(marker.lower() in head.lower() for marker in CHALLENGE_MARKERS)


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


def backoff_delay(attempt: int, retry_after: str | None, *, base_delay: float, max_delay: float) -> float:
    hinted = retry_after_seconds(retry_after)
    if hinted is not None:
        return min(max_delay, hinted) + random.uniform(0.25, 1.75)
    return min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0.5, 2.5)


def polite_fetch_wait(min_interval: float) -> None:
    global _LAST_FETCH_AT
    if min_interval <= 0:
        return
    elapsed = time.monotonic() - _LAST_FETCH_AT
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed + random.uniform(0.05, 0.35))
    _LAST_FETCH_AT = time.monotonic()


def default_headers(referer: str = "https://www.hltv.org/") -> dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Referer": referer,
    }


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def parse_hltv_date(value: str | None) -> str:
    text = clean_text(value) or ""
    direct = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if direct:
        return direct.group(1)

    match = re.search(
        r"([A-Za-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\s+(20\d{2})",
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    month = MONTHS.get(match.group(1).lower())
    if not month:
        return ""
    return f"{int(match.group(3)):04d}-{month:02d}-{int(match.group(2)):02d}"


def first_css_text(selector: Selector, selectors: list[str]) -> str | None:
    for query in selectors:
        value = clean_text(selector.css(query).get())
        if value:
            return value
    return None


def team_from_listing(selector: Selector, number: int, include_score: bool = False) -> dict[str, Any]:
    team: dict[str, Any] = {
        "name": first_css_text(
            selector,
            [
                f"div.team{number} .match-teamname::text",
                f"div.team{number} .matchTeamName::text",
                f"div.team{number} .team::text",
                f".team{number} .team::text",
            ],
        ),
        "logo": selector.css(f"div.team{number} img::attr(src), .team{number} img::attr(src)").get(),
    }
    if include_score:
        scores = [
            clean_text(item)
            for item in selector.css(
                "td.result-score span::text, .result-score span::text, "
                ".score-won::text, .score-lost::text"
            ).getall()
        ]
        scores = [item for item in scores if item and item != "-"]
        team["score"] = scores[number - 1] if len(scores) >= number else None
    return team


def parse_upcoming_matches_fallback(html: str) -> list[dict[str, Any]]:
    selector = Selector(text=html)
    parsed: list[dict[str, Any]] = []
    for section in selector.css("div.matches-list-section"):
        date = parse_hltv_date(
            section.css(".matches-list-headline::text, .matchDayHeadline::text").get()
        )
        for match in section.css("div.match-zone-wrapper, a.match"):
            link = match.css("a.match-info::attr(href), a.match::attr(href), a::attr(href)").get()
            team1 = team_from_listing(match, 1)
            team2 = team_from_listing(match, 2)
            if not link or not team1.get("name") or not team2.get("name"):
                continue
            parsed.append(
                {
                    "hour": first_css_text(match, [".match-time::text", ".matchTime::text"]),
                    "date": date,
                    "link": link,
                    "meta": first_css_text(match, [".match-meta::text", ".matchMeta::text"]),
                    "event": first_css_text(
                        match,
                        [
                            ".match-event::attr(data-event-headline)",
                            ".matchEventName::text",
                            ".event-name::text",
                        ],
                    ),
                    "team1": team1,
                    "team2": team2,
                }
            )
    return parsed


def parse_results_fallback(html: str) -> list[dict[str, Any]]:
    selector = Selector(text=html)
    parsed: list[dict[str, Any]] = []
    sublists = selector.css("div.allres .results-sublist")
    if not sublists:
        sublists = selector.css("div.results-sublist")
    if not sublists:
        sublists = [selector]
    for sublist in sublists:
        date = parse_hltv_date(sublist.css(".standard-headline::text").get())
        for result in sublist.css("div.result-con"):
            link = result.css("a.a-reset::attr(href), a::attr(href)").get()
            match_id = match_id_from_link(link)
            team1 = team_from_listing(result, 1, include_score=True)
            team2 = team_from_listing(result, 2, include_score=True)
            if not link or not match_id or not team1.get("name") or not team2.get("name"):
                continue
            parsed.append(
                {
                    "id": match_id,
                    "link": link,
                    "map": first_css_text(result, [".map-text::text"]),
                    "event": first_css_text(result, [".event-name::text"]),
                    "date": date,
                    "team1": team1,
                    "team2": team2,
                }
            )
    return parsed


def _apply_ca_env() -> None:
    """Propaga el CA bundle corporativo a curl_cffi/requests via variables de entorno.

    Solo actua si HLTV_CA_BUNDLE apunta a un fichero existente (redes con
    inspeccion TLS). En un PC sin restricciones no hay bundle y se usa certifi.
    """
    global _CA_ENV_APPLIED
    if _CA_ENV_APPLIED:
        return
    _CA_ENV_APPLIED = True
    if HLTV_CA_BUNDLE and Path(HLTV_CA_BUNDLE).exists():
        for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            os.environ.setdefault(var, HLTV_CA_BUNDLE)
        _FETCH_STATS["ca_bundle"] = HLTV_CA_BUNDLE
        log(f"CA bundle corporativo activo: {HLTV_CA_BUNDLE}", force=True)


def _scrapling_response_parts(resp: Any) -> tuple[int | None, str]:
    """Extrae (status, html) de una Response de scrapling de forma defensiva."""
    status = getattr(resp, "status", None)
    html = getattr(resp, "html_content", None)
    if html is None:
        body = getattr(resp, "body", None)
        if isinstance(body, bytes):
            html = body.decode("utf-8", "replace")
        elif isinstance(body, str):
            html = body
        else:
            html = str(resp) if resp is not None else ""
    return status, (html or "")


def _persist_cf_from_response(resp: Any) -> None:
    """Guarda cf_clearance + user_agent acuñados por el navegador stealth.

    Asi los tiers HTTP posteriores (scrapling impersonate, requests) reusan la
    cookie. El binding cf_clearance <-> IP + UA + JA3 exige reusar la misma UA.
    """
    try:
        cf_value = None
        cookies = getattr(resp, "cookies", None)
        if isinstance(cookies, dict):
            cf_value = cookies.get("cf_clearance")
        elif cookies:
            for c in cookies:
                name = getattr(c, "name", None) or (c.get("name") if isinstance(c, dict) else None)
                if name == "cf_clearance":
                    cf_value = getattr(c, "value", None) or (c.get("value") if isinstance(c, dict) else None)
                    break
        if not cf_value:
            return
        ua = None
        req = getattr(resp, "request", None)
        req_headers = getattr(req, "headers", None) if req is not None else None
        if isinstance(req_headers, dict):
            ua = req_headers.get("User-Agent") or req_headers.get("user-agent")
        payload = read_json(CF_SESSION, {}) if CF_SESSION.exists() else {}
        payload["cf_clearance"] = cf_value
        if ua:
            payload["user_agent"] = ua
        payload["captured_at"] = now_utc()
        payload["source"] = "scrapling_stealth"
        write_json(CF_SESSION, payload)
        log("cf_clearance acuñado por navegador stealth y guardado en cf_session.json", force=True)
    except Exception as exc:  # pragma: no cover - best effort
        log(f"no se pudo persistir cf_clearance del navegador: {exc}")


def _scrapling_impersonate_get(url: str, cookies: dict[str, str] | None, timeout: int) -> tuple[int | None, str]:
    kwargs: dict[str, Any] = {
        "impersonate": SCRAPLING_IMPERSONATE,
        "stealthy_headers": True,
        "timeout": timeout,
    }
    if SCRAPLING_PROXY:
        kwargs["proxy"] = SCRAPLING_PROXY
    if cookies:
        kwargs["cookies"] = cookies
    try:
        resp = _ScraplingFetcher.get(url, **kwargs)
    except TypeError:
        # Firma distinta segun version: reintenta sin cookies/proxy.
        resp = _ScraplingFetcher.get(url, impersonate=SCRAPLING_IMPERSONATE, stealthy_headers=True, timeout=timeout)
    return _scrapling_response_parts(resp)


def _scrapling_stealth_solve(url: str) -> tuple[int | None, str] | None:
    """Tier 2: navegador stealth que resuelve Cloudflare y acuña cf_clearance."""
    global _STEALTH_SOLVES
    if _ScraplingStealthy is None or _STEALTH_SOLVES >= SCRAPLING_STEALTH_MAX_SOLVES:
        return None
    _STEALTH_SOLVES += 1
    _FETCH_STATS["stealth_solves"] += 1
    kwargs: dict[str, Any] = {
        "headless": SCRAPLING_STEALTH_HEADLESS,
        "solve_cloudflare": True,
        "block_webrtc": True,
        "disable_resources": True,
        "network_idle": False,
        "timeout": SCRAPLING_STEALTH_TIMEOUT_MS,
    }
    if SCRAPLING_PROXY:
        kwargs["proxy"] = SCRAPLING_PROXY
        kwargs["geoip"] = True
    log(f"STEALTH solve_cloudflare (#{_STEALTH_SOLVES}): {url}", force=True)
    started = time.monotonic()
    try:
        resp = _ScraplingStealthy.fetch(url, **kwargs)
    except Exception as exc:
        elapsed = time.monotonic() - started
        log(f"STEALTH solve_cloudflare failed after {elapsed:.1f}s: {type(exc).__name__}: {exc}", force=True)
        raise
    elapsed = time.monotonic() - started
    log(f"STEALTH solve_cloudflare finished in {elapsed:.1f}s: {url}", force=True)
    _persist_cf_from_response(resp)
    return _scrapling_response_parts(resp)


def _retry_after_cf_refresh(url: str, timeout: int, interval: float) -> str | None:
    fresh = read_json(CF_SESSION, {}) if CF_SESSION.exists() else {}
    cookies = {"cf_clearance": fresh["cf_clearance"]} if fresh.get("cf_clearance") else None
    if not cookies:
        return None
    try:
        ensure_fetch_budget(url)
        wait_for_domain_cooldown()
        polite_fetch_wait(interval)
        _FETCH_STATS["http_attempts"] += 1
        status, text = _scrapling_impersonate_get(url, cookies, timeout)
        if status == 200 and text and not is_cloudflare_challenge(text):
            register_fetch_success("scrapling", url)
            fetch_cache_put(url, text)
            log(f"OK scrapling after cf_session refresh {len(text)} bytes: {url}")
            return text
    except FetchBudgetExceeded:
        raise
    except Exception as exc:
        register_fetch_error("scrapling_after_cf_refresh", url, exc)
    return None


def _scrapling_fetch(
    url: str,
    cookies: dict[str, str] | None,
    timeout: int,
    base: float,
    max_wait: float,
    interval: float,
) -> str | None:
    """Tiers 1 (impersonate) y 2 (stealth solve). Devuelve HTML o None (fall-through).

    Reutiliza todas las guardas compartidas (presupuesto, cooldown, backoff,
    deteccion de bloqueo, cache). Si no obtiene HTML valido, devuelve None y el
    fetch principal cae a requests/cloudscraper.
    """
    if not SCRAPLING_ENABLED or _ScraplingFetcher is None:
        return None
    _apply_ca_env()

    # Tier 1: HTTP con impersonation TLS (rapido).
    attempts = max(2, SCRAPLING_TIER1_ATTEMPTS)
    for attempt in range(attempts):
        ensure_fetch_budget(url)
        try:
            log(f"GET scrapling {attempt + 1}/{attempts}: {url}")
            started = time.monotonic()
            wait_for_domain_cooldown()
            polite_fetch_wait(interval)
            _FETCH_STATS["http_attempts"] += 1
            status, text = _scrapling_impersonate_get(url, cookies, timeout)
            challenge = is_cloudflare_challenge(text)
            elapsed = time.monotonic() - started
            if status == 200 and text and not challenge:
                register_fetch_success("scrapling", url)
                fetch_cache_put(url, text)
                log(f"OK scrapling HTTP 200 {len(text)} bytes {elapsed:.1f}s: {url}")
                return text
            if (status in BLOCK_HTTP_CODES) or challenge:
                reason = "cloudflare_challenge" if challenge else f"HTTP {status}"
                if cookies and "/stats/" in url:
                    register_fetch_soft_block("scrapling", url, reason)
                    log("scrapling blocked on stats URL with cf_session; trying requests before long backoff", force=True)
                    return None
                delay = backoff_delay(attempt, None, base_delay=base, max_delay=max_wait)
                delay = max(delay, register_fetch_block("scrapling", url, reason))
                log(f"BLOCK scrapling {reason}; wait {delay:.1f}s: {url}")
                time.sleep(delay)
                continue
            if text:
                register_fetch_success("scrapling", url)
                fetch_cache_put(url, text)
                return text
        except FetchBudgetExceeded:
            raise
        except Exception as exc:
            register_fetch_error("scrapling", url, exc)
            if attempt < attempts - 1:
                time.sleep(backoff_delay(attempt, None, base_delay=base, max_delay=max_wait))

    # Tier 2: navegador stealth que resuelve el challenge y acuña cookie.
    if SCRAPLING_STEALTH_ENABLED and _ScraplingStealthy is not None:
        stealth_problem = False
        try:
            ensure_fetch_budget(url)
            wait_for_domain_cooldown()
            result = _scrapling_stealth_solve(url)
            if result is not None:
                status, text = result
                if status in (None, 200) and text and not is_cloudflare_challenge(text):
                    register_fetch_success("scrapling_stealth", url)
                    fetch_cache_put(url, text)
                    log(f"OK scrapling_stealth {len(text)} bytes: {url}")
                    return text
                # Cookie acuñada: reintenta el tier 1 barato con la nueva cf_clearance.
                fresh = read_json(CF_SESSION, {}) if CF_SESSION.exists() else {}
                cookies2 = {"cf_clearance": fresh["cf_clearance"]} if fresh.get("cf_clearance") else None
                if cookies2:
                    try:
                        polite_fetch_wait(interval)
                        _FETCH_STATS["http_attempts"] += 1
                        status, text = _scrapling_impersonate_get(url, cookies2, timeout)
                        if status == 200 and text and not is_cloudflare_challenge(text):
                            register_fetch_success("scrapling", url)
                            fetch_cache_put(url, text)
                            return text
                    except Exception as exc:
                        register_fetch_error("scrapling", url, exc)
                stealth_problem = True
        except FetchBudgetExceeded:
            raise
        except Exception as exc:
            register_fetch_error("scrapling_stealth", url, exc)
            stealth_problem = True
        if stealth_problem and maybe_refresh_cf_session_once("scrapling_stealth_failed_or_challenged", url):
            refreshed_html = _retry_after_cf_refresh(url, timeout, interval)
            if refreshed_html is not None:
                return refreshed_html

    return None


def fetch_html(
    link: str,
    timeout: int = 45,
    *,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    min_interval: float | None = None,
    use_cache: bool = True,
) -> str:
    """Descarga HTML de HLTV con bajo volumen, backoff y sesión persistente.

    Esta función es el camino principal de `start.py`; los spiders Scrapy quedan
    como fallback. Preferimos tardar más y conservar el run incompleto antes que
    aceptar silenciosamente HTML de bloqueo como si fuese dato real.
    """

    url = hltv_url(link)
    if use_cache:
        cached = fetch_cache_get(url)
        if cached is not None:
            return cached
    check_url_quarantine(url)
    attempts = max_attempts or FETCH_MAX_ATTEMPTS
    base = FETCH_BASE_DELAY if base_delay is None else base_delay
    max_wait = FETCH_MAX_DELAY if max_delay is None else max_delay
    interval = FETCH_MIN_INTERVAL if min_interval is None else min_interval
    headers = default_headers()
    cf_payload = read_json(CF_SESSION, {}) if CF_SESSION.exists() else {}
    if cf_payload.get("user_agent"):
        headers["User-Agent"] = cf_payload["user_agent"]
    cookies = {"cf_clearance": cf_payload["cf_clearance"]} if cf_payload.get("cf_clearance") else None
    last_error: Exception | None = None

    # Tiers 1/2: Scrapling (impersonate TLS -> navegador stealth). Si devuelve
    # None, cae al camino clasico requests -> cloudscraper de mas abajo.
    prefer_requests = bool(cookies) and time.monotonic() < _REQUESTS_PREFERRED_UNTIL
    if prefer_requests:
        log(f"cf_session recently refreshed; trying requests before scrapling: {url}", force=True)
    else:
        scrapling_html = _scrapling_fetch(url, cookies, timeout, base, max_wait, interval)
        if scrapling_html is not None:
            return scrapling_html
    # El navegador stealth pudo acuñar una cf_clearance nueva: reutilizala.
    if not cookies:
        cf_payload = read_json(CF_SESSION, {}) if CF_SESSION.exists() else {}
        if cf_payload.get("user_agent"):
            headers["User-Agent"] = cf_payload["user_agent"]
        cookies = {"cf_clearance": cf_payload["cf_clearance"]} if cf_payload.get("cf_clearance") else None

    for attempt in range(attempts):
        ensure_fetch_budget(url)
        try:
            log(f"GET requests {attempt + 1}/{attempts}: {url}")
            started = time.monotonic()
            wait_for_domain_cooldown()
            polite_fetch_wait(interval)
            _FETCH_STATS["http_attempts"] += 1
            response = http_session().get(url, timeout=timeout, headers=headers, cookies=cookies)
            text = response.text or ""
            challenge = is_cloudflare_challenge(text) or (
                str(response.headers.get("cf-mitigated", "")).strip().lower() == "challenge"
            )
            elapsed = time.monotonic() - started
            if response.status_code == 200 and not challenge:
                register_fetch_success("requests", url)
                fetch_cache_put(url, text)
                log(f"OK requests HTTP 200 {len(text)} bytes {elapsed:.1f}s: {url}")
                return text
            if response.status_code in BLOCK_HTTP_CODES or challenge:
                last_error = HltvFetchError(
                    f"HLTV bloqueo/challenge HTTP {response.status_code}",
                    url=url,
                    status_code=response.status_code,
                    source="requests",
                )
                delay = backoff_delay(attempt, response.headers.get("Retry-After"), base_delay=base, max_delay=max_wait)
                reason = "cloudflare_challenge" if challenge else f"HTTP {response.status_code}"
                delay = max(delay, register_fetch_block("requests", url, reason))
                log(f"BLOCK requests {reason}; wait {delay:.1f}s: {url}")
                time.sleep(delay)
                continue
            response.raise_for_status()
            register_fetch_success("requests", url)
            fetch_cache_put(url, text)
            log(f"OK requests HTTP {response.status_code} {len(text)} bytes {elapsed:.1f}s: {url}")
            return text
        except Exception as exc:
            last_error = exc
            register_fetch_error("requests", url, exc)
            if attempt < attempts - 1:
                delay = backoff_delay(attempt, None, base_delay=base, max_delay=max_wait)
                log(f"ERROR requests {type(exc).__name__}: {exc}; wait {delay:.1f}s: {url}")
                time.sleep(delay)

    if cloudscraper is None:
        if looks_like_cf_or_waf_problem(last_error) and maybe_refresh_cf_session_once("requests_failed_no_cloudscraper", url):
            return fetch_html(
                link,
                timeout=timeout,
                max_attempts=min(3, attempts),
                base_delay=base,
                max_delay=max_wait,
                min_interval=interval,
                use_cache=use_cache,
            )
        quarantine_url(url, f"cloudscraper_not_installed: {last_error}")
        raise RuntimeError(f"HLTV fetch failed and cloudscraper is not installed: {last_error}")

    log(f"FALLBACK cloudscraper: {url}")
    scraper = cloudscraper.create_scraper()
    cloud_attempts = max(2, attempts // 2)
    for attempt in range(cloud_attempts):
        ensure_fetch_budget(url)
        try:
            log(f"GET cloudscraper {attempt + 1}/{cloud_attempts}: {url}")
            started = time.monotonic()
            wait_for_domain_cooldown()
            polite_fetch_wait(interval)
            _FETCH_STATS["http_attempts"] += 1
            response = scraper.get(url, timeout=timeout, headers=headers)
            text = response.text or ""
            challenge = is_cloudflare_challenge(text) or (
                str(response.headers.get("cf-mitigated", "")).strip().lower() == "challenge"
            )
            elapsed = time.monotonic() - started
            if response.status_code == 200 and not challenge:
                register_fetch_success("cloudscraper", url)
                fetch_cache_put(url, text)
                log(f"OK cloudscraper HTTP 200 {len(text)} bytes {elapsed:.1f}s: {url}")
                return text
            if response.status_code in BLOCK_HTTP_CODES or challenge:
                last_error = HltvFetchError(
                    f"HLTV bloqueo/challenge HTTP {response.status_code}",
                    url=url,
                    status_code=response.status_code,
                    source="cloudscraper",
                )
                delay = backoff_delay(attempt, response.headers.get("Retry-After"), base_delay=base, max_delay=max_wait)
                reason = "cloudflare_challenge" if challenge else f"HTTP {response.status_code}"
                delay = max(delay, register_fetch_block("cloudscraper", url, reason))
                log(f"BLOCK cloudscraper {reason}; wait {delay:.1f}s: {url}")
                time.sleep(delay)
                continue
            response.raise_for_status()
            register_fetch_success("cloudscraper", url)
            fetch_cache_put(url, text)
            log(f"OK cloudscraper HTTP {response.status_code} {len(text)} bytes {elapsed:.1f}s: {url}")
            return text
        except Exception as exc:
            last_error = exc
            register_fetch_error("cloudscraper", url, exc)
            if attempt < cloud_attempts - 1:
                delay = backoff_delay(attempt, None, base_delay=base, max_delay=max_wait)
                log(f"ERROR cloudscraper {type(exc).__name__}: {exc}; wait {delay:.1f}s: {url}")
                time.sleep(delay)

    if looks_like_cf_or_waf_problem(last_error) and maybe_refresh_cf_session_once("requests_and_cloudscraper_failed", url):
        return fetch_html(
            link,
            timeout=timeout,
            max_attempts=min(3, attempts),
            base_delay=base,
            max_delay=max_wait,
            min_interval=interval,
            use_cache=use_cache,
        )
    quarantine_url(url, str(last_error or "unknown fetch failure"))
    raise RuntimeError(f"HLTV fetch failed after {attempts} attempts for {url}: {last_error}")


def team_from_html(selector: Selector, number: int) -> dict[str, Any]:
    link = (
        selector.css(f"div.team{number}-gradient a::attr(href)").get()
        or selector.css(f"a.dropdownTeam.team{number}::attr(href)").get()
    )
    name = (
        selector.css(f"div.team{number}-gradient .teamName::text").get()
        or selector.css(f"a.dropdownTeam.team{number} .teamName::text").get()
    )
    logo = (
        selector.css(f"div.team{number}-gradient img.logo::attr(src)").get()
        or selector.css(f"a.dropdownTeam.team{number} img.logo::attr(src)").get()
    )
    score = selector.css(f".team{number}-gradient > div:nth-child(2)::text").get()
    return {
        "name": name.strip() if name else None,
        "id": link.split("/")[2] if link and link.startswith("/team/") else None,
        "link": link,
        "logo": logo,
        "score": score.strip() if score else None,
    }


def parse_match_basic_html(html: str) -> dict[str, Any] | None:
    selector = Selector(text=html)
    team1 = team_from_html(selector, 1)
    team2 = team_from_html(selector, 2)
    if not team1.get("name") or not team2.get("name"):
        return None
    return {
        "match": {
            "date": selector.css(".teamsBox .date::text, .teamsBoxDropdown .date::text").get(),
            "hour": selector.css(".teamsBox .time::text, .teamsBoxDropdown .time::text").get(),
            "event": selector.css(".teamsBox .event a::text, .teamsBox .event::text").get(),
            "team1": team1,
            "team2": team2,
        },
        "maps": [],
        "stats": [],
        "source": "html_fallback",
    }


def parse_match_detail_html(html: str) -> dict[str, Any] | None:
    if PF is None:
        return parse_match_basic_html(html)
    selector = Selector(text=html)
    teams_box = PF.get_parser("match_teams_box").parse(selector.css(".teamsBox"))
    if not (teams_box or {}).get("team1", {}).get("name"):
        basic = parse_match_basic_html(html)
        return basic
    maps_score = PF.get_parser("map_holders").parse(selector)
    player_stats = PF.get_parser("table_stats").parse(selector.css("#all-content"))
    return {"match": teams_box, "maps": maps_score, "stats": player_stats}


def detail_from_upcoming_row(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "match": {
            "date": match.get("date"),
            "hour": match.get("hour"),
            "event": match.get("event"),
            "team1": {
                "name": (match.get("team1") or {}).get("name"),
                "id": (match.get("team1") or {}).get("id"),
                "link": (match.get("team1") or {}).get("link"),
                "logo": (match.get("team1") or {}).get("logo"),
                "score": None,
            },
            "team2": {
                "name": (match.get("team2") or {}).get("name"),
                "id": (match.get("team2") or {}).get("id"),
                "link": (match.get("team2") or {}).get("link"),
                "logo": (match.get("team2") or {}).get("logo"),
                "score": None,
            },
        },
        "maps": [],
        "stats": [],
        "source": "upcoming_row_fallback",
    }


ANALYTICS_MAP_NAMES = {
    "ancient",
    "anubis",
    "cache",
    "dust2",
    "inferno",
    "mirage",
    "nuke",
    "overpass",
    "train",
    "vertigo",
}


def analytics_link_from_match(match_link: str | None) -> str | None:
    if not match_link:
        return None
    match = re.search(r"/matches/(\d+)/([^?#/]+)", match_link)
    if not match:
        return None
    return f"/betting/analytics/{match.group(1)}/{match.group(2)}"


def parse_analytics_metric_line(line: str) -> dict[str, Any] | None:
    match = re.match(r"^(\d+)%\s+(\d+)%\s+(-|\d+%)\s+(\d+)$", line.strip())
    if not match:
        return None
    win_raw = match.group(3)
    return {
        "first_pick_pct": int(match.group(1)),
        "first_ban_pct": int(match.group(2)),
        "win_pct": None if win_raw == "-" else int(win_raw.rstrip("%")),
        "played": int(match.group(4)),
    }


def parse_analytics_percentage(value: str | None) -> float | None:
    if not value:
        return None
    text = clean_text(value) or ""
    if text in {"-", "--"}:
        return None
    return parse_float(text.replace("%", ""))


def analytics_team_key(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def parse_analytics_standin_names(text: str) -> list[str]:
    match = re.search(r"stand-?ins?\s*:\s*(.+?)\s+instead\s+of\s+", text, flags=re.I)
    if not match:
        return []
    return [clean_text(name) or "" for name in match.group(1).split(",") if clean_text(name)]


def parse_analytics_html(html: str, match_id: str, match_link: str, team_names: list[str]) -> dict[str, Any]:
    """Parsea las tablas estables del Analytics Center de HLTV.

    Se conserva también el texto por si HLTV ajusta el HTML. Los campos
    estructurados no toman nada de votos, streams ni promociones: solo
    información deportiva pre-partido expuesta por la propia página.
    """
    selector = Selector(text=html)
    lines = [clean_text(line) for line in selector.xpath("//body//text()").getall()]
    lines = [line for line in lines if line]
    lower_lines = [line.lower() for line in lines]

    def section(start_terms: list[str], stop_terms: list[str]) -> list[str]:
        start = None
        for idx, line in enumerate(lower_lines):
            if any(term in line for term in start_terms):
                start = idx + 1
                break
        if start is None:
            return []
        stop = len(lines)
        for idx in range(start, len(lines)):
            if any(term in lower_lines[idx] for term in stop_terms):
                stop = idx
                break
        return lines[start:stop]

    summary_lines = section(
        ["analytics summary", "match analytics"],
        ["head to head", "map handicap", "map stats", "about this matchup"],
    )
    insight_terms = (
        "better form",
        "better ranked",
        "worse ranked",
        "won ",
        "core lineup",
        "matches with core",
        "stand-in",
        "standin",
        "past 30 days",
        "current event",
        "maps in the past",
    )
    insights = [line for line in summary_lines if any(term in line.lower() for term in insight_terms)][:80]

    structured_insights: list[dict[str, str]] = []
    core_lineup: dict[str, dict[str, Any]] = {}
    standins: dict[str, list[str]] = {}
    for card in selector.css(".analytics-insights-container"):
        team = clean_text(card.css(".analytics-insights-team-header .team-name::text, .analytics-insights-team-header .team-name *::text").get())
        if not team:
            team = clean_text(card.css(".team-name::text").get())
        for row in card.css(".analytics-insights-insight"):
            text = clean_text(" ".join(row.css(".analytics-insights-info *::text, .analytics-insights-info::text").getall()))
            if not text:
                continue
            classes = " ".join(row.css(".analytics-insights-indicator::attr(class)").getall()).lower()
            direction = "against" if "against" in classes else "in_favor" if "favor" in classes else "unknown"
            structured_insights.append({"team": team or "", "direction": direction, "text": text})
            if team and text not in insights and any(term in text.lower() for term in insight_terms):
                insights.append(text)
            if not team:
                continue
            core = re.search(r"(.+?)\s+has\s+played\s+less\s+than\s+(\d+)\s+matches\s+with\s+core", text, flags=re.I)
            if core:
                core_lineup[team] = {
                    "players": [clean_text(name) or "" for name in core.group(1).split(",") if clean_text(name)],
                    "matches_lt": int(core.group(2)),
                }
            names = parse_analytics_standin_names(text)
            if names:
                standins[team] = names

    event_metadata: dict[str, Any] = {}
    for info in selector.css(".analytics-event-info .analytics-info"):
        label = (clean_text(" ".join(info.css(".analytics-info-sub-title *::text, .analytics-info-sub-title::text").getall())) or "").lower()
        value = clean_text(" ".join(info.css(".analytics-info-header *::text, .analytics-info-header::text").getall()))
        if not label or not value:
            continue
        if "prize" in label:
            event_metadata["prize_pool"] = parse_money_amount(value)
            event_metadata["prize_pool_raw"] = value
        elif "teams competing" in label:
            event_metadata["teams_competing"] = parse_int(value)
        elif label == "event":
            event_metadata["name"] = value

    map_stats: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str]] = set()
    # HLTV da una fila por equipo; la celda del mapa usa rowspan y solo aparece
    # en la primera fila. Mantener `current_map` evita el parser frágil basado
    # en proximidad de texto que dejaba la tabla vacía.
    for table in selector.css("table"):
        if not table.css(".analytics-map-stats-team"):
            continue
        current_map = None
        for row in table.css("tbody > tr"):
            current_map = clean_text(row.css(".analytics-map-name::text").get()) or current_map
            team = clean_text(row.css(".maps-team-name::text").get())
            if not current_map or not team:
                continue
            row_key = (current_map.lower(), team.lower())
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            map_stats.append(
                {
                    "map": current_map,
                    "team": team,
                    "first_pick_pct": parse_analytics_percentage(row.css(".analytics-map-stats-pick-percentage::text").get()),
                    "first_ban_pct": parse_analytics_percentage(row.css(".analytics-map-stats-ban-percentage::text").get()),
                    "win_pct": parse_analytics_percentage(row.css(".analytics-map-stats-win-percentage::text").get()),
                    "played": parse_int(row.css(".analytics-map-stats-played::text").get()),
                    "comment": clean_text(" ".join(row.css(".analytics-map-stats-comment *::text, .analytics-map-stats-comment::text").getall())),
                }
            )

    # Fallback para snapshots HTML antiguos, y para que los tests offline sigan
    # cubriendo un HTML mínimo sin las clases actuales de HLTV.
    if not map_stats:
        for idx, line in enumerate(lines):
            map_name = line.strip()
            if map_name.lower() not in ANALYTICS_MAP_NAMES:
                continue
            for team in team_names:
                if not team:
                    continue
                team_idx = next((probe for probe in range(idx + 1, min(idx + 28, len(lines))) if lines[probe].lower() == team.lower()), None)
                if team_idx is None:
                    continue
                metrics = None
                metric_idx = None
                for probe in range(team_idx + 1, min(team_idx + 7, len(lines))):
                    metrics = parse_analytics_metric_line(lines[probe])
                    if metrics:
                        metric_idx = probe
                        break
                if not metrics:
                    continue
                row_key = (map_name.lower(), team.lower())
                if row_key in seen_rows:
                    continue
                seen_rows.add(row_key)
                comment = None
                if metric_idx is not None and metric_idx + 1 < len(lines):
                    next_line = lines[metric_idx + 1]
                    if (
                        next_line.lower() not in ANALYTICS_MAP_NAMES
                        and next_line.lower() not in {name.lower() for name in team_names if name}
                        and not parse_analytics_metric_line(next_line)
                    ):
                        comment = next_line
                map_stats.append({"map": map_name, "team": team, **metrics, "comment": comment})

    series_stats: dict[str, dict[str, Any]] = {}
    for table in selector.css("table.analytics-handicap-table"):
        side_class = " ".join(table.css("::attr(class)").getall()).lower()
        side = "team1" if "team1" in side_class else "team2" if "team2" in side_class else "unknown"
        team = clean_text(table.css(".team-name::text").get())
        if not team:
            continue
        counts = clean_text(table.css(".match-map-count::text").get()) or ""
        count_match = re.search(r"(\d+)\s+matches\s*,\s*(\d+)\s+maps", counts, flags=re.I)
        stats: dict[str, Any] = {
            "team": team,
            "side": side,
            "matches": int(count_match.group(1)) if count_match else None,
            "maps": int(count_match.group(2)) if count_match else None,
            "score_distribution": {},
            "overtime_pct": None,
        }
        for row in table.css("tbody > tr"):
            cells = [clean_text(text) or "" for text in row.css("td::text").getall()]
            label = next((cell for cell in cells if re.search(r"(?:[012]\s*-\s*[012]\s+(?:wins|losses)|overtimes)", cell, flags=re.I)), "")
            value = clean_text(row.css(".handicap-data::text").get())
            pct = parse_analytics_percentage(value)
            if not label or pct is None:
                continue
            normalized = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
            if normalized == "overtimes":
                stats["overtime_pct"] = pct
            else:
                stats["score_distribution"][normalized] = pct
        series_stats[side] = stats

    map_handicap: list[dict[str, Any]] = []
    for container in selector.css(".analytics-handicap-map-container"):
        classes = " ".join(container.css("::attr(class)").getall()).lower()
        side = "team1" if "team1" in classes else "team2" if "team2" in classes else "unknown"
        team = team_names[0] if side == "team1" and team_names else team_names[1] if side == "team2" and len(team_names) > 1 else None
        if not team:
            continue
        overall = {}
        for row in container.css(".analytics-handicap-map-data-overall-container .analytics-handicap-map-data"):
            values = [cleaned for text in row.css("div::text").getall() if (cleaned := clean_text(text))]
            if len(values) < 2:
                continue
            value = parse_float(values[0])
            label = values[-1].lower()
            if "lost in wins" in label:
                overall["avg_rounds_lost_in_wins"] = value
            elif "won in losses" in label:
                overall["avg_rounds_won_in_losses"] = value
        if overall:
            map_handicap.append({"team": team, "map": "overall", **overall})
        for row in container.css("table tbody > tr"):
            map_name = clean_text(row.css(".mapname::text").get())
            values = [parse_float(value) for value in row.css(".analytics-handicap-map-data-avg::text").getall()]
            if not map_name or len(values) < 2:
                continue
            map_handicap.append(
                {
                    "team": team,
                    "map": map_name,
                    "avg_rounds_lost_in_wins": values[0],
                    "avg_rounds_won_in_losses": values[1],
                }
            )

    return {
        "available": bool(lines),
        "match_id": match_id,
        "url": hltv_url(analytics_link_from_match(match_link) or match_link),
        "captured_at": now_utc(),
        "summary_lines": summary_lines[:120],
        "insights": insights[:80],
        "structured_insights": structured_insights,
        "core_lineup": core_lineup,
        "standins": standins,
        "event_metadata": event_metadata,
        "series_stats": series_stats,
        "map_handicap": map_handicap,
        "map_stats": map_stats,
        "line_count": len(lines),
    }


def scrape_match_analytics(match_link: str, run_dir: Path, match_id: str, team_names: list[str]) -> dict[str, Any]:
    analytics_link = analytics_link_from_match(match_link)
    output = run_dir / "analytics" / f"{match_id}.json"
    if not analytics_link:
        log(f"analytics {match_id}: no analytics link")
        payload = {"available": False, "match_id": match_id, "reason": "analytics_link_not_found", "captured_at": now_utc()}
        write_json(output, payload)
        return payload
    try:
        log(f"analytics {match_id}: fetching {analytics_link}")
        html = fetch_html(analytics_link)
        raw_html = save_raw_html(run_dir, "analytics_page", match_id, analytics_link, html)
        payload = parse_analytics_html(html, match_id, analytics_link, team_names)
        payload["raw_html"] = raw_html
        log(
            f"analytics {match_id}: available={payload.get('available')} lines={payload.get('line_count')} "
            f"map_rows={len(payload.get('map_stats') or [])} "
            f"core_flags={len(payload.get('core_lineup') or {})}"
        )
    except Exception as exc:
        log(f"analytics {match_id}: ERROR {exc}")
        payload = {
            "available": False,
            "match_id": match_id,
            "url": hltv_url(analytics_link),
            "error": str(exc),
            "captured_at": now_utc(),
        }
    payload["source_file"] = str(output.relative_to(DATA_ROOT))
    write_json(output, payload)
    return payload


def parse_float(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return float(text.strip().replace(",", "."))
    except ValueError:
        return None


def parse_int(text: Any) -> int | None:
    if text is None:
        return None
    match = re.search(r"-?\d+", str(text).replace(",", ""))
    return int(match.group(0)) if match else None


def parse_money_amount(value: Any) -> int | None:
    """Parse source amounts such as '$5K', 'EUR 20,000' or '$1.2M'."""
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*([kmb])?", text, re.IGNORECASE)
    if not match:
        return None
    number_text = match.group(1)
    suffix = (match.group(2) or "").lower()
    if suffix:
        number = float(number_text.replace(",", "."))
        multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix]
        return int(round(number * multiplier))
    digits = re.sub(r"[^0-9]", "", number_text)
    return int(digits) if digits else None


def clean_texts(node) -> list[str]:
    return [text.strip() for text in node.css("::text").getall() if text.strip()]


def text_join(node) -> str:
    return " ".join(clean_texts(node))


def parse_stat_float(text: str | None) -> float | None:
    if text is None:
        return None
    value = text.strip().replace("%", "").replace("+", "").replace(",", ".")
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_pair(text: str | None) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    match = re.search(r"(-?\d+)\s*:\s*(-?\d+)", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def parse_count_with_parenthetical(text: str | None) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    first = parse_int(text)
    paren = re.search(r"\(([-+]?\d+)\)", text)
    return first, int(paren.group(1)) if paren else None


def mapstats_id_from_link(link: str | None) -> str | None:
    if not link:
        return None
    match = re.search(r"/mapstatsid/(\d+)", link)
    return match.group(1) if match else None


def stats_player_id_from_link(link: str | None) -> str | None:
    if not link:
        return None
    match = re.search(r"/stats/players/(\d+)", link)
    return match.group(1) if match else None


def stats_team_id_from_link(link: str | None) -> str | None:
    if not link:
        return None
    match = re.search(r"/stats/teams/(\d+)", link)
    return match.group(1) if match else None


def parse_veto_html(html: str) -> dict[str, Any]:
    selector = Selector(text=html)
    boxes = selector.css(".veto-box")
    meta_text = text_join(boxes[0]) if boxes else ""
    steps: list[dict[str, Any]] = []
    for box in boxes:
        for div in box.css(".padding > div"):
            text = text_join(div)
            step_match = re.match(r"^(\d+)\.\s+(.+)$", text)
            if not step_match:
                continue
            order = int(step_match.group(1))
            body = step_match.group(2).strip()
            action_match = re.match(r"(.+?)\s+(removed|picked)\s+(.+)$", body, flags=re.I)
            if action_match:
                action_raw = action_match.group(2).lower()
                steps.append(
                    {
                        "step_order": order,
                        "team_name": action_match.group(1).strip(),
                        "action": "ban" if action_raw == "removed" else "pick",
                        "raw_action": action_raw,
                        "map_name": action_match.group(3).strip(),
                        "raw_text": text,
                    }
                )
                continue
            decider_match = re.match(r"(.+?)\s+was left over$", body, flags=re.I)
            if decider_match:
                steps.append(
                    {
                        "step_order": order,
                        "team_name": None,
                        "action": "decider",
                        "raw_action": "left_over",
                        "map_name": decider_match.group(1).strip(),
                        "raw_text": text,
                    }
                )
    return {"meta": meta_text, "context": parse_match_context_meta(meta_text), "steps": steps}


def parse_mapstats_team(selector, side: str) -> dict[str, Any]:
    team = selector.css(f".match-info-box .team-{side}")
    link = team.css("a[href*='/stats/teams/']::attr(href)").get()
    return {
        "name": team.css("a[href*='/stats/teams/']::text").get(),
        "hltv_id": stats_team_id_from_link(link),
        "link": link,
        "score": parse_int(team.css(".bold::text").get()),
    }


def parse_breakdown_row(row) -> dict[str, Any]:
    right = row.css(".right")
    right_text = text_join(right)
    score_left, score_right = parse_pair(right_text)
    bracket_pairs = re.findall(r"\(\s*(\d+)\s*:\s*(\d+)\s*\)", right_text)
    overtime_left = overtime_right = 0
    for left, right_score in bracket_pairs[2:]:
        overtime_left += int(left)
        overtime_right += int(right_score)

    side_rounds = {
        "left": {"ct": None, "t": None},
        "right": {"ct": None, "t": None},
        "overtime_left": overtime_left or None,
        "overtime_right": overtime_right or None,
    }
    colored = []
    for span in right.xpath(".//span[contains(@class,'t-color') or contains(@class,'ct-color')]"):
        class_name = span.attrib.get("class", "")
        value = parse_int(span.css("::text").get())
        if value is None:
            continue
        colored.append(("ct" if "ct-color" in class_name else "t", value))
    if len(colored) >= 4:
        for slot, (side_name, value) in zip(
            ("left", "right", "left", "right"),
            colored[:4],
        ):
            side_rounds[slot][side_name] = value

    return {
        "raw": right_text,
        "score_left": score_left,
        "score_right": score_right,
        "halves": [{"left": int(a), "right": int(b)} for a, b in bracket_pairs[:2]],
        "side_rounds": side_rounds,
    }


def parse_mapstats_info(selector: Selector) -> dict[str, Any]:
    box = selector.css(".match-info-box")
    direct_texts = [text.strip() for text in box.xpath("./text()").getall() if text.strip()]
    info: dict[str, Any] = {
        "event": box.xpath("./a/text()").get(),
        "event_link": box.xpath("./a/@href").get(),
        "datetime_text": box.css("span[data-unix]::text").get(),
        "datetime_unix_ms": parse_int(box.css("span[data-unix]::attr(data-unix)").get()),
        "map_name": direct_texts[0] if direct_texts else None,
        "team_left": parse_mapstats_team(selector, "left"),
        "team_right": parse_mapstats_team(selector, "right"),
    }
    rows = selector.css(".match-info-row")
    for row in rows:
        label = (row.css(".bold::text").get() or "").strip().lower()
        value = text_join(row.css(".right"))
        if label == "breakdown":
            info["breakdown"] = parse_breakdown_row(row)
        elif label == "team rating 3.0":
            left, right = parse_pair(value)
            # Ratings are decimal, so parse the raw pair separately.
            dec = re.search(r"([-+]?\d+(?:\.\d+)?)\s*:\s*([-+]?\d+(?:\.\d+)?)", value)
            info["team_rating_3_0"] = {
                "left": float(dec.group(1)) if dec else left,
                "right": float(dec.group(2)) if dec else right,
                "raw": value,
            }
        elif label == "first kills":
            left, right = parse_pair(value)
            info["first_kills"] = {"left": left, "right": right, "raw": value}
        elif label == "clutches won":
            left, right = parse_pair(value)
            info["clutches_won"] = {"left": left, "right": right, "raw": value}
    return info


def first_cell_text(row, selector: str) -> str | None:
    base_selector = selector.removesuffix("::text")
    nodes = row.css(base_selector)
    values = [value.strip() for value in nodes.xpath(".//text()").getall() if value.strip()]
    return " ".join(values) if values else None


def parse_player_stats_row(row) -> dict[str, Any]:
    player_link = row.css("td.st-player a::attr(href)").get()
    op_kills, op_deaths = parse_pair(first_cell_text(row, "td.st-opkd.traditional-data::text"))
    eco_op_kills, eco_op_deaths = parse_pair(first_cell_text(row, "td.st-opkd.eco-adjusted-data::text"))
    kills, headshots = parse_count_with_parenthetical(first_cell_text(row, "td.st-kills.traditional-data::text"))
    eco_kills, eco_headshots = parse_count_with_parenthetical(first_cell_text(row, "td.st-kills.eco-adjusted-data::text"))
    assists, flash_assists = parse_count_with_parenthetical(first_cell_text(row, "td.st-assists::text"))
    deaths, traded_deaths = parse_count_with_parenthetical(first_cell_text(row, "td.st-deaths.traditional-data::text"))
    eco_deaths, eco_traded_deaths = parse_count_with_parenthetical(first_cell_text(row, "td.st-deaths.eco-adjusted-data::text"))
    kast = (
        first_cell_text(row, "td.st-kast.gtSmartphone-only.traditional-data::text")
        or first_cell_text(row, "td.st-kast.traditional-data::text")
    )
    eco_kast = (
        first_cell_text(row, "td.st-kast.gtSmartphone-only.eco-adjusted-data::text")
        or first_cell_text(row, "td.st-kast.eco-adjusted-data::text")
    )
    return {
        "player_name": text_join(row.css("td.st-player")) or None,
        "player_hltv_id": stats_player_id_from_link(player_link),
        "player_link": player_link,
        "opening_kills": op_kills,
        "opening_deaths": op_deaths,
        "eco_opening_kills": eco_op_kills,
        "eco_opening_deaths": eco_op_deaths,
        "multi_kill_rounds": parse_int(first_cell_text(row, "td.st-mks::text")),
        "kast": parse_stat_float(kast),
        "eco_kast": parse_stat_float(eco_kast),
        "clutches_won": parse_int(first_cell_text(row, "td.st-clutches::text")),
        "kills": kills,
        "headshots": headshots,
        "eco_kills": eco_kills,
        "eco_headshots": eco_headshots,
        "assists": assists,
        "flash_assists": flash_assists,
        "deaths": deaths,
        "traded_deaths": traded_deaths,
        "eco_deaths": eco_deaths,
        "eco_traded_deaths": eco_traded_deaths,
        "adr": parse_stat_float(first_cell_text(row, "td.st-adr.traditional-data::text")),
        "eco_adr": parse_stat_float(first_cell_text(row, "td.st-adr.eco-adjusted-data::text")),
        "round_swing": parse_stat_float(first_cell_text(row, "td.st-roundSwing::text")),
        "rating": parse_stat_float(first_cell_text(row, "td.st-rating::text")),
    }


def parse_mapstats_tables(selector: Selector, info: dict[str, Any]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    team_lookup = {
        (info.get("team_left") or {}).get("name"): info.get("team_left") or {},
        (info.get("team_right") or {}).get("name"): info.get("team_right") or {},
    }
    for table in selector.css("table.stats-table"):
        class_name = table.attrib.get("class", "")
        if "ctstats" in class_name:
            side = "ct"
        elif "tstats" in class_name:
            side = "t"
        else:
            side = "total"
        header = text_join(table.css("thead th:first-child"))
        team = team_lookup.get(header, {"name": header})
        for row in table.css("tbody tr"):
            parsed = parse_player_stats_row(row)
            if parsed.get("player_name"):
                parsed.update(
                    {
                        "team_name": team.get("name") or header,
                        "team_hltv_id": team.get("hltv_id"),
                        "team_link": team.get("link"),
                        "side": side,
                    }
                )
                stats.append(parsed)
    return stats


def parse_mapstats_html(html: str, source_link: str) -> dict[str, Any]:
    selector = Selector(text=html)
    info = parse_mapstats_info(selector)
    match_link = selector.css("a.match-page-link::attr(href)").get()
    parsed = {
        "source_link": source_link,
        "mapstats_id": mapstats_id_from_link(source_link),
        "match_link": match_link,
        "match_id": match_id_from_link(match_link),
        "captured_at": now_utc(),
        "info": info,
        "player_stats": parse_mapstats_tables(selector, info),
    }
    return parsed


def extract_mapstats_links(html: str) -> list[str]:
    selector = Selector(text=html)
    seen: set[str] = set()
    links: list[str] = []
    for href in selector.css("a::attr(href)").getall():
        if "/stats/matches/mapstatsid/" not in href:
            continue
        clean = href.split("?")[0]
        if clean not in seen:
            seen.add(clean)
            links.append(clean)
    return links


def scrape_match_assets(match_link: str, output_path: Path, delay: float = 0.5) -> dict[str, Any]:
    run_dir = output_path.parents[2]
    match_id = match_id_from_link(match_link) or slug_from_link(match_link)
    log(f"assets {match_id}: fetching match page")
    html = fetch_html(match_link)
    html_sources = [save_raw_html(run_dir, "match_page", match_id, match_link, html)]
    existing = read_json(output_path, {}) if output_path.exists() else {}
    assets = existing if isinstance(existing, dict) and existing.get("match_link") == match_link else {
        "match_id": match_id_from_link(match_link),
        "match_link": match_link,
        "captured_at": now_utc(),
        "veto": parse_veto_html(html),
        "mapstats_links": extract_mapstats_links(html),
        "mapstats": [],
        "errors": [],
        "raw_html": html_sources,
    }
    assets["captured_at"] = assets.get("captured_at") or now_utc()
    assets["veto"] = assets.get("veto") or parse_veto_html(html)
    assets["mapstats_links"] = assets.get("mapstats_links") or extract_mapstats_links(html)
    assets["mapstats"] = assets.get("mapstats") or []
    assets["errors"] = assets.get("errors") or []
    assets["raw_html"] = (assets.get("raw_html") or []) + html_sources
    seen_links = {item.get("source_link") for item in assets["mapstats"] if isinstance(item, dict)}
    log(f"assets {match_id}: veto_steps={len((assets.get('veto') or {}).get('steps') or [])} mapstats_links={len(assets['mapstats_links'])}")
    write_json(output_path, assets)
    db_mark_fetch_state(
        "match_assets",
        str(match_id),
        "partial",
        f"checkpoint match_page maps={len(assets['mapstats'])}/{len(assets['mapstats_links'])}",
    )
    for index, link in enumerate(assets["mapstats_links"], start=1):
        if link in seen_links:
            log(f"assets {match_id}: mapstats {index}/{len(assets['mapstats_links'])} already in partial JSON; skip {link}")
            continue
        try:
            log(f"assets {match_id}: mapstats {index}/{len(assets['mapstats_links'])} {link}")
            time.sleep(delay)
            mapstats_id = mapstats_id_from_link(link) or slug_from_link(link)
            map_html = load_persisted_raw_html("mapstats", mapstats_id) or fetch_html(link)
            meta = save_raw_html(run_dir, "mapstats", mapstats_id_from_link(link) or slug_from_link(link), link, map_html)
            parsed = parse_mapstats_html(map_html, link)
            parsed["raw_html"] = meta
            assets["mapstats"].append(parsed)
            seen_links.add(link)
            write_json(output_path, assets)
            db_mark_fetch_state(
                "match_assets",
                str(match_id),
                "partial",
                f"checkpoint mapstats maps={len(assets['mapstats'])}/{len(assets['mapstats_links'])}",
            )
        except KeyboardInterrupt:
            write_json(output_path, assets)
            db_mark_fetch_state(
                "match_assets",
                str(match_id),
                "partial",
                f"interrupted maps={len(assets['mapstats'])}/{len(assets['mapstats_links'])}",
            )
            raise
        except Exception as exc:
            log(f"assets {match_id}: mapstats ERROR {link}: {exc}")
            assets["errors"].append({"link": link, "error": str(exc)})
            write_json(output_path, assets)
            db_mark_fetch_state(
                "match_assets",
                str(match_id),
                "partial",
                f"error mapstats maps={len(assets['mapstats'])}/{len(assets['mapstats_links'])}",
            )
    write_json(output_path, assets)
    db_mark_fetch_state(
        "match_assets",
        str(match_id),
        "ok" if not assets.get("errors") else "partial",
        f"done maps={len(assets['mapstats'])}/{len(assets['mapstats_links'])} errors={len(assets['errors'])}",
    )
    return assets


def parse_odds(html: str) -> dict[str, Any]:
    selector = Selector(text=html)
    providers = []
    for row in selector.css("tr.provider"):
        bookmaker = (
            row.css("img.compare-logo::attr(title)").get()
            or row.css("img.compare-logo::attr(alt)").get()
            or row.css("a.betting-logo-link::attr(aria-label)").get()
        )
        if bookmaker:
            bookmaker = bookmaker.replace("Logo for ", "").replace("Go to ", "").strip()
        odds_values = [parse_float(value) for value in row.css("td.odds-cell a::text").getall()]
        odds_values = [value for value in odds_values if value is not None]
        if len(odds_values) >= 2:
            odd1, odd2 = odds_values[0], odds_values[1]
            implied1 = 1.0 / odd1
            implied2 = 1.0 / odd2
            overround = implied1 + implied2
            providers.append(
                {
                    "bookmaker": bookmaker or "unknown",
                    "team1_decimal": odd1,
                    "team2_decimal": odd2,
                    "team1_implied_prob_norm": implied1 / overround,
                    "team2_implied_prob_norm": implied2 / overround,
                    "overround": overround,
                }
            )

    if not providers:
        return {
            "available": False,
            "bookmaker_count": 0,
            "providers": [],
            "average": None,
        }

    return {
        "available": True,
        "bookmaker_count": len(providers),
        "providers": providers,
        "average": {
            "team1_decimal": sum(row["team1_decimal"] for row in providers) / len(providers),
            "team2_decimal": sum(row["team2_decimal"] for row in providers) / len(providers),
            "team1_implied_prob_norm": sum(row["team1_implied_prob_norm"] for row in providers) / len(providers),
            "team2_implied_prob_norm": sum(row["team2_implied_prob_norm"] for row in providers) / len(providers),
            "overround": sum(row["overround"] for row in providers) / len(providers),
        },
    }


def compact_odds_point(snapshot: dict[str, Any], run_dir: Path) -> dict[str, Any] | None:
    odds = snapshot.get("odds") or {}
    average = odds.get("average") or {}
    if not odds.get("available") or not average:
        return None
    source_file = snapshot.get("source_file") or str(
        (run_dir / "match_snapshots" / f"{snapshot['id']}.json").relative_to(DATA_ROOT)
    )
    return {
        "captured_at": snapshot.get("captured_at"),
        "run_id": run_dir.name,
        "source_file": source_file,
        "bookmaker_count": odds.get("bookmaker_count", 0),
        "team1_decimal": average.get("team1_decimal"),
        "team2_decimal": average.get("team2_decimal"),
        "team1_implied_prob_norm": average.get("team1_implied_prob_norm"),
        "team2_implied_prob_norm": average.get("team2_implied_prob_norm"),
        "overround": average.get("overround"),
        "providers": odds.get("providers") or [],
    }


def append_odds_point(record: dict[str, Any], odds: dict[str, Any], run_dir: Path, source_file: str) -> bool:
    point = compact_odds_point(
        {
            "id": record.get("id"),
            "captured_at": now_utc(),
            "source_file": source_file,
            "odds": odds,
        },
        run_dir,
    )
    if not point:
        return False
    record.setdefault("odds_history", [])
    duplicate = any(
        existing.get("team1_decimal") == point.get("team1_decimal")
        and existing.get("team2_decimal") == point.get("team2_decimal")
        and existing.get("bookmaker_count") == point.get("bookmaker_count")
        and str(existing.get("captured_at", ""))[:10] == str(point.get("captured_at", ""))[:10]
        for existing in record["odds_history"]
    )
    if not duplicate:
        record["odds_history"].append(point)
    record.setdefault("opening_odds", point)
    record["latest_odds_point"] = point
    record["latest_odds"] = odds
    if record.get("status") == "completed":
        closing = observed_closing_odds(record, point)
        if closing:
            record.setdefault("closing_odds", closing)
    return True


def has_usable_odds(record: dict[str, Any]) -> bool:
    if record.get("opening_odds") or record.get("latest_odds_point"):
        return True
    if record.get("odds_history"):
        return True
    latest = record.get("latest_odds") or {}
    return bool(latest.get("available") and latest.get("bookmaker_count"))


def parse_record_date(record: dict[str, Any]) -> datetime.date | None:
    raw = record.get("date") or ((record.get("upcoming_row") or {}).get("date"))
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw)[:10], "%Y-%m-%d").date()
    except ValueError:
        parsed = parse_hltv_date(str(raw))
        if parsed:
            try:
                return datetime.strptime(parsed, "%Y-%m-%d").date()
            except ValueError:
                return None
    return None


def is_recent_or_active_record(record: dict[str, Any], window_days: int) -> bool:
    if record.get("status") != "completed":
        return True
    match_date = parse_record_date(record)
    if match_date is None:
        return False
    today = datetime.now(timezone.utc).date()
    return today - timedelta(days=window_days) <= match_date <= today + timedelta(days=1)


def match_row_from_record(record: dict[str, Any]) -> dict[str, Any]:
    row = dict(record.get("upcoming_row") or {})
    detail_match = (record.get("detail") or {}).get("match") or {}
    row.setdefault("link", record.get("link"))
    row.setdefault("date", record.get("date"))
    row.setdefault("hour", record.get("hour"))
    row.setdefault("meta", record.get("format"))
    row.setdefault("event", record.get("event"))
    for key in ("team1", "team2"):
        if row.get(key):
            continue
        team = detail_match.get(key) or {}
        row[key] = {
            "name": team.get("name"),
            "id": team.get("id"),
            "link": team.get("link"),
            "logo": team.get("logo"),
        }
    return row


def recover_recent_data_gaps(
    master: dict[str, Any],
    run_dir: Path,
    *,
    capture_analytics: bool,
    window_days: int,
    delay: float,
) -> dict[str, Any]:
    """Reintenta datos recientes que quedaron incompletos por bloqueo o parser.

    La recuperación es deliberadamente point-in-time: no inventa datos pasados.
    Si hoy/mañana aún se exponen odds, Analytics o detalle, se guardan con el
    timestamp actual; si ya desaparecieron, queda registrado como error.
    """

    recovery_index: list[dict[str, Any]] = []
    candidates = [
        record
        for record in master.values()
        if record.get("link") and is_recent_or_active_record(record, window_days)
    ]
    candidates.sort(
        key=lambda record: (
            0 if record.get("status") != "completed" else 1,
            parse_record_date(record) or datetime.max.date(),
            str(record.get("id") or ""),
        )
    )
    log(f"same-day recovery: {len(candidates)} candidates in last {window_days} day(s)")

    for candidate_index, record in enumerate(candidates, start=1):
        match_id = str(record.get("id") or match_id_from_link(record.get("link")) or "")
        if not match_id:
            continue
        log(f"[recovery {candidate_index}/{len(candidates)}] {record_label(record)}")
        item: dict[str, Any] = {"id": match_id, "actions": [], "errors": []}

        if not has_usable_odds(record):
            try:
                log(f"[recovery {candidate_index}/{len(candidates)}] odds missing -> fetch")
                html = fetch_html(record["link"])
                raw_html = save_raw_html(run_dir, "recovery_match_page", match_id, record["link"], html)
                odds = parse_odds(html)
                output = run_dir / "same_day_recovery" / f"{match_id}_odds.json"
                payload = {
                    "match_id": match_id,
                    "captured_at": now_utc(),
                    "source": "same_day_gap_recovery",
                    "raw_html": raw_html,
                    "odds": odds,
                }
                write_json(output, payload)
                if append_odds_point(record, odds, run_dir, str(output.relative_to(DATA_ROOT))):
                    item["actions"].append("odds_recovered")
                    log(f"[recovery {candidate_index}/{len(candidates)}] odds recovered bookmakers={odds.get('bookmaker_count')}")
                else:
                    item["actions"].append("odds_still_unavailable")
                    log(f"[recovery {candidate_index}/{len(candidates)}] odds still unavailable")
            except Exception as exc:
                log(f"[recovery {candidate_index}/{len(candidates)}] odds ERROR {exc}")
                item["errors"].append({"kind": "odds", "error": str(exc)})
            time.sleep(delay)

        detail = record.get("detail")
        needs_detail = (
            not isinstance(detail, dict)
            or (detail.get("source") == "upcoming_row_fallback")
            or (record.get("status") == "completed" and not is_completed_detail(detail))
        )
        if needs_detail:
            try:
                log(f"[recovery {candidate_index}/{len(candidates)}] detail missing/stale -> fetch")
                detail_path = run_dir / "same_day_recovery" / f"{match_id}_detail.json"
                recovered_detail, logs = scrape_match_detail(record["link"], detail_path)
                if recovered_detail:
                    record["detail"] = recovered_detail
                    result = result_from_detail(recovered_detail)
                    if result:
                        record.update(result)
                    item["actions"].append("detail_recovered")
                    log(f"[recovery {candidate_index}/{len(candidates)}] detail recovered status={record.get('status')}")
                elif logs:
                    item["errors"].append({"kind": "detail", "error": logs[-500:]})
                    log(f"[recovery {candidate_index}/{len(candidates)}] detail unavailable")
            except Exception as exc:
                log(f"[recovery {candidate_index}/{len(candidates)}] detail ERROR {exc}")
                item["errors"].append({"kind": "detail", "error": str(exc)})
            time.sleep(delay)

        existing_analytics = record.get("analytics") or {}
        if capture_analytics and not existing_analytics.get("available"):
            try:
                log(f"[recovery {candidate_index}/{len(candidates)}] analytics missing -> fetch")
                analytics = scrape_match_analytics(
                    record["link"],
                    run_dir,
                    match_id,
                    team_names_for_analytics(record, record.get("detail")),
                )
                record["analytics"] = analytics
                if analytics.get("source_file"):
                    record["latest_analytics_file"] = analytics["source_file"]
                item["actions"].append("analytics_recovered" if analytics.get("available") else "analytics_still_unavailable")
            except Exception as exc:
                log(f"[recovery {candidate_index}/{len(candidates)}] analytics ERROR {exc}")
                item["errors"].append({"kind": "analytics", "error": str(exc)})
            time.sleep(delay)

        if item["actions"] or item["errors"]:
            record["last_gap_recovery_at"] = now_utc()
            recovery_index.append(item)

    summary = {
        "window_days": window_days,
        "candidates": len(candidates),
        "touched": len(recovery_index),
        "actions": sum(len(item.get("actions") or []) for item in recovery_index),
        "errors": sum(len(item.get("errors") or []) for item in recovery_index),
        "index": recovery_index,
    }
    write_json(run_dir / "same_day_recovery" / "index.json", summary)
    return summary


def roster_signature(squad: list[dict[str, Any]]) -> str:
    ids = sorted(str(player.get("id") or player.get("name") or "").strip() for player in squad)
    return "|".join([player_id for player_id in ids if player_id])


def update_roster_history(team_profiles: list[dict[str, Any]], run_dir: Path, captured_at: str) -> dict[str, Any]:
    history = read_json(MASTER_ROSTERS, {})
    updated = 0
    for item in team_profiles:
        profile = item.get("profile") or {}
        team_id = str(item.get("id") or "")
        if not team_id:
            continue
        squad = profile.get("squad") or []
        signature = roster_signature(squad)
        if not signature:
            continue
        entry = history.setdefault(
            team_id,
            {
                "team_id": team_id,
                "team_name": profile.get("name") or (item.get("source") or {}).get("name"),
                "snapshots": [],
            },
        )
        entry["team_name"] = profile.get("name") or entry.get("team_name")
        point = {
            "captured_at": captured_at,
            "run_id": run_dir.name,
            "signature": signature,
            "player_ids": [str(player.get("id")) for player in squad if player.get("id")],
            "player_names": [player.get("name") for player in squad if player.get("name")],
            "roster_size": len(squad),
        }
        existing = entry.setdefault("snapshots", [])
        if not existing or existing[-1].get("run_id") != point["run_id"]:
            existing.append(point)
            updated += 1
    write_json(MASTER_ROSTERS, history)
    return {"teams": len(history), "snapshots_added": updated, "file": str(MASTER_ROSTERS)}


def load_first_item(path: Path) -> Any:
    data = read_json(path, [])
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    return data


def is_completed_detail(detail: dict[str, Any]) -> bool:
    match = detail.get("match") or {}
    team1 = match.get("team1") or {}
    team2 = match.get("team2") or {}
    return str(team1.get("score") or "").isdigit() and str(team2.get("score") or "").isdigit()


def result_from_detail(detail: dict[str, Any]) -> dict[str, Any] | None:
    if not is_completed_detail(detail):
        return None
    match = detail["match"]
    score1 = int(match["team1"]["score"])
    score2 = int(match["team2"]["score"])
    return {
        "status": "completed",
        "completed_at": now_utc(),
        "score": {
            "team1": score1,
            "team2": score2,
        },
        "winner": match["team1"]["name"] if score1 > score2 else match["team2"]["name"],
        "detail": detail,
    }


def scrape_match_detail(
    match_link: str,
    output_path: Path,
    *,
    initial_html: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Obtiene el detalle; reutiliza el HTML ya descargado por el snapshot."""
    try:
        html = initial_html if initial_html is not None else fetch_html(match_link)
        detail = parse_match_detail_html(html)
        if detail and (detail.get("match") or {}).get("team1", {}).get("name"):
            write_json(output_path, [detail])
            return detail, "snapshot_html" if initial_html is not None else "direct_fetch"
    except Exception as exc:
        direct_error = str(exc)
    else:
        direct_error = "direct parser returned no match"

    match_path = match_link.removeprefix("/matches/")
    ok, logs = run_spider(
        "hltv_match",
        output_path,
        spider_args=["-a", f"match={match_path}"],
        timeout=600,
    )
    if not ok:
        return None, f"direct_fetch_failed={direct_error}\n{logs}"
    detail = load_first_item(output_path)
    if not isinstance(detail, dict) or not (detail.get("match") or {}).get("team1", {}).get("name"):
        return None, f"direct_fetch_failed={direct_error}\n{logs}"
    return detail, f"direct_fetch_failed={direct_error}\n{logs}"


def parse_event_metadata_from_match_html(html: str) -> dict[str, Any]:
    selector = Selector(text=html)
    link = selector.css(".teamsBox .event a::attr(href), .teamsBoxDropdown .event a::attr(href)").get()
    name = clean_text(selector.css(".teamsBox .event a::text, .teamsBox .event::text").get())
    event_id = None
    if link:
        match = re.search(r"/events/(\d+)", link)
        event_id = match.group(1) if match else None
    return {
        "name": name,
        "hltv_event_id": event_id,
        "source": "match_page",
    }


def parse_prematch_lineups_html(html: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extrae la alineación anunciada y sus stats visibles de la ficha HLTV.

    No se mezcla con `match_lineups`, que representa quién jugó de verdad una
    vez finalizada la serie. Esta es una foto pre-partido, append-only.
    """
    selector = Selector(text=html)
    detail_match = (detail or {}).get("match") or {}
    output: dict[str, Any] = {}
    for number in (1, 2):
        side = f"team{number}"
        raw = selector.css(f"[data-team{number}-players-data]::attr(data-team{number}-players-data)").get()
        if not raw:
            continue
        try:
            players_payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        players: list[dict[str, Any]] = []
        for player in players_payload.values() if isinstance(players_payload, dict) else []:
            player_id = str(player.get("playerId") or "").strip()
            if not player_id:
                continue
            players.append(
                {
                    "hltv_player_id": player_id,
                    "nickname": clean_text(player.get("nickname")),
                    "profile_link": player.get("profileLinkUrl"),
                    "stats_link": player.get("statsLinkUrl"),
                    "rating": parse_float(player.get("rating")),
                    "kpr": parse_float(player.get("kpr")),
                    "dpr": parse_float(player.get("dpr")),
                    "kast": parse_analytics_percentage(player.get("kast")),
                    "adr": parse_float(player.get("adr")),
                    "multi_kill_rating": parse_float(player.get("multiKillRating")),
                    "round_swing": parse_analytics_percentage(player.get("roundSwing")),
                    "is_standin": False,
                }
            )
        team = detail_match.get(side) or {}
        output[side] = {
            "team_name": team.get("name"),
            "hltv_team_id": str(team.get("id") or "") or None,
            "players": players,
        }
    return output


def mark_prematch_standins(lineups: dict[str, Any], analytics: dict[str, Any]) -> None:
    standins = analytics.get("standins") or {}
    if not isinstance(standins, dict):
        return
    for lineup in lineups.values():
        if not isinstance(lineup, dict):
            continue
        team_key = analytics_team_key(lineup.get("team_name"))
        standin_keys = {
            analytics_team_key(player)
            for team, players in standins.items()
            if analytics_team_key(team) == team_key
            for player in (players or [])
        }
        if not standin_keys:
            continue
        for player in lineup.get("players") or []:
            player["is_standin"] = analytics_team_key(player.get("nickname")) in standin_keys


def team_names_for_analytics(record: dict[str, Any], detail: dict[str, Any] | None = None) -> list[str]:
    detail_match = (detail or record.get("detail") or {}).get("match") or {}
    upcoming = record.get("upcoming_row") or {}
    return [
        (detail_match.get("team1") or {}).get("name")
        or (upcoming.get("team1") or {}).get("name")
        or "",
        (detail_match.get("team2") or {}).get("name")
        or (upcoming.get("team2") or {}).get("name")
        or "",
    ]


def scrape_team_profile(team: dict[str, Any], output_dir: Path) -> dict[str, Any] | None:
    team_id = team.get("id")
    link = team.get("link")
    if not team_id or not link:
        return None
    output = output_dir / f"{team_id}_{slug_from_link(link)}.json"
    if PF is not None:
        try:
            html = fetch_html(f"{link}#tab-matchesBox")
            profile = PF.get_parser("team_profile").parse(Selector(text=html))
            if isinstance(profile, dict) and profile.get("name"):
                write_json(output, [profile])
                return {"id": str(team_id), "source": team, "profile": profile}
        except Exception:
            pass
    ok, _logs = run_spider("hltv_team", output, spider_args=["-a", f"team={link}"], timeout=600)
    if not ok:
        return None
    profile = load_first_item(output)
    if not isinstance(profile, dict) or not profile.get("name"):
        return None
    return {"id": str(team_id), "source": team, "profile": profile}


def scrape_upcoming(run_dir: Path) -> list[dict[str, Any]]:
    output = run_dir / "raw" / "upcoming_matches.json"
    try:
        log("upcoming: fetching /matches")
        html = fetch_html("/matches")
        save_raw_html(run_dir, "upcoming_page", "matches", "/matches", html)
        data: list[dict[str, Any] | None] = []
        if PF is not None:
            selector = Selector(text=html)
            matches_sections = selector.css("div.matches-list-section")
            data = PF.get_parser("upcoming_matches").parse(matches_sections) or []
        data = [match for match in data if match and match.get("link")]
        if not data:
            data = parse_upcoming_matches_fallback(html)
        write_json(output, data)
        matches = [match for match in data if match and match.get("link")]
        log(f"upcoming: parsed {len(matches)} matches")
        return matches
    except Exception as exc:
        log(f"upcoming: direct fetch ERROR {exc}; trying scrapy fallback")
        write_json(run_dir / "logs" / "upcoming_direct_error.log.json", {"error": str(exc)})

    ok, logs = run_spider("hltv_upcoming_matches", output, timeout=600)
    if not ok:
        log("upcoming: scrapy fallback failed")
        write_json(run_dir / "logs" / "upcoming_error.log.json", {"logs": logs})
        return []
    data = read_json(output, [])
    matches = [match for match in data if match and match.get("link")]
    log(f"upcoming: scrapy fallback parsed {len(matches)} matches")
    return matches


def scrape_recent_results(
    run_dir: Path,
    pages: int = 3,
    target_ids: set[str] | None = None,
    target_info: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch `/results` only as much as the DB worklist needs.

    `target_ids is None` means legacy fallback: DB unavailable, keep the old
    bounded scan. An empty set means SQLite says there are no pending results,
    so skip `/results` completely.
    """
    results_by_id: dict[str, dict[str, Any]] = {}
    if target_ids is not None:
        target_ids = {str(item) for item in target_ids if item}
        if not target_ids:
            log("recent results: DB has no pending_result matches; skipping /results")
            write_json(run_dir / "recent_results_index.json", results_by_id)
            write_json(
                run_dir / "recent_results_manifest.json",
                {"mode": "db_targeted", "target_count": 0, "pages_fetched": 0, "skipped": True},
            )
            return results_by_id
        log(f"recent results: DB-targeted scan for {len(target_ids)} pending ids (max_pages={pages})")
    pages_fetched = 0
    for page in range(pages):
        offset = page * 100
        output = run_dir / "raw" / "recent_results" / f"results_offset_{offset}.json"
        try:
            log(f"recent results: fetching offset={offset} ({page + 1}/{pages})")
            html = fetch_html(f"/results?offset={offset}")
            pages_fetched += 1
            save_raw_html(run_dir, "results_page", f"results_offset_{offset}", f"/results?offset={offset}", html)
            data: list[dict[str, Any] | None] = []
            if PF is not None:
                selector = Selector(text=html)
                sublists = selector.css("div.allres .results-sublist")
                data = PF.get_parser("results").parse(sublists) or []
            data = [item for item in data if item and item.get("id")]
            if not data:
                data = parse_results_fallback(html)
            write_json(output, data)
            for item in data:
                match_id = str(item.get("id") or "")
                if match_id:
                    results_by_id[match_id] = item
            log(f"recent results: offset={offset} parsed {len(data)} rows")
            if target_ids is not None:
                missing = target_ids - set(results_by_id)
                log(f"recent results: DB targets found={len(target_ids) - len(missing)}/{len(target_ids)}")
                if not missing:
                    log("recent results: all DB pending ids found; stopping before older offsets")
                    break
                log(
                    "recent results: still missing "
                    + "; ".join(format_pending_missing(mid, target_info) for mid in sorted(missing)),
                    force=VERBOSE,
                )
            time.sleep(0.2)
            continue
        except Exception as exc:
            log(f"recent results: offset={offset} direct ERROR {exc}; trying scrapy fallback")
            write_json(run_dir / "logs" / f"recent_results_direct_error_{offset}.log.json", {"error": str(exc)})

        ok, _logs = run_spider("hltv_results", output, spider_args=["-a", f"offset={offset}"], timeout=600)
        if not ok:
            log(f"recent results: offset={offset} scrapy fallback failed")
            continue
        pages_fetched += 1
        fallback_rows = read_json(output, [])
        for item in fallback_rows:
            match_id = str(item.get("id") or "")
            if match_id:
                results_by_id[match_id] = item
        log(f"recent results: offset={offset} scrapy fallback parsed {len(fallback_rows)} rows")
        if target_ids is not None:
            missing = target_ids - set(results_by_id)
            log(f"recent results: DB targets found={len(target_ids) - len(missing)}/{len(target_ids)}")
            if not missing:
                log("recent results: all DB pending ids found; stopping before older offsets")
                break
            log(
                "recent results: still missing "
                + "; ".join(format_pending_missing(mid, target_info) for mid in sorted(missing)),
                force=VERBOSE,
            )
    write_json(run_dir / "recent_results_index.json", results_by_id)
    write_json(
        run_dir / "recent_results_manifest.json",
        {
            "mode": "legacy_scan" if target_ids is None else "db_targeted",
            "target_count": None if target_ids is None else len(target_ids),
            "targets_found": None if target_ids is None else len(target_ids & set(results_by_id)),
            "pages_fetched": pages_fetched,
            "max_pages": pages,
        },
    )
    log(f"recent results: unique matches={len(results_by_id)}")
    return results_by_id


def update_pending_matches(
    master: dict[str, Any],
    run_dir: Path,
    recent_results: dict[str, dict[str, Any]],
    capture_analytics: bool = True,
) -> list[dict[str, Any]]:
    updates = []
    db_pending = db_pending_match_ids()
    if db_pending is None:
        pending = [record for record in master.values() if record.get("status") != "completed" and record.get("link")]
        log("pending updates: DB unavailable, using master JSON fallback")
    else:
        pending = [
            record for record in master.values()
            if str(record.get("id") or "") in db_pending and record.get("link")
        ]
        skipped = max(0, len(db_pending) - len(pending))
        if skipped:
            note_freshness_skip(skipped)
            log(f"pending updates: {skipped} DB pending ids not present in master JSON")
    log(f"pending updates: {len(pending)} pending matches")
    for index, record in enumerate(pending, start=1):
        match_id = record["id"]
        log(f"[pending {index}/{len(pending)}] {record_label(record)}")
        update_item: dict[str, Any] = {"id": match_id}
        if str(match_id) not in recent_results:
            update_item["status"] = "still_pending_recent_results_miss"
            updates.append(update_item)
            log(f"[pending {index}/{len(pending)}] not found in recent /results; detail skipped")
            continue
        try:
            log(f"[pending {index}/{len(pending)}] odds snapshot")
            odds_html = fetch_html(record["link"])
            raw_html = save_raw_html(run_dir, "pending_match_page", match_id, record["link"], odds_html)
            odds = parse_odds(odds_html)
            odds_output = run_dir / "updated_pending_details" / f"{match_id}_odds.json"
            write_json(
                odds_output,
                {
                    "match_id": match_id,
                    "captured_at": now_utc(),
                    "raw_html": raw_html,
                    "odds": odds,
                },
            )
            if append_odds_point(record, odds, run_dir, str(odds_output.relative_to(DATA_ROOT))):
                update_item["odds"] = "captured"
                log(f"[pending {index}/{len(pending)}] odds captured bookmakers={odds.get('bookmaker_count')}")
            elif odds.get("available"):
                update_item["odds"] = "duplicate"
                log(f"[pending {index}/{len(pending)}] odds duplicate bookmakers={odds.get('bookmaker_count')}")
            else:
                update_item["odds"] = "unavailable"
                log(f"[pending {index}/{len(pending)}] odds unavailable")
        except Exception as exc:
            update_item["odds_error"] = str(exc)
            log(f"[pending {index}/{len(pending)}] odds ERROR {exc}")

        log(f"[pending {index}/{len(pending)}] detail/result snapshot")
        detail_path = run_dir / "updated_pending_details" / f"{match_id}.json"
        detail, logs = scrape_match_detail(record["link"], detail_path)
        result = result_from_detail(detail) if detail else None

        existing_analytics = record.get("analytics") or {}
        if capture_analytics and not existing_analytics.get("available"):
            analytics = scrape_match_analytics(
                record["link"],
                run_dir,
                match_id,
                team_names_for_analytics(record, detail),
            )
            record["analytics"] = analytics
            if analytics.get("source_file"):
                record["latest_analytics_file"] = analytics["source_file"]

        fallback = recent_results.get(match_id)
        if result is None and fallback:
            score1 = parse_float(fallback.get("team1", {}).get("score"))
            score2 = parse_float(fallback.get("team2", {}).get("score"))
            if score1 is not None and score2 is not None:
                result = {
                    "status": "completed",
                    "completed_at": now_utc(),
                    "score": {"team1": int(score1), "team2": int(score2)},
                    "winner": fallback["team1"]["name"] if score1 > score2 else fallback["team2"]["name"],
                    "result_row": fallback,
                }

        if result:
            record.update(result)
            if record.get("latest_odds_point") and not record.get("closing_odds"):
                closing = observed_closing_odds(record, record["latest_odds_point"])
                if closing:
                    record["closing_odds"] = closing
            record["last_updated_at"] = now_utc()
            update_item.update({"status": "completed", "winner": result.get("winner")})
            updates.append(update_item)
        elif logs:
            update_item.update({"status": "still_pending", "logs_tail": logs[-500:]})
            updates.append(update_item)
        elif len(update_item) > 1:
            update_item["status"] = "still_pending"
            updates.append(update_item)
        log(f"[pending {index}/{len(pending)}] status={update_item.get('status', record.get('status'))}")
        time.sleep(0.2)
    write_json(run_dir / "pending_updates.json", updates)
    log(f"pending updates: touched={len(updates)}")
    return updates


def collect_teams_from_details(details: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    teams: dict[str, dict[str, Any]] = {}
    for detail in details:
        match = detail.get("match") or {}
        for key in ("team1", "team2"):
            team = match.get(key) or {}
            team_id = team.get("id")
            if team_id and team.get("link"):
                teams[str(team_id)] = team
    return teams


def collect_team_profiles(teams: dict[str, dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    profiles = []
    output_dir = run_dir / "team_profiles"
    team_list = []
    skipped = 0
    cache_hits = 0
    for team in teams.values():
        team_id = str(team.get("id") or "")
        if db_entity_is_fresh("team_profile", team_id):
            cached = db_latest_team_profile(team_id)
            if cached:
                profiles.append(cached)
                skipped += 1
                cache_hits += 1
                continue
        team_list.append(team)
    if skipped:
        note_freshness_skip(skipped)
    log(f"team profiles: {len(team_list)} teams to fetch; cache_hits={cache_hits}; skipped_fresh={skipped}")
    for index, team in enumerate(team_list, start=1):
        log(f"[team {index}/{len(team_list)}] {team.get('name') or team.get('id') or '?'}")
        profile = scrape_team_profile(team, output_dir)
        team_id = str(team.get("id") or "")
        if profile:
            profiles.append(profile)
            db_mark_fetch_state("team_profile", team_id, "ok")
            log(f"[team {index}/{len(team_list)}] ok")
        else:
            db_mark_fetch_state("team_profile", team_id, "error", "team profile unavailable")
            log(f"[team {index}/{len(team_list)}] unavailable")
        time.sleep(0.2)
    write_json(run_dir / "team_profiles.json", profiles)
    log(f"team profiles: collected={len(profiles)} fetched={len(team_list)} cache_hits={cache_hits} skipped_fresh={skipped}")
    return profiles


def player_stat_value_present(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "-", "--", "n/a", "na"}


def player_snapshot_has_core_stats(rows: list[dict[str, Any]]) -> bool:
    return any(
        all(player_stat_value_present((player.get("stats") or {}).get(label)) for label in PLAYER_CORE_STAT_LABELS)
        for player in rows
    )


def player_snapshot_is_known_unavailable(rows: list[dict[str, Any]]) -> bool:
    for player in rows:
        try:
            if int(player.get("maps")) <= 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def choose_cached_player_snapshot(rows: list[dict[str, Any]], min_maps: int = 10) -> dict[str, Any] | None:
    complete = [dict(player) for player in rows if player_snapshot_has_core_stats([player])]
    priorities = {"past3months": 0, "past6months": 1, "past12months": 2}

    def sort_key(player: dict[str, Any]) -> tuple[int, int]:
        time_filter = str(player.get("time_filter") or "")
        try:
            maps = int(player.get("maps") or 0)
        except (TypeError, ValueError):
            maps = 0
        return priorities.get(time_filter, 10), -maps

    representative = sorted(
        (
            player
            for player in complete
            if isinstance(player.get("maps"), (int, float)) and int(player["maps"]) >= min_maps
        ),
        key=sort_key,
    )
    if representative:
        selected = representative[0]
        selected["selection_reason"] = f"cached_recent_window_with_at_least_{min_maps}_maps"
        selected["fetch_origin"] = "cache"
        return selected
    if complete:
        selected = max(complete, key=lambda player: int(player.get("maps") or 0))
        selected["selection_reason"] = f"cached_max_maps_below_{min_maps}"
        selected["fetch_origin"] = "cache"
        return selected
    unavailable = [dict(player) for player in rows if player_snapshot_is_known_unavailable([player])]
    if unavailable:
        selected = sorted(unavailable, key=sort_key)[0]
        selected["stats"] = {}
        selected["selection_reason"] = "cached_known_unavailable"
        selected["fetch_origin"] = "cache"
        return selected
    return None


def player_snapshot_quality(snapshots: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    field_coverage = {
        label: sum(
            1
            for rows in snapshots.values()
            if any(player_stat_value_present((player.get("stats") or {}).get(label)) for player in rows)
        )
        for label in PLAYER_CORE_STAT_LABELS
    }
    complete_players = sum(1 for rows in snapshots.values() if player_snapshot_has_core_stats(rows))
    unavailable_players = sum(
        1
        for rows in snapshots.values()
        if not player_snapshot_has_core_stats(rows) and player_snapshot_is_known_unavailable(rows)
    )
    return {
        "core_complete_players": complete_players,
        "known_unavailable_players": unavailable_players,
        "resolved_players": complete_players + unavailable_players,
        "field_coverage": field_coverage,
    }


def team_profiles_players_are_fresh(team_profiles_file: Path) -> tuple[bool, int, int]:
    profiles = read_json(team_profiles_file, [])
    if not profiles:
        return True, 0, 0
    player_ids = player_ids_from_team_profiles_payload(profiles)
    if not player_ids:
        return False, 0, 0
    fresh_ids = db_fresh_entity_keys("player_stats", player_ids)
    fresh = len(fresh_ids)
    snapshots = db_latest_player_snapshots(player_ids)
    resolved = sum(
        1
        for player_id in player_ids
        if player_snapshot_has_core_stats(snapshots.get(player_id, []))
        or player_snapshot_is_known_unavailable(snapshots.get(player_id, []))
    )
    return fresh == len(player_ids) and resolved == len(player_ids), fresh, resolved


def player_ids_from_team_profiles_payload(profiles: list[dict[str, Any]]) -> set[str]:
    player_ids: set[str] = set()
    for profile in profiles:
        for player in ((profile.get("profile") or {}).get("squad") or []):
            player_id = str(player.get("id") or "")
            if player_id:
                player_ids.add(player_id)
    return player_ids


def player_stats_profiles_with_announced_lineups(
    profiles: list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Add match-specific lineup players to the player-stat candidate set.

    The result is intentionally separate from ``team_profiles.json``: an
    announced lineup is authoritative for that match, but must not rewrite the
    team's persistent roster history.
    """
    expanded = copy.deepcopy(profiles)
    by_team_id = {
        str(item.get("id") or ""): item
        for item in expanded
        if str(item.get("id") or "")
    }
    lineup_players_seen: set[str] = set()
    players_added: set[str] = set()
    synthetic_teams = 0

    for snapshot in snapshots:
        for lineup in (snapshot.get("prematch_lineups") or {}).values():
            if not isinstance(lineup, dict):
                continue
            team_id = str(lineup.get("hltv_team_id") or "")
            if not team_id:
                continue
            item = by_team_id.get(team_id)
            if item is None:
                item = {
                    "id": team_id,
                    "profile": {
                        "name": lineup.get("team_name"),
                        "squad": [],
                        "source": "prematch_announced_lineup_candidate",
                    },
                }
                expanded.append(item)
                by_team_id[team_id] = item
                synthetic_teams += 1
            profile = item.setdefault("profile", {})
            squad = profile.setdefault("squad", [])
            existing_ids = {
                str(player.get("id") or "")
                for player in squad
                if str(player.get("id") or "")
            }
            for player in lineup.get("players") or []:
                player_id = str(player.get("hltv_player_id") or "")
                if not player_id:
                    continue
                lineup_players_seen.add(player_id)
                if player_id in existing_ids:
                    continue
                squad.append(
                    {
                        "id": player_id,
                        "name": player.get("nickname") or player_id,
                        "link": player.get("profile_link"),
                        "source": "prematch_announced_lineup_candidate",
                    }
                )
                existing_ids.add(player_id)
                players_added.add(player_id)

    return expanded, {
        "lineup_players_seen": len(lineup_players_seen),
        "lineup_players_added": len(players_added),
        "synthetic_teams_added": synthetic_teams,
        "players_requested": len(player_ids_from_team_profiles_payload(expanded)),
    }


def player_ids_from_team_profiles(team_profiles_file: Path) -> set[str]:
    return player_ids_from_team_profiles_payload(read_json(team_profiles_file, []))


def materialize_cached_player_stats(team_profiles_file: Path, output: Path, year: int) -> dict[str, Any] | None:
    player_ids = player_ids_from_team_profiles(team_profiles_file)
    snapshots = db_latest_player_snapshots(player_ids)
    selected = {
        player_id: player
        for player_id, rows in snapshots.items()
        if (player := choose_cached_player_snapshot(rows)) is not None
    }
    covered_ids = set(selected)
    if not covered_ids:
        return None
    results = [{
        "time_filter": "adaptive",
        "selection_mode": "adaptive_recent_min_maps",
        "players": sorted(selected.values(), key=lambda item: str(item.get("id") or "")),
        "source": "BBDD.player_stat_snapshots",
    }]
    quality = player_snapshot_quality({
        player_id: [player]
        for player_id, player in selected.items()
    })
    payload = {
        "ok": True,
        "source": "BBDD.player_stat_snapshots",
        "materialized_from_cache": True,
        "year": year,
        "captured_at": now_utc(),
        "players_requested": len(player_ids),
        "players_loaded": len(covered_ids),
        "coverage": len(covered_ids) / max(len(player_ids), 1),
        "core_complete_players": quality["core_complete_players"],
        "known_unavailable_players": quality["known_unavailable_players"],
        "field_coverage": quality["field_coverage"],
        "comparisons_collected": len(results),
        "comparisons_failed": 0,
        "results": results,
    }
    write_json(output, payload)
    return payload


def filter_team_profiles_players(
    profiles: list[dict[str, Any]],
    player_ids: set[str],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for item in profiles:
        profile = dict(item.get("profile") or {})
        squad = [
            player
            for player in profile.get("squad") or []
            if str(player.get("id") or "") in player_ids
        ]
        if not squad:
            continue
        copied = dict(item)
        profile["squad"] = squad
        copied["profile"] = profile
        filtered.append(copied)
    return filtered


def flatten_player_results(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    players: dict[str, dict[str, Any]] = {}
    for comparison in payload.get("results") or []:
        for raw_player in comparison.get("players") or []:
            player_id = str(raw_player.get("id") or "")
            if player_id:
                players[player_id] = dict(raw_player)
    return players


def merge_player_stats_payload(
    *,
    year: int,
    requested_ids: set[str],
    cached_players: dict[str, dict[str, Any]],
    fetched_payload: dict[str, Any],
    refresh_ids: set[str] | None = None,
) -> dict[str, Any]:
    fetched_players = flatten_player_results(fetched_payload)
    captured_at = now_utc()
    for player in fetched_players.values():
        player["fetch_origin"] = "online"
        player["captured_at"] = captured_at
    merged = dict(cached_players)
    merged.update(fetched_players)
    refresh_ids = requested_ids if refresh_ids is None else refresh_ids
    stale_fallback = sorted((refresh_ids - set(fetched_players)) & set(cached_players))
    return {
        "source": "https://www.hltv.org/stats/players/compare",
        "year": year,
        "time_filters": fetched_payload.get("time_filters") or ["past3months", "past6months", "past12months"],
        "selection_mode": "adaptive_recent_min_maps",
        "min_maps": int(fetched_payload.get("min_maps") or 10),
        "players_requested": len(requested_ids),
        "players_loaded": len(merged),
        "players_fetched_online": len(fetched_players),
        "players_loaded_from_cache": len(set(merged) - set(fetched_players)),
        "players_stale_fallback": len(stale_fallback),
        "stale_fallback_ids": stale_fallback,
        "comparisons_collected": int(fetched_payload.get("comparisons_collected") or 0),
        "comparisons_failed": int(fetched_payload.get("comparisons_failed") or 0),
        "requests_attempted": int(fetched_payload.get("requests_attempted") or 0),
        "stopped_reason": fetched_payload.get("stopped_reason"),
        "failures": fetched_payload.get("failures") or [],
        "results": [{
            "time_filter": "adaptive",
            "selection_mode": "adaptive_recent_min_maps",
            "min_maps": int(fetched_payload.get("min_maps") or 10),
            "players": sorted(merged.values(), key=lambda item: str(item.get("id") or "")),
        }],
    }


def collect_player_stats(team_profiles_file: Path, run_dir: Path, year: int, delay: float) -> dict[str, Any]:
    output = run_dir / f"player_compare_stats_{year}.json"
    if not team_profiles_file.exists():
        log("player stats: missing team_profiles.json")
        return {"ok": False, "reason": "missing_team_profiles"}
    profiles = read_json(team_profiles_file, [])
    requested_ids = player_ids_from_team_profiles_payload(profiles)
    snapshots = db_latest_player_snapshots(requested_ids)
    fresh_ids = db_fresh_entity_keys("player_stats", requested_ids)
    resolved_ids = {
        player_id
        for player_id, rows in snapshots.items()
        if player_snapshot_has_core_stats(rows) or player_snapshot_is_known_unavailable(rows)
    }
    reusable_ids = fresh_ids & resolved_ids
    fetch_ids = requested_ids - reusable_ids
    fresh_count = len(fresh_ids)
    complete_count = len(resolved_ids)
    if not fetch_ids:
        cached = materialize_cached_player_stats(team_profiles_file, output, year)
        if cached:
            note_freshness_skip(fresh_count)
            log(
                "player stats: materialized from BBDD cache; "
                f"players_fresh={fresh_count} players_loaded={cached.get('players_loaded')} "
                f"core_complete={cached.get('core_complete_players')} "
                f"known_unavailable={cached.get('known_unavailable_players')}"
            )
            return {
                "ok": True,
                "skipped_by_freshness": True,
                "file": str(output),
                "players_fresh": fresh_count,
                "players_requested": cached.get("players_requested"),
                "players_loaded": cached.get("players_loaded"),
                "coverage": cached.get("coverage"),
                "core_complete_players": cached.get("core_complete_players"),
                "known_unavailable_players": cached.get("known_unavailable_players"),
                "field_coverage": cached.get("field_coverage"),
                "comparisons_collected": cached.get("comparisons_collected"),
                "comparisons_failed": 0,
                "source": "BBDD.player_stat_snapshots",
            }
        log("player stats: freshness said ok, but BBDD cache was empty; fetching")
    elif fresh_count == len(requested_ids) and complete_count < fresh_count:
        log(
            "player stats: fresh cache has unresolved core fields; forcing profile refresh "
            f"resolved={complete_count}/{fresh_count}",
            force=True,
        )
    cached_players = {
        player_id: player
        for player_id, rows in snapshots.items()
        if (player := choose_cached_player_snapshot(rows)) is not None
    }
    filtered_profiles_file = run_dir / "_player_stats_fetch_profiles.json"
    scraper_output = run_dir / f"_player_stats_fetch_{year}.json"
    write_json(filtered_profiles_file, filter_team_profiles_players(profiles, fetch_ids))
    log(
        "player stats: selective refresh "
        f"requested={len(requested_ids)} reusable_cache={len(reusable_ids)} "
        f"fetch_online={len(fetch_ids)} stale_cache_fallback={len(fetch_ids & set(cached_players))}"
    )

    def refresh_cf_session(reason: str) -> tuple[bool, str]:
        helper = SCRAPY_ROOT / "hltv_scraper" / "grab_cf.py"
        log(f"player stats: refreshing cf_session ({reason})", force=True)
        return run_cmd([str(PYTHON_EXE), str(helper)], SCRAPER_PROJECT, timeout=240, stream=True)

    def should_refresh_cf(payload: dict[str, Any], logs: str, ok: bool) -> bool:
        text = " ".join(
            [
                str(payload.get("stopped_reason") or ""),
                str(payload.get("reason") or ""),
                logs or "",
            ]
        ).lower()
        if any(marker in text for marker in ("cloudflare", "challenge", "cf_session", "cf_clearance", "403")):
            return True
        return (not ok) and int(payload.get("comparisons_collected") or 0) == 0

    if not CF_SESSION.exists():
        try:
            ok_refresh, refresh_logs = refresh_cf_session("missing_cf_session")
            if not ok_refresh or not CF_SESSION.exists():
                return {
                    "ok": False,
                    "reason": "cf_session_missing_and_refresh_failed",
                    "logs_tail": refresh_logs[-1000:],
                }
        except Exception as exc:
            log(f"player stats: cf_session refresh ERROR {exc}")
            return {"ok": False, "reason": f"cf_session_missing_and_refresh_failed: {exc}"}

    cmd = [
        str(PYTHON_EXE),
        str(COMPARE_SCRIPT),
        "--team-profiles-file",
        str(filtered_profiles_file),
        "--output-file",
        str(scraper_output),
        "--year",
        str(year),
        "--time-filters",
        "past3months,past6months,past12months",
        "--adaptive-time-filter",
        "--min-maps",
        "10",
        "--limit",
        "1000",
        "--delay",
        str(delay),
        "--max-retries",
        "3",
        "--retry-base-delay",
        "3.0",
        "--retry-max-delay",
        "90.0",
        "--max-requests",
        "1000",
        "--max-cloudflare-streak",
        "1",
    ]
    if VERBOSE:
        cmd.append("--verbose")
    log(f"player stats: starting compare scrape year={year} delay={delay}s")
    ok, logs = run_cmd(cmd, SCRAPER_PROJECT, timeout=1800, stream=VERBOSE)
    fetched_payload = read_json(scraper_output, {}) if scraper_output.exists() else {}
    cf_session_refresh_attempted = False
    cf_session_refresh_ok = None
    first_stopped_reason = fetched_payload.get("stopped_reason")
    if should_refresh_cf(fetched_payload, logs, ok):
        cf_session_refresh_attempted = True
        refresh_ok, refresh_logs = refresh_cf_session(str(fetched_payload.get("stopped_reason") or "compare_problem"))
        cf_session_refresh_ok = refresh_ok
        logs = (logs or "") + "\n[cf_refresh]\n" + (refresh_logs or "")
        if refresh_ok:
            log("player stats: retrying compare scrape after cf_session refresh", force=True)
            ok, retry_logs = run_cmd(cmd, SCRAPER_PROJECT, timeout=1800, stream=VERBOSE)
            logs = (logs or "") + "\n[retry_after_cf_refresh]\n" + (retry_logs or "")
            fetched_payload = read_json(scraper_output, {}) if scraper_output.exists() else fetched_payload
    payload = merge_player_stats_payload(
        year=year,
        requested_ids=requested_ids,
        cached_players=cached_players,
        fetched_payload=fetched_payload,
        refresh_ids=fetch_ids,
    )
    write_json(output, payload)
    collected = int(fetched_payload.get("comparisons_collected") or 0)
    failed = int(fetched_payload.get("comparisons_failed") or 0)
    stopped_reason = fetched_payload.get("stopped_reason")
    loaded = int(payload.get("players_loaded") or 0)
    complete_coverage = loaded == len(requested_ids)
    semantic_ok = not stopped_reason and failed == 0 and complete_coverage
    log(
        f"player stats: collected={collected} failed={failed} loaded={loaded}/{len(requested_ids)} "
        f"online={payload.get('players_fetched_online')} cache={payload.get('players_loaded_from_cache')} "
        f"stopped={stopped_reason or 'no'} ok={semantic_ok}"
    )
    problem_text = " ".join(
        [
            str(fetched_payload.get("stopped_reason") or ""),
            str(fetched_payload.get("reason") or ""),
            logs or "",
        ]
    ).lower()
    if stopped_reason and collected == 0:
        status = "blocked" if any(marker in problem_text for marker in ("cloudflare", "challenge", "403", "cf_session")) else "error"
        note = str(stopped_reason or fetched_payload.get("reason") or "player compare scrape failed")
        for player_id in fetch_ids:
            db_mark_fetch_state("player_stats", player_id, status, note[:500])
    return {
        "ok": semantic_ok,
        "partial": not semantic_ok and loaded > 0,
        "file": str(output),
        "players_requested": len(requested_ids),
        "players_loaded": loaded,
        "players_fetched_online": payload.get("players_fetched_online"),
        "players_loaded_from_cache": payload.get("players_loaded_from_cache"),
        "players_stale_fallback": payload.get("players_stale_fallback"),
        "comparisons_collected": collected,
        "comparisons_failed": failed,
        "stopped_reason": stopped_reason,
        "first_stopped_reason": first_stopped_reason,
        "selection_mode": fetched_payload.get("selection_mode"),
        "min_maps": fetched_payload.get("min_maps"),
        "requests_attempted": fetched_payload.get("requests_attempted"),
        "cf_session_refresh_attempted": cf_session_refresh_attempted,
        "cf_session_refresh_ok": cf_session_refresh_ok,
        "cf_session_refresh_hint": (
            ".\\SCRAPER\\hltv-scraper-api\\.venv\\Scripts\\python.exe "
            ".\\SCRAPER\\hltv-scraper-api\\hltv_scraper\\hltv_scraper\\grab_cf.py"
            if "Cloudflare" in str(stopped_reason or logs)
            else None
        ),
        "logs_tail": logs[-1000:],
    }


def asset_file_exists(meta: dict[str, Any]) -> bool:
    source_file = meta.get("source_file")
    return bool(source_file and (DATA_ROOT / source_file).exists())


def collect_completed_match_assets(
    master: dict[str, Any],
    run_dir: Path,
    limit: int,
    delay: float,
) -> dict[str, Any]:
    candidates = []
    skipped_fresh = 0
    for record in master.values():
        if record.get("status") != "completed" or not record.get("link"):
            continue
        meta = record.get("hltv_assets") or {}
        if meta.get("status") == "ok" and asset_file_exists(meta):
            continue
        match_id = str(record.get("id") or match_id_from_link(record.get("link")) or "")
        if db_match_has_coverage(match_id, "has_box_score") or db_entity_is_fresh("match_assets", match_id):
            skipped_fresh += 1
            continue
        candidates.append(record)
    if skipped_fresh:
        note_freshness_skip(skipped_fresh)

    if limit > 0:
        candidates = candidates[:limit]

    index = []
    log(f"completed assets: {len(candidates)} missing/partial completed matches (limit={limit}) skipped_fresh={skipped_fresh}")
    for item_index, record in enumerate(candidates, start=1):
        match_id = str(record.get("id") or match_id_from_link(record.get("link")) or "")
        if not match_id:
            continue
        log(f"[assets {item_index}/{len(candidates)}] {record_label(record)}")
        output = run_dir / "match_assets" / match_id / "assets.json"
        try:
            assets = scrape_match_assets(record["link"], output, delay=delay)
            source_file = str(output.relative_to(DATA_ROOT))
            meta = {
                "status": "ok" if not assets.get("errors") else "partial",
                "captured_at": assets.get("captured_at"),
                "run_id": run_dir.name,
                "source_file": source_file,
                "veto_steps": len((assets.get("veto") or {}).get("steps") or []),
                "mapstats_links": len(assets.get("mapstats_links") or []),
                "mapstats_maps": len(assets.get("mapstats") or []),
                "errors": assets.get("errors") or [],
            }
            record["hltv_assets"] = meta
            index.append({"id": match_id, **meta})
            db_status = "ok" if meta["status"] == "ok" else "partial"
            db_mark_fetch_state("match_assets", match_id, db_status, f"maps={meta['mapstats_maps']} errors={len(meta['errors'])}")
            log(f"[assets {item_index}/{len(candidates)}] status={meta['status']} maps={meta['mapstats_maps']} errors={len(meta['errors'])}")
        except Exception as exc:
            text = str(exc).lower()
            status = "blocked" if any(marker in text for marker in ("cloudflare", "challenge", "403", "cf_", "turnstile")) else "error"
            db_mark_fetch_state("match_assets", match_id, status, str(exc)[:500])
            log(f"[assets {item_index}/{len(candidates)}] ERROR {exc}")
            index.append({"id": match_id, "status": status, "error": str(exc)})
        time.sleep(delay)

    summary = {
        "eligible_missing": len(candidates),
        "captured": sum(1 for item in index if item.get("status") in {"ok", "partial"}),
        "errors": sum(1 for item in index if item.get("status") == "error"),
        "skipped_by_freshness": skipped_fresh,
        "limit": limit,
        "index": index,
    }
    write_json(run_dir / "match_assets_index.json", summary)
    log(f"completed assets: captured={summary['captured']} errors={summary['errors']}")
    return summary


def parse_ranking_html(html: str, ranking_type: str) -> dict[str, Any]:
    selector = Selector(text=html)
    date_text = selector.css("div.regional-ranking-header-text::text").get()
    ranked_teams = selector.css("div.ranked-team.standard-box")
    ranking = []
    for team in ranked_teams:
        link = team.css(".lineup-con .more a.moreLink::attr(href)").get()
        points_text = team.css("span.points::text").get()
        ranking.append(
            {
                "position": parse_int(team.css("span.position::text").get()),
                "name": team.css(".name::text").get(),
                "id": link.split("/")[2] if link and link.startswith("/team/") else None,
                "link": link,
                "logo": team.css("span.team-logo img::attr(src)").get(),
                "points": parse_int(points_text),
                "points_raw": points_text,
                "players": team.css("div.playersLine .rankingNicknames span::text").getall(),
            }
        )
    return {
        "ranking_type": ranking_type,
        "date_text": date_text,
        "captured_at": now_utc(),
        "source": "https://www.hltv.org/ranking/teams" if ranking_type == "hltv" else "https://www.hltv.org/valve-ranking/teams",
        "ranking": [item for item in ranking if item.get("name")],
    }


def collect_rankings(run_dir: Path) -> dict[str, Any]:
    output_dir = run_dir / "rankings"
    summary: dict[str, Any] = {}
    for ranking_type, link in {
        "hltv": "/ranking/teams",
        "valve": "/valve-ranking/teams",
    }.items():
        entity_type = "ranking_hltv" if ranking_type == "hltv" else "ranking_valve"
        if db_entity_is_fresh(entity_type, "global"):
            note_freshness_skip()
            log(f"ranking {ranking_type}: skipped by fetch_state freshness")
            summary[ranking_type] = {"ok": True, "skipped_by_freshness": True, "count": 0}
            continue
        try:
            log(f"ranking {ranking_type}: fetching {link}")
            html = fetch_html(link)
            raw_html = save_raw_html(run_dir, "ranking", ranking_type, link, html)
            payload = parse_ranking_html(html, ranking_type)
            payload["raw_html"] = raw_html
            output = output_dir / f"{ranking_type}.json"
            write_json(output, payload)
            summary[ranking_type] = {
                "ok": True,
                "count": len(payload.get("ranking") or []),
                "file": str(output.relative_to(DATA_ROOT)),
                "captured_at": payload.get("captured_at"),
            }
            log(f"ranking {ranking_type}: count={len(payload.get('ranking') or [])}")
        except Exception as exc:
            log(f"ranking {ranking_type}: ERROR {exc}")
            summary[ranking_type] = {"ok": False, "error": str(exc)}
    write_json(run_dir / "rankings_index.json", summary)
    return summary


def build_data_quality_report(
    run_dir: Path,
    manifest: dict[str, Any],
    master: dict[str, Any],
    snapshots: list[dict[str, Any]],
    team_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    odds_available = sum(1 for snapshot in snapshots if (snapshot.get("odds") or {}).get("available"))
    analytics_available = sum(1 for snapshot in snapshots if (snapshot.get("analytics") or {}).get("available"))
    pending = sum(1 for record in master.values() if record.get("status") != "completed")
    completed = sum(1 for record in master.values() if record.get("status") == "completed")
    completed_with_assets = sum(
        1
        for record in master.values()
        if record.get("status") == "completed" and asset_file_exists(record.get("hltv_assets") or {})
    )
    profile_roster_sizes = [
        len((profile.get("profile") or {}).get("squad") or [])
        for profile in team_profiles
    ]
    raw_html_files = list((run_dir / "raw_html").glob("**/*.html.gz"))
    analytics_files = list((run_dir / "analytics").glob("*.json"))
    player_stats_step = (manifest.get("steps") or {}).get("player_stats") or {}
    rankings_step = (manifest.get("steps") or {}).get("rankings") or {}
    match_assets_step = (manifest.get("steps") or {}).get("match_assets") or {}
    recovery_step = (manifest.get("steps") or {}).get("same_day_recovery") or {}
    report = {
        "run_id": run_dir.name,
        "generated_at": now_utc(),
        "summary": {
            "upcoming_matches": len(snapshots),
            "odds_available": odds_available,
            "odds_coverage": odds_available / len(snapshots) if snapshots else 0.0,
            "analytics_available": analytics_available,
            "analytics_coverage": analytics_available / len(snapshots) if snapshots else 0.0,
            "analytics_files": len(analytics_files),
            "team_profiles": len(team_profiles),
            "avg_roster_size": (sum(profile_roster_sizes) / len(profile_roster_sizes)) if profile_roster_sizes else None,
            "master_completed": completed,
            "master_pending": pending,
            "completed_with_assets": completed_with_assets,
            "completed_asset_coverage": completed_with_assets / completed if completed else 0.0,
            "raw_html_files": len(raw_html_files),
        },
        "player_stats": player_stats_step,
        "rankings": rankings_step,
        "match_assets": match_assets_step,
        "same_day_recovery": recovery_step,
        "warnings": [],
    }
    if snapshots and odds_available / len(snapshots) < 0.5:
        report["warnings"].append("odds_coverage_below_50pct")
    if completed and completed_with_assets / completed < 0.8:
        report["warnings"].append("completed_asset_coverage_below_80pct")
    if player_stats_step and not player_stats_step.get("ok"):
        report["warnings"].append("player_stats_not_ok")
    if any(not value.get("ok") for value in rankings_step.values() if isinstance(value, dict)):
        report["warnings"].append("ranking_capture_error")
    if recovery_step.get("errors"):
        report["warnings"].append("same_day_recovery_errors")
    write_json(run_dir / "data_quality_report.json", report)
    return report


def iter_completed_details(master: dict[str, Any]) -> list[dict[str, Any]]:
    details = []
    for record in master.values():
        result = record.get("result") or record
        detail = result.get("detail") if isinstance(result, dict) else None
        if isinstance(detail, dict):
            details.append(detail)
        elif isinstance(record.get("detail"), dict):
            details.append(record["detail"])
    return details


def compute_map_winrates(master: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    teams: dict[str, dict[str, dict[str, int]]] = {}
    for record in master.values():
        detail = record.get("detail") or (record.get("result") or {}).get("detail")
        if not isinstance(detail, dict):
            continue
        match = detail.get("match") or {}
        team1 = (match.get("team1") or {}).get("name")
        team2 = (match.get("team2") or {}).get("name")
        for item in detail.get("maps") or []:
            map_name = item.get("map_name")
            score = item.get("score") or {}
            if not map_name or not team1 or not team2:
                continue
            s1 = parse_float(score.get(team1))
            s2 = parse_float(score.get(team2))
            if s1 is None or s2 is None or s1 == s2:
                continue
            for team, won in [(team1, s1 > s2), (team2, s2 > s1)]:
                bucket = teams.setdefault(team, {}).setdefault(map_name, {"played": 0, "wins": 0})
                bucket["played"] += 1
                bucket["wins"] += int(won)

    output = {}
    for team, maps in teams.items():
        output[team] = {
            map_name: {
                "played": stats["played"],
                "wins": stats["wins"],
                "winrate": stats["wins"] / stats["played"] if stats["played"] else None,
            }
            for map_name, stats in maps.items()
        }
    write_json(run_dir / "map_winrates_from_completed_details.json", output)
    return output


def build_match_snapshot(match: dict[str, Any], run_dir: Path, capture_analytics: bool = True) -> dict[str, Any]:
    match_id = match_id_from_link(match.get("link"))
    if not match_id:
        raise ValueError(f"Cannot parse match id from {match.get('link')}")

    detail_path = run_dir / "match_details" / f"{match_id}.json"
    odds = {"available": False, "bookmaker_count": 0, "providers": [], "average": None}
    match_context = None
    prematch_lineups: dict[str, Any] = {}
    event_metadata: dict[str, Any] = {}
    odds_error = None
    html_error = None
    html = None
    try:
        html = fetch_html(match["link"])
        raw_html_meta = save_raw_html(run_dir, "match_snapshot", match_id, match["link"], html)
        match_context = (parse_veto_html(html).get("context") or None)
        odds = parse_odds(html)
        event_metadata = parse_event_metadata_from_match_html(html)
    except Exception as exc:
        raw_html_meta = None
        html_error = str(exc)
        odds_error = str(exc)

    detail, detail_logs = scrape_match_detail(match["link"], detail_path, initial_html=html)
    if detail is None:
        detail = detail_from_upcoming_row(match)

    detail_match = (detail or {}).get("match") or {}
    team_names = [
        (detail_match.get("team1") or {}).get("name") or ((match.get("team1") or {}).get("name") or ""),
        (detail_match.get("team2") or {}).get("name") or ((match.get("team2") or {}).get("name") or ""),
    ]
    analytics = (
        scrape_match_analytics(match["link"], run_dir, match_id, team_names)
        if capture_analytics
        else {"available": False, "reason": "skipped"}
    )
    analytics_event = analytics.get("event_metadata") if isinstance(analytics, dict) else None
    if isinstance(analytics_event, dict):
        event_metadata = {**event_metadata, **{key: value for key, value in analytics_event.items() if value is not None}}
    if html is not None:
        prematch_lineups = parse_prematch_lineups_html(html, detail)
        mark_prematch_standins(prematch_lineups, analytics)

    snapshot_path = run_dir / "match_snapshots" / f"{match_id}.json"

    snapshot = {
        "id": match_id,
        "captured_at": now_utc(),
        "source_file": str(snapshot_path.relative_to(DATA_ROOT)),
        "data_quality": {
            "real_pre_match_snapshot": True,
            "legacy_backfill": False,
            "training_weight_hint": "high_after_result",
        },
        "source": "https://www.hltv.org/",
        "upcoming_row": match,
        "link": match.get("link"),
        "date": match.get("date"),
        "hour": match.get("hour"),
        "format": match.get("meta"),
        "event": match.get("event"),
        "detail": detail,
        "detail_logs_tail": detail_logs[-500:] if detail_logs else "",
        "odds": odds,
        "match_context": match_context,
        "odds_error": odds_error,
        "html_error": html_error,
        "raw_html": raw_html_meta,
        "analytics": analytics,
        "event_metadata": event_metadata,
        "prematch_lineups": prematch_lineups,
        "status": "completed" if detail and is_completed_detail(detail) else "pending",
    }
    write_json(snapshot_path, snapshot)
    return snapshot


def upcoming_match_datetime(match: dict[str, Any]) -> datetime | None:
    date_text = parse_hltv_date(match.get("date"))
    hour = clean_text(match.get("hour")) or ""
    if not date_text or not re.fullmatch(r"\d{1,2}:\d{2}", hour):
        return None
    try:
        return datetime.strptime(f"{date_text} {hour}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def observed_closing_odds(
    record: dict[str, Any],
    point: dict[str, Any] | None,
    *,
    max_age_hours: float = CLOSING_ODDS_MAX_AGE_HOURS,
) -> dict[str, Any] | None:
    """Return a closing quote only when it was observed shortly pre-kickoff."""
    if not point or not point.get("captured_at"):
        return None
    kickoff = upcoming_match_datetime(record)
    if kickoff is None:
        return None
    try:
        captured = datetime.fromisoformat(
            str(point["captured_at"]).replace("Z", "+00:00")
        )
    except ValueError:
        return None
    seconds_to_start = int((kickoff - captured).total_seconds())
    if seconds_to_start < 0 or seconds_to_start > max_age_hours * 3600:
        return None
    qualified = dict(point)
    qualified["quality"] = "closing_observed"
    qualified["seconds_to_start"] = seconds_to_start
    qualified["is_observed_closing"] = True
    return qualified


def scheduled_snapshot_refresh_reason(record: dict[str, Any], match: dict[str, Any], now: datetime | None = None) -> str | None:
    """Decide si corresponde una foto incremental, sin tocar la primera."""
    if record.get("status") == "completed":
        return None
    if not has_usable_odds(record):
        return "missing_odds"
    analytics = record.get("analytics") or {}
    if not analytics.get("available"):
        return "missing_analytics"
    if not record.get("prematch_lineups"):
        return "missing_advertised_lineup"
    last_raw = record.get("last_prematch_refresh_at") or record.get("last_seen_at") or record.get("first_seen_at")
    if not last_raw:
        return "no_previous_refresh"
    try:
        last_refresh = datetime.fromisoformat(str(last_raw).replace("Z", "+00:00"))
    except ValueError:
        return "invalid_refresh_timestamp"
    now = now or datetime.now(timezone.utc)
    interval_hours = PREMATCH_REFRESH_HOURS
    scheduled_at = upcoming_match_datetime(match)
    if scheduled_at is not None and 0 <= (scheduled_at - now).total_seconds() <= PREMATCH_NEAR_START_WINDOW_HOURS * 3600:
        interval_hours = PREMATCH_NEAR_START_REFRESH_HOURS
    if now - last_refresh >= timedelta(hours=max(0.25, interval_hours)):
        return f"ttl_{interval_hours:g}h"
    return None


def materialize_persistent_match_snapshot(
    master: dict[str, Any],
    match: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any] | None:
    """Reutiliza la foto PRE-MATCH original dentro del run actual.

    La idempotencia evita otra petición a HLTV, pero el run debe seguir
    conteniendo todos los upcoming para que enriquecimiento y web no queden
    vacíos. `captured_at` y la evidencia original se conservan intactos.
    """
    match_id = str(match.get("id") or match_id_from_link(match.get("link") or "") or "")
    record = master.get(match_id) or {}
    if not match_id or not record:
        return None

    candidate_files: list[str] = []
    if record.get("latest_snapshot_file"):
        candidate_files.append(str(record["latest_snapshot_file"]))
    candidate_files.extend(
        str(item) for item in reversed(record.get("snapshot_files") or []) if item
    )

    snapshot: dict[str, Any] | None = None
    for candidate in dict.fromkeys(candidate_files):
        source_path = Path(candidate)
        if not source_path.is_absolute():
            source_path = DATA_ROOT / Path(candidate.replace("\\", os.sep))
        payload = read_json(source_path, None)
        if isinstance(payload, list):
            payload = payload[0] if payload else None
        if isinstance(payload, dict) and str(payload.get("id") or "") == match_id:
            snapshot = dict(payload)
            break

    if snapshot is None:
        detail = record.get("detail")
        if not isinstance(detail, dict):
            return None
        snapshot = {
            "id": match_id,
            "captured_at": record.get("first_seen_at") or record.get("last_seen_at"),
            "data_quality": record.get("data_quality") or {
                "real_pre_match_snapshot": True,
                "legacy_backfill": False,
            },
            "source": "persistent_master_record",
            "detail": detail,
            "odds": record.get("latest_odds") or {
                "available": False,
                "bookmaker_count": 0,
                "providers": [],
                "average": None,
            },
            "analytics": record.get("analytics") or {},
            "match_context": record.get("match_context"),
            "event_metadata": record.get("event_metadata") or {},
            "prematch_lineups": record.get("prematch_lineups") or {},
        }

    snapshot["upcoming_row"] = match
    snapshot["link"] = match.get("link") or record.get("link") or snapshot.get("link")
    snapshot["date"] = match.get("date") or record.get("date") or snapshot.get("date")
    snapshot["hour"] = match.get("hour") or record.get("hour") or snapshot.get("hour")
    snapshot["format"] = match.get("meta") or record.get("format") or snapshot.get("format")
    snapshot["event"] = match.get("event") or record.get("event") or snapshot.get("event")
    snapshot["status"] = "completed" if record.get("status") == "completed" else "pending"
    quality = dict(snapshot.get("data_quality") or {})
    quality["materialized_from_persistent_snapshot"] = True
    quality["original_captured_at"] = snapshot.get("captured_at")
    snapshot["data_quality"] = quality
    write_json(run_dir / "match_snapshots" / f"{match_id}.json", snapshot)
    return snapshot


def upsert_master_record(master: dict[str, Any], snapshot: dict[str, Any], run_dir: Path) -> None:
    match_id = snapshot["id"]
    record = master.get(match_id, {"id": match_id, "first_seen_at": snapshot["captured_at"]})
    was_completed = record.get("status") == "completed"
    snapshot_file = str((run_dir / "match_snapshots" / f"{match_id}.json").relative_to(DATA_ROOT))
    record.update(
        {
            "id": match_id,
            "link": snapshot.get("link"),
            "date": snapshot.get("date"),
            "hour": snapshot.get("hour"),
            "event": snapshot.get("event"),
            "format": snapshot.get("format"),
            "last_seen_at": snapshot["captured_at"],
            "last_prematch_refresh_at": snapshot["captured_at"],
            "status": "completed" if was_completed else snapshot.get("status", "pending"),
            "latest_snapshot_file": snapshot_file,
            "latest_odds": snapshot.get("odds"),
            "data_quality": snapshot.get("data_quality"),
        }
    )
    if snapshot.get("detail"):
        record["detail"] = snapshot["detail"]
    if snapshot.get("analytics"):
        record["analytics"] = snapshot["analytics"]
        if snapshot["analytics"].get("source_file"):
            record["latest_analytics_file"] = snapshot["analytics"]["source_file"]
    if snapshot.get("match_context"):
        record["match_context"] = snapshot["match_context"]
    if snapshot.get("event_metadata"):
        record["event_metadata"] = snapshot["event_metadata"]
    if snapshot.get("prematch_lineups"):
        record["prematch_lineups"] = snapshot["prematch_lineups"]
    if snapshot["status"] == "completed":
        result = result_from_detail(snapshot["detail"])
        if result:
            record.update(result)
    odds_point = compact_odds_point(snapshot, run_dir)
    if odds_point:
        record.setdefault("odds_history", [])
        if not any(point.get("captured_at") == odds_point["captured_at"] for point in record["odds_history"]):
            record["odds_history"].append(odds_point)
        record.setdefault("opening_odds", odds_point)
        record["latest_odds_point"] = odds_point
        if record.get("status") == "completed":
            closing = observed_closing_odds(record, odds_point)
            if closing:
                record.setdefault("closing_odds", closing)
    record.setdefault("snapshot_files", [])
    if snapshot_file not in record["snapshot_files"]:
        record["snapshot_files"].append(snapshot_file)
    record.setdefault("first_prematch_snapshot_file", snapshot_file)
    master[match_id] = record


def main() -> int:
    global VERBOSE
    parser = argparse.ArgumentParser(description="Daily HLTV start: snapshot pre-match data and update pending results.")
    parser.add_argument("--max-matches", type=int, default=0, help="Debug limit. 0 means all upcoming matches.")
    parser.add_argument("--player-delay", type=float, default=0.75)
    parser.add_argument("--skip-player-stats", action="store_true")
    parser.add_argument("--skip-team-profiles", action="store_true")
    parser.add_argument("--skip-match-assets", action="store_true", help="No captura veto/mapstats de partidos completados.")
    parser.add_argument(
        "--match-assets-limit",
        type=int,
        default=int(os.environ.get("BBDD_ASSETS_BACKFILL_LIMIT", "20")),
        help="Maximo de partidos completados a backfillear; 0 = todos.",
    )
    parser.add_argument("--match-assets-delay", type=float, default=0.5, help="Retardo entre peticiones de assets HLTV.")
    parser.add_argument("--skip-analytics", action="store_true", help="No captura HLTV betting analytics de partidos upcoming.")
    parser.add_argument("--skip-rankings", action="store_true", help="No captura rankings actuales HLTV/Valve.")
    parser.add_argument("--skip-warmup", action="store_true", help="No hace warm-up inicial de sesion HLTV.")
    parser.add_argument("--skip-same-day-recovery", action="store_true", help="No reintenta huecos recientes ya conocidos en master.")
    parser.add_argument("--same-day-recovery-window-days", type=int, default=2, help="Dias hacia atras para recuperar odds/detalle/analytics recientes.")
    parser.add_argument("--recovery-delay", type=float, default=1.5, help="Retardo entre peticiones de recuperacion de huecos.")
    parser.add_argument("--allow-empty-scrape", action="store_true", help="Permite runs sin recent ni upcoming (solo debug/offline).")
    parser.add_argument("--no-promote", action="store_true", help="No publica este run ni actualiza master.")
    parser.add_argument("--verbose", action="store_true", help="Muestra progreso, URLs, reintentos y bloqueos durante el scrape.")
    args = parser.parse_args()
    VERBOSE = bool(args.verbose)
    promote_run = not args.no_promote and args.max_matches == 0

    current_run_id = run_id()
    run_dir = RUNS_DIR / current_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    master: dict[str, Any] = read_json(MASTER_MATCHES, {})
    log(f"run started: {current_run_id} promote={promote_run} run_dir={run_dir}", force=VERBOSE)

    manifest: dict[str, Any] = {
        "run_id": current_run_id,
        "started_at": now_utc(),
        "source": "https://www.hltv.org/",
        "settings": vars(args),
        "steps": {},
    }
    write_json(run_dir / "manifest.json", manifest)

    warmup_result = {"enabled": False, "reason": "skipped"}
    if not args.skip_warmup:
        log("phase: session warm-up", force=VERBOSE)
        warmup_result = warm_up_hltv_session(run_dir)
    manifest["steps"]["warmup"] = warmup_result
    write_json(run_dir / "manifest.json", manifest)

    log("phase: recent results", force=VERBOSE)
    db_pending_for_results = db_pending_match_ids()
    db_pending_info = db_pending_match_info()
    if db_pending_for_results is None:
        log("recent results phase: DB unavailable, using legacy bounded scan", force=VERBOSE)
    else:
        log(f"recent results phase: DB pending targets={len(db_pending_for_results)}", force=VERBOSE)
    recent_results = scrape_recent_results(run_dir, target_ids=db_pending_for_results, target_info=db_pending_info)
    recent_manifest = read_json(run_dir / "recent_results_manifest.json", {})
    manifest["steps"]["recent_results"] = {"count": len(recent_results), **recent_manifest}
    write_json(run_dir / "manifest.json", manifest)

    log("phase: pending updates", force=VERBOSE)
    pending_updates = update_pending_matches(master, run_dir, recent_results, capture_analytics=not args.skip_analytics)
    manifest["steps"]["pending_updates"] = {"count": len(pending_updates)}
    write_json(run_dir / "manifest.json", manifest)

    recovery_result = {"touched": 0, "reason": "skipped"}
    if not args.skip_same_day_recovery:
        log("phase: same-day recovery", force=VERBOSE)
        recovery_result = recover_recent_data_gaps(
            master,
            run_dir,
            capture_analytics=not args.skip_analytics,
            window_days=max(0, args.same_day_recovery_window_days),
            delay=max(0.0, args.recovery_delay),
        )
        if promote_run:
            write_json(MASTER_MATCHES, master)
    manifest["steps"]["same_day_recovery"] = {
        key: value for key, value in recovery_result.items() if key != "index"
    }
    write_json(run_dir / "manifest.json", manifest)

    log("phase: upcoming matches", force=VERBOSE)
    upcoming = scrape_upcoming(run_dir)
    if args.max_matches:
        upcoming = upcoming[: args.max_matches]
    manifest["steps"]["upcoming"] = {"count": len(upcoming)}
    write_json(run_dir / "manifest.json", manifest)
    if not args.allow_empty_scrape and not recent_results and not upcoming:
        manifest["failed_at"] = now_utc()
        manifest["failure"] = (
            "HLTV scrape returned zero recent results and zero upcoming matches. "
            "Refresh cf_session.json or inspect spiders/parsers before accepting this run."
        )
        manifest["steps"]["fetch_diagnostics"] = fetch_diagnostics()
        write_json(run_dir / "manifest.json", manifest)
        raise RuntimeError(manifest["failure"])

    snapshots = []
    reused_photographed = 0
    refreshed_photographed = 0
    missing_persistent_snapshot = 0
    for index, match in enumerate(upcoming, start=1):
        log(f"[snapshot {index}/{len(upcoming)}] {match_label(match)}", force=VERBOSE)
        match_id = str(match.get("id") or match_id_from_link(match.get("link") or "") or "")
        if db_match_already_photographed(match_id):
            refresh_reason = scheduled_snapshot_refresh_reason(master.get(match_id) or {}, match)
            if refresh_reason:
                log(
                    f"[snapshot {index}/{len(upcoming)}] refreshing scheduled pre-match data ({refresh_reason})",
                    force=VERBOSE,
                )
                snapshot = build_match_snapshot(match, run_dir, capture_analytics=not args.skip_analytics)
                quality = dict(snapshot.get("data_quality") or {})
                quality["scheduled_refresh"] = True
                quality["refresh_reason"] = refresh_reason
                snapshot["data_quality"] = quality
                snapshots.append(snapshot)
                refreshed_photographed += 1
                upsert_master_record(master, snapshot, run_dir)
                if promote_run:
                    write_json(MASTER_MATCHES, master)
                odds = snapshot.get("odds") or {}
                analytics = snapshot.get("analytics") or {}
                lineups = snapshot.get("prematch_lineups") or {}
                log(
                    f"[snapshot {index}/{len(upcoming)}] refreshed odds={odds.get('available')} "
                    f"books={odds.get('bookmaker_count')} analytics={analytics.get('available')} "
                    f"lineup_players={sum(len((side or {}).get('players') or []) for side in lineups.values())}",
                    force=VERBOSE,
                )
                time.sleep(0.2)
                continue
            cached_snapshot = materialize_persistent_match_snapshot(master, match, run_dir)
            if cached_snapshot is not None:
                snapshots.append(cached_snapshot)
                reused_photographed += 1
                note_freshness_skip()
                log(
                    f"[snapshot {index}/{len(upcoming)}] reused persistent PRE-MATCH snapshot "
                    f"captured_at={cached_snapshot.get('captured_at')}",
                    force=VERBOSE,
                )
                continue
            missing_persistent_snapshot += 1
            log(
                f"[snapshot {index}/{len(upcoming)}] DB says photographed but snapshot is missing; recapturing",
                force=True,
            )
        snapshot = build_match_snapshot(match, run_dir, capture_analytics=not args.skip_analytics)
        snapshots.append(snapshot)
        upsert_master_record(master, snapshot, run_dir)
        if promote_run:
            write_json(MASTER_MATCHES, master)
        odds = snapshot.get("odds") or {}
        analytics = snapshot.get("analytics") or {}
        context = snapshot.get("match_context") or {}
        log(
            f"[snapshot {index}/{len(upcoming)}] saved status={snapshot.get('status')} "
            f"odds={odds.get('available')} books={odds.get('bookmaker_count')} "
            f"analytics={analytics.get('available')} env={context.get('environment') or 'unknown'}",
            force=VERBOSE,
        )
        time.sleep(0.2)
    write_json(run_dir / "matches_snapshot_index.json", snapshots)
    manifest["steps"]["match_snapshots"] = {
        "count": len(snapshots),
        "captured_online": len(snapshots) - reused_photographed,
        "refreshed_scheduled_prematch": refreshed_photographed,
        "reused_from_persistent_snapshot": reused_photographed,
        "persistent_snapshot_missing_recaptured": missing_persistent_snapshot,
        "skipped_already_photographed": 0,
    }
    write_json(run_dir / "manifest.json", manifest)

    details = [snapshot["detail"] for snapshot in snapshots if isinstance(snapshot.get("detail"), dict)]
    teams = collect_teams_from_details(details)
    write_json(run_dir / "teams_index.json", teams)
    manifest["steps"]["teams_detected"] = {"count": len(teams)}
    write_json(run_dir / "manifest.json", manifest)

    team_profiles = []
    if not args.skip_team_profiles:
        log("phase: team profiles", force=VERBOSE)
        team_profiles = collect_team_profiles(teams, run_dir)
    manifest["steps"]["team_profiles"] = {"count": len(team_profiles)}
    write_json(run_dir / "manifest.json", manifest)

    if team_profiles and promote_run:
        roster_history = update_roster_history(team_profiles, run_dir, manifest["started_at"])
    elif team_profiles:
        roster_history = {
            "teams": len(team_profiles),
            "snapshots_added": 0,
            "skipped": "not_promoted",
        }
    else:
        roster_history = {"teams": 0}
    manifest["steps"]["roster_history"] = roster_history
    write_json(run_dir / "manifest.json", manifest)

    player_stats_profiles, player_stats_candidates = player_stats_profiles_with_announced_lineups(
        team_profiles,
        snapshots,
    )
    player_stats_profiles_file = run_dir / "player_stats_profiles.json"
    write_json(player_stats_profiles_file, player_stats_profiles)
    manifest["steps"]["player_stats_candidates"] = player_stats_candidates
    write_json(run_dir / "manifest.json", manifest)

    player_stats_result = {"ok": False, "reason": "skipped"}
    if not args.skip_player_stats and player_stats_profiles:
        log("phase: player compare stats", force=VERBOSE)
        player_stats_result = collect_player_stats(
            player_stats_profiles_file,
            run_dir,
            datetime.now().year,
            args.player_delay,
        )
    manifest["steps"]["player_stats"] = player_stats_result
    write_json(run_dir / "manifest.json", manifest)

    rankings_result = {"ok": False, "reason": "skipped"}
    if not args.skip_rankings:
        log("phase: rankings", force=VERBOSE)
        rankings_result = collect_rankings(run_dir)
    manifest["steps"]["rankings"] = rankings_result
    write_json(run_dir / "manifest.json", manifest)

    match_assets_result = {"captured": 0, "reason": "skipped"}
    if not args.skip_match_assets:
        log("phase: completed match assets", force=VERBOSE)
        asset_limit = args.match_assets_limit
        if not promote_run and asset_limit == 50:
            asset_limit = 3
        match_assets_result = collect_completed_match_assets(
            master,
            run_dir,
            limit=asset_limit,
            delay=args.match_assets_delay,
        )
        if promote_run:
            write_json(MASTER_MATCHES, master)
    manifest["steps"]["match_assets"] = {
        key: value for key, value in match_assets_result.items() if key != "index"
    }
    write_json(run_dir / "manifest.json", manifest)

    map_winrates = compute_map_winrates(master, run_dir)
    log(f"map winrates: teams={len(map_winrates)}", force=VERBOSE)
    manifest["steps"]["map_winrates"] = {"teams": len(map_winrates)}
    quality_report = build_data_quality_report(run_dir, manifest, master, snapshots, team_profiles)
    log(
        "data quality: "
        + json.dumps(quality_report.get("summary") or {}, ensure_ascii=False, sort_keys=True),
        force=VERBOSE,
    )
    manifest["steps"]["data_quality"] = {
        "file": str((run_dir / "data_quality_report.json").relative_to(DATA_ROOT)),
        "warnings": quality_report.get("warnings") or [],
        **(quality_report.get("summary") or {}),
    }
    manifest["steps"]["fetch_diagnostics"] = fetch_diagnostics()
    write_json(run_dir / "fetch_diagnostics.json", manifest["steps"]["fetch_diagnostics"])
    manifest["finished_at"] = now_utc()
    write_json(run_dir / "manifest.json", manifest)

    if promote_run:
        write_json(MASTER_MATCHES, master)
        write_json(
            MASTER_MANIFEST,
            {
                "last_run_id": current_run_id,
                "last_run_at": manifest["finished_at"],
                "total_matches_seen": len(master),
                "pending_matches": sum(1 for record in master.values() if record.get("status") != "completed"),
                "completed_matches": sum(1 for record in master.values() if record.get("status") == "completed"),
            },
        )
    else:
        manifest["not_promoted"] = True
        manifest["not_promoted_reason"] = "--no-promote or --max-matches debug run"
        write_json(run_dir / "manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nDaily snapshot saved to: {run_dir}")
    if not promote_run:
        print("Debug run not promoted to PIPELINE/master/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
