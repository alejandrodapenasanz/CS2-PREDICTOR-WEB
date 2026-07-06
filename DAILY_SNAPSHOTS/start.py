from __future__ import annotations

import argparse
import os
import gzip
import hashlib
import json
import math
import random
import re
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
SCRAPER_PROJECT = ROOT / "SCRAPPER" / "hltv-scraper-api"
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
    from DAILY_SNAPSHOTS.match_context import parse_match_context_meta
except Exception:  # pragma: no cover - direct script execution
    from match_context import parse_match_context_meta

DATA_ROOT = ROOT / "DAILY_SNAPSHOTS"
RUNS_DIR = DATA_ROOT / "runs"
MASTER_DIR = DATA_ROOT / "master"
MASTER_MATCHES = MASTER_DIR / "matches.json"
MASTER_MANIFEST = MASTER_DIR / "manifest.json"
MASTER_ROSTERS = MASTER_DIR / "roster_history.json"

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


def looks_like_cf_or_waf_problem(error: Exception | None) -> bool:
    text = str(error or "").lower()
    return any(marker in text for marker in ("cloudflare", "challenge", "403", "429", "bloqueo", "cf-chl"))


def maybe_refresh_cf_session_once(reason: str, url: str) -> bool:
    global _CF_SESSION_REFRESHED_THIS_RUN, _HTTP_SESSION, _FETCH_DOMAIN_COOLDOWN_UNTIL, _FETCH_BLOCK_STREAK
    if _CF_SESSION_REFRESHED_THIS_RUN:
        return False
    helper = SCRAPY_ROOT / "hltv_scraper" / "grab_cf.py"
    if not helper.exists():
        return False
    _CF_SESSION_REFRESHED_THIS_RUN = True
    _FETCH_STATS["cf_session_refresh_attempts"] += 1
    log(f"cf_session refresh triggered by {reason}: {url}", force=True)
    ok, logs = run_cmd([str(PYTHON_EXE), str(helper)], SCRAPER_PROJECT, timeout=240, stream=True)
    if ok and CF_SESSION.exists():
        _FETCH_STATS["cf_session_refresh_successes"] += 1
        _HTTP_SESSION = None
        _FETCH_BLOCK_STREAK = 0
        _FETCH_DOMAIN_COOLDOWN_UNTIL = min(_FETCH_DOMAIN_COOLDOWN_UNTIL, time.monotonic() + 5.0)
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
    resp = _ScraplingStealthy.fetch(url, **kwargs)
    _persist_cf_from_response(resp)
    return _scrapling_response_parts(resp)


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
        except FetchBudgetExceeded:
            raise
        except Exception as exc:
            register_fetch_error("scrapling_stealth", url, exc)

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


def parse_analytics_html(html: str, match_id: str, match_link: str, team_names: list[str]) -> dict[str, Any]:
    selector = Selector(text=html)
    lines = [
        clean_text(line)
        for line in selector.xpath("//body//text()").getall()
    ]
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
        "past 30 days",
        "current event",
        "maps in the past",
    )
    insights = [
        line for line in summary_lines
        if any(term in line.lower() for term in insight_terms)
    ][:80]

    map_stats: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str]] = set()
    for idx, line in enumerate(lines):
        map_name = line.strip()
        if map_name.lower() not in ANALYTICS_MAP_NAMES:
            continue
        for team in team_names:
            if not team:
                continue
            team_idx = None
            for probe in range(idx + 1, min(idx + 28, len(lines))):
                if lines[probe].lower() == team.lower():
                    team_idx = probe
                    break
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
                    and "first pick" not in next_line.lower()
                    and "first ban" not in next_line.lower()
                ):
                    comment = next_line
            map_stats.append({"map": map_name, "team": team, **metrics, "comment": comment})

    return {
        "available": bool(lines),
        "match_id": match_id,
        "url": hltv_url(analytics_link_from_match(match_link) or match_link),
        "captured_at": now_utc(),
        "summary_lines": summary_lines[:120],
        "insights": insights,
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
        log(f"analytics {match_id}: available={payload.get('available')} lines={payload.get('line_count')}")
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
    assets = {
        "match_id": match_id_from_link(match_link),
        "match_link": match_link,
        "captured_at": now_utc(),
        "veto": parse_veto_html(html),
        "mapstats_links": extract_mapstats_links(html),
        "mapstats": [],
        "errors": [],
        "raw_html": html_sources,
    }
    log(f"assets {match_id}: veto_steps={len((assets.get('veto') or {}).get('steps') or [])} mapstats_links={len(assets['mapstats_links'])}")
    for index, link in enumerate(assets["mapstats_links"], start=1):
        try:
            log(f"assets {match_id}: mapstats {index}/{len(assets['mapstats_links'])} {link}")
            time.sleep(delay)
            map_html = fetch_html(link)
            meta = save_raw_html(run_dir, "mapstats", mapstats_id_from_link(link) or slug_from_link(link), link, map_html)
            parsed = parse_mapstats_html(map_html, link)
            parsed["raw_html"] = meta
            assets["mapstats"].append(parsed)
        except Exception as exc:
            log(f"assets {match_id}: mapstats ERROR {link}: {exc}")
            assets["errors"].append({"link": link, "error": str(exc)})
    write_json(output_path, assets)
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
        record.setdefault("closing_odds", point)
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


def scrape_match_detail(match_link: str, output_path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        html = fetch_html(match_link)
        detail = parse_match_detail_html(html)
        if detail and (detail.get("match") or {}).get("team1", {}).get("name"):
            write_json(output_path, [detail])
            return detail, "direct_fetch"
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


def scrape_recent_results(run_dir: Path, pages: int = 3) -> dict[str, dict[str, Any]]:
    results_by_id: dict[str, dict[str, Any]] = {}
    for page in range(pages):
        offset = page * 100
        output = run_dir / "raw" / "recent_results" / f"results_offset_{offset}.json"
        try:
            log(f"recent results: fetching offset={offset} ({page + 1}/{pages})")
            html = fetch_html(f"/results?offset={offset}")
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
            time.sleep(0.2)
            continue
        except Exception as exc:
            log(f"recent results: offset={offset} direct ERROR {exc}; trying scrapy fallback")
            write_json(run_dir / "logs" / f"recent_results_direct_error_{offset}.log.json", {"error": str(exc)})

        ok, _logs = run_spider("hltv_results", output, spider_args=["-a", f"offset={offset}"], timeout=600)
        if not ok:
            log(f"recent results: offset={offset} scrapy fallback failed")
            continue
        fallback_rows = read_json(output, [])
        for item in fallback_rows:
            match_id = str(item.get("id") or "")
            if match_id:
                results_by_id[match_id] = item
        log(f"recent results: offset={offset} scrapy fallback parsed {len(fallback_rows)} rows")
    write_json(run_dir / "recent_results_index.json", results_by_id)
    log(f"recent results: unique matches={len(results_by_id)}")
    return results_by_id


def update_pending_matches(
    master: dict[str, Any],
    run_dir: Path,
    recent_results: dict[str, dict[str, Any]],
    capture_analytics: bool = True,
) -> list[dict[str, Any]]:
    updates = []
    pending = [record for record in master.values() if record.get("status") != "completed" and record.get("link")]
    log(f"pending updates: {len(pending)} pending matches")
    for index, record in enumerate(pending, start=1):
        match_id = record["id"]
        log(f"[pending {index}/{len(pending)}] {record_label(record)}")
        update_item: dict[str, Any] = {"id": match_id}
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
                record["closing_odds"] = record["latest_odds_point"]
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
    team_list = list(teams.values())
    log(f"team profiles: {len(team_list)} teams")
    for index, team in enumerate(team_list, start=1):
        log(f"[team {index}/{len(team_list)}] {team.get('name') or team.get('id') or '?'}")
        profile = scrape_team_profile(team, output_dir)
        if profile:
            profiles.append(profile)
            log(f"[team {index}/{len(team_list)}] ok")
        else:
            log(f"[team {index}/{len(team_list)}] unavailable")
        time.sleep(0.2)
    write_json(run_dir / "team_profiles.json", profiles)
    log(f"team profiles: collected={len(profiles)}")
    return profiles


def collect_player_stats(team_profiles_file: Path, run_dir: Path, year: int, delay: float) -> dict[str, Any]:
    output = run_dir / f"player_compare_stats_{year}.json"
    if not team_profiles_file.exists():
        log("player stats: missing team_profiles.json")
        return {"ok": False, "reason": "missing_team_profiles"}

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
        str(team_profiles_file),
        "--output-file",
        str(output),
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
        "3",
    ]
    if VERBOSE:
        cmd.append("--verbose")
    log(f"player stats: starting compare scrape year={year} delay={delay}s")
    ok, logs = run_cmd(cmd, SCRAPER_PROJECT, timeout=1800, stream=VERBOSE)
    payload = read_json(output, {}) if output.exists() else {}
    cf_session_refresh_attempted = False
    cf_session_refresh_ok = None
    first_stopped_reason = payload.get("stopped_reason")
    if should_refresh_cf(payload, logs, ok):
        cf_session_refresh_attempted = True
        refresh_ok, refresh_logs = refresh_cf_session(str(payload.get("stopped_reason") or "compare_problem"))
        cf_session_refresh_ok = refresh_ok
        logs = (logs or "") + "\n[cf_refresh]\n" + (refresh_logs or "")
        if refresh_ok:
            log("player stats: retrying compare scrape after cf_session refresh", force=True)
            ok, retry_logs = run_cmd(cmd, SCRAPER_PROJECT, timeout=1800, stream=VERBOSE)
            logs = (logs or "") + "\n[retry_after_cf_refresh]\n" + (retry_logs or "")
            payload = read_json(output, {}) if output.exists() else payload
    collected = int(payload.get("comparisons_collected") or 0)
    failed = int(payload.get("comparisons_failed") or 0)
    log(f"player stats: collected={collected} failed={failed} ok={ok}")
    return {
        "ok": ok or collected > 0,
        "partial": (not ok and collected > 0) or failed > 0,
        "file": str(output),
        "players_requested": payload.get("players_requested"),
        "comparisons_collected": collected,
        "comparisons_failed": failed,
        "stopped_reason": payload.get("stopped_reason"),
        "first_stopped_reason": first_stopped_reason,
        "selection_mode": payload.get("selection_mode"),
        "min_maps": payload.get("min_maps"),
        "requests_attempted": payload.get("requests_attempted"),
        "cf_session_refresh_attempted": cf_session_refresh_attempted,
        "cf_session_refresh_ok": cf_session_refresh_ok,
        "cf_session_refresh_hint": (
            ".\\SCRAPPER\\hltv-scraper-api\\.venv\\Scripts\\python.exe "
            ".\\SCRAPPER\\hltv-scraper-api\\hltv_scraper\\hltv_scraper\\grab_cf.py"
            if "Cloudflare" in str(payload.get("stopped_reason") or logs)
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
    for record in master.values():
        if record.get("status") != "completed" or not record.get("link"):
            continue
        meta = record.get("hltv_assets") or {}
        if meta.get("status") == "ok" and asset_file_exists(meta):
            continue
        candidates.append(record)

    if limit > 0:
        candidates = candidates[:limit]

    index = []
    log(f"completed assets: {len(candidates)} missing/partial completed matches (limit={limit})")
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
            log(f"[assets {item_index}/{len(candidates)}] status={meta['status']} maps={meta['mapstats_maps']} errors={len(meta['errors'])}")
        except Exception as exc:
            log(f"[assets {item_index}/{len(candidates)}] ERROR {exc}")
            index.append({"id": match_id, "status": "error", "error": str(exc)})
        time.sleep(delay)

    summary = {
        "eligible_missing": len(candidates),
        "captured": sum(1 for item in index if item.get("status") in {"ok", "partial"}),
        "errors": sum(1 for item in index if item.get("status") == "error"),
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
    detail, detail_logs = scrape_match_detail(match["link"], detail_path)

    odds = {"available": False, "bookmaker_count": 0, "providers": [], "average": None}
    match_context = None
    odds_error = None
    html_error = None
    try:
        html = fetch_html(match["link"])
        raw_html_meta = save_raw_html(run_dir, "match_snapshot", match_id, match["link"], html)
        match_context = (parse_veto_html(html).get("context") or None)
        odds = parse_odds(html)
        if detail is None:
            detail = parse_match_basic_html(html)
    except Exception as exc:
        raw_html_meta = None
        html_error = str(exc)
        odds_error = str(exc)

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

    snapshot = {
        "id": match_id,
        "captured_at": now_utc(),
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
        "status": "completed" if detail and is_completed_detail(detail) else "pending",
    }
    write_json(run_dir / "match_snapshots" / f"{match_id}.json", snapshot)
    return snapshot


def upsert_master_record(master: dict[str, Any], snapshot: dict[str, Any], run_dir: Path) -> None:
    match_id = snapshot["id"]
    record = master.get(match_id, {"id": match_id, "first_seen_at": snapshot["captured_at"]})
    record.update(
        {
            "id": match_id,
            "link": snapshot.get("link"),
            "date": snapshot.get("date"),
            "hour": snapshot.get("hour"),
            "event": snapshot.get("event"),
            "format": snapshot.get("format"),
            "last_seen_at": snapshot["captured_at"],
            "status": snapshot.get("status", "pending"),
            "latest_snapshot_file": str((run_dir / "match_snapshots" / f"{match_id}.json").relative_to(DATA_ROOT)),
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
            record.setdefault("closing_odds", odds_point)
    record.setdefault("snapshot_files", [])
    record["snapshot_files"].append(record["latest_snapshot_file"])
    master[match_id] = record


def main() -> int:
    global VERBOSE
    parser = argparse.ArgumentParser(description="Daily HLTV start: snapshot pre-match data and update pending results.")
    parser.add_argument("--max-matches", type=int, default=0, help="Debug limit. 0 means all upcoming matches.")
    parser.add_argument("--player-delay", type=float, default=0.75)
    parser.add_argument("--skip-player-stats", action="store_true")
    parser.add_argument("--skip-team-profiles", action="store_true")
    parser.add_argument("--skip-match-assets", action="store_true", help="No captura veto/mapstats de partidos completados.")
    parser.add_argument("--match-assets-limit", type=int, default=50, help="Maximo de partidos completados a backfillear; 0 = todos.")
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
    recent_results = scrape_recent_results(run_dir)
    manifest["steps"]["recent_results"] = {"count": len(recent_results)}
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
    for index, match in enumerate(upcoming, start=1):
        log(f"[snapshot {index}/{len(upcoming)}] {match_label(match)}", force=VERBOSE)
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
    manifest["steps"]["match_snapshots"] = {"count": len(snapshots)}
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

    player_stats_result = {"ok": False, "reason": "skipped"}
    if not args.skip_player_stats and team_profiles:
        log("phase: player compare stats", force=VERBOSE)
        player_stats_result = collect_player_stats(
            run_dir / "team_profiles.json",
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
        print("Debug run not promoted to DAILY_SNAPSHOTS/master/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
