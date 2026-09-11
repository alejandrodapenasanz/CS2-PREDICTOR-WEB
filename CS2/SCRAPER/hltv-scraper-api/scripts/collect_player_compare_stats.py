from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from parsel import Selector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_FILE = PROJECT_ROOT / "hltv_scraper" / "cf_session.json"
BLOCK_HTTP_CODES = {403, 429, 500, 502, 503, 504, 522, 524}
CHALLENGE_MARKERS = (
    "Just a moment",
    "cf-chl",
    "Checking your browser",
    "Enable JavaScript and cookies",
)
COMPARE_BLOCK_COOLDOWN_BASE = 90.0
COMPARE_BLOCK_COOLDOWN_MAX = 1200.0
COMPARE_BLOCK_STREAK_THRESHOLD = 3
COMPARE_CIRCUIT_BREAKER_SLEEP = 420.0
_COMPARE_COOLDOWN_UNTIL = 0.0
_COMPARE_BLOCK_STREAK = 0


class CloudflareChallengeStop(RuntimeError):
    """Raised when compare pages are persistently behind a Cloudflare challenge."""


NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?%?$|^-?\d+$")
KNOWN_STATS = [
    "Rating 3.0",
    "Rating 2.1",
    "Rating 2.0",
    "Rating 1.0",
    "KPR",
    "DPR",
    "APR",
    "KAST",
    "Impact",
    "ADR",
    "Average Damage per Round",
    "K-D Diff",
    "K/D Diff",
    "Surviving",
    "Multi-kill rating",
    "Round Swing",
    "HS %",
    "Headshots",
    "Opening KPR",
    "Opening DPR",
    "Opening attempts",
    "Opening success",
    "AWP KPR",
    "Flash assists",
    "Maps",
    "Rounds",
]

# The compare page deliberately exposes a compact comparison table.  HLTV's
# individual player page is the source for the remaining core pre-match stats.
STAT_ALIASES = {
    "assists per round": "APR",
    "deaths per round": "DPR",
    "kills per round": "KPR",
    "kills / round": "KPR",
    "assists / round": "APR",
    "deaths / round": "DPR",
    "damage / round": "ADR",
    "average damage per round": "ADR",
    "impact rating": "Impact",
    "headshot %": "HS %",
    "opening kills per round": "Opening KPR",
    "opening deaths per round": "Opening DPR",
    "flash assists per round": "Flash assists",
    "maps played": "Maps",
}
PLAYER_DETAIL_REQUIRED_STATS = ("Rating 3.0", "KPR", "DPR", "APR", "KAST", "Impact", "ADR")


@dataclass(frozen=True)
class PlayerRef:
    id: str
    name: str
    slug: str
    link: str


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_print(message: str, *, file=None) -> None:
    file = file or sys.stdout
    encoding = getattr(file, "encoding", None) or "utf-8"
    print(message.encode(encoding, errors="replace").decode(encoding), file=file, flush=True)


def slug_from_link(link: str, fallback_name: str) -> str:
    if link:
        parts = link.strip("/").split("/")
        if len(parts) >= 3:
            return parts[2]
    return re.sub(r"[^a-z0-9]+", "-", fallback_name.lower()).strip("-")


def load_players(team_profiles_file: Path) -> list[PlayerRef]:
    data = read_json(team_profiles_file)
    players: dict[str, PlayerRef] = {}
    for team in data if isinstance(data, list) else []:
        profile = team.get("profile") or {}
        for player in profile.get("squad") or []:
            player_id = player.get("id")
            link = player.get("link")
            name = player.get("name") or ""
            if not player_id or not link or not name:
                continue
            players.setdefault(
                player_id,
                PlayerRef(
                    id=str(player_id),
                    name=name,
                    slug=slug_from_link(link, name),
                    link=link,
                ),
            )
    return list(players.values())


def load_session(session_file: Path) -> dict[str, str]:
    if not session_file.exists():
        raise FileNotFoundError(
            f"No existe {session_file}. Ejecuta primero: python hltv_scraper/hltv_scraper/grab_cf.py"
        )
    session = read_json(session_file)
    if "cf_clearance" not in session or "user_agent" not in session:
        raise ValueError(f"Sesión inválida en {session_file}")
    return session


def compare_url(
    p1: PlayerRef,
    p2: PlayerRef,
    time_filter: str,
    match_filter: str,
    map_filter: str,
) -> str:
    return (
        "https://www.hltv.org/stats/players/compare/"
        f"{p1.id}/{p1.slug}/{p2.id}/{p2.slug}"
        f"?p1TimeFilter={time_filter}&p2TimeFilter={time_filter}"
        f"&p1MatchFilter={match_filter}&p2MatchFilter={match_filter}"
        f"&p1MapFilter={map_filter}&p2MapFilter={map_filter}"
    )


def _shift_months(value: date, months: int) -> date:
    target_month = value.month - months
    target_year = value.year
    while target_month <= 0:
        target_month += 12
        target_year -= 1
    return date(target_year, target_month, min(value.day, monthrange(target_year, target_month)[1]))


def player_stats_url(player: PlayerRef, time_filter: str, as_of: date | None = None) -> str:
    """Build an individual HLTV stats URL matching the chosen adaptive window."""
    as_of = as_of or date.today()
    params: dict[str, str] = {}
    match = re.fullmatch(r"past(\d+)months", time_filter)
    if match:
        params = {
            "startDate": _shift_months(as_of, int(match.group(1))).isoformat(),
            "endDate": as_of.isoformat(),
        }
    elif time_filter.isdigit() and len(time_filter) == 4:
        params = {"startDate": f"{time_filter}-01-01", "endDate": as_of.isoformat()}
    suffix = f"?{urlencode(params)}" if params else ""
    return f"https://www.hltv.org/stats/players/{player.id}/{player.slug}{suffix}"


def clean_texts(selector: Selector) -> list[str]:
    texts = []
    for text in selector.css("body ::text").getall():
        cleaned = " ".join(text.split())
        if cleaned:
            texts.append(cleaned)
    return texts


def is_numeric(value: str) -> bool:
    return bool(NUMERIC_RE.match(value.strip()))


def nearest_numeric_before(texts: list[str], index: int, window: int = 4) -> str | None:
    for cursor in range(index - 1, max(-1, index - window - 1), -1):
        if is_numeric(texts[cursor]):
            return texts[cursor]
    return None


def nearest_numeric_after(texts: list[str], index: int, window: int = 5) -> str | None:
    for cursor in range(index + 1, min(len(texts), index + window + 1)):
        if is_numeric(texts[cursor]):
            return texts[cursor]
    return None


def canonical_stat(text: str) -> str | None:
    lowered = " ".join(text.lower().split())
    for stat in KNOWN_STATS:
        if lowered == stat.lower():
            return stat
    return STAT_ALIASES.get(lowered)


def parse_stats_rows(selector: Selector) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selector.css("div.stats-row, div.stats-compare-row, tr"):
        texts = [" ".join(text.split()) for text in row.css("::text").getall()]
        texts = [text for text in texts if text]
        if len(texts) < 2:
            continue

        label = next((canonical_stat(text) for text in texts if canonical_stat(text)), None)
        if not label:
            continue

        label_index = next(i for i, text in enumerate(texts) if canonical_stat(text) == label)
        left = nearest_numeric_before(texts, label_index)
        right = nearest_numeric_after(texts, label_index)
        rows.append({"stat": label, "player1": left, "player2": right, "raw": texts})
    return rows


def parse_stats_from_text(selector: Selector) -> list[dict[str, Any]]:
    texts = clean_texts(selector)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, text in enumerate(texts):
        stat = canonical_stat(text)
        if not stat or stat in seen:
            continue
        left = nearest_numeric_before(texts, index)
        right = nearest_numeric_after(texts, index)
        if left is None and right is None:
            continue
        rows.append({"stat": stat, "player1": left, "player2": right})
        seen.add(stat)
    return rows


def parse_compare_column(column: Selector) -> tuple[int | None, dict[str, str]]:
    """Parse one player's column without borrowing values from the other side."""
    map_text = _first_clean(column.css(".selector-map-count::text").getall())
    map_match = re.search(r"Based on\s+(\d+)\s+maps", map_text or "", flags=re.IGNORECASE)
    stats: dict[str, str] = {}
    for row in column.css("div.stats-row"):
        texts = [" ".join(text.split()) for text in row.css("::text").getall()]
        texts = [text for text in texts if text]
        label = next((canonical_stat(text) for text in texts if canonical_stat(text)), None)
        if not label:
            continue
        value = next(
            (
                text
                for text in reversed(texts)
                if canonical_stat(text) is None and has_stat_value(text) and is_numeric(text)
            ),
            None,
        )
        if value is not None:
            stats[label] = value
    return (int(map_match.group(1)) if map_match else None), stats


def parse_compare_page(html: str, p1: PlayerRef, p2: PlayerRef, url: str, time_filter: str) -> dict[str, Any]:
    selector = Selector(text=html)
    title = selector.css("title::text").get()
    columns = selector.css(".stats-player-compare > .columns > .col.stats-rows")
    if len(columns) >= 2:
        p1_maps, p1_stats = parse_compare_column(columns[0])
        p2_maps, p2_stats = parse_compare_column(columns[1])
        stat_names = list(dict.fromkeys([*p1_stats, *p2_stats]))
        rows = [
            {
                "stat": stat,
                "player1": p1_stats.get(stat),
                "player2": p2_stats.get(stat),
            }
            for stat in stat_names
        ]
        return {
            "url": url,
            "time_filter": time_filter,
            "title": title,
            "players": [
                {
                    "id": p1.id,
                    "name": p1.name,
                    "slug": p1.slug,
                    "link": p1.link,
                    "maps": p1_maps,
                    "time_filter": time_filter,
                    "stats": p1_stats,
                },
                {
                    "id": p2.id,
                    "name": p2.name,
                    "slug": p2.slug,
                    "link": p2.link,
                    "maps": p2_maps,
                    "time_filter": time_filter,
                    "stats": p2_stats,
                },
            ],
            "comparison_rows": rows,
        }

    # Compatibility fallback for older HLTV markup.
    maps = [int(match) for match in re.findall(r"Based on\s+(\d+)\s+maps", " ".join(clean_texts(selector)))]
    rows = parse_stats_rows(selector)
    text_rows = parse_stats_from_text(selector)
    if rows:
        by_stat = {row["stat"]: row for row in rows}
        for row in text_rows:
            if row["stat"] not in by_stat:
                rows.append(row)
                by_stat[row["stat"]] = row
    else:
        rows = text_rows

    p1_stats = {
        str(row["stat"]): str(row["player1"])
        for row in rows
        if row.get("stat") is not None and row.get("player1") is not None
    }
    p2_stats = {
        str(row["stat"]): str(row["player2"])
        for row in rows
        if row.get("stat") is not None and row.get("player2") is not None
    }

    return {
        "url": url,
        "time_filter": time_filter,
        "title": title,
        "players": [
            {
                "id": p1.id,
                "name": p1.name,
                "slug": p1.slug,
                "link": p1.link,
                "maps": maps[0] if len(maps) >= 1 else None,
                "time_filter": time_filter,
                "stats": p1_stats,
            },
            {
                "id": p2.id,
                "name": p2.name,
                "slug": p2.slug,
                "link": p2.link,
                "maps": maps[1] if len(maps) >= 2 else None,
                "time_filter": time_filter,
                "stats": p2_stats,
            },
        ],
        "comparison_rows": rows,
    }


def _first_clean(values: list[str]) -> str | None:
    for value in values:
        cleaned = " ".join(value.split())
        if cleaned:
            return cleaned
    return None


def parse_player_profile_page(html: str, player: PlayerRef, url: str, time_filter: str) -> dict[str, Any]:
    """Extract the individual fields omitted by HLTV's player comparison table."""
    selector = Selector(text=html)
    stats: dict[str, str] = {}

    for wrapper in selector.css(".player-summary-stat-box-rating-wrapper"):
        label = _first_clean(wrapper.css(".player-summary-stat-box-data-description-text::text").getall())
        value = _first_clean(wrapper.css(".player-summary-stat-box-rating-data-text::text").getall())
        canonical = canonical_stat(label or "")
        if canonical and value is not None:
            stats.setdefault(canonical, value)

    for wrapper in selector.css(".player-summary-stat-box-data-wrapper"):
        label = _first_clean(wrapper.css(".player-summary-stat-box-data-text.traditionalData::text").getall())
        value = _first_clean(wrapper.css(".player-summary-stat-box-data.traditionalData::text").getall())
        canonical = canonical_stat(label or "")
        if canonical and value is not None:
            stats.setdefault(canonical, value)

    for row in selector.css("div.stats-row"):
        values = [" ".join(value.split()) for value in row.xpath("./span/text()").getall()]
        values = [value for value in values if value]
        if len(values) < 2:
            continue
        canonical = canonical_stat(values[0])
        if canonical:
            stats.setdefault(canonical, values[1])

    maps = stats.pop("Maps", None)
    map_match = re.search(r"\d+", str(maps or ""))
    return {
        "url": url,
        "time_filter": time_filter,
        "maps": int(map_match.group()) if map_match else None,
        "stats": stats,
    }


def has_stat_value(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "-", "--", "n/a", "na"}


def player_stats_complete(stats: dict[str, Any] | None) -> bool:
    stats = stats or {}
    return all(has_stat_value(stats.get(key)) for key in PLAYER_DETAIL_REQUIRED_STATS)


def comparison_details_complete(comparison: dict[str, Any]) -> bool:
    players = comparison.get("players") or []
    return bool(players) and all(player_stats_complete(player.get("stats")) for player in players)


def merge_player_profile_stats(player: dict[str, Any], detail: dict[str, Any]) -> None:
    compare_stats = dict(player.get("stats") or {})
    profile_stats = dict(detail.get("stats") or {})
    profile_maps = detail.get("maps")
    # A zero-map profile is authoritative. Keeping compare-page values here can
    # leak the other column's values when HLTV renders one side as unavailable.
    merged = {} if profile_maps == 0 else dict(compare_stats)
    # HLTV renders unavailable profile values as "-".  They are not a value
    # and must neither replace a compare-page metric nor satisfy completeness.
    merged.update({key: value for key, value in profile_stats.items() if has_stat_value(value)})
    player["compare_stats"] = compare_stats
    player["profile_stats"] = profile_stats
    player["stats"] = merged
    player["profile_url"] = detail.get("url")
    player["profile_time_filter"] = detail.get("time_filter")
    if profile_maps is not None:
        player["maps"] = profile_maps
    player["detail_status"] = "ok" if player_stats_complete(merged) else "partial"


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


def backoff_delay(attempt: int, retry_after: str | None, base_delay: float, max_delay: float) -> float:
    hinted = retry_after_seconds(retry_after)
    if hinted is not None:
        return min(max_delay, hinted) + random.uniform(0.25, 1.75)
    return min(max_delay, base_delay * (2**attempt)) + random.uniform(0.5, 2.5)


def wait_compare_cooldown(verbose: bool) -> None:
    remaining = _COMPARE_COOLDOWN_UNTIL - time.monotonic()
    if remaining <= 0:
        return
    if verbose:
        safe_print(f"  compare cooldown active; sleeping {remaining:.1f}s")
    time.sleep(remaining)


def register_compare_success(verbose: bool) -> None:
    global _COMPARE_BLOCK_STREAK
    if verbose and _COMPARE_BLOCK_STREAK:
        safe_print("  compare block streak reset after success")
    _COMPARE_BLOCK_STREAK = 0


def register_compare_block(reason: str, verbose: bool, request_label: str = "request") -> float:
    global _COMPARE_BLOCK_STREAK, _COMPARE_COOLDOWN_UNTIL
    _COMPARE_BLOCK_STREAK += 1
    adaptive = min(
        COMPARE_BLOCK_COOLDOWN_MAX,
        COMPARE_BLOCK_COOLDOWN_BASE * (2 ** max(0, _COMPARE_BLOCK_STREAK - 1)),
    ) + random.uniform(2.0, 8.0)
    if _COMPARE_BLOCK_STREAK >= COMPARE_BLOCK_STREAK_THRESHOLD:
        adaptive = max(adaptive, COMPARE_CIRCUIT_BREAKER_SLEEP + random.uniform(10.0, 45.0))
    _COMPARE_COOLDOWN_UNTIL = max(_COMPARE_COOLDOWN_UNTIL, time.monotonic() + adaptive)
    if verbose:
        safe_print(f"  BLOCK {request_label} {reason}; streak={_COMPARE_BLOCK_STREAK}; cooldown {adaptive:.1f}s")
    return adaptive


def fetch_compare_page_resilient(
    url: str,
    session: dict[str, str],
    *,
    http: requests.Session,
    timeout: int = 45,
    max_retries: int = 6,
    base_delay: float = 3.0,
    max_delay: float = 180.0,
    verbose: bool = False,
    max_cloudflare_streak: int = 3,
    request_label: str = "compare",
) -> str:
    headers = {
        "User-Agent": session["user_agent"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Referer": "https://www.hltv.org/stats",
    }
    cookies = {"cf_clearance": session["cf_clearance"]}
    attempts = max(1, max_retries)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            if verbose:
                safe_print(f"  GET {request_label} attempt {attempt + 1}/{attempts}: {url}")
            wait_compare_cooldown(verbose)
            started = time.monotonic()
            response = http.get(url, headers=headers, cookies=cookies, timeout=timeout)
            text = response.text or ""
            challenge = is_cloudflare_challenge(text)
            elapsed = time.monotonic() - started
            if response.status_code == 200 and not challenge:
                register_compare_success(verbose)
                if verbose:
                    safe_print(f"  OK {request_label} HTTP 200 {len(text)} bytes {elapsed:.1f}s")
                return text
            if response.status_code in BLOCK_HTTP_CODES or challenge:
                last_error = RuntimeError(f"HLTV bloqueo/challenge HTTP {response.status_code}")
                reason = "cloudflare_challenge" if challenge else f"HTTP {response.status_code}"
                delay = register_compare_block(reason, verbose, request_label)
                if challenge and max_cloudflare_streak > 0 and _COMPARE_BLOCK_STREAK >= max_cloudflare_streak:
                    raise CloudflareChallengeStop(
                        f"Cloudflare challenge persistente en /stats/players/{request_label} "
                        f"(streak={_COMPARE_BLOCK_STREAK}). Refresca cf_session.json o reintenta mas tarde."
                    )
                if attempt < attempts - 1:
                    delay = max(
                        delay,
                        backoff_delay(attempt, response.headers.get("Retry-After"), base_delay, max_delay),
                    )
                    if verbose:
                        safe_print(f"  BLOCK {request_label} {reason}; wait {delay:.1f}s")
                    time.sleep(delay)
                continue
            response.raise_for_status()
            register_compare_success(verbose)
            if verbose:
                safe_print(f"  OK {request_label} HTTP {response.status_code} {len(text)} bytes {elapsed:.1f}s")
            return text
        except CloudflareChallengeStop:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                delay = backoff_delay(attempt, None, base_delay, max_delay)
                if verbose:
                    safe_print(f"  ERROR compare {type(exc).__name__}: {exc}; wait {delay:.1f}s", file=sys.stderr)
                time.sleep(delay)
    raise RuntimeError(f"HLTV compare fetch failed after {attempts} attempts: {last_error}")


def fetch_compare_page(url: str, session: dict[str, str], timeout: int = 30) -> str:
    response = requests.get(
        url,
        headers={
            "User-Agent": session["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.hltv.org/stats",
        },
        cookies={"cf_clearance": session["cf_clearance"]},
        timeout=timeout,
    )
    if response.status_code == 403 or "Just a moment" in response.text[:5000]:
        raise RuntimeError("HLTV devolvió Cloudflare challenge/403; refresca cf_session.json")
    response.raise_for_status()
    return response.text


def pairwise(players: list[PlayerRef]) -> list[tuple[PlayerRef, PlayerRef]]:
    pairs = []
    for index in range(0, len(players), 2):
        p1 = players[index]
        p2 = players[index + 1] if index + 1 < len(players) else players[index]
        pairs.append((p1, p2))
    return pairs


def comparison_key(item: dict[str, Any], fallback_time_filter: str) -> tuple[str, str, str]:
    players = item.get("players") or []
    p1 = str((players[0] if len(players) > 0 else {}).get("id") or "")
    p2 = str((players[1] if len(players) > 1 else {}).get("id") or "")
    time_filter = str(item.get("time_filter") or fallback_time_filter)
    return (p1, p2, time_filter)


def failure_key(item: dict[str, Any]) -> tuple[str, str, str]:
    p1 = str((item.get("player1") or {}).get("id") or "")
    p2 = str((item.get("player2") or {}).get("id") or "")
    time_filter = str(item.get("time_filter") or "")
    if not time_filter:
        match = re.search(r"[?&]p1TimeFilter=([^&]+)", str(item.get("url") or ""))
        if match:
            time_filter = match.group(1)
    return (p1, p2, time_filter)


def player_maps(player: dict[str, Any] | None) -> int:
    if not player:
        return 0
    try:
        return int(player.get("maps") or 0)
    except (TypeError, ValueError):
        return 0


def select_adaptive_player(attempts: list[dict[str, Any]], player_index: int, min_maps: int) -> dict[str, Any] | None:
    candidates = []
    for attempt in attempts:
        players = attempt.get("players") or []
        if len(players) <= player_index:
            continue
        player = dict(players[player_index])
        player["selected_from_time_filter"] = player.get("time_filter") or attempt.get("time_filter")
        candidates.append(player)
        if player_maps(player) >= min_maps:
            player["selection_reason"] = f"first_recent_window_with_at_least_{min_maps}_maps"
            return player
    if not candidates:
        return None
    fallback = max(candidates, key=player_maps)
    fallback["selection_reason"] = f"fallback_max_maps_below_{min_maps}"
    return fallback


def adaptive_result(
    attempts: list[dict[str, Any]],
    *,
    p1: PlayerRef,
    p2: PlayerRef,
    min_maps: int,
) -> dict[str, Any]:
    selected1 = select_adaptive_player(attempts, 0, min_maps) or {
        "id": p1.id,
        "name": p1.name,
        "slug": p1.slug,
        "link": p1.link,
        "maps": None,
        "time_filter": "unknown",
        "stats": {},
        "selection_reason": "missing",
    }
    selected2 = select_adaptive_player(attempts, 1, min_maps) or {
        "id": p2.id,
        "name": p2.name,
        "slug": p2.slug,
        "link": p2.link,
        "maps": None,
        "time_filter": "unknown",
        "stats": {},
        "selection_reason": "missing",
    }
    return {
        "url": attempts[-1].get("url") if attempts else "",
        "time_filter": "adaptive",
        "selection_mode": "adaptive_recent_min_maps",
        "min_maps": min_maps,
        "attempted_time_filters": [attempt.get("time_filter") for attempt in attempts],
        "players": [selected1, selected2],
        "comparison_rows": [],
    }


def adaptive_pair_satisfied(attempts: list[dict[str, Any]], min_maps: int) -> bool:
    if not attempts:
        return False
    players = attempts[-1].get("players") or []
    if len(players) < 2:
        return False
    return player_maps(players[0]) >= min_maps and player_maps(players[1]) >= min_maps


def output_payload(
    *,
    year: int,
    time_filters: list[str],
    players_requested: int,
    results: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    stopped_reason: str | None = None,
    requests_attempted: int | None = None,
    selection_mode: str | None = None,
    min_maps: int | None = None,
) -> dict[str, Any]:
    payload = {
        "source": "https://www.hltv.org/stats/players/compare",
        "year": year,
        "time_filters": time_filters,
        "players_requested": players_requested,
        "comparisons_collected": len(results),
        "comparisons_failed": len(failures),
        "results": results,
        "failures": failures,
    }
    if stopped_reason:
        payload["stopped_reason"] = stopped_reason
    if requests_attempted is not None:
        payload["requests_attempted"] = requests_attempted
    if selection_mode:
        payload["selection_mode"] = selection_mode
    if min_maps is not None:
        payload["min_maps"] = min_maps
    return payload


def enrich_comparison_with_player_profiles(
    comparison: dict[str, Any],
    *,
    session: dict[str, str],
    http: requests.Session,
    args: argparse.Namespace,
    requests_used: list[int],
) -> str | None:
    """Fill core stats from player pages, retaining compare-page-only metrics.

    Returns a stop reason after checkpointable partial work.  A later invocation
    resumes only the missing player profiles instead of re-fetching comparisons.
    """
    for player in comparison.get("players") or []:
        if player_stats_complete(player.get("stats")):
            player.setdefault("detail_status", "ok")
            continue
        if args.max_requests > 0 and requests_used[0] >= args.max_requests:
            player["detail_status"] = "request_budget_reached"
            return "request_budget_reached"
        player_ref = PlayerRef(
            id=str(player.get("id") or ""),
            name=str(player.get("name") or ""),
            slug=str(player.get("slug") or ""),
            link=str(player.get("link") or ""),
        )
        if not player_ref.id or not player_ref.slug:
            player["detail_status"] = "missing_player_identity"
            continue
        time_filter = str(player.get("selected_from_time_filter") or player.get("time_filter") or "past3months")
        url = player_stats_url(player_ref, time_filter)
        safe_print(f"  PROFILE {player_ref.name} [{time_filter}]")
        try:
            requests_used[0] += 1
            html = fetch_compare_page_resilient(
                url,
                session,
                http=http,
                max_retries=args.max_retries,
                base_delay=args.retry_base_delay,
                max_delay=args.retry_max_delay,
                verbose=args.verbose,
                max_cloudflare_streak=args.max_cloudflare_streak,
                request_label="player_profile",
            )
            detail = parse_player_profile_page(html, player_ref, url, time_filter)
            merge_player_profile_stats(player, detail)
            safe_print(
                f"  PROFILE {'OK' if player['detail_status'] == 'ok' else 'PARTIAL'} "
                f"{player_ref.name}: fields={len(detail.get('stats') or {})} maps={player.get('maps')}"
            )
        except CloudflareChallengeStop as exc:
            player["detail_status"] = "blocked"
            player["detail_error"] = str(exc)
            return "cloudflare_challenge"
        except Exception as exc:
            player["detail_status"] = "error"
            player["detail_error"] = f"{type(exc).__name__}: {exc}"
            safe_print(f"  PROFILE FAIL {player_ref.name}: {player['detail_error']}", file=sys.stderr)
        finally:
            if args.delay:
                time.sleep(args.delay)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrapea /stats/players/compare de HLTV para jugadores de team_profiles.json."
    )
    parser.add_argument("--team-profiles-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--year", type=int, default=date.today().year)
    parser.add_argument(
        "--time-filters",
        default="past3months,past6months,past12months",
        help="Filtros HLTV separados por coma. Usa 'year' para el año indicado por --year.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-delay", type=float, default=3.0)
    parser.add_argument("--retry-max-delay", type=float, default=180.0)
    parser.add_argument("--session-file", default=str(DEFAULT_SESSION_FILE))
    parser.add_argument("--match-filter", default="allMatches")
    parser.add_argument("--map-filter", default="allMaps")
    parser.add_argument("--verbose", action="store_true", help="Muestra cada intento HTTP y esperas de backoff.")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=1000,
        help="Presupuesto maximo de paginas compare nuevas por run; 0 = sin limite.",
    )
    parser.add_argument(
        "--adaptive-time-filter",
        action="store_true",
        help="Elige la ventana mas reciente con muestra suficiente: 3m -> 6m -> 12m.",
    )
    parser.add_argument(
        "--min-maps", type=int, default=10, help="Mapas minimos por jugador para aceptar una ventana reciente."
    )
    parser.add_argument(
        "--max-cloudflare-streak",
        type=int,
        default=3,
        help="Corta y guarda parcial tras N challenges Cloudflare consecutivos; 0 = no cortar.",
    )
    parser.add_argument(
        "--no-player-profile-details",
        dest="player_profile_details",
        action="store_false",
        help="No completa ADR/DPR/Impact desde el perfil individual de cada jugador.",
    )
    parser.set_defaults(player_profile_details=True)
    args = parser.parse_args()

    team_profiles_file = Path(args.team_profiles_file)
    output_file = Path(args.output_file)
    session_file = Path(args.session_file)

    players = load_players(team_profiles_file)[: args.limit]
    session = load_session(session_file)
    time_filters = [
        (str(args.year) if item.strip() == "year" else item.strip())
        for item in args.time_filters.split(",")
        if item.strip()
    ]
    existing = read_json(output_file) if output_file.exists() else {}
    results = list(existing.get("results") or [])
    failures = list(existing.get("failures") or [])
    completed = {comparison_key(item, str(args.year)) for item in results}
    result_by_key = {comparison_key(item, str(args.year)): item for item in results}
    http = requests.Session()
    pairs = pairwise(players)
    total_requests = len(pairs) * len(time_filters) + (len(players) if args.player_profile_details else 0)
    current = 0
    new_requests = 0
    safe_print(
        f"Player compare scrape: players={len(players)} pairs={len(pairs)} "
        f"time_filters={len(time_filters)} requests={total_requests} already_done={len(completed)} "
        f"max_new_requests={args.max_requests}"
    )

    if args.adaptive_time_filter:
        total_pairs = len(pairs)
        safe_print(
            f"Adaptive time filter enabled: try {', '.join(time_filters)}; "
            f"accept first window with >= {args.min_maps} maps per player."
        )
        for pair_index, (p1, p2) in enumerate(pairs, start=1):
            key = (p1.id, p2.id, "adaptive")
            if key in completed:
                existing_result = result_by_key.get(key)
                if args.player_profile_details and existing_result and not comparison_details_complete(existing_result):
                    safe_print(f"[{pair_index}/{total_pairs}] RESUME PROFILE DETAILS {p1.name} vs {p2.name} [adaptive]")
                    request_counter = [new_requests]
                    detail_stop = enrich_comparison_with_player_profiles(
                        existing_result,
                        session=session,
                        http=http,
                        args=args,
                        requests_used=request_counter,
                    )
                    new_requests = request_counter[0]
                    write_json(
                        output_file,
                        output_payload(
                            year=args.year,
                            time_filters=time_filters,
                            players_requested=len(players),
                            results=results,
                            failures=failures,
                            stopped_reason=detail_stop,
                            requests_attempted=new_requests,
                            selection_mode="adaptive_recent_min_maps",
                            min_maps=args.min_maps,
                        ),
                    )
                    if detail_stop:
                        safe_print(f"STOP player profile enrichment: {detail_stop}; partial output saved")
                        return 0
                else:
                    safe_print(f"[{pair_index}/{total_pairs}] SKIP {p1.name} vs {p2.name} [adaptive]")
                continue
            attempts_for_pair: list[dict[str, Any]] = []
            for time_filter in time_filters:
                if args.max_requests > 0 and new_requests >= args.max_requests:
                    safe_print(
                        f"STOP request budget reached: {new_requests}/{args.max_requests} new compare pages. "
                        "Partial output saved; next start will resume."
                    )
                    write_json(
                        output_file,
                        output_payload(
                            year=args.year,
                            time_filters=time_filters,
                            players_requested=len(players),
                            results=results,
                            failures=failures,
                            stopped_reason="request_budget_reached",
                            requests_attempted=new_requests,
                            selection_mode="adaptive_recent_min_maps",
                            min_maps=args.min_maps,
                        ),
                    )
                    return 0
                url = compare_url(p1, p2, time_filter, args.match_filter, args.map_filter)
                try:
                    safe_print(f"[{pair_index}/{total_pairs}] FETCH {p1.name} vs {p2.name} [{time_filter}]")
                    new_requests += 1
                    html = fetch_compare_page_resilient(
                        url,
                        session,
                        http=http,
                        max_retries=args.max_retries,
                        base_delay=args.retry_base_delay,
                        max_delay=args.retry_max_delay,
                        verbose=args.verbose,
                        max_cloudflare_streak=args.max_cloudflare_streak,
                    )
                    parsed = parse_compare_page(html, p1, p2, url, time_filter)
                    attempts_for_pair.append(parsed)
                    parsed_players = parsed.get("players") or []
                    p1_maps = player_maps(parsed_players[0] if len(parsed_players) > 0 else None)
                    p2_maps = player_maps(parsed_players[1] if len(parsed_players) > 1 else None)
                    safe_print(
                        f"[{pair_index}/{total_pairs}] OK {p1.name} vs {p2.name} [{time_filter}] maps={p1_maps}/{p2_maps}"
                    )
                    if adaptive_pair_satisfied(attempts_for_pair, args.min_maps):
                        break
                except CloudflareChallengeStop as exc:
                    failures.append(
                        {
                            "player1": p1.__dict__,
                            "player2": p2.__dict__,
                            "time_filter": "adaptive",
                            "url": url,
                            "error": str(exc),
                        }
                    )
                    safe_print(f"[{pair_index}/{total_pairs}] STOP {exc}", file=sys.stderr)
                    write_json(
                        output_file,
                        output_payload(
                            year=args.year,
                            time_filters=time_filters,
                            players_requested=len(players),
                            results=results,
                            failures=failures,
                            stopped_reason=str(exc),
                            requests_attempted=new_requests,
                            selection_mode="adaptive_recent_min_maps",
                            min_maps=args.min_maps,
                        ),
                    )
                    return 0
                except Exception as exc:
                    failures.append(
                        {
                            "player1": p1.__dict__,
                            "player2": p2.__dict__,
                            "time_filter": time_filter,
                            "url": url,
                            "error": str(exc),
                        }
                    )
                    safe_print(
                        f"[{pair_index}/{total_pairs}] FAIL {p1.name} vs {p2.name} [{time_filter}]: {exc}",
                        file=sys.stderr,
                    )
                    break
                finally:
                    if args.delay:
                        time.sleep(args.delay)

            if attempts_for_pair:
                selected = adaptive_result(attempts_for_pair, p1=p1, p2=p2, min_maps=args.min_maps)
                results.append(selected)
                completed.add(key)
                result_by_key[key] = selected
                failures = [failure for failure in failures if failure_key(failure) != key]
                chosen = [player.get("time_filter") for player in selected.get("players") or []]
                maps = [player.get("maps") for player in selected.get("players") or []]
                safe_print(f"[{pair_index}/{total_pairs}] SELECT {p1.name} vs {p2.name} windows={chosen} maps={maps}")

                # Save the comparison before profile enrichment so Ctrl+C, a request
                # budget, or Cloudflare can resume without repeating it.
                write_json(
                    output_file,
                    output_payload(
                        year=args.year,
                        time_filters=time_filters,
                        players_requested=len(players),
                        results=results,
                        failures=failures,
                        requests_attempted=new_requests,
                        selection_mode="adaptive_recent_min_maps",
                        min_maps=args.min_maps,
                    ),
                )
                if args.player_profile_details:
                    request_counter = [new_requests]
                    detail_stop = enrich_comparison_with_player_profiles(
                        selected,
                        session=session,
                        http=http,
                        args=args,
                        requests_used=request_counter,
                    )
                    new_requests = request_counter[0]
                    if detail_stop:
                        write_json(
                            output_file,
                            output_payload(
                                year=args.year,
                                time_filters=time_filters,
                                players_requested=len(players),
                                results=results,
                                failures=failures,
                                stopped_reason=detail_stop,
                                requests_attempted=new_requests,
                                selection_mode="adaptive_recent_min_maps",
                                min_maps=args.min_maps,
                            ),
                        )
                        safe_print(f"STOP player profile enrichment: {detail_stop}; partial output saved")
                        return 0

            write_json(
                output_file,
                output_payload(
                    year=args.year,
                    time_filters=time_filters,
                    players_requested=len(players),
                    results=results,
                    failures=failures,
                    requests_attempted=new_requests,
                    selection_mode="adaptive_recent_min_maps",
                    min_maps=args.min_maps,
                ),
            )

        failures = [failure for failure in failures if failure_key(failure) not in completed]
        write_json(
            output_file,
            output_payload(
                year=args.year,
                time_filters=time_filters,
                players_requested=len(players),
                results=results,
                failures=failures,
                requests_attempted=new_requests,
                selection_mode="adaptive_recent_min_maps",
                min_maps=args.min_maps,
            ),
        )
        safe_print(f"Saved {len(results)} adaptive comparisons to {output_file}")
        return 0 if not failures else 2

    for p1, p2 in pairs:
        for time_filter in time_filters:
            current += 1
            key = (p1.id, p2.id, time_filter)
            if key in completed:
                safe_print(f"[{current}/{total_requests}] SKIP {p1.name} vs {p2.name} [{time_filter}]")
                continue
            if args.max_requests > 0 and new_requests >= args.max_requests:
                safe_print(
                    f"STOP request budget reached: {new_requests}/{args.max_requests} new compare pages. "
                    "Partial output saved; next start will resume."
                )
                write_json(
                    output_file,
                    output_payload(
                        year=args.year,
                        time_filters=time_filters,
                        players_requested=len(players),
                        results=results,
                        failures=failures,
                        stopped_reason="request_budget_reached",
                        requests_attempted=new_requests,
                    ),
                )
                return 0
            url = compare_url(p1, p2, time_filter, args.match_filter, args.map_filter)
            try:
                safe_print(f"[{current}/{total_requests}] FETCH {p1.name} vs {p2.name} [{time_filter}]")
                new_requests += 1
                html = fetch_compare_page_resilient(
                    url,
                    session,
                    http=http,
                    max_retries=args.max_retries,
                    base_delay=args.retry_base_delay,
                    max_delay=args.retry_max_delay,
                    verbose=args.verbose,
                    max_cloudflare_streak=args.max_cloudflare_streak,
                )
                parsed = parse_compare_page(html, p1, p2, url, time_filter)
                results.append(parsed)
                completed.add(key)
                failures = [failure for failure in failures if failure_key(failure) != key]
                safe_print(f"[{current}/{total_requests}] OK {p1.name} vs {p2.name} [{time_filter}]")
            except CloudflareChallengeStop as exc:
                failures.append(
                    {
                        "player1": p1.__dict__,
                        "player2": p2.__dict__,
                        "time_filter": time_filter,
                        "url": url,
                        "error": str(exc),
                    }
                )
                safe_print(f"[{current}/{total_requests}] STOP {exc}", file=sys.stderr)
                write_json(
                    output_file,
                    output_payload(
                        year=args.year,
                        time_filters=time_filters,
                        players_requested=len(players),
                        results=results,
                        failures=failures,
                        stopped_reason=str(exc),
                        requests_attempted=new_requests,
                    ),
                )
                return 0
            except Exception as exc:
                failures.append(
                    {
                        "player1": p1.__dict__,
                        "player2": p2.__dict__,
                        "time_filter": time_filter,
                        "url": url,
                        "error": str(exc),
                    }
                )
                safe_print(
                    f"[{current}/{total_requests}] FAIL {p1.name} vs {p2.name} [{time_filter}]: {exc}", file=sys.stderr
                )
            write_json(
                output_file,
                output_payload(
                    year=args.year,
                    time_filters=time_filters,
                    players_requested=len(players),
                    results=results,
                    failures=failures,
                    requests_attempted=new_requests,
                ),
            )
            if args.delay:
                time.sleep(args.delay)

    failures = [failure for failure in failures if failure_key(failure) not in completed]
    write_json(
        output_file,
        output_payload(
            year=args.year,
            time_filters=time_filters,
            players_requested=len(players),
            results=results,
            failures=failures,
            requests_attempted=new_requests,
        ),
    )
    safe_print(f"Saved {len(results)} comparisons to {output_file}")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
