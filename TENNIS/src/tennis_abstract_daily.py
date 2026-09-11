"""Resumable daily Tennis Abstract player acquisition, in shadow mode.

The native inventory comes from validated ATP/WTA Elo report links, never
from guessed name slugs. Source dates and identities remain quarantined from
Elo/features until their independent mapping contracts are established.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
import re
from typing import Any, Callable
from urllib.parse import parse_qsl, urlsplit

from .responsible_http import RateLimitedError, ResponsibleHttpError, WafBlockedError
from .tennis_abstract_access import (
    ORIGIN,
    TennisAbstractAcquisitionClient,
    validate_acquisition_url,
    ACCESS_RECOVERY_VERSION,
    browser_recovery_authorized,
)
from .tennis_abstract_elo import load_latest_elo
from .tennis_abstract_browser import TennisAbstractPlayerTimeout
from .tennis_abstract_history_audit import fetch_player_history
from .tennis_abstract_store import DEFAULT_STORE, TennisAbstractStore, acquisition_lock, utc_text

PARSER_CONTRACT_VERSION = 2  # Operator-approved quarantine of incomplete literal rows.
PLAYER_TIMEOUT_RETRY_SECONDS = 1800
MAX_CONSECUTIVE_PLAYER_TIMEOUTS = 3
TIMEOUT_STREAK_PAUSE_SECONDS = 900


def isolated_player_timeout(error: Exception) -> TennisAbstractPlayerTimeout | None:
    """Recover only the browser's typed evidence through the common client's wrappers."""

    if type(error) not in {ResponsibleHttpError, TennisAbstractPlayerTimeout}:
        return None
    current: BaseException | None = error
    for _ in range(16):
        if isinstance(current, TennisAbstractPlayerTimeout):
            return current
        if current is None or isinstance(current, (WafBlockedError, RateLimitedError)):
            return None
        current = current.__cause__
    return None


def pause_applies(pause: dict | None, now: str) -> bool:
    """A parser revision may retry a schema failure, never override a WAF/429 pause."""

    if not pause or pause["retry_at_utc"] <= now:
        return False
    revised_schema = (
        pause.get("type") in {"ValueError", "SyntaxError"}
        and pause.get("parser_contract_version", 1) < PARSER_CONTRACT_VERSION
    )
    return not revised_schema


def can_recover_local_waf_pause(pause: dict | None) -> bool:
    """Allow one recovery of an old LOCAL WAF pause, never a server Retry-After."""

    return bool(
        pause
        and pause.get("type") == "WafBlockedError"
        and pause.get("access_recovery_version", 0) < ACCESS_RECOVERY_VERSION
        and not pause.get("server_retry_after_utc")
        and browser_recovery_authorized()
    )


def can_explicitly_retry_local_failure(pause: dict | None) -> bool:
    """An operator-triggered browser retry cannot erase a provider rate-limit deadline."""

    return bool(
        pause
        and pause.get("type")
        in {"WafBlockedError", "ValueError", "SyntaxError", "ResponsibleHttpError", "RuntimeError"}
        and not pause.get("server_retry_after_utc")
    )


@dataclass(frozen=True)
class InventoryPlayer:
    """Native source identity plus provenance of the discovered profile link."""

    gender: str
    key: str
    name: str
    profile_url: str
    inventory_url: str
    inventory_captured_at_utc: str
    rank: int
    selection_evidence: tuple[dict[str, str], ...] = ()


def inventory_player(row: dict) -> InventoryPlayer:
    """Accept an exact known report link, without fuzzy player-name mapping."""

    gender = str(row["gender"])
    parsed = urlsplit(str(row["player_url"]))
    expected = {"M": "/cgi-bin/player.cgi", "F": "/cgi-bin/wplayer.cgi"}.get(gender)
    if (
        parsed.scheme != "https"
        or parsed.netloc not in {"tennisabstract.com", "www.tennisabstract.com"}
        or parsed.path != expected
        or parsed.fragment
    ):
        raise ValueError("Unrecognized Tennis Abstract inventory profile link.")
    url = ORIGIN + parsed.path + "?" + parsed.query
    validate_acquisition_url(url)
    query = parse_qsl(parsed.query)
    key = query[0][1]
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", key) is None:
        raise ValueError(
            "Modern/numeric profile keys require a separately inspected history contract."
        )
    return InventoryPlayer(
        gender,
        key,
        str(row["player_name"]),
        url,
        str(row["source_url"]),
        utc_text(datetime.fromisoformat(str(row["retrieved_at_utc"]))),
        int(row["elo_rank"]),
    )


def load_inventory() -> list[InventoryPlayer]:
    """Use only validated local source reports, refreshed by the launcher beforehand."""

    players = [
        inventory_player(row)
        for gender in ("M", "F")
        for row in load_latest_elo(gender).to_dict("records")
    ]
    identifiers = [(player.gender, player.key) for player in players]
    if not players or len(identifiers) != len(set(identifiers)):
        raise ValueError("Empty or duplicate Tennis Abstract player inventory.")
    return sorted(players, key=lambda player: (player.rank, player.gender, player.key))


def utc_now() -> datetime:
    """Timestamp after each completed fetch, including retries across midnight."""

    return datetime.now(UTC)


def acquisition_status(
    *,
    store_path: Path = DEFAULT_STORE,
    players: list[InventoryPlayer] | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> dict[str, Any]:
    """Read committed progress without network, writes or the acquisition writer lock."""

    inventory = load_inventory() if players is None else players
    identifiers = {(player.gender, player.key) for player in inventory}
    if not inventory or len(identifiers) != len(inventory):
        raise ValueError("Status requires a nonempty inventory with unique native identities.")
    if store_path.resolve().name != DEFAULT_STORE.name:
        raise ValueError("Use a dedicated tennis_abstract.sqlite3 source store.")
    now = utc_text(clock())
    report: dict[str, Any] = {
        "checked_at_utc": now,
        "utc_date": now[:10],
        "status": "not_started",
        "inventory_players": len(inventory),
        "selection_scope": "ta_elo_inventory"
        if players is None
        else "explicit_native_player_selection",
        "coverage_scope": "all_stored_players",
        "stored_inventory_players": 0,
        "never_acquired": len(inventory),
        "refreshed_today": 0,
        "pending_today": len(inventory),
        "last_success_utc": None,
        "pause": None,
        "deferred_player_timeouts": [],
        "last_run": None,
        "production_changed": False,
        "mode": "shadow_acquisition_not_model_input",
        "model_ready_rows": 0,
        "database_path": str(store_path.resolve()),
    }
    if not store_path.exists():
        return report
    with closing(TennisAbstractStore(store_path, read_only=True)) as store:
        # One read transaction: counts and timestamps describe the same committed snapshot.
        store.connection.execute("BEGIN")
        checks = store.checks()
        successes = [
            str(checks[identity]["last_success_utc"])
            for identity in identifiers
            if checks.get(identity, {}).get("last_success_utc")
        ]
        today = sum(value[:10] == now[:10] for value in successes)
        pause = store.state("pause")
        player_pauses = store.state("player_timeouts") or {}
        report["last_agenda_selection"] = store.state("last_agenda_selection")
        last = store.state("last_report")
        report.update(store.coverage())
        report.update(
            status="deferred"
            if pause_applies(pause, now)
            else ("up_to_date" if today == len(inventory) else "pending"),
            stored_inventory_players=len(successes),
            never_acquired=len(inventory) - len(successes),
            refreshed_today=today,
            pending_today=len(inventory) - today,
            last_success_utc=max(successes, default=None),
            pause=pause,
            deferred_player_timeouts=[
                item
                for item in player_pauses.values()
                if item["retry_at_utc"] > now and tuple(item["identity"]) in identifiers
            ],
            last_run={
                key: last.get(key)
                for key in ("status", "selection_scope", "inventory_players", "finished_at_utc")
            }
            if last
            else None,
        )
        return report


def update_daily(
    *,
    store_path: Path = DEFAULT_STORE,
    players: list[InventoryPlayer] | None = None,
    client: TennisAbstractAcquisitionClient | None = None,
    clock: Callable[[], datetime] = utc_now,
    max_profiles: int | None = None,
    browser_recovery: bool = False,
    progress: Callable[[str], None] | None = None,
    selection_context: dict[str, Any] | None = None,
) -> dict:
    """Commit each successful player; pause safely with visible pending coverage."""

    if max_profiles is not None and (isinstance(max_profiles, bool) or max_profiles <= 0):
        raise ValueError("max_profiles must be positive, or omitted for the full inventory.")
    if browser_recovery and not browser_recovery_authorized():
        raise ValueError("Explicit browser recovery requires dated operator authorization.")
    inventory = load_inventory() if players is None else players
    if not inventory:
        raise ValueError("An empty inventory cannot count as a successful daily update.")
    identifiers = [(player.gender, player.key) for player in inventory]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate player identity in acquisition inventory.")
    start = utc_text(clock())
    day = start[:10]
    with acquisition_lock(store_path), closing(TennisAbstractStore(store_path)) as store:
        checks = store.checks()
        pending = [
            player
            for player in inventory
            if (checks.get((player.gender, player.key), {}).get("last_success_utc") or "")[:10]
            != day
        ]
        # A rejected/failed player does not permanently starve untouched players.
        pending.sort(
            key=lambda player: (
                checks.get((player.gender, player.key), {}).get("last_attempt_utc", ""),
                player.rank,
                player.gender,
                player.key,
            )
        )
        report: dict[str, Any] = {
            "started_at_utc": start,
            "utc_date": day,
            "status": "already_refreshed",
            "inventory_players": len(inventory),
            "selection_scope": "daily_agenda"
            if selection_context is not None
            else ("ta_elo_inventory" if players is None else "explicit_native_player_selection"),
            "agenda_selection": selection_context,
            "coverage_scope": "all_stored_players",
            "already_refreshed": len(inventory) - len(pending),
            "refreshed": 0,
            "attempted": 0,
            "player_timeouts": [],
            "skipped_player_timeouts": 0,
            "pending": len(pending),
            "failure": None,
            "database_path": str(store.path),
            "production_changed": False,
            "mode": "shadow_acquisition_not_model_input",
            "parser_contract_version": PARSER_CONTRACT_VERSION,
            "oldest_inventory_capture_utc": min(p.inventory_captured_at_utc for p in inventory),
        }
        pause = store.state("pause")
        player_pauses = store.state("player_timeouts") or {}
        if selection_context is not None:
            store.set_state("last_agenda_selection", selection_context)
        recover_waf = can_recover_local_waf_pause(pause) if client is None else False
        explicit_local_retry = browser_recovery and can_explicitly_retry_local_failure(pause)
        if pending and pause_applies(pause, start) and not (recover_waf or explicit_local_retry):
            report.update(status="deferred", failure=pause)
        elif pending:
            if pause:
                store.set_state("pause", None)
            active = client or TennisAbstractAcquisitionClient(
                start_in_browser=recover_waf or browser_recovery
            )
            timeout_streak = 0
            try:
                for player in pending:
                    if max_profiles is not None and report["attempted"] >= max_profiles:
                        break
                    player_identity = f"{player.gender}:{player.key}"
                    player_pause = player_pauses.get(player_identity)
                    if player_pause and player_pause["retry_at_utc"] > utc_text(clock()):
                        report["skipped_player_timeouts"] += 1
                        report["status"] = "partial"
                        continue
                    report["attempted"] += 1
                    if progress:
                        progress(
                            f"{report['attempted']}/{len(pending)} {player.gender}:{player.key}"
                        )
                    try:
                        history = fetch_player_history(
                            active, player.gender, player.key, quarantine_incomplete=True
                        )
                        store.save(history, captured_at=clock(), inventory=asdict(player))
                    except (ResponsibleHttpError, ValueError, SyntaxError, OSError) as exc:
                        at = clock()
                        timeout = isolated_player_timeout(exc)
                        store.record_failure(player.gender, player.key, at, str(timeout or exc))
                        if timeout is not None:
                            timeout_streak += 1
                            item = {
                                "type": "TennisAbstractPlayerTimeout",
                                "player": player_identity,
                                "identity": [player.gender, player.key],
                                "message": str(timeout),
                                "attempted_at_utc": utc_text(at),
                                "retry_at_utc": utc_text(
                                    at + timedelta(seconds=PLAYER_TIMEOUT_RETRY_SECONDS)
                                ),
                            }
                            player_pauses[player_identity] = item
                            store.set_state("player_timeouts", player_pauses)
                            report["player_timeouts"].append(item)
                            report["status"] = "partial"
                            if progress:
                                progress(
                                    f"TIMEOUT {player_identity}; ficha aplazada hasta "
                                    f"{item['retry_at_utc']}; se conserva el ultimo dato valido."
                                )
                            if timeout_streak < MAX_CONSECUTIVE_PLAYER_TIMEOUTS:
                                continue
                            streak_failure = {
                                "type": "ConsecutivePlayerTimeouts",
                                "message": f"{timeout_streak} timeouts consecutivos; pausa local breve.",
                                "retry_at_utc": utc_text(
                                    at + timedelta(seconds=TIMEOUT_STREAK_PAUSE_SECONDS)
                                ),
                            }
                            store.set_state("pause", streak_failure)
                            report.update(status="deferred", failure=streak_failure)
                            break
                        failure: dict[str, Any] = {
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "player": f"{player.gender}:{player.key}",
                            "parser_contract_version": PARSER_CONTRACT_VERSION,
                            "access_recovery_version": ACCESS_RECOVERY_VERSION,
                        }
                        if isinstance(exc, RateLimitedError):
                            failure["retry_at_utc"] = utc_text(
                                datetime.fromtimestamp(exc.retry_at, UTC)
                            )
                        else:
                            # WAF/schema/network errors are not retried repeatedly in this run.
                            failure["retry_at_utc"] = utc_text(at + timedelta(days=1))
                        if isinstance(exc, WafBlockedError):
                            failure["status_code"] = exc.response.status_code
                        store.set_state("pause", failure)
                        report.update(
                            status="deferred" if isinstance(exc, RateLimitedError) else "failed",
                            failure=failure,
                        )
                        break
                    timeout_streak = 0
                    if player_identity in player_pauses:
                        del player_pauses[player_identity]
                        store.set_state("player_timeouts", player_pauses)
                    report["refreshed"] += 1
                    report["pending"] -= 1
                    report["status"] = "partial" if report["pending"] else "completed"
            finally:
                if client is None:
                    active.close()
        report.update(store.coverage())
        if (
            selection_context is not None
            and (
                selection_context.get("unresolved_players")
                or selection_context.get("unusable_agenda_rows")
            )
            and report["status"] in {"completed", "already_refreshed"}
        ):
            report["status"] = "partial_selection"
        report["finished_at_utc"] = utc_text(clock())
        store.set_state("last_report", report)
        return report
