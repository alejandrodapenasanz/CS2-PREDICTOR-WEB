"""Bounded daily detail recovery, independent from model/feature ingestion.

Reuse the existing result mappers, sanctioned store and approved HTTP clients.
Completed settlements are NOT a reason to stop searching for missing sets.
"""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable

import pandas as pd

from ..config import OPERATIONS_DATABASE_PATH, PLAYER_MAPPING_DATABASE_PATH, STATE_ROOT
from ..tennis_explorer import refresh_daily_results
from ..tennisratio import load_mapped_results
from ..tennisratio.client import TennisRatioClient
from ..tennisratio.parser import parse_profile_html
from .result_details import coverage, mark_attempt, pending, replay_details, result_status
from .result_sources import (
    build_tennis_explorer_mapped_result_snapshot,
    build_tennisratio_result_snapshot,
)
from .store import OperationsStore
from .types import OperationsValidationError, OperationsConflictError


def _blocked(exc: Exception) -> bool:
    """Stop the source batch on WAF/rate-limit evidence, never bypass it."""
    return any(
        token in f"{type(exc).__name__}: {exc}".lower()
        for token in ("blocked", "waf", "cloudflare", "429", "403", "rate limit", "cooldown")
    )


def _accept(
    store: OperationsStore,
    rows: list[dict[str, Any]],
    source: str,
    now: datetime,
    errors: list[str] | None = None,
) -> int:
    """Use only the sanctioned append-only reconciliation API, one row at a time."""
    settled = 0
    for row in rows:
        row = {
            **row,
            "observed_at_utc": row.get("observed_at_utc")
            or row.get("retrieved_at_utc")
            or now.isoformat(),
        }
        # Canonical JSON also makes retrying identical evidence idempotent.
        encoded = json.dumps(row, sort_keys=True, default=str)
        run_id = "result-detail:" + hashlib.sha256((source + encoded).encode()).hexdigest()
        try:
            report = store.reconcile_observations(
                run_id,
                pd.DataFrame([row]),
                source_system=source,
                observed_at_utc=row["observed_at_utc"],
                metadata={"pipeline": "bounded_result_details_v1", "feature_source": False},
            )
            settled += report.settlements_inserted
        except (ValueError, TypeError, OperationsValidationError, OperationsConflictError) as exc:
            if errors is None:
                raise
            errors.append(f"{row.get('source_match_id')}: {type(exc).__name__}: {exc}")
    return settled


def _profile_urls(database: Path) -> dict[tuple[str, str], str]:
    """Read already observed exact profile URLs, never guess a URL from a name."""
    if not database.is_file():
        return {}
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        return {
            (gender, "tennisratio:" + key.removeprefix("tennisratio:")): url
            for gender, key, url in conn.execute(
                "SELECT gender,source_player_key,source_url FROM player_observations ORDER BY first_seen_at_utc"
            )
        }


def profile_observations(
    profile: Any, payload: Any, targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join only a unique exact date/gender/native-player pair; no fuzzy names.

    Current captures may settle earlier events but do NOT enter any Elo or
    feature source. Result/availability clocks remain distinct in the payload.
    """
    subject = "tennisratio:" + profile.source_player_key.removeprefix("tennisratio:")
    candidates: dict[tuple[str, frozenset[str]], list[Any]] = {}
    for match in profile.matches:
        if not match.rival_source_key or match.effective_date > payload.retrieved_at_utc.date():
            continue
        rival = "tennisratio:" + match.rival_source_key.removeprefix("tennisratio:")
        candidates.setdefault(
            (match.effective_date.isoformat(), frozenset((subject, rival))), []
        ).append(match)
    output = []
    for target in targets:
        pair = frozenset((target["player_1_slug"], target["player_2_slug"]))
        same_targets = [
            r
            for r in targets
            if r["gender"] == target["gender"]
            and r["match_date"] == target["match_date"]
            and frozenset((r["player_1_slug"], r["player_2_slug"])) == pair
        ]
        matches = candidates.get((target["match_date"], pair), [])
        if len(same_targets) != 1 or len(matches) != 1 or target["gender"] != profile.gender:
            continue
        match = matches[0]
        winner = subject if match.result == "Win" else next(iter(pair - {subject}))
        first_wins = winner == target["player_1_slug"]
        # ProfileMatch.score is oriented to the profile player, not always winner.
        score = match.score_winner_perspective or match.score
        orientation = "winner" if match.score_winner_perspective else "player_1"
        row = {
            "source_match_id": target["source_match_id"],
            "match_date": target["match_date"],
            "player_1_slug": target["player_1_slug"],
            "player_2_slug": target["player_2_slug"],
            "status": "finished",
            "winner_slug": winner,
            "winner_side": "player_1" if first_wins else "player_2",
            "player_1_sets_won": match.winner_sets_won if first_wins else match.loser_sets_won,
            "player_2_sets_won": match.loser_sets_won if first_wins else match.winner_sets_won,
            "sets_score": score,
            "score_orientation": orientation,
            "score_subject_slug": subject,
            "source_url": payload.source_url,
            "observed_at_utc": payload.retrieved_at_utc.isoformat(),
            "retrieved_at_utc": payload.retrieved_at_utc.isoformat(),
            "source_snapshot_sha256": payload.sha256,
            "result_evidence": "tennisratio_exact_native_pair_date",
            "source_result_payload": match.raw_payload,
        }
        if not match.score_winner_perspective:
            row["score_orientation"] = "subject"
        row["status"] = result_status(row)
        output.append(row)
    return output


def recover_results(
    *,
    database_path: Path = OPERATIONS_DATABASE_PATH,
    ratio_database_path: Path = STATE_ROOT / "data" / "processed" / "tennisratio.sqlite3",
    mapping_database_path: Path = PLAYER_MAPPING_DATABASE_PATH,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    max_dates: int = 3,
    max_profiles: int = 10,
    result_refresher: Callable[..., Any] = refresh_daily_results,
    ratio_loader: Callable[..., Any] = load_mapped_results,
    profile_client: Any = None,
    exclude_fetched_dates: tuple[date, ...] = (),
) -> dict[str, Any]:
    """Enrich past results locally first, then limited profiles/dates with backoff.

    A failure leaves the match queued; it cannot stop predictions. Database
    integrity errors still propagate to the caller as explicit failures.
    """
    now = clock().astimezone(UTC)
    report: dict[str, Any] = {
        "reviewed": 0,
        "new_results": 0,
        "enriched_results": 0,
        "sets_added": 0,
        "settled": 0,
        "errors": [],
        "profiles_requested": 0,
    }
    with OperationsStore(database_path, clock=clock) as store:
        before = coverage(store.connection, now)
        sets_before = store.connection.execute("SELECT COUNT(*) FROM result_sets").fetchone()[0]
        report["local_replay"] = replay_details(store.connection)
        queue = [
            r
            for r in pending(store.connection, now)
            if not r["next_attempt_at_utc"]
            or datetime.fromisoformat(r["next_attempt_at_utc"]) <= now
        ]
        dates = list(dict.fromkeys(r["match_date"] for r in queue))[: max(0, max_dates)]
        targets = [r for r in queue if r["match_date"] in dates]
        report["reviewed"] = len(targets)
        errors_by_date: dict[str, str] = {}
        # Reuse stored Ratio results, also for settled-but-incomplete matches.
        for day in dates:
            try:
                snapshot = build_tennisratio_result_snapshot(
                    store,
                    pending_date=date.fromisoformat(day),
                    as_of_date=now.date() + timedelta(days=1),
                    loader=ratio_loader,
                    include_settled=True,
                )
                if snapshot:
                    rows = [
                        dict(r)
                        for r in snapshot.matches.to_dict(orient="records")
                        if pd.Timestamp(r["observed_at_utc"]) <= now
                    ]
                    report["settled"] += _accept(store, rows, "tennisratio", now, report["errors"])
            except Exception as exc:
                errors_by_date[day] = f"local TennisRatio: {type(exc).__name__}: {exc}"
        remaining = {r["source_match_id"] for r in pending(store.connection, now)}
        targets = [r for r in targets if r["source_match_id"] in remaining]
        urls = _profile_urls(ratio_database_path) if max_profiles > 0 and targets else {}
        client = profile_client
        visited: set[str] = set()
        try:
            for target in targets:
                url = next(
                    (
                        urls[(target["gender"], slug)]
                        for slug in (target["player_1_slug"], target["player_2_slug"])
                        if (target["gender"], slug) in urls
                    ),
                    None,
                )
                if not url or url in visited or len(visited) >= max_profiles:
                    continue
                visited.add(url)
                report["profiles_requested"] += 1
                try:
                    client = client or TennisRatioClient(clock=clock)
                    payload = client.get(url, retrieved_at_utc=clock(), cache_mode="refresh")
                    profile = parse_profile_html(payload.content, source_url=payload.source_url)
                    rows = profile_observations(profile, payload, targets)
                    report["settled"] += _accept(store, rows, "tennisratio", now, report["errors"])
                except Exception as exc:
                    errors_by_date[target["match_date"]] = (
                        f"TennisRatio: {type(exc).__name__}: {exc}"
                    )
                    if _blocked(exc):
                        break
        finally:
            if client is not None and profile_client is None:
                client.close()
        # Native fallback results are retained even without a model prediction.
        # Cross-source mapping continues to use the established strict mapper.
        explorer_blocked = False
        for day in dates:
            day_targets = [r for r in pending(store.connection, now) if r["match_date"] == day]
            if not day_targets:
                continue
            if date.fromisoformat(day) not in exclude_fetched_dates and not explorer_blocked:
                try:
                    snapshot = result_refresher(date.fromisoformat(day), clock=clock)
                    mapped = build_tennis_explorer_mapped_result_snapshot(
                        store,
                        snapshot,
                        mapping_database_path=mapping_database_path,
                        include_settled=True,
                    )
                    keys = {r["source_match_id"] for r in day_targets}
                    rows_by_id = {
                        str(r["source_match_id"]): dict(r)
                        for r in snapshot.matches.to_dict(orient="records")
                        if str(r.get("source_match_id")) in keys
                    }
                    rows_by_id.update(
                        {
                            str(r["source_match_id"]): dict(r)
                            for r in mapped.matches.to_dict(orient="records")
                            if str(r["source_match_id"]) in keys
                        }
                    )
                    report["settled"] += _accept(
                        store,
                        list(rows_by_id.values()),
                        "tennis_explorer",
                        snapshot.retrieved_at_utc,
                        report["errors"],
                    )
                except Exception as exc:
                    errors_by_date[day] = f"Tennis Explorer: {type(exc).__name__}: {exc}"
                    explorer_blocked = _blocked(exc)
            mark_attempt(
                store.connection,
                [r["source_match_id"] for r in day_targets],
                now,
                errors_by_date.get(day),
            )
        after = coverage(store.connection, now)
        conflicts = store.connection.execute(
            "SELECT COUNT(*) FROM result_detail_conflicts"
        ).fetchone()[0]
        report.update(
            coverage=after,
            pending=after["pending"],
            errors=report["errors"] + list(errors_by_date.values()),
            detail_conflicts=conflicts,
            new_results=max(0, after["with_winner"] - before["with_winner"]),
            enriched_results=max(0, after["with_detailed_sets"] - before["with_detailed_sets"]),
            sets_added=store.connection.execute("SELECT COUNT(*) FROM result_sets").fetchone()[0]
            - sets_before,
        )
    print("[TENNIS resultados] " + json.dumps(report, ensure_ascii=False), flush=True)
    return report
