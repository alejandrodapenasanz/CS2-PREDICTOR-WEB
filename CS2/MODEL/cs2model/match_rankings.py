"""Causal, opt-in learner inputs from dated ranking badges on match pages.

Kept separate from legacy global ranking features: collecting a new source
must not change the feature recipe of an already published artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import sqlite3
from typing import Any

MATCH_RANKING_VERSION = "dated_match_badges_v1"
MATCH_RANKING_DIFF_COLUMNS = [
    "match_ranking_hltv_position_advantage",
    "match_ranking_valve_position_advantage",
]
MATCH_RANKING_SYM_COLUMNS = [
    "match_ranking_available",
    "match_ranking_hltv_available",
    "match_ranking_valve_available",
    "match_ranking_edition_age_days_max",
]
MATCH_RANKING_COLUMNS = MATCH_RANKING_DIFF_COLUMNS + MATCH_RANKING_SYM_COLUMNS


@dataclass(frozen=True)
class RankingObservation:
    """Only factual fields; neither the source match's result nor its date."""

    edition: date
    captured: datetime
    position: int
    source_sha256: str
    ranking_url: str
    match_url: str


RankingStore = dict[tuple[str, str], list[RankingObservation]]


def load_match_ranking_store(conn: sqlite3.Connection) -> RankingStore:
    """Read the additive acquisition contract; older databases remain usable."""
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='match_team_ranking_observations'"
    ).fetchone():
        return {}
    store: RankingStore = {}
    for row in conn.execute(
        """SELECT hltv_team_id, ranking_type, ranking_date, captured_at_utc,
                  position, source_sha256, ranking_url, match_url
           FROM match_team_ranking_observations"""
    ):
        team, kind, edition_text, captured_text, position, digest, ranking_url, match_url = row
        try:
            edition = date.fromisoformat(str(edition_text))
            captured = datetime.fromisoformat(str(captured_text).replace("Z", "+00:00"))
            if captured.tzinfo is None:
                continue  # A local/unknown time is not proof of UTC availability.
            captured = captured.astimezone(UTC)
            rank = int(position)
        except (ValueError, TypeError, OverflowError):
            continue
        if (
            not team
            or kind not in {"hltv", "valve"}
            or rank <= 0
            or rank != position
            or edition > captured.date()
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            continue
        observation = RankingObservation(edition, captured, rank, digest, str(ranking_url), str(match_url))
        store.setdefault((str(team), str(kind)), []).append(observation)
    for observations in store.values():
        observations.sort(key=lambda item: (item.edition, item.captured, item.source_sha256, item.match_url))
    return store


def _latest_asof(
    store: RankingStore, team_id: str | int | None, kind: str, day: date
) -> tuple[RankingObservation | None, str]:
    eligible = [
        item for item in store.get((str(team_id), kind), []) if item.edition < day and item.captured.date() < day
    ]
    if not eligible:
        return None, "no_pre_day_evidence"
    latest_edition = max(item.edition for item in eligible)
    latest = [item for item in eligible if item.edition == latest_edition]
    if len({item.position for item in latest}) != 1:
        return None, "conflicting_positions_in_edition"
    # Repeated captures do not make an old edition fresh or inflate coverage.
    return min(latest, key=lambda item: (item.captured, item.source_sha256, item.match_url)), "ok"


def match_ranking_payload_asof(
    store: RankingStore,
    team1_id: str | int | None,
    team2_id: str | int | None,
    match_dt: datetime | None,
) -> dict[str, Any]:
    """One feature row per match; compare only the SAME exact ranking edition.

    The cutoff is the start of civil D, even if the caller has an exact kickoff.
    Future captures/conflicts cannot change an older prediction's inputs.
    """
    features = dict.fromkeys(MATCH_RANKING_COLUMNS, 0.0)
    evidence: dict[str, Any] = {"contract": MATCH_RANKING_VERSION, "sources": {}}
    if match_dt is None:
        evidence["reason"] = "missing_match_date"
        return {"match_ranking_features": features, "match_ranking_evidence": evidence}
    day = match_dt.date()
    for kind in ("hltv", "valve"):
        first, first_reason = _latest_asof(store, team1_id, kind, day)
        second, second_reason = _latest_asof(store, team2_id, kind, day)
        audit: dict[str, Any] = {"team1_status": first_reason, "team2_status": second_reason}
        evidence["sources"][kind] = audit
        if first is None or second is None:
            audit["reason"] = "missing_or_conflicting_team_evidence"
            continue
        if first.edition != second.edition:
            audit["reason"] = "different_editions"
            continue
        features[f"match_ranking_{kind}_position_advantage"] = float(second.position - first.position)
        features[f"match_ranking_{kind}_available"] = 1.0
        features["match_ranking_available"] = 1.0
        features["match_ranking_edition_age_days_max"] = max(
            features["match_ranking_edition_age_days_max"], float((day - first.edition).days)
        )
        audit.update(
            reason="ok",
            edition=first.edition.isoformat(),
            captured_at_utc=[item.captured.isoformat() for item in (first, second)],
            source_sha256=[item.source_sha256 for item in (first, second)],
            ranking_urls=[item.ranking_url for item in (first, second)],
            match_urls=[item.match_url for item in (first, second)],
        )
    return {"match_ranking_features": features, "match_ranking_evidence": evidence}
