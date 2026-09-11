"""Exact schedule-to-Sackmann metadata; no result or same-day evidence.

This is a serving adapter, not a new model feature family. Unknown rounds and
formats stay absent; the existing fitted preprocessing handles their nulls.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
import re


CONTRACT_VERSION = "schedule_match_format_v1"
MAIN_DRAW_ROUNDS = frozenset({"R128", "R64", "R32", "R16", "QF", "SF", "F"})
SACKMANN_ROUNDS = MAIN_DRAW_ROUNDS | {"Q1", "Q2", "Q3", "Q4", "RR", "BR", "ER"}
_ROUND_NAMES = {
    "final": "F",
    "finals": "F",
    "semifinal": "SF",
    "semifinals": "SF",
    "quarterfinal": "QF",
    "quarterfinals": "QF",
    "round robin": "RR",
}


def normalize_round(value: object) -> str | None:
    """Translate only unambiguous draw sizes/names, never ordinal rounds."""

    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.upper() in SACKMANN_ROUNDS:
        return raw.upper()
    token = " ".join(raw.casefold().replace("-", "").split())
    if token in _ROUND_NAMES:
        return _ROUND_NAMES[token]
    match = re.fullmatch(r"round of (128|64|32|16|8|4|2)", token)
    if match:
        return {"8": "QF", "4": "SF", "2": "F"}.get(match[1], f"R{match[1]}")
    match = re.fullmatch(r"qualifying (?:round )?([1-4])", token)
    return f"Q{match[1]}" if match else None


@dataclass(frozen=True)
class MatchFormat:
    """Resolved inputs plus an explicit reason for each absent value."""

    best_of: int | None
    round: str | None
    best_of_reason: str
    round_reason: str


def resolve_match_format(
    *,
    match_date: date,
    captured_at_utc: datetime | None,
    gender: str,
    source_family: str,
    tournament_level: object,
    tournament: object,
    round_raw: object,
    round_evidence: object,
) -> MatchFormat:
    """Use only authenticated pre-D schedule context and standard singles rules.

    ATP/WTA/Challenger/ITF singles use three sets in the modern rule regime.
    Grand Slam men's MAIN draw requires five. Male Slam qualifying is left
    unknown because format exceptions require exact event/rule evidence.
    Special formats and historical non-Slam rules are deliberately not guessed.
    """

    if (
        captured_at_utc is None
        or captured_at_utc.tzinfo is None
        or captured_at_utc.astimezone(UTC).date() >= match_date
    ):
        return MatchFormat(None, None, "no_pre_date_evidence", "no_pre_date_evidence")
    if source_family != "tennisratio":
        return MatchFormat(None, None, "unsupported_source", "unsupported_source")
    round_code = normalize_round(round_raw)
    round_reason = "exact_schedule_round" if round_code else "round_missing_or_ambiguous"
    if not isinstance(round_evidence, str) or round_evidence not in {
        "match",
        "tournament_group",
        "match_and_group",
    }:
        round_code = None
        round_reason = "round_missing_or_conflicting"
    level = tournament_level if isinstance(tournament_level, str) else ""
    title = tournament.casefold() if isinstance(tournament, str) else ""
    if (
        any(token in title for token in ("qualies", "qualifying", "qualification"))
        and round_code in MAIN_DRAW_ROUNDS
    ):
        return MatchFormat(None, None, "qualifying_draw_ambiguous", "qualifying_round_ambiguous")
    special = any(
        x in title
        for x in (
            "next gen",
            "nextgen",
            "davis cup",
            "exhibition",
            "doubles",
            "junior",
            "wheelchair",
        )
    )
    incompatible_level = (gender == "M" and level == "WTA") or (
        gender == "F" and level in {"ATP", "Challengers"}
    )
    if gender not in {"M", "F"} or special or not title.strip() or incompatible_level:
        return MatchFormat(None, round_code, "format_not_accredited", round_reason)
    if level == "Grand Slams":
        if round_code in MAIN_DRAW_ROUNDS:
            return MatchFormat(
                5 if gender == "M" else 3, round_code, "slam_main_draw_singles", round_reason
            )
        if gender == "F" and round_code in {"Q1", "Q2", "Q3"}:
            return MatchFormat(3, round_code, "slam_women_qualifying", round_reason)
        return MatchFormat(None, round_code, "slam_draw_format_unverified", round_reason)
    if match_date.year >= 2009 and level in {"ATP", "WTA", "Challengers", "Futures"}:
        return MatchFormat(3, round_code, "standard_tour_singles", round_reason)
    return MatchFormat(None, round_code, "level_format_unverified", round_reason)
