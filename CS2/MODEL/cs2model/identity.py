"""Stable team identity and provisional-participant guards."""

from __future__ import annotations

import re
from typing import Any


_PROVISIONAL_PATTERNS = (
    re.compile(r"^\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:tbd|tba|to be determined|unknown)\s*$", re.IGNORECASE),
    re.compile(r"\b(?:winner|loser)\b", re.IGNORECASE),
    re.compile(r"\b(?:ganador|perdedor)\b", re.IGNORECASE),
)


def is_provisional_team_name(value: Any) -> bool:
    """Return True for bracket placeholders that are not a resolved team."""
    name = str(value or "").strip()
    return any(pattern.search(name) for pattern in _PROVISIONAL_PATTERNS)


def resolved_team(team: Any) -> bool:
    """A resolved HLTV participant has a non-placeholder name and stable id."""
    return (
        isinstance(team, dict)
        and not is_provisional_team_name(team.get("name"))
        and bool(str(team.get("id") or team.get("hltv_id") or "").strip())
    )


def choose_match_team(record: dict[str, Any], side: str) -> dict[str, Any] | None:
    """Choose the authoritative participant without erasing prematch snapshots.

    For a completed match, the final detail page is the match fact. Before the
    result, the direct/upcoming participant remains the current announced fact.
    """
    direct = record.get(side)
    detail = ((record.get("detail") or {}).get("match") or {}).get(side)
    if str(record.get("status") or "").lower() == "completed" and resolved_team(detail):
        return detail
    if resolved_team(direct):
        return direct
    if resolved_team(detail):
        return detail
    if isinstance(direct, dict) and direct.get("name"):
        return direct
    if isinstance(detail, dict) and detail.get("name"):
        return detail
    return None
