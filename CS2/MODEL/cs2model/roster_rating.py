"""Causal roster-sensitive transforms for the rating challenger.

The current announced five is compared with the last complete *actual* five
that had already been observed for the organisation.  This module never owns
match ordering: :class:`ChronologicalState` calls it before observing the
current result and only stores the actual five afterwards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

from .config import RosterRatingConfig
from .glicko2 import Rating


@dataclass(frozen=True)
class ObservedLineup:
    """A complete lineup learned only after its match was observed."""

    player_ids: frozenset[str]
    observed_at: datetime


@dataclass(frozen=True)
class RosterAdjustment:
    """Immutable preview of the adjustment that applies before a match."""

    comparable: bool = False
    core_changed: bool = False
    retained_players: int = 0
    replacements: int = 0
    credit_fraction: float = 1.0
    previous_player_ids: frozenset[str] = frozenset()
    current_player_ids: frozenset[str] = frozenset()


def _player_identifier(player: Any) -> str:
    if isinstance(player, Mapping):
        for key in ("hltv_player_id", "player_id", "id"):
            value = player.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""
    return str(player).strip() if player is not None else ""


def complete_player_ids(lineup: Any, lineup_size: int = 5) -> frozenset[str] | None:
    """Return stable player IDs only for an exact, complete lineup."""

    payload = lineup
    if isinstance(payload, Mapping):
        payload = payload.get("players")
    if not isinstance(payload, Iterable) or isinstance(payload, (str, bytes, Mapping)):
        return None
    identifiers = frozenset(
        identifier for identifier in (_player_identifier(player) for player in payload) if identifier
    )
    return identifiers if len(identifiers) == int(lineup_size) else None


def evaluate_roster_change(
    current_lineup: Any,
    previous_lineup: ObservedLineup | None,
    match_at: datetime | None,
    config: RosterRatingConfig,
) -> RosterAdjustment:
    """Compare point-in-time announced players with the last observed actual five."""

    current = complete_player_ids(current_lineup, config.lineup_size)
    if current is None or previous_lineup is None or match_at is None:
        return RosterAdjustment(current_player_ids=current or frozenset())
    previous = previous_lineup.player_ids
    if len(previous) != config.lineup_size or previous_lineup.observed_at >= match_at:
        return RosterAdjustment(current_player_ids=current)
    age_days = (match_at - previous_lineup.observed_at).total_seconds() / 86400.0
    if age_days > config.lookback_days:
        return RosterAdjustment(current_player_ids=current)

    retained = len(current & previous)
    replacements = config.lineup_size - retained
    core_changed = replacements >= config.core_change_min_replacements
    if core_changed:
        raw_fraction = retained / config.lineup_size
        credit_fraction = config.minimum_credit_fraction + ((1.0 - config.minimum_credit_fraction) * raw_fraction)
    else:
        credit_fraction = 1.0
    return RosterAdjustment(
        comparable=True,
        core_changed=core_changed,
        retained_players=retained,
        replacements=replacements,
        credit_fraction=float(credit_fraction),
        previous_player_ids=previous,
        current_player_ids=current,
    )


def adjust_glicko_rating(
    rating: Rating,
    adjustment: RosterAdjustment,
    config: RosterRatingConfig,
) -> Rating:
    """Discount credit around neutral and inflate RD without a hard reset.

    With ``f`` equal to ``credit_fraction``::

        r' = neutral + f * (r - neutral)
        RD'^2 = f * RD^2 + (1-f) * RD_max^2

    The floor in ``f`` preserves some organisational history even if none of
    the previous five remains.  Sigma, period and games stay traceable.
    """

    if not adjustment.core_changed:
        return rating.copy()
    fraction = adjustment.credit_fraction
    result = rating.copy()
    result.rating = config.neutral_rating + fraction * (rating.rating - config.neutral_rating)
    result.rd = min(
        config.maximum_rd,
        math.sqrt(fraction * rating.rd * rating.rd + (1.0 - fraction) * config.maximum_rd * config.maximum_rd),
    )
    return result


def adjust_elo_rating(
    rating: float,
    adjustment: RosterAdjustment,
    config: RosterRatingConfig,
) -> float:
    """Apply the same proportional credit rule to the Elo challenger."""

    if not adjustment.core_changed:
        return float(rating)
    return float(config.neutral_rating + adjustment.credit_fraction * (float(rating) - config.neutral_rating))
