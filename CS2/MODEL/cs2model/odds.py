"""Point-in-time contract for CS2 opening odds.

Only a two-way decimal market whose first observation is strictly before the
published UTC kickoff is eligible.  The helper deliberately ignores stored
normalized probabilities and recomputes the de-vigged pair from decimal odds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


ODDS_FEATURE_COLUMNS: tuple[str, ...] = (
    "opening_odds_prob_centered",
    "opening_odds_confidence",
    "opening_bookmaker_count_log",
    "odds_available",
)

MIN_DECIMAL_ODDS = 1.001
MAX_DECIMAL_ODDS = 100.0
MIN_OVERROUND = 0.90
MAX_OVERROUND = 1.50


@dataclass(frozen=True)
class OpeningOdds:
    """Validated first two-way opening market for one match."""

    fair_prob_team1: float
    fair_prob_team2: float
    decimal_team1: float
    decimal_team2: float
    overround: float
    bookmaker_count: int
    captured_at_utc: str
    source: str
    recovered: bool = False

    def feature_values(self, *, reverse: bool = False) -> dict[str, float]:
        """Return symmetric model features for the requested team order."""

        probability = self.fair_prob_team2 if reverse else self.fair_prob_team1
        return {
            "opening_odds_prob_centered": probability - 0.5,
            "opening_odds_confidence": abs(probability - 0.5),
            "opening_bookmaker_count_log": math.log1p(max(self.bookmaker_count, 0)),
            "odds_available": 1.0,
        }

    def as_dict(self) -> dict[str, Any]:
        """Serialize the auditable normalized contract."""

        return {
            "available": True,
            "fair_prob_team1": self.fair_prob_team1,
            "fair_prob_team2": self.fair_prob_team2,
            "decimal_team1": self.decimal_team1,
            "decimal_team2": self.decimal_team2,
            "overround": self.overround,
            "bookmaker_count": self.bookmaker_count,
            "captured_at_utc": self.captured_at_utc,
            "source": self.source,
            "recovered": self.recovered,
        }


def no_odds_features() -> dict[str, float]:
    """Native-NaN representation used by the mixed LightGBM candidate."""

    return {
        "opening_odds_prob_centered": math.nan,
        "opening_odds_confidence": math.nan,
        "opening_bookmaker_count_log": math.nan,
        "odds_available": 0.0,
    }


def _utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _positive_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not MIN_DECIMAL_ODDS <= parsed <= MAX_DECIMAL_ODDS:
        return None
    return parsed


def devig_two_way(decimal_team1: Any, decimal_team2: Any) -> tuple[float, float, float] | None:
    """Normalize a coherent two-way decimal market to a probability sum of 1."""

    odds1 = _positive_number(decimal_team1)
    odds2 = _positive_number(decimal_team2)
    if odds1 is None or odds2 is None:
        return None
    implied1 = 1.0 / odds1
    implied2 = 1.0 / odds2
    overround = implied1 + implied2
    if not MIN_OVERROUND <= overround <= MAX_OVERROUND:
        return None
    fair1 = implied1 / overround
    fair2 = implied2 / overround
    if not (0.0 < fair1 < 1.0 and 0.0 < fair2 < 1.0):
        return None
    if not math.isclose(fair1 + fair2, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return None
    return fair1, fair2, overround


def _market_values(payload: Mapping[str, Any]) -> tuple[Any, Any]:
    average = payload.get("average")
    source = average if isinstance(average, Mapping) else payload
    return (
        source.get("team1_decimal") or source.get("odds_t1"),
        source.get("team2_decimal") or source.get("odds_t2"),
    )


def validate_opening_odds(
    payload: Mapping[str, Any] | None,
    *,
    captured_at_utc: Any,
    kickoff_utc: Any,
    source: str,
    recovered: bool = False,
) -> OpeningOdds | None:
    """Validate one opening observation against the strict pre-kickoff cutoff."""

    if not isinstance(payload, Mapping):
        return None
    captured = _utc_datetime(captured_at_utc)
    kickoff = _utc_datetime(kickoff_utc)
    if captured is None or kickoff is None or captured >= kickoff:
        return None
    decimal1, decimal2 = _market_values(payload)
    normalized = devig_two_way(decimal1, decimal2)
    if normalized is None:
        return None
    fair1, fair2, overround = normalized
    average = payload.get("average")
    source_payload = average if isinstance(average, Mapping) else payload
    try:
        bookmaker_count = int(payload.get("bookmaker_count") or 0)
    except (TypeError, ValueError):
        bookmaker_count = 0
    if bookmaker_count <= 0:
        providers = payload.get("providers")
        bookmaker_count = len(providers) if isinstance(providers, list) else 1
    return OpeningOdds(
        fair_prob_team1=fair1,
        fair_prob_team2=fair2,
        decimal_team1=float(source_payload.get("team1_decimal") or source_payload.get("odds_t1")),
        decimal_team2=float(source_payload.get("team2_decimal") or source_payload.get("odds_t2")),
        overround=overround,
        bookmaker_count=max(bookmaker_count, 1),
        captured_at_utc=captured.isoformat().replace("+00:00", "Z"),
        source=source,
        recovered=bool(recovered),
    )


def resolve_opening_odds(
    *,
    current_market: Mapping[str, Any] | None,
    current_captured_at_utc: Any,
    stored_opening: Mapping[str, Any] | None,
    odds_history: Iterable[Mapping[str, Any]] = (),
    kickoff_utc: Any,
) -> OpeningOdds | None:
    """Resolve the earliest valid opening, recovering it when today's scrape failed.

    Stored opening/history precede the current market because production must
    use the first observed market, not a later price that happened to be
    available during the current run.
    """

    current = validate_opening_odds(
        current_market,
        captured_at_utc=current_captured_at_utc,
        kickoff_utc=kickoff_utc,
        source="current_snapshot",
    )
    stored_candidates: list[tuple[datetime, Mapping[str, Any], str]] = []
    if isinstance(stored_opening, Mapping):
        captured = _utc_datetime(stored_opening.get("captured_at") or stored_opening.get("captured_at_utc"))
        if captured is not None:
            stored_candidates.append((captured, stored_opening, "stored_opening"))
    for item in odds_history:
        if not isinstance(item, Mapping):
            continue
        captured = _utc_datetime(item.get("captured_at") or item.get("captured_at_utc"))
        if captured is not None:
            stored_candidates.append((captured, item, "stored_history"))
    for captured, payload, source in sorted(stored_candidates, key=lambda item: item[0]):
        validated = validate_opening_odds(
            payload,
            captured_at_utc=captured,
            kickoff_utc=kickoff_utc,
            source=source,
            recovered=current is None,
        )
        if validated is not None:
            return validated
    return current


def training_opening_odds(row: Mapping[str, Any]) -> OpeningOdds | None:
    """Apply the same contract to one database training row."""

    return validate_opening_odds(
        {
            "team1_decimal": row.get("opening_odds_decimal_t1"),
            "team2_decimal": row.get("opening_odds_decimal_t2"),
            "bookmaker_count": row.get("opening_bookmaker_count"),
        },
        captured_at_utc=row.get("opening_odds_captured_at"),
        kickoff_utc=row.get("kickoff_utc"),
        source="training_database_first_opening",
    )


__all__ = [
    "ODDS_FEATURE_COLUMNS",
    "OpeningOdds",
    "devig_two_way",
    "no_odds_features",
    "resolve_opening_odds",
    "training_opening_odds",
    "validate_opening_odds",
]
