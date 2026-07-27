"""Shared Best Opportunity policy used by enrichment, storage and backtests."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

POLICY_VERSION = "best_opportunity_v1"
MIN_DECISION_CONFIDENCE = 0.60
MIN_RELIABILITY = 0.45


def opportunity_score(decision_confidence: float, reliability_score: float) -> float:
    """Return the operational ranking score in the [0, 1] interval."""
    confidence = max(0.0, min(1.0, float(decision_confidence)))
    reliability = max(0.0, min(1.0, float(reliability_score)))
    return confidence * reliability


def is_opportunity_eligible(
    decision_confidence: float,
    reliability_score: float,
    *,
    min_confidence: float = MIN_DECISION_CONFIDENCE,
    min_reliability: float = MIN_RELIABILITY,
) -> bool:
    """Apply the same default eligibility gates as the web dashboard."""
    return (
        float(decision_confidence) >= float(min_confidence)
        and float(reliability_score) >= float(min_reliability)
    )


def annotate_opportunities(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Freeze score and run/day ranks in each enriched prediction.

    Rank 1 is the best eligible card. Ineligible cards retain a score but do
    not receive a rank, which prevents a weak-data pick from being labelled as
    the best opportunity merely because no stronger match is available.
    """
    eligible: list[dict[str, Any]] = []
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for entry in entries:
        prediction = entry.setdefault("prediction", {})
        confidence = float(
            prediction.get("decision_confidence")
            or max(
                float(prediction.get("decision_prob_team1") or 0.5),
                1.0 - float(prediction.get("decision_prob_team1") or 0.5),
            )
        )
        reliability = float(prediction.get("reliability_score") or 0.0)
        score = opportunity_score(confidence, reliability)
        is_eligible = is_opportunity_eligible(confidence, reliability)
        prediction.update(
            {
                "opportunity_score": score,
                "opportunity_eligible": is_eligible,
                "opportunity_rank": None,
                "opportunity_rank_day": None,
                "is_best_opportunity": False,
                "opportunity_policy_version": POLICY_VERSION,
                "opportunity_min_confidence": MIN_DECISION_CONFIDENCE,
                "opportunity_min_reliability": MIN_RELIABILITY,
            }
        )
        if is_eligible:
            eligible.append(entry)
            by_day[str(entry.get("date") or "")].append(entry)

    ordering = lambda row: (
        -float(row["prediction"]["opportunity_score"]),
        -float(row["prediction"]["decision_confidence"]),
        str(row.get("id") or ""),
    )
    for rank, entry in enumerate(sorted(eligible, key=ordering), start=1):
        entry["prediction"]["opportunity_rank"] = rank

    for day_entries in by_day.values():
        for rank, entry in enumerate(sorted(day_entries, key=ordering), start=1):
            prediction = entry["prediction"]
            prediction["opportunity_rank_day"] = rank
            prediction["is_best_opportunity"] = rank == 1

    return entries
