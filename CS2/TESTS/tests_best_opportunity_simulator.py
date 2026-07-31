"""Backtest the frozen Best Opportunity policy against completed matches.

The simulator keeps one prediction per match: the latest snapshot captured
strictly before the scheduled start. It then ranks eligible matches by UTC
match day, so repeated pipeline runs cannot duplicate a pick.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIPELINE.opportunity import (  # noqa: E402
    MIN_DECISION_CONFIDENCE,
    MIN_RELIABILITY,
    POLICY_VERSION,
    is_opportunity_eligible,
    opportunity_score,
)

DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_JSON = ROOT / "MODEL" / "results" / "BEST_OPPORTUNITY_BACKTEST.json"
DEFAULT_MD = ROOT / "MODEL" / "results" / "BEST_OPPORTUNITY_BACKTEST.md"


@dataclass(frozen=True)
class Pick:
    hltv_match_id: str
    match_day: str
    datetime_utc: str
    predicted_at_utc: str
    favorite: str
    opponent: str
    confidence: float
    reliability: float
    opportunity_score: float
    correct: bool


def wilson_interval(correct: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def load_latest_prematch_predictions(db_path: Path) -> list[Pick]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT
                p.*,
                m.datetime_utc,
                m.team1_id,
                m.team2_id,
                m.winner_team_id,
                t1.name AS team1_name,
                t2.name AS team2_name,
                ROW_NUMBER() OVER (
                    PARTITION BY p.hltv_match_id
                    ORDER BY datetime(p.predicted_at_utc) DESC, p.prediction_id DESC
                ) AS snapshot_rank
            FROM predictions p
            JOIN matches m ON m.hltv_match_id = p.hltv_match_id
            JOIN teams t1 ON t1.team_id = m.team1_id
            JOIN teams t2 ON t2.team_id = m.team2_id
            WHERE m.status = 'completed'
              AND m.winner_team_id IS NOT NULL
              AND m.datetime_utc IS NOT NULL
              AND datetime(p.predicted_at_utc) < datetime(m.datetime_utc)
        )
        SELECT * FROM latest WHERE snapshot_rank = 1
        ORDER BY datetime(datetime_utc), hltv_match_id
        """
    ).fetchall()
    conn.close()

    picks: list[Pick] = []
    for row in rows:
        quality = json.loads(row["data_quality_json"] or "{}")
        if quality.get("real_pre_match_snapshot") is False:
            continue
        probability = row["decision_prob_team1"]
        if probability is None:
            probability = row["risk_adjusted_prob_team1"]
        if probability is None:
            probability = row["prob_team1"]
        probability = float(probability)
        confidence = float(
            row["decision_confidence"]
            if row["decision_confidence"] is not None
            else max(probability, 1.0 - probability)
        )
        reliability = float(row["reliability_score"] or 0.0)
        team1_is_pick = probability >= 0.5
        favorite_id = row["team1_id"] if team1_is_pick else row["team2_id"]
        picks.append(
            Pick(
                hltv_match_id=str(row["hltv_match_id"]),
                match_day=str(row["datetime_utc"])[:10],
                datetime_utc=str(row["datetime_utc"]),
                predicted_at_utc=str(row["predicted_at_utc"]),
                favorite=str(row["team1_name"] if team1_is_pick else row["team2_name"]),
                opponent=str(row["team2_name"] if team1_is_pick else row["team1_name"]),
                confidence=confidence,
                reliability=reliability,
                opportunity_score=opportunity_score(confidence, reliability),
                correct=int(favorite_id) == int(row["winner_team_id"]),
            )
        )
    return picks


def summarize(picks: list[Pick]) -> dict[str, Any]:
    correct = sum(int(pick.correct) for pick in picks)
    total = len(picks)
    low, high = wilson_interval(correct, total)
    return {
        "n": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else None,
        "wilson_95_low": low if total else None,
        "wilson_95_high": high if total else None,
        "match_days": len({pick.match_day for pick in picks}),
    }


def select_daily_rank(picks: list[Pick], limit: int) -> list[Pick]:
    by_day: dict[str, list[Pick]] = defaultdict(list)
    for pick in picks:
        by_day[pick.match_day].append(pick)
    selected: list[Pick] = []
    for day in sorted(by_day):
        ranked = sorted(
            by_day[day],
            key=lambda pick: (
                -pick.opportunity_score,
                -pick.confidence,
                pick.hltv_match_id,
            ),
        )
        selected.extend(ranked[:limit])
    return selected


def run_backtest(db_path: Path) -> dict[str, Any]:
    all_picks = load_latest_prematch_predictions(db_path)
    eligible = [
        pick
        for pick in all_picks
        if is_opportunity_eligible(pick.confidence, pick.reliability)
    ]
    top1 = select_daily_rank(eligible, 1)
    top3 = select_daily_rank(eligible, 3)
    return {
        "policy": {
            "version": POLICY_VERSION,
            "formula": "decision_confidence * reliability_score",
            "min_decision_confidence": MIN_DECISION_CONFIDENCE,
            "min_reliability": MIN_RELIABILITY,
            "selection": "latest real prematch snapshot; rank within UTC match day",
        },
        "sample": {
            "unique_completed_matches_with_prematch_prediction": len(all_picks),
            "first_match_day": min((pick.match_day for pick in all_picks), default=None),
            "last_match_day": max((pick.match_day for pick in all_picks), default=None),
        },
        "strategies": {
            "all_latest_predictions": summarize(all_picks),
            "all_eligible_cards": summarize(eligible),
            "best_one_per_day": summarize(top1),
            "best_three_per_day": summarize(top3),
        },
        "best_one_per_day_picks": [asdict(pick) for pick in top1],
    }


def render_markdown(result: dict[str, Any]) -> str:
    policy = result["policy"]
    sample = result["sample"]
    lines = [
        "# Best Opportunity backtest",
        "",
        f"- Policy: `{policy['version']}`.",
        f"- Score: `{policy['formula']}`.",
        (
            f"- Gates: confidence >= {policy['min_decision_confidence']:.0%}; "
            f"reliability >= {policy['min_reliability']:.0%}."
        ),
        f"- Deduplication: {policy['selection']}.",
        (
            "- Sample: "
            f"{sample['unique_completed_matches_with_prematch_prediction']} matches, "
            f"{sample['first_match_day']} to {sample['last_match_day']}."
        ),
        "",
        "| Strategy | Picks | Correct | Accuracy | Wilson 95% CI | Days |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "all_latest_predictions": "All latest prematch predictions",
        "all_eligible_cards": "All eligible Best Opportunity cards",
        "best_one_per_day": "Best Opportunity #1 per day",
        "best_three_per_day": "Top 3 Best Opportunity per day",
    }
    for key, metrics in result["strategies"].items():
        accuracy = metrics["accuracy"]
        interval = (
            f"{metrics['wilson_95_low']:.1%}-{metrics['wilson_95_high']:.1%}"
            if accuracy is not None
            else "n/a"
        )
        lines.append(
            f"| {labels[key]} | {metrics['n']} | {metrics['correct']} | "
            f"{accuracy:.2%} | {interval} | {metrics['match_days']} |"
            if accuracy is not None
            else f"| {labels[key]} | 0 | 0 | n/a | n/a | 0 |"
        )
    lines.extend(
        [
            "",
            "Accuracy measures winner selection only. It is not an ROI backtest and "
            "does not imply that every eligible card has positive expected value.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest the Best Opportunity ranking.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    result = run_backtest(args.db)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.md_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.md_output.write_text(render_markdown(result), encoding="utf-8")

    print(f"Database: {args.db}")
    print(
        "Sample: "
        f"{result['sample']['unique_completed_matches_with_prematch_prediction']} "
        "unique completed matches"
    )
    for name, metrics in result["strategies"].items():
        accuracy = metrics["accuracy"]
        print(
            f"{name}: picks={metrics['n']} correct={metrics['correct']} "
            f"accuracy={accuracy:.2%}"
            if accuracy is not None
            else f"{name}: picks=0 correct=0 accuracy=n/a"
        )
    print(f"Saved: {args.md_output}")
    print(f"Saved: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
