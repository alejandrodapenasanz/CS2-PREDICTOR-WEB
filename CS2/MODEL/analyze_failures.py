"""Analyze closed prediction failures against persisted point-in-time snapshots.

This script is deliberately descriptive: it does not assume why a match failed.
It measures where failures concentrate, then maps those observations to
testable hypotheses grounded in CS mechanics and betting-model literature.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DAILY_ROOT = ROOT / "PIPELINE"
RUNS_DIR = DAILY_ROOT / "runs"
MASTER_MATCHES = DAILY_ROOT / "master" / "matches.json"
OUTPUT_DIR = ROOT / "MODEL" / "results"

sys.path.insert(0, str(ROOT))
try:
    from PIPELINE.enrich_predictions import (
        build_market_blend_policy,
        decision_probability_team1,
        reliability_score,
    )
except Exception:  # pragma: no cover - fallback for isolated report runs
    def build_market_blend_policy(master: dict[str, Any]) -> dict[str, Any]:
        return {"available": False, "source": "unavailable_import_fallback"}

    def reliability_score(entry: dict[str, Any], features: dict[str, Any]) -> float:
        score = 1.0
        min_matches = min(features.get("matches_team1") or 0, features.get("matches_team2") or 0)
        if min_matches < 5:
            score *= 0.35
        elif min_matches < 12:
            score *= 0.60
        elif min_matches < 25:
            score *= 0.85
        rd = ((features.get("glicko_rd_team1") or 160) + (features.get("glicko_rd_team2") or 160)) / 2
        if rd > 140:
            score *= 0.55
        elif rd > 110:
            score *= 0.78
        elif rd > 90:
            score *= 0.92
        if entry.get("format") == "bo1":
            score *= 0.82
        return max(0.05, min(1.0, score))

    def decision_probability_team1(
        model_prob_team1: float,
        odds_prob_team1: float | None,
        reliability: float,
        market: dict[str, Any] | None = None,
        policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        risk_adjusted = 0.5 + max(0.05, min(1.0, reliability)) * (model_prob_team1 - 0.5)
        if odds_prob_team1 is None:
            return {"prob_team1": risk_adjusted, "risk_adjusted_prob_team1": risk_adjusted}
        return {
            "prob_team1": 0.85 * model_prob_team1 + 0.15 * odds_prob_team1,
            "risk_adjusted_prob_team1": risk_adjusted,
        }


SOURCES = [
    {
        "label": "Walsh & Joshi (2024), Machine learning for sports betting",
        "url": "https://www.sciencedirect.com/science/article/pii/S266682702400015X",
        "use": "Calibration/log loss should dominate raw accuracy for betting and Kelly sizing.",
    },
    {
        "label": "Hegarty & Whelan (2024), testing sports betting market efficiency",
        "url": "https://www.ucd.ie/economics/t4media/WP24_03.pdf",
        "use": "Normalized implied probabilities are the correct odds-derived market benchmark.",
    },
    {
        "label": "Petri et al. (2021), Bandit Modeling of Map Selection in CS:GO",
        "url": "https://arxiv.org/abs/2106.08888",
        "use": "Pick/ban and map selection can move map and match win probability, especially in even matches.",
    },
    {
        "label": "Broms & Nordansjo (2024), Predicting Counter-Strike Matches",
        "url": "https://lup.lub.lu.se/student-papers/record/9145457/file/9145459.pdf",
        "use": "HLTV-style player/team variables can predict above 50%, but odds remain a hard benchmark.",
    },
    {
        "label": "Svec (2022), Predicting Counter-Strike Game Outcomes with ML",
        "url": "https://wiki.control.fel.cvut.cz/mediawiki/images/e/e9/P_2022_svec_ondrej.pdf",
        "use": "Player, roster/team representations and Elo-like baselines are natural CS match features.",
    },
]


CATEGORY_FLAGS = {
    "market_sparse_or_missing": {"NO_ODDS", "LOW_ODDS_SOURCES", "LOW_MARKET_CONSENSUS"},
    "market_disagreement": {"MODEL_ODDS_DISAGREE", "VALUE_EDGE", "ODDS_STRONG_DRIFT"},
    "map_veto_pool_uncertain": {"LOW_MAP_POOL_DATA"},
    "roster_player_uncertain": {"ROSTER_RECENT_CHANGE", "STANDIN_RISK", "LOW_PLAYER_STATS", "PLAYER_FORM_BACKFILL_ONLY"},
    "low_team_history": {"LOW_TEAM_HISTORY", "UNKNOWN_EVENT", "CALIBRATION_LOW_SAMPLE"},
    "format_variance": {"BO1_HIGH_VARIANCE"},
    "schedule_fatigue": {"FATIGUE_BACK_TO_BACK", "MULTI_MATCH_DAY", "HIGH_SCHEDULE_DENSITY"},
    "low_operational_reliability": {"LOW_RELIABILITY_OVERCONFIDENCE", "DECISION_SHRUNK_TO_COINFLIP"},
    "incentive_context_unknown": {"INCENTIVE_UNKNOWN", "INCENTIVE_RISK"},
}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def actual_team1_win(record: dict[str, Any]) -> int | None:
    score = record.get("score") or {}
    try:
        s1 = int(score.get("team1"))
        s2 = int(score.get("team2"))
    except (TypeError, ValueError):
        detail = (record.get("result") or {}).get("detail") or record.get("detail") or {}
        match = detail.get("match") or {}
        try:
            s1 = int((match.get("team1") or {}).get("score"))
            s2 = int((match.get("team2") or {}).get("score"))
        except (TypeError, ValueError):
            return None
    if s1 == s2:
        return None
    return 1 if s1 > s2 else 0


def market_prob_team1(entry: dict[str, Any]) -> float | None:
    prediction = entry.get("prediction") or {}
    if prediction.get("odds_prob_team1") is not None:
        return safe_prob(prediction.get("odds_prob_team1"))
    market = ((entry.get("controls") or {}).get("market") or {})
    for source in (market.get("latest") or {}, (entry.get("odds") or {}).get("average") or {}, market.get("opening") or {}):
        value = source.get("team1_implied_prob_norm")
        prob = safe_prob(value)
        if prob is not None:
            return prob
    return None


def safe_prob(value: Any) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if 0 < p < 1:
        return p
    return None


def entry_reliability(entry: dict[str, Any]) -> float:
    prediction = entry.get("prediction") or {}
    stored = safe_prob(prediction.get("reliability_score"))
    if stored is not None:
        return stored
    return reliability_score(entry, entry.get("features") or {})


def probabilities(entry: dict[str, Any], policy: dict[str, Any] | None = None) -> dict[str, float | None]:
    pred = entry.get("prediction") or {}
    model_p = safe_prob(pred.get("model_prob_team1"))
    if model_p is None:
        model_p = safe_prob(pred.get("blended_prob_team1"))
    if model_p is None:
        return {"model": None, "market": None, "risk_adjusted": None, "decision": None}
    market_p = market_prob_team1(entry)
    rel = entry_reliability(entry)
    market = ((entry.get("controls") or {}).get("market") or {})
    decision = decision_probability_team1(model_p, market_p, rel, market=market, policy=policy)
    return {
        "model": model_p,
        "market": market_p,
        "risk_adjusted": safe_prob(pred.get("risk_adjusted_prob_team1")) or decision.get("risk_adjusted_prob_team1"),
        "decision": safe_prob(pred.get("decision_prob_team1")) or decision.get("prob_team1"),
    }


def load_prediction_rows() -> list[dict[str, Any]]:
    master = read_json(MASTER_MATCHES, {})
    policy = build_market_blend_policy(master)
    actuals = {str(mid): actual_team1_win(record) for mid, record in master.items()}
    actuals = {mid: y for mid, y in actuals.items() if y is not None}
    rows: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.glob("*/predictions_enriched.json")):
        for entry in read_json(path, []):
            mid = str(entry.get("id") or "")
            if mid not in actuals:
                continue
            probs = probabilities(entry, policy)
            if probs["model"] is None:
                continue
            p = float(probs["model"])
            y = int(actuals[mid])
            flags = [str(flag.get("code")) for flag in entry.get("flags") or [] if flag.get("code")]
            row = {
                "run_id": path.parent.name,
                "match_id": mid,
                "captured_at": entry.get("captured_at") or "",
                "date": entry.get("date"),
                "hour": entry.get("hour"),
                "event": entry.get("event"),
                "format": entry.get("format") or "unknown",
                "team1": ((entry.get("team1") or {}).get("name") or ""),
                "team2": ((entry.get("team2") or {}).get("name") or ""),
                "actual": y,
                "model_prob_team1": p,
                "risk_adjusted_prob_team1": probs["risk_adjusted"],
                "decision_prob_team1": probs["decision"],
                "market_prob_team1": probs["market"],
                "model_side": "team1" if p >= 0.5 else "team2",
                "actual_side": "team1" if y else "team2",
                "model_confidence": max(p, 1 - p),
                "decision_confidence": max(float(probs["decision"]), 1 - float(probs["decision"])) if probs["decision"] is not None else None,
                "reliability": entry_reliability(entry),
                "flags": flags,
                "odds_available": probs["market"] is not None,
                "raw_entry": entry,
            }
            row["model_correct"] = int(row["model_side"] == row["actual_side"])
            if row["decision_prob_team1"] is not None:
                row["decision_side"] = "team1" if float(row["decision_prob_team1"]) >= 0.5 else "team2"
                row["decision_correct"] = int(row["decision_side"] == row["actual_side"])
            else:
                row["decision_side"] = None
                row["decision_correct"] = None
            if row["market_prob_team1"] is not None:
                row["market_side"] = "team1" if float(row["market_prob_team1"]) >= 0.5 else "team2"
                row["market_correct"] = int(row["market_side"] == row["actual_side"])
            else:
                row["market_side"] = None
                row["market_correct"] = None
            rows.append(row)
    return rows


def latest_by_match(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["match_id"]
        current = latest.get(key)
        marker = (row.get("captured_at") or "", row.get("run_id") or "")
        current_marker = ((current or {}).get("captured_at") or "", (current or {}).get("run_id") or "")
        if current is None or marker > current_marker:
            latest[key] = row
    return list(latest.values())


def log_loss(y: int, p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def metric_block(rows: list[dict[str, Any]], prob_key: str = "model_prob_team1", correct_key: str | None = None) -> dict[str, Any]:
    vals = [row for row in rows if row.get(prob_key) is not None]
    if not vals:
        return {"n": 0}
    correct_key = correct_key or ("decision_correct" if prob_key == "decision_prob_team1" else "model_correct")
    y = [int(row["actual"]) for row in vals]
    p = [float(row[prob_key]) for row in vals]
    return {
        "n": len(vals),
        "accuracy": sum(int(row.get(correct_key) == 1) for row in vals) / len(vals),
        "log_loss": sum(log_loss(yy, pp) for yy, pp in zip(y, p)) / len(vals),
        "brier": sum((yy - pp) ** 2 for yy, pp in zip(y, p)) / len(vals),
        "avg_confidence": sum(max(pp, 1 - pp) for pp in p) / len(vals),
    }


def grouped_metrics(
    rows: list[dict[str, Any]],
    key_func: Callable[[dict[str, Any]], str],
    prob_key: str = "model_prob_team1",
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_func(row)].append(row)
    out = []
    for key in sorted(groups):
        block = metric_block(groups[key], prob_key=prob_key)
        out.append({"group": key, **block})
    return out


def confidence_band(row: dict[str, Any]) -> str:
    c = row["model_confidence"]
    if c < 0.55:
        return "50-55"
    if c < 0.65:
        return "55-65"
    if c < 0.80:
        return "65-80"
    return "80+"


def reliability_band(row: dict[str, Any]) -> str:
    r = float(row["reliability"])
    if r < 0.25:
        return "0-25"
    if r < 0.45:
        return "25-45"
    if r < 0.65:
        return "45-65"
    if r < 0.85:
        return "65-85"
    return "85-100"


def flag_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for row in rows:
        for flag in row["flags"]:
            totals[flag] += 1
            if not row["model_correct"]:
                failures[flag] += 1
    out = []
    for flag, n in totals.most_common():
        fail_n = failures[flag]
        out.append({
            "flag": flag,
            "n": n,
            "failures": fail_n,
            "failure_rate_when_present": fail_n / n if n else None,
            "share_of_failures": fail_n / max(1, sum(1 for row in rows if not row["model_correct"])),
        })
    return out


def category_stats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for category, flags in CATEGORY_FLAGS.items():
        if category == "low_operational_reliability":
            subset = [
                row for row in rows
                if float(row.get("reliability") or 1.0) < 0.35
                and float(row.get("model_confidence") or 0.5) >= 0.65
            ]
        else:
            subset = [row for row in rows if flags.intersection(row["flags"])]
        fail_subset = [row for row in subset if not row["model_correct"]]
        out.append({
            "category": category,
            "flags": sorted(flags),
            "n_present": len(subset),
            "failures_present": len(fail_subset),
            "failure_rate_when_present": len(fail_subset) / len(subset) if subset else None,
            "share_of_all_failures": len(fail_subset) / max(1, sum(1 for row in rows if not row["model_correct"])),
        })
    out.sort(key=lambda row: (row["failures_present"], row["n_present"]), reverse=True)
    return out


def market_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    odds_rows = [row for row in rows if row.get("market_prob_team1") is not None]
    if not odds_rows:
        return {"n": 0}
    return {
        "model": metric_block(odds_rows, "model_prob_team1", "model_correct"),
        "market": metric_block(odds_rows, "market_prob_team1", "market_correct"),
        "decision": metric_block(odds_rows, "decision_prob_team1", "decision_correct"),
        "disagreements": [
            summarize_case(row) for row in odds_rows
            if row.get("market_side") and row["market_side"] != row["model_side"]
        ],
    }


def summarize_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "match_id": row["match_id"],
        "run_id": row["run_id"],
        "date": row["date"],
        "format": row["format"],
        "match": f"{row['team1']} vs {row['team2']}",
        "event": row["event"],
        "actual_side": row["actual_side"],
        "model_side": row["model_side"],
        "model_prob_team1": round(float(row["model_prob_team1"]), 4),
        "decision_side": row.get("decision_side"),
        "decision_prob_team1": round(float(row["decision_prob_team1"]), 4) if row.get("decision_prob_team1") is not None else None,
        "market_side": row.get("market_side"),
        "market_prob_team1": round(float(row["market_prob_team1"]), 4) if row.get("market_prob_team1") is not None else None,
        "confidence": round(float(row["model_confidence"]), 4),
        "decision_confidence": round(float(row["decision_confidence"]), 4) if row.get("decision_confidence") is not None else None,
        "reliability": round(float(row["reliability"]), 4),
        "flags": row["flags"],
    }


def build_report(rows: list[dict[str, Any]], latest: list[dict[str, Any]], analysis: dict[str, Any]) -> str:
    generated = analysis["generated_at"]
    lines: list[str] = [
        "# Failure analysis - CS2 Predictor\n",
        f"\nGenerated: {generated}\n",
        "\n## Scientific framing\n\n",
        "This report separates observations from hypotheses. A flag co-occurring with a missed prediction is not proof of causality; it is a candidate mechanism to test as more closed matches accumulate.\n",
        "\n### Literature anchors\n\n",
    ]
    for source in SOURCES:
        lines.append(f"- {source['label']}: {source['use']}  \n  {source['url']}\n")

    policy = analysis.get("market_blend_policy") or {}
    lines.append("\n## Market Blend Policy\n\n")
    if policy.get("available"):
        lines.append(
            f"- Status: learned by expanding walk-forward on {policy.get('n_closed_odds_matches')} closed odds matches.\n"
            f"- Production market weight: {policy.get('production_market_weight')}\n"
            f"- Walk-forward metrics: {policy.get('walk_forward_metrics')}\n"
        )
    else:
        lines.append(
            f"- Status: dynamic conservative prior; learned blend disabled until "
            f"{policy.get('min_samples_for_learning')} closed point-in-time odds matches.\n"
            f"- Current closed odds matches: {policy.get('n_closed_odds_matches')}.\n"
        )

    def add_metric_table(title: str, blocks: dict[str, Any]) -> None:
        lines.append(f"\n## {title}\n\n")
        lines.append("| Probability | N | Accuracy | Log loss | Brier | Avg confidence |\n")
        lines.append("|---|---:|---:|---:|---:|---:|\n")
        for name, block in blocks.items():
            if not block or not block.get("n"):
                continue
            lines.append(
                f"| {name} | {block['n']} | {block['accuracy']:.3f} | {block['log_loss']:.4f} | "
                f"{block['brier']:.4f} | {block['avg_confidence']:.3f} |\n"
            )

    add_metric_table("Closed predictions", analysis["overall"])
    add_metric_table("Latest snapshot per match", analysis["latest"])

    latest_market = analysis.get("market_latest") or {}
    if latest_market.get("model", {}).get("n"):
        add_metric_table(
            "Latest matches with odds only",
            {
                "model": latest_market.get("model"),
                "market": latest_market.get("market"),
                "decision": latest_market.get("decision"),
            },
        )

    lines.append("\n## Where the model misses\n\n")
    lines.append("| Segment | Group | N | Accuracy | Log loss |\n")
    lines.append("|---|---|---:|---:|---:|\n")
    for segment_name in ("by_format", "by_confidence", "by_reliability", "by_odds"):
        for row in analysis["latest_segments"][segment_name]:
            if row.get("n"):
                lines.append(
                    f"| {segment_name} | {row['group']} | {row['n']} | "
                    f"{row['accuracy']:.3f} | {row['log_loss']:.4f} |\n"
                )

    lines.append("\n## Failure hypotheses from flags\n\n")
    lines.append("| Hypothesis | N present | Failures | Fail rate | Share of failures |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for row in analysis["latest_failure_categories"]:
        rate = row["failure_rate_when_present"]
        share = row["share_of_all_failures"]
        lines.append(
            f"| {row['category']} | {row['n_present']} | {row['failures_present']} | "
            f"{rate:.3f} | {share:.3f} |\n" if rate is not None else
            f"| {row['category']} | {row['n_present']} | {row['failures_present']} | - | {share:.3f} |\n"
        )

    lines.append("\n## High-confidence misses\n\n")
    misses = analysis["high_confidence_misses"]
    if not misses:
        lines.append("No high-confidence misses in the selected sample.\n")
    else:
        lines.append("| Match | Date | Format | Actual | Model | Conf | Rel | Flags |\n")
        lines.append("|---|---|---|---|---|---:|---:|---|\n")
        for case in misses[:20]:
            lines.append(
                f"| {case['match']} | {case['date']} | {case['format']} | {case['actual_side']} | "
                f"{case['model_side']} | {case['confidence']:.3f} | {case['reliability']:.3f} | "
                f"{', '.join(case['flags'][:8])} |\n"
            )

    lines.append("\n## Interpretation\n\n")
    lines.append(
        "- The strongest local warning is calibration under sparse live data: latest closed snapshots have worse log loss than the long walk-forward training metric, so stake sizing should use calibrated/operational probabilities rather than raw accuracy.\n"
    )
    lines.append(
        "- Market odds are a hard benchmark. In the local odds subset, the decision layer (model plus normalized market probability) should be monitored separately from the pure stats model.\n"
    )
    lines.append(
        "- Map pool and veto uncertainty appear in most misses. The literature on CS map selection supports treating map/veto as a first-class variable, but this project still needs more point-in-time map/veto outcomes before training a reliable compositional Bo3 model.\n"
    )
    lines.append(
        "- BO1 and low-roster/player-history cases should remain stake-capped or downgraded; they are not necessarily impossible to predict, but the current evidence is thinner and variance is higher.\n"
    )

    lines.append("\n## Next TESTS\n\n")
    lines.append("- Recompute this report after every scrape/result update; require at least 100-200 closed odds matches before promoting Model B stats+odds as a production model.\n")
    lines.append("- Once analytics/veto coverage has enough closed outcomes, backtest a map-compositional Bo3 model: estimate map win probabilities, simulate likely veto/picks, then aggregate series probability.\n")
    lines.append("- Track error by roster-change recency and player L5/L10 coverage; if the failure rate remains elevated, make roster/player coverage a formal model feature or stronger stake penalty.\n")
    return "".join(lines)


def run() -> dict[str, Any]:
    rows = load_prediction_rows()
    market_policy = build_market_blend_policy(read_json(MASTER_MATCHES, {}))
    latest = latest_by_match(rows)
    failures = [row for row in latest if not row["model_correct"]]
    high_confidence_misses = [
        summarize_case(row) for row in sorted(failures, key=lambda r: r["model_confidence"], reverse=True)
        if row["model_confidence"] >= 0.65
    ]
    analysis = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": SOURCES,
        "market_blend_policy": market_policy,
        "counts": {
            "all_closed_snapshot_predictions": len(rows),
            "latest_closed_matches": len(latest),
            "latest_failures": len(failures),
        },
        "overall": {
            "model": metric_block(rows, "model_prob_team1", "model_correct"),
            "risk_adjusted": metric_block(rows, "risk_adjusted_prob_team1", "model_correct"),
            "decision": metric_block(rows, "decision_prob_team1", "decision_correct"),
        },
        "latest": {
            "model": metric_block(latest, "model_prob_team1", "model_correct"),
            "risk_adjusted": metric_block(latest, "risk_adjusted_prob_team1", "model_correct"),
            "decision": metric_block(latest, "decision_prob_team1", "decision_correct"),
        },
        "latest_segments": {
            "by_format": grouped_metrics(latest, lambda row: row["format"]),
            "by_confidence": grouped_metrics(latest, confidence_band),
            "by_reliability": grouped_metrics(latest, reliability_band),
            "by_odds": grouped_metrics(latest, lambda row: "odds" if row["odds_available"] else "no_odds"),
        },
        "latest_flag_stats": flag_stats(latest),
        "latest_failure_categories": category_stats(latest),
        "market_latest": market_eval(latest),
        "market_all_snapshots": market_eval(rows),
        "high_confidence_misses": high_confidence_misses,
        "latest_failures": [summarize_case(row) for row in sorted(failures, key=lambda r: r["model_confidence"], reverse=True)],
    }
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze closed prediction failures.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    analysis = run()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "failure_analysis.json", analysis)
    report = build_report([], [], analysis)
    (out / "FAILURE_ANALYSIS.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output_json": str(out / "failure_analysis.json"),
        "output_md": str(out / "FAILURE_ANALYSIS.md"),
        "latest_closed_matches": analysis["counts"]["latest_closed_matches"],
        "latest_failures": analysis["counts"]["latest_failures"],
        "latest_model_accuracy": analysis["latest"]["model"].get("accuracy"),
        "latest_decision_accuracy": analysis["latest"]["decision"].get("accuracy"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
