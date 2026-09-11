"""Evaluate frozen operational predictions, grouped by exact artifact hash."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from cs2model.config import DEFAULT_CONFIG_PATH, load_config
from cs2model.segment_calibration import segment_calibration_suite

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_OUTPUT = ROOT / "MODEL" / "results" / "live_ledger_evaluation.json"


def summarize(rows: list[sqlite3.Row]) -> dict:
    if not rows:
        return {"n": 0, "accuracy": None, "log_loss": None, "brier": None}
    return {
        "n": len(rows),
        "accuracy": sum(int(row["prediction_correct"]) for row in rows) / len(rows),
        "log_loss": sum(float(row["realized_log_loss"]) for row in rows) / len(rows),
        "brier": sum(float(row["realized_brier"]) for row in rows) / len(rows),
    }


def _segment_rows(rows: list[sqlite3.Row]) -> list[dict]:
    output: list[dict] = []
    for row in rows:
        try:
            features = json.loads(row["features_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            features = {}
        output.append(
            {
                "actual": int(row["actual_team1_win"]),
                "prob_team1": float(row["prob_team1"]),
                "elo_diff": features.get("elo_diff"),
                "environment": row["environment"],
                "stage": row["stage"],
                "format": f"bo{int(row['best_of'])}" if row["best_of"] else None,
            }
        )
    return output


def _segment_report(rows: list[sqlite3.Row], config_path: Path) -> dict:
    policy = load_config(config_path).segment_calibration
    report = segment_calibration_suite(
        _segment_rows(rows),
        min_n=policy.min_live_segment_rows,
        n_bins=policy.reliability_bins,
        min_absolute_gap=policy.min_absolute_gap,
        min_ece=policy.min_ece,
    )
    decision_allowed = len(rows) >= policy.min_live_total_rows_for_decisions
    report["live_decision_policy"] = {
        "decision_allowed": decision_allowed,
        "evaluated_rows": len(rows),
        "min_total_rows": policy.min_live_total_rows_for_decisions,
        "note": (
            "Live sample has reached the descriptive review threshold; any model change still requires a temporal backtest and promotion gate."
            if decision_allowed
            else "Current live sample does not decide by segment; use the temporal backtest for model decisions."
        ),
    }
    return report


def evaluate(db_path: Path, config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT pl.artifact_sha256, pl.model_version, pl.prediction_correct,
               pl.realized_log_loss, pl.realized_brier, pl.actual_team1_win,
               pl.prob_team1, pl.features_json, m.environment, m.stage, m.best_of
        FROM prediction_ledger AS pl
        JOIN matches AS m ON m.match_id=pl.match_id
        WHERE pl.ledger_status='evaluated' AND pl.artifact_sha256 IS NOT NULL
          AND pl.actual_team1_win IS NOT NULL
        ORDER BY pl.result_filled_at_utc, pl.ledger_id
        """
    ).fetchall()
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[(row["artifact_sha256"], row["model_version"])].append(row)
    conn.close()
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "prediction_ledger_frozen_prematch_only",
        "all": summarize(rows),
        "segment_calibration": _segment_report(rows, config_path),
        "artifacts": [
            {
                "artifact_sha256": artifact_hash,
                "model_version": model_version,
                **summarize(group_rows),
                "segment_calibration": _segment_report(group_rows, config_path),
            }
            for (artifact_hash, model_version), group_rows in grouped.items()
        ],
    }


def _console_summary(report: dict) -> dict:
    calibration = report["segment_calibration"]
    segments: dict[str, list[dict]] = {}
    for dimension in ("by_elo_gap", "by_stage", "by_environment"):
        segments[dimension] = [
            {
                "segment": name,
                "n": values.get("n", 0),
                "gap": values.get("calibration_gap"),
                "ece": values.get("ece_10"),
                "status": values.get("status"),
            }
            for name, values in calibration.get(dimension, {}).items()
        ]
    return {
        "generated_at_utc": report["generated_at_utc"],
        "all": report["all"],
        "live_decision_policy": calibration["live_decision_policy"],
        "segments": segments,
        "output_contains_full_reliability_curves": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--verbose-json", action="store_true")
    args = parser.parse_args()
    report = evaluate(Path(args.db), Path(args.config))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            report if args.verbose_json else _console_summary(report),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
