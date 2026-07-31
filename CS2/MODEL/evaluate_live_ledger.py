"""Evaluate frozen operational predictions, grouped by exact artifact hash."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

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


def evaluate(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT artifact_sha256, model_version, prediction_correct,
               realized_log_loss, realized_brier
        FROM prediction_ledger
        WHERE ledger_status='evaluated' AND artifact_sha256 IS NOT NULL
        ORDER BY result_filled_at_utc, ledger_id
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
        "artifacts": [
            {
                "artifact_sha256": artifact_hash,
                "model_version": model_version,
                **summarize(group_rows),
            }
            for (artifact_hash, model_version), group_rows in grouped.items()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    report = evaluate(Path(args.db))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
