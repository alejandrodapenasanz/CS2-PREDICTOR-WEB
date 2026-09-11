"""Update the live drift report from frozen pre-match predictions in SQLite."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "MODEL"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from cs2model import dataio
from cs2model.config import DEFAULT_CONFIG_PATH, load_config, set_runtime_config
from cs2model.drift import build_drift_report
from cs2model.features import build_training_frame


DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_OUTPUT = ROOT / "MODEL" / "results" / "drift_live.json"


def load_closed_predictions(db_path: Path) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            WITH eligible AS (
                SELECT
                    p.hltv_match_id,
                    p.prob_team1,
                    p.predicted_at_utc,
                    m.datetime_utc,
                    m.score_t1,
                    m.score_t2,
                    m.environment,
                    m.stage,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.match_id
                        ORDER BY p.predicted_at_utc DESC, p.prediction_id DESC
                    ) AS row_num
                FROM predictions p
                JOIN matches m ON m.match_id = p.match_id
                WHERE m.status = 'completed'
                  AND m.score_t1 IS NOT NULL
                  AND m.score_t2 IS NOT NULL
                  AND m.score_t1 <> m.score_t2
                  AND p.predicted_at_utc <= m.datetime_utc
            )
            SELECT * FROM eligible WHERE row_num = 1 ORDER BY datetime_utc, hltv_match_id
            """
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "match_id": str(row["hltv_match_id"]),
            "date": str(row["datetime_utc"] or "")[:10],
            "actual": int(row["score_t1"] > row["score_t2"]),
            "prob_team1": float(row["prob_team1"]),
            "environment": row["environment"],
            "stage": row["stage"],
        }
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor causal de drift del modelo en produccion")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--config", default=os.environ.get("CS2_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    config = load_config(args.config)
    set_runtime_config(config)
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    predictions = load_closed_predictions(db_path)
    matches = dataio.load_training_rows_from_db(db_path)
    _features, _labels, regime_meta, _state = build_training_frame(
        matches, form_half_life=config.training.form_half_life_days
    )
    for match, meta in zip(matches, regime_meta):
        match["patch_version"] = meta.get("patch_version")
        match["map_pool_regime"] = meta.get("map_pool_regime")
    report = build_drift_report(predictions, matches, config.drift)
    report["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report["source"] = str(db_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"drift={report['status']} closed_predictions={report['n_predictions']} "
        f"clv_n={report['clv'].get('n', 0)} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
