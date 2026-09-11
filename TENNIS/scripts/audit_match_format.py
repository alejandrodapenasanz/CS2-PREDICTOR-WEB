"""Run the actual daily inference pipeline without publishing or settling.

Usage: python scripts/audit_match_format.py [--date YYYY-MM-DD]
An explicit --replay-at UTC timestamp enables a labelled, non-official replay.
The report is stdout JSON; no operational prediction or model is mutated.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.daily_pipeline.pipeline import run_daily_prediction_pipeline  # noqa: E402
from src.daily_pipeline.format_report import match_format_report  # noqa: E402


def main() -> int:
    """Execute causal inference, never the append-only operational writer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat)
    parser.add_argument("--replay-at", type=datetime.fromisoformat)
    args = parser.parse_args()
    replay = args.replay_at
    if replay is not None and (
        replay.tzinfo is None or args.date is None or replay.date() != args.date
    ):
        parser.error("--replay-at requires a timezone and the same civil --date")
    run = run_daily_prediction_pipeline(
        args.date, clock=(lambda: replay) if replay else None, publish=False
    )
    report = match_format_report(run.predictions)
    report.update(
        {
            "date": run.match_date.isoformat(),
            "mode": "diagnostic_replay_not_official" if replay else "read_only_current",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "pipeline": "ok",
            "published": False,
        }
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
