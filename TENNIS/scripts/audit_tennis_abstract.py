"""Inspect authorized Tennis Abstract histories with Scrapling, without publication.

Usage from TENNIS: python scripts/audit_tennis_abstract.py
Optional --player M:JannikSinner --player F:ArynaSabalenka selects exact keys.
Prints JSON evidence. Only the common HTTP cache is written; model, features,
source precedence and the sacred operational database are never changed.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tennis_abstract_access import TennisAbstractAcquisitionClient  # noqa: E402
from src.tennis_abstract_history_audit import audit_player  # noqa: E402
from src.responsible_http import ResponsibleHttpError, WafBlockedError  # noqa: E402


def main() -> int:
    """Report sample coverage honestly; do not promote an acquisition experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", action="append", default=None)
    args = parser.parse_args()
    selected = args.player or ["M:JannikSinner", "F:ArynaSabalenka"]
    client = TennisAbstractAcquisitionClient()
    reports = []
    failure: dict[str, object] | None = None
    try:
        for item in selected:
            gender, key = item.split(":", 1)
            reports.append(audit_player(client, gender, key))
    except (ResponsibleHttpError, ValueError, OSError) as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, WafBlockedError):
            failure.update(
                {
                    "status_code": exc.response.status_code,
                    "url": exc.response.url,
                    "action": "stopped_no_automatic_retry",
                }
            )
    finally:
        client.close()
    print(
        json.dumps(
            {
                "audited_at_utc": datetime.now(UTC).isoformat(),
                "production_changed": False,
                "players": reports,
                "failure": failure,
            },
            sort_keys=True,
        )
    )
    return 1 if failure is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
