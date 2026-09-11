#!/usr/bin/env python3
"""Publish two CS2 posts and generate the incremental opportunities report.

Inputs are --date YYYY-MM-DD (today by default) and optional --dry-run.
Execute directly with Python or through TELEGRAM/run_telegram.ps1.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys
from typing import Sequence
from zoneinfo import ZoneInfo


TELEGRAM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TELEGRAM_ROOT.parent
SRC_ROOT = TELEGRAM_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cs2_telegram.client import TelegramAPIError, TelegramClient  # noqa: E402
from cs2_telegram.config import ConfigurationError, load_settings  # noqa: E402
from cs2_telegram.formatter import (  # noqa: E402
    TelegramMessageTooLongError,
    render_messages,
)
from cs2_telegram.publisher import (  # noqa: E402
    PublicationDeliveryError,
    publish_daily_messages,
)
from cs2_telegram.report import (  # noqa: E402
    DailyReportError,
    existing_report_match_ids,
    render_daily_report,
    write_daily_report,
)
from cs2_telegram.source import (  # noqa: E402
    PredictionSourceError,
    load_latest_daily_picks,
    load_upcoming_opportunities,
)
from cs2_telegram.state import PublicationStateError, PublishStateStore  # noqa: E402


MADRID_TIME_ZONE = ZoneInfo("Europe/Madrid")


def _iso_date(value: str) -> date:
    """Parse one strict ISO date for argparse with a concise error."""

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def _configure_utf8_output() -> None:
    """Use UTF-8 console streams when Python exposes safe reconfiguration."""

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (OSError, ValueError):
            # Embedded or already-detached streams can reject reconfiguration.
            # Output remains usable through the host-provided fallback stream.
            continue


def _today_in_madrid() -> date:
    """Return the current publication date in the Madrid project timezone."""

    return datetime.now(MADRID_TIME_ZONE).date()


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without reading configuration or prediction data."""

    parser = argparse.ArgumentParser(description="Publish exactly two English CS2 prediction posts to Telegram.")
    parser.add_argument(
        "--date",
        dest="target_date",
        type=_iso_date,
        default=_today_in_madrid(),
        metavar="YYYY-MM-DD",
        help="prediction date (default: today in Europe/Madrid)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print both messages without network access or state writes",
    )
    return parser


def _print_preview(
    best_message: str,
    others_message: str,
    daily_report: str,
) -> None:
    """Print both Telegram payloads and the report without writing state."""

    print("=== BEST OPPORTUNITIES ===")
    print(best_message)
    print("\n=== TODAY'S OTHER PICKS ===")
    print(others_message)
    print("\n=== DAILY REPORT (NOT WRITTEN) ===")
    print(daily_report, end="" if daily_report.endswith("\n") else "\n")


def _print_outcome(outcome: object) -> None:
    """Print a compact non-secret summary of a publication attempt."""

    best = getattr(outcome, "best")
    others = getattr(outcome, "others")
    target_date = getattr(outcome, "target_date")
    print(f"Telegram publication for {target_date}:")
    print(
        f"  best opportunity: {best.status}"
        + (f" (message_id={best.message_id})" if best.message_id is not None else "")
    )
    print(
        f"  other picks: {others.status}"
        + (f" (message_id={others.message_id})" if others.message_id is not None else "")
    )


def run(target_date: date, *, dry_run: bool) -> int:
    """Publish one date and export unseen current/future opportunities."""

    daily = load_latest_daily_picks(PROJECT_ROOT, target_date)
    report_run_id, upcoming = load_upcoming_opportunities(PROJECT_ROOT, target_date)
    best_message, others_message = render_messages(daily)

    if dry_run:
        _print_preview(best_message, others_message, render_daily_report(upcoming))
        return 0

    settings = load_settings(TELEGRAM_ROOT / ".env", require_token=True)
    client = TelegramClient(
        settings.bot_token,
        timeout_seconds=settings.timeout_seconds,
    )
    store = PublishStateStore(settings.state_db_path)
    for existing_id in existing_report_match_ids(TELEGRAM_ROOT):
        store.mark_reported(match_id=existing_id, source_run_id="legacy-daily-report")
    reported_ids = store.reported_match_ids()
    new_opportunities = tuple(pick for pick in upcoming if pick.match_id not in reported_ids)
    outcome = publish_daily_messages(
        client=client,
        store=store,
        channel_id=settings.channel_id,
        target_date=daily.target_date,
        source_run_id=daily.source_run_id,
        best_message=best_message,
        others_message=others_message,
    )
    _print_outcome(outcome)
    report_path = write_daily_report(new_opportunities, TELEGRAM_ROOT)
    for pick in new_opportunities:
        store.mark_reported(
            match_id=pick.match_id,
            match_date=pick.match_date,
            hltv_url=pick.hltv_url,
            source_run_id=report_run_id,
        )
    print(f"Daily opportunities report written to: {report_path} ({len(new_opportunities)} new match(es))")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the CLI and convert expected failures into exit code one."""

    _configure_utf8_output()
    arguments = build_parser().parse_args(argv)
    try:
        return run(arguments.target_date, dry_run=arguments.dry_run)
    except (
        ConfigurationError,
        PredictionSourceError,
        TelegramMessageTooLongError,
        TelegramAPIError,
        PublicationDeliveryError,
        PublicationStateError,
        DailyReportError,
    ) as exc:
        print(f"Telegram publisher error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
