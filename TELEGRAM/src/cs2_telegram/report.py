"""Render and atomically persist newly detected qualified opportunities."""

from __future__ import annotations

from collections.abc import Iterable
import os
from pathlib import Path
import re
import tempfile

from .models import PredictionPick

REPORT_FILENAME = "daily_report.txt"
CHANNEL_HANDLE = "@cs2DailyPicks"


class DailyReportError(RuntimeError):
    """Report a failed read or atomic replacement of the local report."""


def render_daily_report(picks: Iterable[PredictionPick]) -> str:
    """Render all supplied current and future qualified opportunities."""

    selected = tuple(picks)
    lines = ["CS2 BEST OPPORTUNITIES", ""]
    if not selected:
        lines.append("Status: No new qualifying opportunities are available.")
    for index, pick in enumerate(selected):
        if index:
            lines.extend(["", "----------------------------------------", ""])
        lines.extend(
            [
                f"Date: {pick.match_date.isoformat()}",
                f"URL: {pick.hltv_url}",
                f"Winner: {pick.winner}",
                f"Match: {pick.team1} vs {pick.team2}",
                f"Win probability: {pick.win_probability:.2%}",
                f"Confidence: {pick.confidence_label}",
                "",
                "This prediction was calculated using my predictive Machine Learning and Artificial Intelligence model as part of my Data Science PhD thesis.",
                "The probability is a model estimate rather than a guaranteed outcome.",
            ]
        )
    lines.extend(["", "See all model predictions for free on Telegram:", CHANNEL_HANDLE])
    return "\n".join(lines) + "\n"


def existing_report_match_ids(telegram_root: Path | str) -> set[str]:
    """Extract identifiers already present in the current report for migration."""

    path = Path(telegram_root).expanduser().resolve() / REPORT_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    except OSError as exc:
        raise DailyReportError(f"Could not read existing daily report at {path}: {exc}") from exc
    return set(re.findall(r"https://www\.hltv\.org/matches/([^/\s?]+)", text))


def write_daily_report(
    picks: Iterable[PredictionPick],
    telegram_root: Path | str,
) -> Path:
    """Atomically replace ``daily_report.txt`` with the supplied new picks."""

    root = Path(telegram_root).expanduser().resolve()
    destination = root / REPORT_FILENAME
    report = render_daily_report(picks)
    descriptor = -1
    temporary_path: Path | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(dir=root, prefix=".daily_report.", suffix=".tmp", text=True)
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(report)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise DailyReportError(f"Could not atomically write daily report at {destination}: {exc}") from exc
    return destination
