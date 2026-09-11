"""Render validated CS2 picks as two attractive Telegram HTML messages."""

from __future__ import annotations

from html import escape

from .models import DailyPredictions, PredictionPick

TELEGRAM_TEXT_LIMIT = 4096


class TelegramMessageTooLongError(ValueError):
    """Report that a rendered message exceeds Telegram's text limit."""


def _pick_lines(pick: PredictionPick) -> list[str]:
    """Render one prediction block while escaping all source-provided text."""

    team1 = escape(pick.team1, quote=False)
    team2 = escape(pick.team2, quote=False)
    winner = escape(pick.winner, quote=False)
    return [
        f"🎮 <b>Match:</b> {team1} vs {team2}",
        f"🏆 <b>Winner:</b> {winner}",
        f"📊 <b>Win probability:</b> {pick.win_probability:.2%}",
        f"<b>Confidence:</b> {escape(pick.confidence_label, quote=False)}",
    ]


def _check_length(message: str, label: str) -> str:
    """Return a valid message or raise a clear Telegram length error."""

    length = len(message)
    if length > TELEGRAM_TEXT_LIMIT:
        raise TelegramMessageTooLongError(
            f"{label} message is {length} characters; Telegram allows at most "
            f"{TELEGRAM_TEXT_LIMIT}. Reduce the number or length of picks before publishing."
        )
    return message


def format_daily_messages(daily: DailyPredictions) -> tuple[str, str]:
    """Return two English HTML messages: all opportunities, then other picks."""

    date_text = daily.target_date.isoformat()
    best_lines = [
        "🔥 <b>BEST OPPORTUNITIES</b>",
        f"📅 <b>Date:</b> {date_text}",
        "Adjusted win probability &gt; 65% · eligible CS2 picks",
        "",
    ]
    if not daily.opportunities:
        best_lines.append("No qualifying best opportunity is available today.")
    else:
        for index, pick in enumerate(daily.opportunities):
            if index:
                best_lines.extend(["", "──────────", ""])
            best_lines.extend(_pick_lines(pick))
    best_lines.extend(["", "<i>Model estimates — not guarantees.</i>"])

    other_lines = ["📋 <b>TODAY'S OTHER PICKS</b>", f"📅 <b>Date:</b> {date_text}", ""]
    if not daily.others:
        other_lines.append("No other validated predictions are available today.")
    else:
        for index, pick in enumerate(daily.others):
            if index:
                other_lines.extend(["", "──────────", ""])
            other_lines.extend(_pick_lines(pick))
    other_lines.extend(["", "<i>Model estimates — not guarantees.</i>"])

    best_message = _check_length("\n".join(best_lines), "Best opportunity")
    others_message = _check_length("\n".join(other_lines), "Other picks")
    return best_message, others_message


def render_messages(daily: DailyPredictions) -> tuple[str, str]:
    """Compatibility wrapper for :func:`format_daily_messages`."""

    return format_daily_messages(daily)
