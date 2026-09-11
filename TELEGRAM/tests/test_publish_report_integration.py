"""Offline integration tests for publishing and the generated daily report."""

from __future__ import annotations

from datetime import date, datetime
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


TELEGRAM_ROOT = Path(__file__).resolve().parents[1]


def _load_publish_script() -> object:
    """Import the standalone publisher without executing its main block."""

    script_path = TELEGRAM_ROOT / "scripts" / "publish_cs2.py"
    spec = importlib.util.spec_from_file_location(
        "telegram_publish_report_integration_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _daily_stub() -> SimpleNamespace:
    """Return the minimum validated-daily contract needed by the orchestration."""

    return SimpleNamespace(
        target_date=date(2026, 8, 11),
        source_run_id="fixture-run",
    )


def test_default_cli_date_uses_europe_madrid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve an omitted date with Europe/Madrid rather than machine-local time."""

    script = _load_publish_script()
    observed: dict[str, object] = {}

    class FixedDateTime:
        """Provide a deterministic timezone-aware replacement for datetime."""

        @classmethod
        def now(cls, timezone: object) -> datetime:
            """Record the requested timezone and return a fixed Madrid instant."""

            observed["timezone"] = timezone
            return datetime(2026, 8, 11, 0, 5, tzinfo=timezone)

    monkeypatch.setattr(script, "datetime", FixedDateTime)

    arguments = script.build_parser().parse_args([])

    assert getattr(observed["timezone"], "key") == "Europe/Madrid"
    assert arguments.target_date == date(2026, 8, 11)


def test_dry_run_previews_report_without_writing_or_opening_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep dry-run credential-free, network-free, database-free, and read-only."""

    script = _load_publish_script()
    daily = _daily_stub()
    monkeypatch.setattr(script, "load_latest_daily_picks", lambda *_: daily)
    monkeypatch.setattr(script, "load_upcoming_opportunities", lambda *_: ("fixture-run", ()))
    monkeypatch.setattr(script, "render_messages", lambda _: ("BEST", "OTHERS"))
    monkeypatch.setattr(script, "render_daily_report", lambda _: "URL: fixture\n")

    def forbidden(*_: object, **__: object) -> object:
        """Fail if a dry run reaches any side-effect dependency."""

        raise AssertionError("dry-run attempted a forbidden side effect")

    monkeypatch.setattr(script, "load_settings", forbidden)
    monkeypatch.setattr(script, "PublishStateStore", forbidden)
    monkeypatch.setattr(script, "TelegramClient", forbidden)
    monkeypatch.setattr(script, "publish_daily_messages", forbidden)
    monkeypatch.setattr(script, "write_daily_report", forbidden)

    assert script.run(daily.target_date, dry_run=True) == 0
    output = capsys.readouterr().out
    assert "DAILY REPORT (NOT WRITTEN)" in output
    assert "URL: fixture" in output


def test_success_or_idempotent_noop_writes_report_after_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Write the report only after both publication slots are confirmed."""

    script = _load_publish_script()
    daily = _daily_stub()
    calls: list[str] = []
    monkeypatch.setattr(script, "load_latest_daily_picks", lambda *_: daily)
    monkeypatch.setattr(script, "load_upcoming_opportunities", lambda *_: ("fixture-run", ()))
    monkeypatch.setattr(script, "render_messages", lambda _: ("BEST", "OTHERS"))
    monkeypatch.setattr(script, "render_daily_report", lambda _: "REPORT")
    monkeypatch.setattr(
        script,
        "load_settings",
        lambda *_args, **_kwargs: SimpleNamespace(
            bot_token="test-only",
            timeout_seconds=1,
            state_db_path=tmp_path / "state.sqlite3",
            channel_id="@fixture",
        ),
    )
    monkeypatch.setattr(script, "TelegramClient", lambda *_args, **_kwargs: object())
    store = SimpleNamespace(
        reported_match_ids=lambda: set(),
        mark_reported=lambda **_: None,
    )
    monkeypatch.setattr(script, "PublishStateStore", lambda *_: store)
    monkeypatch.setattr(script, "existing_report_match_ids", lambda *_: set())

    def publish(**_: object) -> SimpleNamespace:
        """Return the same outcome shape used for an idempotent no-op."""

        calls.append("publish")
        slot = SimpleNamespace(status="already_sent", message_id=42)
        return SimpleNamespace(target_date=daily.target_date, best=slot, others=slot)

    def write_report(received: object, root: Path) -> Path:
        """Capture report ordering without touching the real daily report."""

        assert received == ()
        assert root == script.TELEGRAM_ROOT
        calls.append("report")
        return tmp_path / "daily_report.txt"

    monkeypatch.setattr(script, "publish_daily_messages", publish)
    monkeypatch.setattr(script, "write_daily_report", write_report)

    assert script.run(daily.target_date, dry_run=False) == 0
    assert calls == ["publish", "report"]


def test_failed_publication_does_not_replace_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preserve the previous report when Telegram delivery is not confirmed."""

    script = _load_publish_script()
    daily = _daily_stub()
    monkeypatch.setattr(script, "load_latest_daily_picks", lambda *_: daily)
    monkeypatch.setattr(script, "load_upcoming_opportunities", lambda *_: ("fixture-run", ()))
    monkeypatch.setattr(script, "render_messages", lambda _: ("BEST", "OTHERS"))
    monkeypatch.setattr(script, "render_daily_report", lambda _: "REPORT")
    monkeypatch.setattr(
        script,
        "load_settings",
        lambda *_args, **_kwargs: SimpleNamespace(
            bot_token="test-only",
            timeout_seconds=1,
            state_db_path=tmp_path / "state.sqlite3",
            channel_id="@fixture",
        ),
    )
    monkeypatch.setattr(script, "TelegramClient", lambda *_args, **_kwargs: object())
    store = SimpleNamespace(
        reported_match_ids=lambda: set(),
        mark_reported=lambda **_: None,
    )
    monkeypatch.setattr(script, "PublishStateStore", lambda *_: store)
    monkeypatch.setattr(script, "existing_report_match_ids", lambda *_: set())

    def fail_publish(**_: object) -> object:
        """Simulate a publication failure before report replacement."""

        raise RuntimeError("delivery failed")

    def forbidden_write(*_: object, **__: object) -> object:
        """Fail if an unconfirmed publication attempts to replace the report."""

        raise AssertionError("report was written after failed publication")

    monkeypatch.setattr(script, "publish_daily_messages", fail_publish)
    monkeypatch.setattr(script, "write_daily_report", forbidden_write)

    with pytest.raises(RuntimeError, match="delivery failed"):
        script.run(daily.target_date, dry_run=False)
