"""Offline checks for the Telegram launcher, documentation, and secret policy."""

from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import sys
from pathlib import Path


TELEGRAM_ROOT = Path(__file__).resolve().parents[1]
BOT_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9_])\d{8,12}:[A-Za-z0-9_-]{30,}(?![A-Za-z0-9_])")


def _load_publish_script() -> object:
    """Import the standalone publisher script without invoking its main block."""

    script_path = TELEGRAM_ROOT / "scripts" / "publish_cs2.py"
    spec = importlib.util.spec_from_file_location("telegram_publish_cs2_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_is_independent_of_the_calling_directory() -> None:
    """Require the launcher to resolve the Python script from its own path."""

    launcher = (TELEGRAM_ROOT / "run_telegram.ps1").read_text(encoding="utf-8")

    assert "$PSScriptRoot" in launcher
    assert "scripts\\publish_cs2.py" in launcher
    assert ".venv\\Scripts\\python.exe" in launcher
    assert "[switch]$DryRun" in launcher
    assert "--dry-run" in launcher
    assert "daily_report.txt" in launcher
    assert "Europe/Madrid" in launcher


def test_readme_documents_secure_read_only_channel_setup() -> None:
    """Keep the mandatory token rotation and channel permissions visible."""

    readme = (TELEGRAM_ROOT / "README.md").read_text(encoding="utf-8")

    for required_text in (
        "@cs2DailyPicks",
        "@CS2PredictorPublisherBot",
        "/revoke",
        "Post Messages",
        "Discussion Group",
        "--dry-run",
        "exactly two",
        "daily_report.txt",
        "Europe/Madrid",
    ):
        assert required_text in readme


def test_readme_documents_handle_only_daily_report_cta() -> None:
    """Keep the report CTA exact and exclude a t.me URL from its example block."""

    readme = (TELEGRAM_ROOT / "README.md").read_text(encoding="utf-8")
    section = re.search(
        r"## Incremental opportunities report.*?```text\n(?P<example>.*?)\n```",
        readme,
        flags=re.DOTALL,
    )

    assert section is not None
    example = section.group("example")
    assert "See all model predictions for free on Telegram:" in example
    assert "\n@cs2DailyPicks" in example
    assert "https://t.me/" not in example


def test_no_bot_token_is_committed_under_telegram() -> None:
    """Reject Bot API token-shaped secrets in all versionable text files."""

    ignored_parts = {".venv", "__pycache__", ".pytest_cache", "data"}
    findings: list[str] = []
    for path in TELEGRAM_ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored_parts for part in path.parts):
            continue
        if path == TELEGRAM_ROOT / ".env":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if BOT_TOKEN_PATTERN.search(content):
            findings.append(str(path.relative_to(TELEGRAM_ROOT)))

    assert findings == [], f"Bot token-shaped secrets found in: {findings}"


def test_local_env_file_is_explicitly_gitignored() -> None:
    """Require the only secret-bearing local environment file to stay unversioned."""

    gitignore_lines = {
        line.strip()
        for line in (TELEGRAM_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".env" in gitignore_lines
    assert "daily_report.txt" in gitignore_lines


def test_python_cli_help_is_offline_and_importable() -> None:
    """Ensure CLI discovery succeeds without credentials or network access."""

    completed = subprocess.run(
        [sys.executable, str(TELEGRAM_ROOT / "scripts" / "publish_cs2.py"), "--help"],
        cwd=TELEGRAM_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--date" in completed.stdout
    assert "--dry-run" in completed.stdout


def test_preview_reconfigures_cp1252_streams_before_printing(monkeypatch) -> None:
    """Ensure emoji previews do not fail on the default Windows cp1252 streams."""

    script = _load_publish_script()
    stdout_bytes = io.BytesIO()
    stderr_bytes = io.BytesIO()
    stdout = io.TextIOWrapper(stdout_bytes, encoding="cp1252", errors="strict")
    stderr = io.TextIOWrapper(stderr_bytes, encoding="cp1252", errors="strict")

    with monkeypatch.context() as patch:
        patch.setattr(sys, "stdout", stdout)
        patch.setattr(sys, "stderr", stderr)
        script._configure_utf8_output()
        script._print_preview("🔥 BEST", "📋 OTHERS", "URL: https://fixture.invalid\n")
        stdout.flush()
        rendered = stdout_bytes.getvalue().decode("utf-8")

    assert "🔥 BEST" in rendered
    assert "📋 OTHERS" in rendered
    assert "DAILY REPORT (NOT WRITTEN)" in rendered
