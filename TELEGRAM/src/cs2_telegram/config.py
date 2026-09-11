"""Load secure, dependency-free configuration for the CS2 Telegram publisher.

The module reads a local ``TELEGRAM/.env`` file and then applies process
environment overrides.  It never logs configuration values and hides the bot
token from dataclass representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
from typing import Mapping


TELEGRAM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = TELEGRAM_ROOT / ".env"
DEFAULT_STATE_DB = TELEGRAM_ROOT / "data" / "telegram_publish_state.sqlite3"
DEFAULT_CHANNEL_ID = "@cs2DailyPicks"
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigurationError(ValueError):
    """Report invalid Telegram configuration without revealing secret values."""


@dataclass(frozen=True)
class TelegramSettings:
    """Immutable settings required by the Telegram publishing transport."""

    bot_token: str = field(repr=False)
    channel_id: str = DEFAULT_CHANNEL_ID
    state_db_path: Path = DEFAULT_STATE_DB
    timeout_seconds: float = 20.0


def _unquote_dotenv_value(raw_value: str, line_number: int) -> str:
    """Return a dotenv value with optional matching quotes removed."""

    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            raise ConfigurationError(f"Unterminated quoted value in Telegram .env on line {line_number}.")
        return value[1:-1]
    # Treat a whitespace-prefixed hash as an inline comment.  Hashes inside a
    # token/value remain untouched.
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def _read_dotenv(path: Path, *, required: bool) -> dict[str, str]:
    """Parse a small, non-interpolating dotenv file without external packages."""

    if not path.exists():
        if required:
            raise ConfigurationError(f"Telegram environment file does not exist: {path}")
        return {}

    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        raise ConfigurationError("Telegram environment file could not be read.") from None

    for line_number, original_line in enumerate(lines, start=1):
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(f"Invalid Telegram .env assignment on line {line_number}.")
        key, raw_value = line.split("=", maxsplit=1)
        key = key.strip()
        if not _ENV_KEY_RE.fullmatch(key):
            raise ConfigurationError(f"Invalid Telegram .env key on line {line_number}.")
        values[key] = _unquote_dotenv_value(raw_value, line_number)
    return values


def _resolve_state_path(raw_path: str | None) -> Path:
    """Resolve a configured state database relative to ``TELEGRAM``."""

    if raw_path is None or not raw_path.strip():
        configured = DEFAULT_STATE_DB
    else:
        configured = Path(raw_path.strip()).expanduser()
    if not configured.is_absolute():
        configured = TELEGRAM_ROOT / configured
    resolved = configured.resolve()
    try:
        resolved.relative_to(TELEGRAM_ROOT.resolve())
    except ValueError:
        raise ConfigurationError("TELEGRAM_STATE_DB_PATH must remain inside the TELEGRAM directory.") from None
    return resolved


def load_settings(
    env_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    require_token: bool = True,
) -> TelegramSettings:
    """Load Telegram settings with environment variables taking precedence.

    Args:
        env_file: Optional explicit dotenv file.  An explicit missing file is an
            error; the default ``TELEGRAM/.env`` is optional.
        environ: Optional environment mapping, primarily for deterministic tests.
        require_token: Require ``TELEGRAM_BOT_TOKEN`` to be non-empty.  This can
            be disabled for a purely local dry run, although a client still
            refuses to send without a real token.

    Returns:
        A validated :class:`TelegramSettings` instance.
    """

    explicit_env_file = env_file is not None
    dotenv_path = Path(env_file).expanduser().resolve() if env_file else DEFAULT_ENV_FILE
    values = _read_dotenv(dotenv_path, required=explicit_env_file)
    source_environment = os.environ if environ is None else environ
    for key in (
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHANNEL_ID",
        "TELEGRAM_STATE_DB_PATH",
        "TELEGRAM_REQUEST_TIMEOUT_SECONDS",
    ):
        if key in source_environment:
            values[key] = source_environment[key]

    token = values.get("TELEGRAM_BOT_TOKEN", "").strip()
    if require_token and not token:
        raise ConfigurationError("TELEGRAM_BOT_TOKEN is required in TELEGRAM/.env or the process environment.")

    channel_id = values.get("TELEGRAM_CHANNEL_ID", DEFAULT_CHANNEL_ID).strip()
    if not channel_id:
        raise ConfigurationError("TELEGRAM_CHANNEL_ID must not be empty.")

    raw_timeout = values.get("TELEGRAM_REQUEST_TIMEOUT_SECONDS", "20").strip()
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError:
        raise ConfigurationError("TELEGRAM_REQUEST_TIMEOUT_SECONDS must be numeric.") from None
    if timeout_seconds <= 0:
        raise ConfigurationError("TELEGRAM_REQUEST_TIMEOUT_SECONDS must be greater than zero.")

    return TelegramSettings(
        bot_token=token,
        channel_id=channel_id,
        state_db_path=_resolve_state_path(values.get("TELEGRAM_STATE_DB_PATH")),
        timeout_seconds=timeout_seconds,
    )
