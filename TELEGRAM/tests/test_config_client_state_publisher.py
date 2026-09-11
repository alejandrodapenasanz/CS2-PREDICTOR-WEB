"""Offline tests for Telegram configuration, transport and idempotent state."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from urllib import parse

from cs2_telegram.client import TelegramAPIError, TelegramClient, TelegramSendResult
from cs2_telegram.config import ConfigurationError, load_settings
from cs2_telegram.publisher import PublicationDeliveryError, publish_daily_messages
from cs2_telegram.state import (
    AmbiguousPublicationError,
    PayloadConflictError,
    PublishStateStore,
)


class _FakeResponse:
    """Provide the subset of urllib response behavior used by the client."""

    def __init__(self, payload: dict[str, object]) -> None:
        """Encode a JSON response payload for later reads."""

        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        """Enter a no-op response context."""

        return self

    def __exit__(self, *args: object) -> None:
        """Exit a no-op response context."""

        return None

    def read(self) -> bytes:
        """Return the encoded response body."""

        return self._body


class _FakeMessageClient:
    """Record publisher calls and optionally simulate an uncertain failure."""

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        """Configure the one-based call number that should fail."""

        self.calls: list[tuple[str, str, str | None]] = []
        self.fail_on_call = fail_on_call

    def send_message(self, chat_id: str, text: str, *, parse_mode: str | None = None) -> TelegramSendResult:
        """Record a send and return a deterministic confirmation."""

        self.calls.append((chat_id, text, parse_mode))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("network result unknown")
        return TelegramSendResult(chat_id=chat_id, message_id=100 + len(self.calls))


class ConfigTests(unittest.TestCase):
    """Verify dotenv precedence, defaults and secret-safe representations."""

    def test_load_settings_uses_env_override_and_hides_token(self) -> None:
        """Load local values while giving the process environment precedence."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text(
                "TELEGRAM_BOT_TOKEN=file-secret\n"
                "TELEGRAM_CHANNEL_ID=@from_file\n"
                "TELEGRAM_STATE_DB_PATH=data/custom.sqlite3\n",
                encoding="utf-8",
            )
            settings = load_settings(
                env_file,
                environ={"TELEGRAM_CHANNEL_ID": "@from_environment"},
            )

        self.assertEqual(settings.bot_token, "file-secret")
        self.assertEqual(settings.channel_id, "@from_environment")
        self.assertNotIn("file-secret", repr(settings))
        self.assertTrue(settings.state_db_path.is_absolute())

    def test_missing_token_is_clear_but_does_not_include_values(self) -> None:
        """Reject a missing credential with a non-secret diagnostic."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text("TELEGRAM_CHANNEL_ID=@channel\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_settings(env_file, environ={})

    def test_state_database_cannot_escape_telegram_directory(self) -> None:
        """Reject a state path outside the dedicated integration directory."""

        with tempfile.TemporaryDirectory() as temporary_directory:
            env_file = Path(temporary_directory) / ".env"
            env_file.write_text(
                "TELEGRAM_BOT_TOKEN=test-secret\nTELEGRAM_STATE_DB_PATH=C:/outside/state.sqlite3\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_settings(env_file, environ={})


class ClientTests(unittest.TestCase):
    """Verify encoded requests, confirmations and token-safe failures."""

    def test_client_posts_encoded_message_and_parses_confirmation(self) -> None:
        """Use an injected transport instead of the real Telegram network."""

        captured: dict[str, object] = {}

        def transport(http_request: object, *, timeout: float) -> _FakeResponse:
            """Capture request fields and return a successful API response."""

            captured["request"] = http_request
            captured["timeout"] = timeout
            return _FakeResponse({"ok": True, "result": {"message_id": 77, "date": 123456}})

        client = TelegramClient("123:test-secret", transport=transport)
        result = client.send_message("@channel", "A & B")
        http_request = captured["request"]
        body = parse.parse_qs(http_request.data.decode("utf-8"))  # type: ignore[attr-defined]

        self.assertEqual(body["chat_id"], ["@channel"])
        self.assertEqual(body["text"], ["A & B"])
        self.assertEqual(result.message_id, 77)

    def test_transport_exception_never_exposes_token(self) -> None:
        """Discard a transport exception even when it embeds the secret URL."""

        token = "123:must-not-leak"

        def transport(http_request: object, *, timeout: float) -> _FakeResponse:
            """Raise an unsafe exception like urllib may produce."""

            raise RuntimeError(f"failed URL containing {token}")

        client = TelegramClient(token, transport=transport)
        with self.assertRaises(TelegramAPIError) as caught:
            client.send_message("@channel", "prediction")
        self.assertNotIn(token, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)


class StateAndPublisherTests(unittest.TestCase):
    """Verify durable idempotency, conflicts, dry runs and fail-closed sends."""

    def setUp(self) -> None:
        """Create an isolated SQLite publication ledger."""

        self._temporary_directory = tempfile.TemporaryDirectory()
        self.store = PublishStateStore(Path(self._temporary_directory.name) / "publish.sqlite3")
        self.target_date = date(2026, 8, 11)

    def tearDown(self) -> None:
        """Remove the isolated ledger."""

        self._temporary_directory.cleanup()

    def test_publisher_sends_exactly_two_and_rerun_is_noop(self) -> None:
        """Send best/others once and return existing confirmations on rerun."""

        client = _FakeMessageClient()
        first = publish_daily_messages(
            client=client,
            store=self.store,
            channel_id="@channel",
            target_date=self.target_date,
            source_run_id="run-1",
            best_message="BEST",
            others_message="OTHERS",
        )
        second = publish_daily_messages(
            client=client,
            store=self.store,
            channel_id="@channel",
            target_date=self.target_date,
            source_run_id="run-2",
            best_message="BEST",
            others_message="OTHERS",
        )

        self.assertEqual(
            client.calls,
            [
                ("@channel", "BEST", "HTML"),
                ("@channel", "OTHERS", "HTML"),
            ],
        )
        self.assertEqual(first.best.status, "sent")
        self.assertEqual(second.best.status, "already_sent")
        self.assertEqual(second.others.status, "already_sent")

    def test_reported_match_ids_are_immutable_and_idempotent(self) -> None:
        """Remember exported report matches independently of the text file."""

        self.store.mark_reported(
            match_id="2396583",
            match_date=self.target_date,
            hltv_url="https://www.hltv.org/matches/2396583/a-vs-b",
            source_run_id="run-1",
        )
        self.store.mark_reported(
            match_id="2396583",
            match_date=self.target_date,
            hltv_url="https://www.hltv.org/matches/2396583/a-vs-b",
            source_run_id="run-2",
        )

        self.assertEqual(self.store.reported_match_ids(), {"2396583"})

    def test_dry_run_does_not_send_or_reserve(self) -> None:
        """Validate both messages without a store or database."""

        client = _FakeMessageClient()
        outcome = publish_daily_messages(
            client=client,
            store=None,
            channel_id="@channel",
            target_date=self.target_date,
            source_run_id="dry-run",
            best_message="BEST",
            others_message="OTHERS",
            dry_run=True,
        )

        self.assertTrue(outcome.dry_run)
        self.assertEqual(client.calls, [])

    def test_payload_conflict_fails_before_any_send(self) -> None:
        """Reject changed predictions for an immutable daily publication key."""

        original_client = _FakeMessageClient()
        publish_daily_messages(
            client=original_client,
            store=self.store,
            channel_id="@channel",
            target_date=self.target_date,
            source_run_id="run-1",
            best_message="BEST",
            others_message="OTHERS",
        )
        conflicting_client = _FakeMessageClient()
        with self.assertRaises(PayloadConflictError):
            publish_daily_messages(
                client=conflicting_client,
                store=self.store,
                channel_id="@channel",
                target_date=self.target_date,
                source_run_id="run-2",
                best_message="CHANGED",
                others_message="OTHERS",
            )
        self.assertEqual(conflicting_client.calls, [])

    def test_uncertain_failure_is_persisted_and_blocks_rerun(self) -> None:
        """Reserve before transport and never retry an ambiguous delivery."""

        failing_client = _FakeMessageClient(fail_on_call=1)
        with self.assertRaises(PublicationDeliveryError):
            publish_daily_messages(
                client=failing_client,
                store=self.store,
                channel_id="@channel",
                target_date=self.target_date,
                source_run_id="run-1",
                best_message="BEST",
                others_message="OTHERS",
            )

        failed = self.store.get("@channel", self.target_date, "best")
        self.assertIsNotNone(failed)
        self.assertEqual(failed.status, "failed")  # type: ignore[union-attr]
        retry_client = _FakeMessageClient()
        with self.assertRaises(AmbiguousPublicationError):
            publish_daily_messages(
                client=retry_client,
                store=self.store,
                channel_id="@channel",
                target_date=self.target_date,
                source_run_id="run-2",
                best_message="BEST",
                others_message="OTHERS",
            )
        self.assertEqual(retry_client.calls, [])


if __name__ == "__main__":
    unittest.main()
