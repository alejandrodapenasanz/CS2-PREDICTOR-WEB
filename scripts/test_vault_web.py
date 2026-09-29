"""Offline checks for the public dashboard's private-state boundary.

Run: python scripts/test_vault_web.py. No HTTP requests or production writes.
"""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vault_dashboard", ROOT / "WEB/serve.py")
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


class DashboardBoundaryTests(unittest.TestCase):
    """Only the generated public data.js can be read from VAULT over HTTP."""

    def setUp(self) -> None:
        """Build a request handler without opening a socket."""
        self.handler = SERVER.DashboardHandler.__new__(SERVER.DashboardHandler)
        self.handler.directory = str(ROOT / "WEB")

    def test_public_data_maps_to_vault(self) -> None:
        """Cache-busting does not change the published artifact's location."""
        self.assertEqual(
            Path(self.handler.translate_path("/data.js?ts=123")),
            ROOT / "VAULT/WEB/data.js",
        )

    def test_traversal_cannot_read_secrets(self) -> None:
        """Encoded/plain traversal never escapes the public static tree."""
        for path in (
            "/../VAULT/TELEGRAM/.env",
            "/%2e%2e/VAULT/TENNIS/BBDD/tennis.sqlite3",
            "/VAULT/CS2/MODEL/artifacts/model.pkl",
            "/data.js/../TELEGRAM/.env",
        ):
            with self.subTest(path=path):
                resolved = Path(self.handler.translate_path(path)).resolve()
                self.assertTrue(resolved.is_relative_to(ROOT / "WEB"))

    def test_listing_disabled(self) -> None:
        """An unknown directory cannot enumerate project files."""
        self.handler.send_error = Mock()
        self.assertIsNone(self.handler.list_directory(str(ROOT / "WEB")))
        self.handler.send_error.assert_called_once_with(404)


if __name__ == "__main__":
    unittest.main()
