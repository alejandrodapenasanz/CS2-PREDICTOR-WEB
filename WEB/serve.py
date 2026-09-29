"""Serve the local dashboard without exposing private VAULT files.

Run Python 3.13: python WEB/serve.py [--port 8000]. Only WEB assets and the
published data.js are served, bound to localhost. Never serve repository root.
"""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from os import PathLike
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

WEB_ROOT = Path(__file__).resolve().parent
DATA_PATH = WEB_ROOT.parent / "VAULT/WEB/data.js"


class DashboardHandler(SimpleHTTPRequestHandler):
    """Map the one public generated artifact; deny directory listings."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Constrain all other requests to the static WEB directory."""
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def translate_path(self, path: str) -> str:
        """Do not expose databases, credentials or models through HTTP."""
        if urlsplit(path).path == "/data.js":
            return str(DATA_PATH)
        return super().translate_path(path)

    def list_directory(self, path: str | PathLike[str]) -> None:
        """Do not list source or data directories."""
        self.send_error(404)
        return None

    def end_headers(self) -> None:
        """Every refresh sees the latest pipeline publication."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    """Start a localhost-only dashboard server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler) as server:
        print(f"Dashboard: http://127.0.0.1:{args.port}/", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
