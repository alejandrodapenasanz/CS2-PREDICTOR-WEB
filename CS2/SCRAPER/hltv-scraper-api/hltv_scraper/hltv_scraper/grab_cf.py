import argparse
import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

import nodriver as uc

CF_TARGET_URL = "https://www.hltv.org/stats"
CF_SESSION_FILE = Path(__file__).resolve().parents[1] / "cf_session.json"


def hltv_target_url(value: str) -> str:
    """Keep the interactive helper scoped to HTTPS pages on HLTV."""
    parts = urlsplit(value)
    if parts.scheme != "https" or parts.netloc not in {"www.hltv.org", "hltv.org"} or parts.username or parts.password:
        raise argparse.ArgumentTypeError("Expected an HTTPS URL on hltv.org.")
    return value


async def grab_cf_session(target_url: str = CF_TARGET_URL) -> bool:
    target_url = hltv_target_url(target_url)
    browser = await uc.start(headless=False)
    try:
        tab = await browser.get(target_url)
        print("Waiting for cf_clearance... (solve the challenge in the browser window if it appears)", flush=True)

        for _ in range(180):
            cookies = await browser.cookies.get_all()
            cf = next((c for c in cookies if c.name == "cf_clearance" and c.value), None)
            if cf:
                ua = await tab.evaluate("navigator.userAgent")
                CF_SESSION_FILE.write_text(
                    json.dumps({"cf_clearance": cf.value, "user_agent": ua}, indent=2),
                    encoding="utf-8",
                )
                print(f"Saved session to {CF_SESSION_FILE}", flush=True)
                return True
            await asyncio.sleep(1)
        print("Failed to obtain cf_clearance within 3 minutes.", flush=True)
        return False
    finally:
        browser.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Open an HLTV page visibly for interactive session renewal.")
    parser.add_argument("--url", type=hltv_target_url, default=CF_TARGET_URL)
    args = parser.parse_args()
    return 0 if uc.loop().run_until_complete(grab_cf_session(args.url)) else 1


if __name__ == "__main__":
    sys.exit(main())
