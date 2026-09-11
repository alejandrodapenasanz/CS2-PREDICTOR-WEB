"""Deterministic, offline boot smoke for the scraper's real entrypoints."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_ROOT = PROJECT_ROOT / "hltv_scraper"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPECTED_SPIDERS = {
    "hltv_big_results",
    "hltv_match",
    "hltv_news",
    "hltv_player",
    "hltv_player_stats_overview",
    "hltv_players_search",
    "hltv_results",
    "hltv_team",
    "hltv_team_matches",
    "hltv_teams_search",
    "hltv_top30",
    "hltv_upcoming_matches",
    "hltv_valve_ranking",
}
EXPECTED_API_RULES = {
    "/api/v1/matches/upcoming",
    "/api/v1/news",
    "/api/v1/players/search/<string:name>",
    "/api/v1/results/",
    "/api/v1/teams/rankings",
}
CLI_ENTRYPOINTS = (
    PROJECT_ROOT / "scripts" / "collect_hltv_data.py",
    PROJECT_ROOT / "scripts" / "collect_player_compare_stats.py",
    SCRAPY_ROOT / "hltv_scraper" / "grab_cf.py",
)


def deterministic_environment() -> dict[str, str]:
    """Return a stable, non-interactive environment for child entrypoints."""

    environment = os.environ.copy()
    environment.update(
        {
            "FLASK_DEBUG": "0",
            "NO_COLOR": "1",
            "PYTHONHASHSEED": "42",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    return environment


def run_checked(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a boot-only command and expose deterministic output on failure."""

    return subprocess.run(
        command,
        cwd=cwd,
        env=deterministic_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


def smoke_flask() -> None:
    """Create the production Flask app and validate its registered API surface."""

    from app import create_app

    application = create_app()
    rules = {rule.rule for rule in application.url_map.iter_rules()}
    missing = EXPECTED_API_RULES - rules
    if missing:
        raise RuntimeError(f"Flask boot missed routes: {sorted(missing)}")


def smoke_scrapy() -> None:
    """Boot the real Scrapy project and require every production spider."""

    result = run_checked([sys.executable, "-m", "scrapy", "list"], cwd=SCRAPY_ROOT)
    spiders = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    missing = EXPECTED_SPIDERS - spiders
    if missing:
        raise RuntimeError(f"Scrapy boot missed spiders: {sorted(missing)}")


def smoke_cli_entrypoints() -> None:
    """Exercise argument-parser boot without requests or opening a browser."""

    for entrypoint in CLI_ENTRYPOINTS:
        run_checked([sys.executable, str(entrypoint), "--help"], cwd=PROJECT_ROOT)


def main() -> int:
    random.seed(42)
    smoke_flask()
    smoke_scrapy()
    smoke_cli_entrypoints()
    print("Scraper entrypoint smoke: ok (seed=42, network=off)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
