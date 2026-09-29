"""Upcoming identities are explicit listing attributes, not detail-page guesses."""

import pytest
from parsel import Selector

from hltv_scraper.hltv_scraper.spiders.parsers.upcoming_match_team import UpcomingMatchTeamParser


@pytest.mark.parametrize("value,expected", [("13679", "13679"), ("", None), ("0", None), ("-1", None), ("TBD", None)])
def test_listing_preserves_only_valid_team_ids(value, expected):
    html = f'<div class="match-zone-wrapper"><div class="match-wrapper" team1="{value}"><div class="team1"><div class="match-teamname">FOKUS</div></div></div></div>'
    team = UpcomingMatchTeamParser.parse(Selector(text=html), 1)
    assert team["id"] == expected
    assert team["name"] == "FOKUS"
