from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIPELINE import start


PLAYER_SCRAPER = ROOT / "SCRAPER" / "hltv-scraper-api" / "scripts" / "collect_player_compare_stats.py"
PLAYER_SPEC = importlib.util.spec_from_file_location("player_compare_stats_test", PLAYER_SCRAPER)
assert PLAYER_SPEC and PLAYER_SPEC.loader
player_compare_stats = importlib.util.module_from_spec(PLAYER_SPEC)
sys.modules[PLAYER_SPEC.name] = player_compare_stats
PLAYER_SPEC.loader.exec_module(player_compare_stats)


class HltvParserTests(unittest.TestCase):
    def test_player_profile_parser_completes_core_snapshot(self) -> None:
        html = """
        <div class="player-summary-stat-box-rating-wrapper">
          <div class="player-summary-stat-box-rating-data-text">1.12</div>
          <div class="player-summary-stat-box-data-description-text">Rating 3.0</div>
        </div>
        <div class="player-summary-stat-box-data-wrapper">
          <div class="player-summary-stat-box-data traditionalData">73.5%</div>
          <div class="player-summary-stat-box-data-text traditionalData">KAST</div>
        </div>
        <div class="stats-row"><span>Kills / round</span><span>0.71</span></div>
        <div class="stats-row"><span>Assists / round</span><span>0.16</span></div>
        <div class="stats-row"><span>Deaths / round</span><span>0.62</span></div>
        <div class="stats-row"><span>Damage / Round</span><span>79.4</span></div>
        <div class="stats-row"><span>Impact rating</span><span>1.14</span></div>
        <div class="stats-row"><span>Maps played</span><span>24</span></div>
        """
        player = player_compare_stats.PlayerRef("1", "Alpha", "alpha", "/stats/players/1/alpha")
        parsed = player_compare_stats.parse_player_profile_page(
            html,
            player,
            "https://www.hltv.org/stats/players/1/alpha",
            "past3months",
        )
        self.assertEqual(parsed["maps"], 24)
        self.assertEqual(parsed["stats"]["Rating 3.0"], "1.12")
        self.assertEqual(parsed["stats"]["KPR"], "0.71")
        self.assertEqual(parsed["stats"]["DPR"], "0.62")
        self.assertEqual(parsed["stats"]["ADR"], "79.4")
        self.assertTrue(player_compare_stats.player_stats_complete(parsed["stats"]))

    def test_player_profile_url_uses_same_calendar_window(self) -> None:
        player = player_compare_stats.PlayerRef("1", "Alpha", "alpha", "/stats/players/1/alpha")
        url = player_compare_stats.player_stats_url(player, "past3months", date(2026, 7, 12))
        self.assertIn("startDate=2026-04-12", url)
        self.assertIn("endDate=2026-07-12", url)

    def test_player_profile_missing_marker_is_not_a_statistic(self) -> None:
        player = {
            "stats": {
                "Rating 3.0": "1.01",
                "KPR": "0.68",
                "APR": "0.17",
                "KAST": "72.0%",
                "Impact": "1.02",
            }
        }
        player_compare_stats.merge_player_profile_stats(
            player,
            {"url": "https://example.test", "time_filter": "past3months", "maps": 0,
             "stats": {"KPR": "-", "DPR": "-", "ADR": "-"}},
        )
        self.assertNotIn("KPR", player["stats"])
        self.assertNotIn("ADR", player["stats"])
        self.assertEqual(player["maps"], 0)
        self.assertFalse(player_compare_stats.player_stats_complete(player["stats"]))

    def test_compare_parser_keeps_empty_player_column_empty(self) -> None:
        html = """
        <div class="stats-section stats-player stats-player-compare">
          <div class="columns">
            <div class="col stats-rows">
              <div class="selector-map-count">Based on 31 maps</div>
              <div class="stats-row"><span>KPR</span><span>0.62</span></div>
              <div class="stats-row"><span>KAST</span><span>70.7%</span></div>
            </div>
            <div class="col stats-rows">
              <div class="selector-map-count">Based on 0 maps</div>
            </div>
          </div>
        </div>
        """
        p1 = player_compare_stats.PlayerRef("1", "Alpha", "alpha", "/player/1/alpha")
        p2 = player_compare_stats.PlayerRef("2", "Beta", "beta", "/player/2/beta")

        parsed = player_compare_stats.parse_compare_page(html, p1, p2, "https://example.test", "past3months")

        self.assertEqual(parsed["players"][0]["maps"], 31)
        self.assertEqual(parsed["players"][0]["stats"]["KPR"], "0.62")
        self.assertEqual(parsed["players"][1]["maps"], 0)
        self.assertEqual(parsed["players"][1]["stats"], {})

    def test_player_profile_filter_keeps_only_ids_due_for_refresh(self) -> None:
        profiles = [{
            "id": "10",
            "profile": {
                "name": "Alpha",
                "squad": [
                    {"id": "101", "name": "Fresh"},
                    {"id": "102", "name": "Due"},
                ],
            },
        }]

        filtered = start.filter_team_profiles_players(profiles, {"102"})

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["profile"]["name"], "Alpha")
        self.assertEqual(filtered[0]["profile"]["squad"], [{"id": "102", "name": "Due"}])
        self.assertEqual(len(profiles[0]["profile"]["squad"]), 2)

    def test_player_payload_merge_prefers_online_refresh_and_keeps_fresh_cache(self) -> None:
        cached = {
            "101": {"id": "101", "maps": 20, "fetch_origin": "cache", "stats": {"KPR": 0.60}},
            "102": {"id": "102", "maps": 12, "fetch_origin": "cache", "stats": {"KPR": 0.61}},
        }
        fetched = {
            "comparisons_collected": 1,
            "comparisons_failed": 0,
            "requests_attempted": 3,
            "results": [{"players": [{"id": "102", "maps": 18, "stats": {"KPR": 0.70}}]}],
        }

        payload = start.merge_player_stats_payload(
            year=2026,
            requested_ids={"101", "102"},
            cached_players=cached,
            fetched_payload=fetched,
            refresh_ids={"102"},
        )
        players = {
            player["id"]: player
            for player in payload["results"][0]["players"]
        }

        self.assertEqual(payload["players_loaded"], 2)
        self.assertEqual(payload["players_fetched_online"], 1)
        self.assertEqual(payload["players_loaded_from_cache"], 1)
        self.assertEqual(payload["players_stale_fallback"], 0)
        self.assertEqual(players["101"]["fetch_origin"], "cache")
        self.assertEqual(players["102"]["fetch_origin"], "online")
        self.assertEqual(players["102"]["stats"]["KPR"], 0.70)

    def test_parse_veto_html(self) -> None:
        html = """
        <div class="standard-box veto-box"><div class="padding preformatted-text">Best of 3</div></div>
        <div class="standard-box veto-box"><div class="padding">
          <div>1. Alpha removed Mirage</div>
          <div>2. Beta picked Inferno</div>
          <div>3. Nuke was left over</div>
        </div></div>
        """
        veto = start.parse_veto_html(html)
        self.assertEqual(len(veto["steps"]), 3)
        self.assertEqual(veto["steps"][0]["action"], "ban")
        self.assertEqual(veto["steps"][1]["action"], "pick")
        self.assertEqual(veto["steps"][2]["action"], "decider")
        self.assertEqual(veto["context"]["best_of"], 3)

    def test_parse_maps_box_context_lan_swiss_incentive(self) -> None:
        html = """
        <div class="standard-box veto-box"><div class="padding preformatted-text">
          Best of 3 (LAN)

          * Swiss round 4 (teams with a 2-1 record). Winner advances to playoffs.
        </div></div>
        """
        context = start.parse_veto_html(html)["context"]
        self.assertEqual(context["best_of"], 3)
        self.assertEqual(context["environment"], "lan")
        self.assertEqual(context["stage"], "swiss")
        self.assertEqual(context["swiss_round"], 4)
        self.assertEqual(context["swiss_record"], "2-1")
        self.assertTrue(context["winner_advances"])
        self.assertTrue(context["high_stakes"])
        self.assertEqual(context["incentive_label"], "winner_advances")

    def test_parse_maps_box_context_losing_team_eliminated(self) -> None:
        html = """
        <div class="standard-box veto-box"><div class="padding preformatted-text">
          Best of 3 (LAN)

          * Swiss round 4 (teams with a 1-2 record). Losing team is eliminated.
        </div></div>
        """
        context = start.parse_veto_html(html)["context"]
        self.assertEqual(context["stage"], "swiss")
        self.assertEqual(context["swiss_record"], "1-2")
        self.assertTrue(context["loser_eliminated"])
        self.assertTrue(context["high_stakes"])
        self.assertEqual(context["incentive_label"], "elimination_match")

    def test_parse_mapstats_html_total_ct_t(self) -> None:
        table = """
        <table class="stats-table totalstats"><thead><tr><th>Alpha</th></tr></thead><tbody>
          <tr><td class="st-player"><a href="/stats/players/1/a">a</a></td>
          <td class="st-opkd traditional-data">2 : 1</td><td class="st-mks">3</td>
          <td class="st-kast gtSmartphone-only traditional-data">70.0%</td><td class="st-clutches">1</td>
          <td class="st-kills traditional-data">20 (10)</td><td class="st-assists">5 (2)</td>
          <td class="st-deaths traditional-data">15 (3)</td><td class="st-adr traditional-data">80.0</td>
          <td class="st-roundSwing">+2.5%</td><td class="st-rating">1.20</td></tr>
        </tbody></table>
        <table class="stats-table ctstats hidden"><thead><tr><th>Alpha</th></tr></thead><tbody>
          <tr><td class="st-player"><a href="/stats/players/1/a">a</a></td><td class="st-rating">1.10</td></tr>
        </tbody></table>
        <table class="stats-table tstats hidden"><thead><tr><th>Alpha</th></tr></thead><tbody>
          <tr><td class="st-player"><a href="/stats/players/1/a">a</a></td><td class="st-rating">1.30</td></tr>
        </tbody></table>
        """
        html = f"""
        <div class="match-info-box">
          <a href="/stats?event=1">Cup</a><span data-unix="1">2026-01-01 10:00</span>
          Map
          <div class="team-left"><a href="/stats/teams/10/alpha">Alpha</a><div class="bold won">13</div></div>
          <div class="team-right"><a href="/stats/teams/20/beta">Beta</a><div class="bold lost">7</div></div>
        </div>
        <div class="match-info-row"><div class="right"><span class="ct-color">7</span> : <span class="t-color">5</span> ( <span class="t-color">6</span> : <span class="ct-color">2</span> )</div><div class="bold">Breakdown</div></div>
        <a class="match-page-link" href="/matches/99/alpha-vs-beta">More info</a>
        {table}
        """
        parsed = start.parse_mapstats_html(html, "/stats/matches/mapstatsid/123/alpha-vs-beta")
        self.assertEqual(parsed["mapstats_id"], "123")
        self.assertEqual(parsed["match_id"], "99")
        self.assertEqual(len(parsed["player_stats"]), 3)
        total = next(row for row in parsed["player_stats"] if row["side"] == "total")
        self.assertEqual(total["kills"], 20)
        self.assertEqual(total["headshots"], 10)
        self.assertEqual(total["flash_assists"], 2)

    def test_parse_analytics_html_map_stats(self) -> None:
        html = """
        <main>
          <h1>Analytics center</h1>
          <section>
            <h2>Analytics summary</h2>
            <p>Alpha is better ranked (#23)</p>
            <p>Beta has won 4 out of the last 5 matches</p>
            <p>Alpha core lineup has only played 12 maps in the past 30 days</p>
          </section>
          <section>
            <h2>Map stats</h2>
            <div>Map Team First pick First ban Win Played Comment</div>
            <div>Mirage</div>
            <div>Alpha</div>
            <div>20% 10% 60% 5</div>
            <div>Comfort pick</div>
            <div>Beta</div>
            <div>0% 25% - 0</div>
          </section>
        </main>
        """
        parsed = start.parse_analytics_html(html, "123", "/betting/analytics/123/alpha-vs-beta", ["Alpha", "Beta"])
        self.assertTrue(parsed["available"])
        self.assertEqual(len(parsed["insights"]), 3)
        self.assertEqual(len(parsed["map_stats"]), 2)
        alpha = next(row for row in parsed["map_stats"] if row["team"] == "Alpha")
        beta = next(row for row in parsed["map_stats"] if row["team"] == "Beta")
        self.assertEqual(alpha["map"], "Mirage")
        self.assertEqual(alpha["first_pick_pct"], 20)
        self.assertEqual(alpha["first_ban_pct"], 10)
        self.assertEqual(alpha["win_pct"], 60)
        self.assertEqual(alpha["played"], 5)
        self.assertEqual(beta["win_pct"], None)

    def test_parse_current_analytics_markup_and_advertised_lineups(self) -> None:
        team1 = {
            "101": {
                "playerId": 101,
                "nickname": "AlphaStar",
                "rating": "1.20",
                "kpr": "0.75",
                "dpr": "0.60",
                "kast": "74.0%",
                "adr": "82.0",
                "multiKillRating": "1.19",
                "roundSwing": "+2.00%",
                "profileLinkUrl": "/player/101/alphastar",
            }
        }
        team2 = {
            "202": {
                "playerId": 202,
                "nickname": "StandIn",
                "rating": "0.90",
                "kpr": "0.60",
                "dpr": "0.72",
                "kast": "68.0%",
                "adr": "66.0",
                "multiKillRating": "0.91",
                "roundSwing": "-1.20%",
                "profileLinkUrl": "/player/202/standin",
            }
        }
        html = f"""
        <div class="teamsBox"><div class="event"><a href="/events/88/test-event">Test event</a></div></div>
        <div data-team1-players-data='{json.dumps(team1)}'></div>
        <div data-team2-players-data='{json.dumps(team2)}'></div>
        <div class="analytics-event-info">
          <div class="analytics-info"><div class="analytics-info-header">Test event</div><div class="analytics-info-sub-title">Event</div></div>
          <div class="analytics-info"><div class="analytics-info-header">$50,000</div><div class="analytics-info-sub-title">Prize pool at event</div></div>
          <div class="analytics-info"><div class="analytics-info-header">16</div><div class="analytics-info-sub-title">Teams competing</div></div>
        </div>
        <div class="analytics-insights-container">
          <div class="analytics-insights-team-header"><div class="team-name">Beta</div></div>
          <div class="analytics-insights-insight"><div class="analytics-insights-indicator against"></div><div class="analytics-insights-info">Beta is playing with stand-ins: StandIn instead of Regular</div></div>
          <div class="analytics-insights-insight"><div class="analytics-insights-indicator against"></div><div class="analytics-insights-info">StandIn has played less than 5 matches with core</div></div>
        </div>
        <table class="table-container"><thead><tr><th class="analytics-map-stats-map">Map</th></tr></thead><tbody>
          <tr><td rowspan="2"><div class="analytics-map-name">Mirage</div></td><td class="analytics-map-stats-team"><div class="maps-team-name">Alpha</div></td><td class="analytics-map-stats-pick-percentage">60%</td><td class="analytics-map-stats-ban-percentage">10%</td><td class="analytics-map-stats-win-percentage">70%</td><td class="analytics-map-stats-played">10</td><td class="analytics-map-stats-comment">Comfort pick</td></tr>
          <tr><td class="analytics-map-stats-team"><div class="maps-team-name">Beta</div></td><td class="analytics-map-stats-pick-percentage">0%</td><td class="analytics-map-stats-ban-percentage">50%</td><td class="analytics-map-stats-win-percentage">40%</td><td class="analytics-map-stats-played">8</td></tr>
        </tbody></table>
        <table class="analytics-handicap-table team1"><thead><tr><th><span class="team-name">Alpha</span><span class="match-map-count">20 matches, 50 maps.</span></th></tr></thead><tbody><tr><td>2 - 0 wins</td><td class="handicap-data">40%</td></tr><tr><td>2 - 1 wins</td><td class="handicap-data">20%</td></tr><tr><td>Overtimes</td><td class="handicap-data">5%</td></tr></tbody></table>
        <table class="analytics-handicap-table team2"><thead><tr><th><span class="team-name">Beta</span><span class="match-map-count">15 matches, 36 maps.</span></th></tr></thead><tbody><tr><td>2 - 0 wins</td><td class="handicap-data">20%</td></tr><tr><td>2 - 1 wins</td><td class="handicap-data">20%</td></tr><tr><td>Overtimes</td><td class="handicap-data">10%</td></tr></tbody></table>
        """
        parsed = start.parse_analytics_html(html, "123", "/matches/123/alpha-vs-beta", ["Alpha", "Beta"])
        self.assertEqual(parsed["event_metadata"], {"name": "Test event", "prize_pool": 50000, "teams_competing": 16})
        self.assertEqual(len(parsed["map_stats"]), 2)
        self.assertEqual(parsed["core_lineup"]["Beta"]["matches_lt"], 5)
        self.assertEqual(parsed["standins"]["Beta"], ["StandIn"])
        self.assertEqual(parsed["series_stats"]["team1"]["score_distribution"]["2_0_wins"], 40.0)

        lineups = start.parse_prematch_lineups_html(
            html,
            {"match": {"team1": {"name": "Alpha", "id": "10"}, "team2": {"name": "Beta", "id": "20"}}},
        )
        start.mark_prematch_standins(lineups, parsed)
        self.assertEqual(lineups["team1"]["players"][0]["hltv_player_id"], "101")
        self.assertTrue(lineups["team2"]["players"][0]["is_standin"])


if __name__ == "__main__":
    unittest.main()
