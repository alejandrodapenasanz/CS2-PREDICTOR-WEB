from __future__ import annotations

import importlib.util
import sys
import unittest
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DAILY_SNAPSHOTS import start


PLAYER_SCRAPER = ROOT / "SCRAPPER" / "hltv-scraper-api" / "scripts" / "collect_player_compare_stats.py"
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
        self.assertEqual(player["stats"]["KPR"], "0.68")
        self.assertNotIn("ADR", player["stats"])
        self.assertFalse(player_compare_stats.player_stats_complete(player["stats"]))

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


if __name__ == "__main__":
    unittest.main()
