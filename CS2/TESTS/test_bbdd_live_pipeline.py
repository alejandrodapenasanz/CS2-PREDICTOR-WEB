from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MODEL"))

from BBDD import build_db, ingest
from cs2model import dataio


def load_daily_start_module():
    spec = importlib.util.spec_from_file_location("daily_start", ROOT / "PIPELINE" / "start.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LiveDatabasePipelineTests(unittest.TestCase):
    def make_db(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        tmp = tempfile.TemporaryDirectory()
        db_path = Path(tmp.name) / "cs2.db"
        conn = build_db.connect_live_db(db_path)
        conn.close()
        return tmp, db_path

    def seed_completed_match(self, db_path: Path) -> None:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("INSERT INTO teams(team_id, name, hltv_id) VALUES (1,'Alpha',10),(2,'Beta',20)")
        conn.execute("INSERT INTO events(event_id, name) VALUES (1,'DB Test Cup')")
        conn.execute(
            """
            INSERT INTO matches(
                match_id, hltv_match_id, event_id, datetime_utc, team1_id, team2_id,
                best_of, stage, environment, status, data_tier, winner_team_id,
                score_t1, score_t2, has_prematch_odds
            )
            VALUES (1,'2390001',1,'2026-01-02T18:00:00Z',1,2,3,'semi','lan',
                    'completed','completed',1,2,1,1)
            """
        )
        conn.executemany(
            """
            INSERT INTO odds(match_id, bookmaker, captured_at_utc, market_type,
                             odds_t1, odds_t2, prob_t1, prob_t2)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            [
                (1, "BookA", "2026-01-02T09:00:00Z", "opening", 1.80, 2.10, 0.538, 0.462),
                (1, "BookB", "2026-01-02T09:00:00Z", "opening", 1.90, 2.00, 0.513, 0.487),
                (1, "BookA", "2026-01-02T17:55:00Z", "closing", 1.20, 4.50, 0.789, 0.211),
            ],
        )
        conn.commit()
        conn.close()

    def test_schema_is_created_and_migration_is_idempotent(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        conn = build_db.connect_live_db(db_path)
        build_db.ensure_live_schema(conn)
        columns = build_db.table_columns(conn, "matches")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()

        self.assertIn("fetch_state", tables)
        self.assertIn("ingest_runs", tables)
        self.assertIn("prematch_lineup_snapshots", tables)
        self.assertIn("match_analytics_snapshots", tables)
        self.assertIn("match_analytics_map_stats", tables)
        self.assertIn("hltv_match_id", columns)
        self.assertIn("prematch_captured_at_utc", columns)
        self.assertIn("result_filled_at_utc", columns)

    def test_estimate_columns_migrate_as_null_without_rewriting_legacy_rows(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        self.seed_completed_match(db_path)
        estimate_columns = (
            "ensemble_disagreement",
            "estimate_band_half_width",
            "estimate_confidence_level",
            "estimate_history_coverage",
        )
        conn = sqlite3.connect(db_path)
        for table in ("predictions", "prediction_ledger"):
            for column in estimate_columns:
                conn.execute(f'ALTER TABLE "{table}" DROP COLUMN "{column}"')
        conn.execute(
            """
            INSERT INTO predictions(
                match_id, hltv_match_id, model_version, predicted_at_utc, prob_team1,
                decision_confidence, reliability_score, prediction_json
            ) VALUES (
                1, '2390001', 'legacy@v1', '2026-01-02T10:00:00Z', 0.7,
                0.7, 0.8, '{}'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO prediction_ledger(
                match_id, hltv_match_id, team1_id, team2_id, kickoff_utc,
                predicted_at_utc, model_version, prob_team1, prediction_json,
                ledger_status, created_at_utc, updated_at_utc
            ) VALUES (
                1, '2390001', 1, 2, '2026-01-02T18:00:00Z',
                '2026-01-02T10:00:00Z', 'legacy@v1', 0.7, '{}',
                'evaluated', '2026-01-02T10:00:00Z', '2026-01-02T20:00:00Z'
            )
            """
        )
        legacy_columns = {
            table: [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]
            for table in ("predictions", "prediction_ledger")
        }
        before = {
            table: conn.execute(
                f'SELECT {", ".join(f"[{column}]" for column in columns)} FROM "{table}"'
            ).fetchone()
            for table, columns in legacy_columns.items()
        }

        build_db.ensure_live_schema(conn)
        build_db.ensure_live_schema(conn)

        after = {
            table: conn.execute(
                f'SELECT {", ".join(f"[{column}]" for column in columns)} FROM "{table}"'
            ).fetchone()
            for table, columns in legacy_columns.items()
        }
        for table in ("predictions", "prediction_ledger"):
            self.assertEqual(after[table], before[table])
            self.assertTrue(set(estimate_columns) <= build_db.table_columns(conn, table))
            migrated = conn.execute(
                f'SELECT {", ".join(estimate_columns)} FROM "{table}"'
            ).fetchone()
            self.assertEqual(migrated, (None, None, None, None))
        conn.close()

    def test_prematch_lineups_and_analytics_are_persisted_structurally(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        self.seed_completed_match(db_path)
        run_dir = Path(tmp.name) / "runs" / "run_analytics"
        (run_dir / "match_snapshots").mkdir(parents=True)
        (run_dir / "analytics").mkdir()
        snapshot = {
            "id": "2390001",
            "captured_at": "2026-01-02T09:00:00Z",
            "source_file": "runs/run_analytics/match_snapshots/2390001.json",
            "prematch_lineups": {
                "team1": {"team_name": "Alpha", "hltv_team_id": "10", "players": [{"hltv_player_id": "101", "nickname": "A", "is_standin": False}]},
                "team2": {"team_name": "Beta", "hltv_team_id": "20", "players": [{"hltv_player_id": "202", "nickname": "B", "is_standin": True}]},
            },
        }
        analytics = {
            "available": True,
            "match_id": "2390001",
            "captured_at": "2026-01-02T09:00:00Z",
            "source_file": "runs/run_analytics/analytics/2390001.json",
            "event_metadata": {"name": "DB Test Cup", "prize_pool": 50000, "teams_competing": 16},
            "core_lineup": {"Beta": {"matches_lt": 5}},
            "series_stats": {
                "team1": {"team": "Alpha", "matches": 20, "maps": 45, "overtime_pct": 4.0},
                "team2": {"team": "Beta", "matches": 18, "maps": 40, "overtime_pct": 7.0},
            },
            "map_stats": [{"team": "Alpha", "map": "Mirage", "first_pick_pct": 50, "first_ban_pct": 10, "win_pct": 60, "played": 10}],
            "map_handicap": [{"team": "Beta", "map": "Mirage", "avg_rounds_lost_in_wins": 7.0, "avg_rounds_won_in_losses": 8.5}],
        }
        (run_dir / "match_snapshots" / "2390001.json").write_text(json.dumps(snapshot), encoding="utf-8")
        (run_dir / "analytics" / "2390001.json").write_text(json.dumps(analytics), encoding="utf-8")

        conn = build_db.connect_live_db(db_path)
        lineups = build_db.insert_prematch_lineup_snapshots(
            conn.cursor(), run_dir, {"2390001": 1}, {"alpha": 1, "beta": 2}, {"10": 1, "20": 2}
        )
        analytics_counts = build_db.insert_match_analytics_snapshots(
            conn.cursor(), run_dir, {"2390001": 1}, {"alpha": 1, "beta": 2}, {"10": 1, "20": 2}
        )
        conn.commit()
        stored_lineups = conn.execute("SELECT COUNT(*) FROM prematch_lineup_snapshots").fetchone()[0]
        stored_analytics = conn.execute("SELECT COUNT(*) FROM match_analytics_snapshots").fetchone()[0]
        stored_maps = conn.execute("SELECT COUNT(*) FROM match_analytics_map_stats").fetchone()[0]
        event = conn.execute("SELECT prize_pool, teams_competing FROM events WHERE event_id=1").fetchone()
        conn.close()

        self.assertEqual(lineups["prematch_lineup_rows"], 2)
        self.assertEqual(analytics_counts["analytics_snapshot_rows"], 1)
        self.assertEqual((stored_lineups, stored_analytics, stored_maps), (2, 1, 1))
        self.assertEqual(tuple(event), (50000, 16))

    def test_training_loader_uses_only_complete_prestart_lineup_and_analytics(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        self.seed_completed_match(db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        for team_id, prefix, ratings in ((1, "a", (1.20, 1.10, 1.00, 0.95, 0.90)), (2, "b", (1.05, 1.00, 0.95, 0.90, 0.85))):
            for index, rating in enumerate(ratings, start=1):
                player_id = team_id * 10 + index
                conn.execute(
                    "INSERT INTO players(player_id, nick, hltv_id) VALUES (?,?,?)",
                    (player_id, f"{prefix}{index}", str(player_id)),
                )
                payload = {
                    "hltv_player_id": str(player_id),
                    "rating": rating,
                    "kpr": rating / 2,
                    "kast": rating * 70,
                    "adr": rating * 75,
                    "multi_kill_rating": rating,
                    "round_swing": rating - 1,
                }
                conn.execute(
                    """
                    INSERT INTO prematch_lineup_snapshots(
                        match_id, team_id, player_id, captured_at_utc, run_id,
                        is_standin, source_file, payload_json
                    ) VALUES (1,?,?,?,?,?,?,?)
                    """,
                    (team_id, player_id, "2026-01-02T09:00:00Z", "pre", 0, f"pre-{player_id}.json", json.dumps(payload)),
                )
                # A deliberately stronger post-start row must be ignored.
                payload["rating"] = 9.99
                conn.execute(
                    """
                    INSERT INTO prematch_lineup_snapshots(
                        match_id, team_id, player_id, captured_at_utc, run_id,
                        is_standin, source_file, payload_json
                    ) VALUES (1,?,?,?,?,?,?,?)
                    """,
                    (team_id, player_id, "2026-01-02T19:00:00Z", "post", 0, f"post-{player_id}.json", json.dumps(payload)),
                )
        pre_analytics = {
            "available": True,
            "captured_at": "2026-01-02T09:00:00Z",
            "event_metadata": {"prize_pool": 50000, "teams_competing": 16},
        }
        post_analytics = {
            "available": True,
            "captured_at": "2026-01-02T19:00:00Z",
            "event_metadata": {"prize_pool": 9999999, "teams_competing": 99},
        }
        for captured_at, source_file, payload in (
            ("2026-01-02T09:00:00Z", "pre-analytics.json", pre_analytics),
            ("2026-01-02T19:00:00Z", "post-analytics.json", post_analytics),
        ):
            conn.execute(
                """
                INSERT INTO match_analytics_snapshots(
                    match_id, captured_at_utc, run_id, source_file, payload_json
                ) VALUES (1,?,?,?,?)
                """,
                (captured_at, "test", source_file, json.dumps(payload)),
            )
        conn.commit()
        conn.close()

        rows = dataio.load_training_rows_from_db(db_path)

        self.assertEqual(len(rows), 1)
        lineup = rows[0]["prematch_lineups"]
        self.assertEqual(len(lineup["team1"]["players"]), 5)
        self.assertAlmostEqual(lineup["team1"]["players"][0]["rating"], 1.20)
        self.assertEqual(rows[0]["event_metadata"], {"prize_pool": 50000, "teams_competing": 16})

    def test_fetch_state_ttl_marks_entities_fresh_until_next_eligible(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        conn = build_db.connect_live_db(db_path)
        ingest.upsert_fetch_state(
            conn,
            "team_profile",
            "10",
            "ok",
            fetched_at="2026-07-07T08:00:00Z",
        )

        self.assertTrue(ingest.entity_is_fresh(conn, "team_profile", "10", "2026-07-08T08:00:00Z"))
        self.assertFalse(ingest.entity_is_fresh(conn, "team_profile", "10", "2026-07-20T08:00:00Z"))
        conn.close()

    def test_fetch_state_upsert_is_idempotent_for_same_evidence(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        conn = build_db.connect_live_db(db_path)
        for _ in range(2):
            ingest.upsert_fetch_state(
                conn,
                "player_stats",
                "101",
                "ok",
                fetched_at="2026-07-16T08:00:00Z",
            )
        fetch_count = conn.execute(
            "SELECT fetch_count FROM fetch_state WHERE entity_type='player_stats' AND entity_key='101'"
        ).fetchone()[0]
        conn.close()

        self.assertEqual(fetch_count, 1)

    def test_blocked_fetch_state_is_skipped_until_next_retry(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        conn = build_db.connect_live_db(db_path)
        ingest.upsert_fetch_state(
            conn,
            "match_assets",
            "2390001",
            "blocked",
            fetched_at="2026-07-07T08:00:00Z",
            note="cloudflare",
        )
        conn.commit()
        conn.close()

        daily_start = load_daily_start_module()
        old_db = daily_start.BBDD_DB
        daily_start.BBDD_DB = db_path
        try:
            self.assertTrue(daily_start.db_entity_is_fresh("match_assets", "2390001", "2026-07-07T08:30:00Z"))
            self.assertFalse(daily_start.db_entity_is_fresh("match_assets", "2390001", "2026-07-07T10:00:00Z"))
        finally:
            daily_start.BBDD_DB = old_db

    def test_player_fetch_state_aggregates_windows_before_setting_status(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        run_dir = Path(tmp.name) / "runs" / "player_windows"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"started_at": "2026-07-16T08:00:00Z"}),
            encoding="utf-8",
        )
        complete = {
            "id": "101",
            "maps": 20,
            "time_filter": "past3months",
            "stats": {
                "Rating 3.0": "1.10",
                "KPR": "0.70",
                "DPR": "0.60",
                "APR": "0.20",
                "KAST": "73.0",
                "Impact": "1.12",
                "ADR": "80.0",
            },
        }
        partial = {
            "id": "101",
            "maps": 50,
            "time_filter": "past6months",
            "stats": {"KPR": "0.68"},
        }
        (run_dir / "player_compare_stats_2026.json").write_text(
            json.dumps({"results": [{"players": [complete]}, {"players": [partial]}]}),
            encoding="utf-8",
        )
        conn = build_db.connect_live_db(db_path)

        ingest.update_fetch_state_from_run(conn, run_dir)
        conn.commit()
        status = conn.execute(
            "SELECT last_status FROM fetch_state WHERE entity_type='player_stats' AND entity_key='101'"
        ).fetchone()[0]
        conn.close()

        self.assertEqual(status, "ok")

    def test_cached_player_rows_do_not_advance_fetch_state(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        run_dir = Path(tmp.name) / "runs" / "cached_players"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"started_at": "2026-07-16T08:00:00Z"}),
            encoding="utf-8",
        )
        (run_dir / "player_compare_stats_2026.json").write_text(
            json.dumps({
                "results": [{
                    "players": [{
                        "id": "101",
                        "fetch_origin": "cache",
                        "captured_at": "2026-07-14T08:00:00Z",
                        "maps": 20,
                        "stats": {},
                    }]
                }]
            }),
            encoding="utf-8",
        )
        conn = build_db.connect_live_db(db_path)

        ingest.update_fetch_state_from_run(conn, run_dir)
        count = conn.execute(
            "SELECT COUNT(*) FROM fetch_state WHERE entity_type='player_stats'"
        ).fetchone()[0]
        conn.close()

        self.assertEqual(count, 0)

    def test_player_cache_does_not_mix_windows_from_older_captures(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        conn = build_db.connect_live_db(db_path)
        conn.execute(
            """
            INSERT INTO player_stat_snapshots(
                hltv_player_id, run_id, captured_at_utc, time_filter, maps,
                rating, kpr, dpr, apr, kast, impact, adr, source_file, payload_json
            ) VALUES ('101','old','2026-07-10T08:00:00Z','past6months',30,
                      1.10,0.70,0.60,0.20,73.0,1.12,80.0,'old.json','{}')
            """
        )
        conn.execute(
            """
            INSERT INTO player_stat_snapshots(
                hltv_player_id, run_id, captured_at_utc, time_filter, maps,
                rating, kpr, source_file, payload_json
            ) VALUES ('101','new','2026-07-16T08:00:00Z','past3months',0,
                      1.10,0.70,'new.json','{}')
            """
        )
        conn.commit()
        conn.close()
        daily_start = load_daily_start_module()
        old_db = daily_start.BBDD_DB
        daily_start.BBDD_DB = db_path
        try:
            rows = daily_start.db_latest_player_snapshots({"101"})
        finally:
            daily_start.BBDD_DB = old_db

        self.assertEqual(len(rows["101"]), 1)
        self.assertEqual(rows["101"][0]["time_filter"], "past3months")
        self.assertEqual(rows["101"][0]["maps"], 0)
        self.assertEqual(rows["101"][0]["stats"]["KPR"], 0.70)
        selected = daily_start.choose_cached_player_snapshot(rows["101"])
        self.assertIsNotNone(selected)
        self.assertEqual(selected["stats"], {})

    def test_player_state_reconcile_promotes_tracked_partial_but_ignores_untracked_history(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        conn = build_db.connect_live_db(db_path)
        captured_at = "2026-07-13T05:33:17Z"
        complete_stats = (1.10, 0.70, 0.60, 0.20, 73.0, 1.12, 80.0)
        conn.execute(
            """
            INSERT INTO player_stat_snapshots(
                hltv_player_id, run_id, captured_at_utc, time_filter, maps,
                rating, kpr, dpr, apr, kast, impact, adr, source_file, payload_json
            ) VALUES ('101','run',?,'past3months',20,?,?,?,?,?,?,?,'stats.json','{}')
            """,
            (captured_at, *complete_stats),
        )
        conn.execute(
            """
            INSERT INTO player_stat_snapshots(
                hltv_player_id, run_id, captured_at_utc, time_filter, maps,
                kpr, source_file, payload_json
            ) VALUES ('101','run',?,'past6months',50,0.68,'stats.json','{}')
            """,
            (captured_at,),
        )
        conn.execute(
            """
            INSERT INTO player_stat_snapshots(
                hltv_player_id, run_id, captured_at_utc, time_filter, maps,
                kpr, source_file, payload_json
            ) VALUES ('999','old',?,'past3months',5,0.60,'old.json','{}')
            """,
            (captured_at,),
        )
        ingest.upsert_fetch_state(
            conn,
            "player_stats",
            "101",
            "partial",
            fetched_at=captured_at,
        )

        touched = ingest.reconcile_known_missing_player_stats(conn)
        tracked = conn.execute(
            "SELECT last_status FROM fetch_state WHERE entity_type='player_stats' AND entity_key='101'"
        ).fetchone()[0]
        untracked = conn.execute(
            "SELECT COUNT(*) FROM fetch_state WHERE entity_type='player_stats' AND entity_key='999'"
        ).fetchone()[0]
        conn.close()

        self.assertEqual(touched, 1)
        self.assertEqual(tracked, "ok")
        self.assertEqual(untracked, 0)

    def test_recent_results_skips_hltv_when_db_has_no_pending_targets(self) -> None:
        daily_start = load_daily_start_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            calls: list[str] = []

            def fail_fetch(link: str, *args, **kwargs):
                calls.append(link)
                raise AssertionError("fetch_html should not be called without DB pending targets")

            old_fetch = daily_start.fetch_html
            daily_start.fetch_html = fail_fetch
            try:
                result = daily_start.scrape_recent_results(run_dir, target_ids=set())
            finally:
                daily_start.fetch_html = old_fetch

            self.assertEqual(result, {})
            self.assertEqual(calls, [])

    def test_recent_results_stops_before_older_offsets_when_targets_found(self) -> None:
        daily_start = load_daily_start_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            calls: list[str] = []

            def fake_fetch(link: str, *args, **kwargs):
                calls.append(link)
                return "<html></html>"

            old_fetch = daily_start.fetch_html
            old_parser = daily_start.parse_results_fallback
            old_pf = daily_start.PF
            old_save = daily_start.save_raw_html
            daily_start.fetch_html = fake_fetch
            daily_start.PF = None
            daily_start.save_raw_html = lambda *args, **kwargs: {"source_file": "test.html.gz"}
            daily_start.parse_results_fallback = lambda html: [{"id": "2390001", "link": "/matches/2390001/a-vs-b"}]
            try:
                result = daily_start.scrape_recent_results(run_dir, pages=3, target_ids={"2390001"})
            finally:
                daily_start.fetch_html = old_fetch
                daily_start.parse_results_fallback = old_parser
                daily_start.PF = old_pf
                daily_start.save_raw_html = old_save

            self.assertIn("2390001", result)
            self.assertEqual(calls, ["/results?offset=0"])

    def test_photographed_upcoming_is_materialized_without_losing_original_timestamp(self) -> None:
        daily_start = load_daily_start_module()
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "PIPELINE"
            old_snapshot = data_root / "runs" / "old" / "match_snapshots" / "2390003.json"
            old_snapshot.parent.mkdir(parents=True)
            original = {
                "id": "2390003",
                "captured_at": "2026-07-09T08:00:00Z",
                "date": "2026-07-10",
                "hour": "18:00",
                "format": "bo3",
                "event": "Test Cup",
                "detail": {"match": {"team1": {"name": "Alpha"}, "team2": {"name": "Beta"}}},
                "odds": {"available": True, "bookmaker_count": 2},
                "data_quality": {"real_pre_match_snapshot": True},
                "status": "pending",
            }
            old_snapshot.write_text(json.dumps(original), encoding="utf-8")
            master = {
                "2390003": {
                    "id": "2390003",
                    "status": "scheduled",
                    "latest_snapshot_file": "runs\\old\\match_snapshots\\2390003.json",
                }
            }
            upcoming = {
                "id": "2390003",
                "link": "/matches/2390003/alpha-vs-beta",
                "date": "2026-07-11",
                "hour": "19:30",
                "meta": "bo3",
                "event": "Test Cup",
            }
            run_dir = data_root / "runs" / "new"
            old_data_root = daily_start.DATA_ROOT
            daily_start.DATA_ROOT = data_root
            try:
                snapshot = daily_start.materialize_persistent_match_snapshot(master, upcoming, run_dir)
            finally:
                daily_start.DATA_ROOT = old_data_root

            self.assertIsNotNone(snapshot)
            self.assertEqual(snapshot["captured_at"], "2026-07-09T08:00:00Z")
            self.assertEqual(snapshot["date"], "2026-07-11")
            self.assertEqual(snapshot["hour"], "19:30")
            self.assertTrue(snapshot["data_quality"]["materialized_from_persistent_snapshot"])
            self.assertTrue((run_dir / "match_snapshots" / "2390003.json").exists())

    def test_result_update_does_not_create_prematch_flags(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        conn = build_db.connect_live_db(db_path)
        conn.execute("INSERT INTO teams(team_id, name, hltv_id) VALUES (1,'Alpha',10),(2,'Beta',20)")
        conn.execute("INSERT INTO events(event_id, name) VALUES (1,'DB Test Cup')")
        conn.execute(
            """
            INSERT INTO matches(
                match_id, hltv_match_id, event_id, datetime_utc, team1_id, team2_id,
                best_of, status, data_tier, prematch_captured_at_utc,
                has_prematch_odds, has_analytics, has_context
            )
            VALUES (1,'2390002',1,'2026-01-02T18:00:00Z',1,2,3,
                    'scheduled','prematch_captured','2026-01-02T08:00:00Z',0,0,0)
            """
        )
        conn.commit()

        ingest.upsert_match(
            conn,
            {
                "id": "2390002",
                "date": "2026-01-02",
                "hour": "18:00",
                "event": "DB Test Cup",
                "format": "bo3",
                "status": "completed",
                "team1": {"name": "Alpha", "id": "10"},
                "team2": {"name": "Beta", "id": "20"},
                "score": {"team1": "2", "team2": "0"},
                "opening_odds": {"available": True},
                "analytics": {"available": True},
                "match_context": {"environment": "lan", "stage": "semi"},
            },
            captured_at="2026-01-02T19:00:00Z",
        )
        row = conn.execute(
            """
            SELECT status, data_tier, prematch_captured_at_utc, result_filled_at_utc,
                   has_prematch_odds, has_analytics, has_context, score_t1, score_t2
            FROM matches WHERE hltv_match_id='2390002'
            """
        ).fetchone()
        conn.close()

        self.assertEqual(row[0], "completed")
        self.assertEqual(row[1], "completed")
        self.assertEqual(row[2], "2026-01-02T08:00:00Z")
        self.assertIsNotNone(row[3])
        self.assertEqual((row[4], row[5], row[6]), (0, 0, 0))
        self.assertEqual((row[7], row[8]), (2, 0))

    def test_scheduled_match_updates_participant_without_rewriting_first_snapshot(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        conn = build_db.connect_live_db(db_path)
        conn.execute("INSERT INTO teams(team_id, name, hltv_id) VALUES (1,'Alpha',10),(2,'Beta',20)")
        conn.execute("INSERT INTO events(event_id, name) VALUES (1,'DB Test Cup')")
        conn.execute(
            """
            INSERT INTO matches(match_id, hltv_match_id, event_id, datetime_utc, team1_id, team2_id,
                                best_of, status, data_tier, prematch_captured_at_utc)
            VALUES (1,'2390004',1,'2026-01-02T18:00:00Z',1,2,3,'scheduled','prematch_captured','2026-01-01T09:00:00Z')
            """
        )
        conn.commit()
        ingest.upsert_match(
            conn,
            {
                "id": "2390004", "date": "2026-01-02", "hour": "19:00", "event": "DB Test Cup", "format": "bo3",
                "status": "scheduled", "team1": {"name": "Alpha", "id": "10"}, "team2": {"name": "ex-Beta", "id": "30"},
                "match_context": {"environment": "online", "stage": "group"},
            },
            captured_at="2026-01-02T08:00:00Z",
        )
        row = conn.execute(
            """
            SELECT t.name, t.hltv_id, m.datetime_utc, m.prematch_captured_at_utc
            FROM matches m JOIN teams t ON t.team_id=m.team2_id WHERE m.hltv_match_id='2390004'
            """
        ).fetchone()
        conn.close()
        self.assertEqual(tuple(row), ("ex-Beta", 30, "2026-01-02T19:00:00Z", "2026-01-01T09:00:00Z"))

    def test_ingest_run_records_request_and_freshness_counters(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        run_dir = Path(tmp.name) / "runs" / "run_freshness"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(json.dumps({"started_at": "2026-07-07T08:00:00Z"}), encoding="utf-8")
        (run_dir / "fetch_diagnostics.json").write_text(
            json.dumps({"http_attempts": 10, "freshness_skipped": 7}),
            encoding="utf-8",
        )
        master = Path(tmp.name) / "matches.json"
        master.write_text("{}", encoding="utf-8")

        ingest.ingest_run(run_dir, db_path, master, backup_dir=None, mirror_backup_dir=None)

        conn = build_db.connect_live_db(db_path)
        row = conn.execute(
            """
            SELECT requests_made, requests_skipped_by_freshness
            FROM ingest_runs
            WHERE run_id='run_freshness'
            ORDER BY ingest_id DESC LIMIT 1
            """
        ).fetchone()
        conn.close()

        self.assertEqual(tuple(row), (10, 7))

    def test_training_loader_reads_db_and_uses_opening_odds_only(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        self.seed_completed_match(db_path)

        rows = dataio.load_training_rows_from_db(db_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "2390001")
        book_a = (1 / 1.80) / ((1 / 1.80) + (1 / 2.10))
        book_b = (1 / 1.90) / ((1 / 1.90) + (1 / 2.00))
        self.assertAlmostEqual(rows[0]["opening_odds_t1"], (book_a + book_b) / 2)
        self.assertEqual(rows[0]["opening_odds_captured_at"], "2026-01-02T09:00:00Z")

    def test_training_loader_deduplicates_historical_seed_and_hltv_copy(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("INSERT INTO teams(team_id, name, hltv_id) VALUES (1,'Alpha',10),(2,'Beta',20)")
        conn.execute("INSERT INTO events(event_id, name) VALUES (1,'Duplicate Cup')")
        base = (1, 1, "2026-01-02T18:00:00Z", 1, 2, 3, "completed", "historical_seed", 1, 2, 1)
        conn.execute(
            """
            INSERT INTO matches(match_id,event_id,datetime_utc,team1_id,team2_id,best_of,
                                status,data_tier,winner_team_id,score_t1,score_t2)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            base,
        )
        conn.execute(
            """
            INSERT INTO matches(match_id,hltv_match_id,event_id,datetime_utc,team1_id,team2_id,best_of,
                                status,data_tier,winner_team_id,score_t1,score_t2)
            VALUES (2,'2390004',1,'2026-01-02T18:00:00Z',1,2,3,
                    'completed','completed',1,2,1)
            """
        )
        conn.commit()
        conn.close()

        rows = dataio.load_training_rows_from_db(db_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "2390004")

    def test_feature_mart_temporal_audit_detects_leakage_rows(self) -> None:
        tmp, db_path = self.make_db()
        self.addCleanup(tmp.cleanup)
        self.seed_completed_match(db_path)
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO match_features(match_id, team_id, data_up_to_utc) VALUES (1,1,'2026-01-02T18:00:00Z')"
        )
        conn.execute(
            "INSERT INTO match_features(match_id, team_id, data_up_to_utc) VALUES (1,2,'2026-01-03T00:00:00Z')"
        )
        leaks = conn.execute(
            """
            SELECT COUNT(*)
            FROM match_features mf
            JOIN matches m ON m.match_id = mf.match_id
            WHERE mf.data_up_to_utc > m.datetime_utc
            """
        ).fetchone()[0]
        conn.close()

        self.assertEqual(leaks, 1)


if __name__ == "__main__":
    unittest.main()
