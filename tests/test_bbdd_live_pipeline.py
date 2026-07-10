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
    spec = importlib.util.spec_from_file_location("daily_start", ROOT / "DAILY_SNAPSHOTS" / "start.py")
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
        self.assertIn("hltv_match_id", columns)
        self.assertIn("prematch_captured_at_utc", columns)
        self.assertIn("result_filled_at_utc", columns)

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
            data_root = Path(tmp) / "DAILY_SNAPSHOTS"
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
        self.assertAlmostEqual(rows[0]["opening_odds_t1"], (0.538 + 0.513) / 2)
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
