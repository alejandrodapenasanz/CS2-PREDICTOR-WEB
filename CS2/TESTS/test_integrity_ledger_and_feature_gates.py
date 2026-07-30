from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "MODEL"
for path in (ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from BBDD import build_db, deduplicate_matches, ingest, repair_integrity
from MODEL import train
from PIPELINE import enrich_predictions
from cs2model.identity import choose_match_team, is_provisional_team_name


class ParticipantIntegrityTests(unittest.TestCase):
    def test_completed_match_prefers_resolved_detail_participant(self) -> None:
        record = {
            "status": "completed",
            "team1": {"name": "Winner of A", "id": ""},
            "detail": {
                "match": {
                    "team1": {"name": "Resolved", "id": "42"},
                }
            },
        }
        self.assertEqual(choose_match_team(record, "team1")["id"], "42")
        self.assertTrue(is_provisional_team_name("Winner of A"))
        self.assertFalse(is_provisional_team_name("WINNERS"))

    def test_ingest_rejects_unconfirmed_participant(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            conn = build_db.connect_live_db(Path(temp) / "confirmed.db")
            record = {
                "id": "101",
                "status": "scheduled",
                "date": "2026-01-02",
                "hour": "12:00",
                "team1": {"name": "Alpha", "id": "1"},
                "team2": {"name": "TBD", "id": ""},
                "event": "Event",
            }
            self.assertEqual(ingest.upsert_match(conn, record, "2026-01-01T10:00:00Z"), 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 0)
            conn.close()

    def test_predictor_rejects_unconfirmed_participant(self) -> None:
        snapshot = {
            "status": "scheduled",
            "date": "2026-01-02",
            "hour": "12:00",
            "upcoming_row": {
                "team1": {"name": "Alpha", "id": "1"},
                "team2": {"name": "Winner of semifinal", "id": ""},
            },
        }
        publishable, reason = enrich_predictions.is_snapshot_publishable(
            snapshot,
            enrich_predictions.parse_match_datetime("2026-01-01", "10:00"),
        )
        self.assertFalse(publishable)
        self.assertEqual(reason, "participants_unconfirmed")

    def test_repair_purges_unconfirmed_normalized_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            conn = build_db.connect_live_db(Path(temp) / "repair.db")
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("INSERT INTO events(name) VALUES ('Event')")
                event_id = conn.execute("SELECT event_id FROM events").fetchone()[0]
                conn.execute("INSERT INTO teams(name,hltv_id) VALUES ('Alpha',1)")
                conn.execute("INSERT INTO teams(name,hltv_id) VALUES ('TBD',NULL)")
                team_ids = [
                    row[0]
                    for row in conn.execute("SELECT team_id FROM teams ORDER BY team_id")
                ]
                conn.execute(
                    """
                    INSERT INTO matches(
                        hltv_match_id,event_id,datetime_utc,datetime_precision,
                        team1_id,team2_id,best_of,status,data_tier
                    ) VALUES ('102',?,'2026-01-02T12:00:00Z','exact',?,?,3,
                              'scheduled','prematch_captured')
                    """,
                    (event_id, *team_ids),
                )
                conn.execute(
                    """
                    INSERT INTO raw_snapshots(
                        kind,hltv_match_id,run_id,captured_at_utc,source_file,payload_json
                    ) VALUES ('match_snapshot','102','run','2026-01-01T10:00:00Z',
                              'raw.json','{}')
                    """
                )
                self.assertEqual(
                    repair_integrity.purge_unconfirmed_live_matches(conn), 1
                )
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 0
                )
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM raw_snapshots").fetchone()[0], 1
                )
            finally:
                conn.close()

    def test_historical_id_backfill_preserves_distinct_double_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            conn = build_db.connect_live_db(Path(temp) / "history.db")
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("INSERT INTO events(name) VALUES ('Cup')")
                conn.execute("INSERT INTO teams(name) VALUES ('Alpha'),('Beta')")
                event_id = conn.execute("SELECT event_id FROM events").fetchone()[0]
                team_ids = [
                    row[0]
                    for row in conn.execute("SELECT team_id FROM teams ORDER BY team_id")
                ]
                for match_id in (1, 2):
                    conn.execute(
                        """
                        INSERT INTO matches(
                            match_id,event_id,datetime_utc,datetime_precision,
                            team1_id,team2_id,best_of,status,data_tier,
                            winner_team_id,score_t1,score_t2
                        ) VALUES (?,?,'2026-01-02','date_only',?,?,3,
                                  'completed','historical_seed',?,2,0)
                        """,
                        (match_id, event_id, *team_ids, team_ids[0]),
                    )
                rows = [
                    {
                        "id": "2390001",
                        "date": "2026-01-02",
                        "event": "Cup",
                        "format": "bo3",
                        "team1": "Alpha",
                        "team2": "Beta",
                        "score1": 2,
                        "score2": 0,
                    },
                    {
                        "id": "2390002",
                        "date": "2026-01-02",
                        "event": "Cup",
                        "format": "bo3",
                        "team1": "Alpha",
                        "team2": "Beta",
                        "score1": 2,
                        "score2": 0,
                    },
                ]
                result = repair_integrity._backfill_match_ids_from_rows(conn, rows)
                ids = [
                    row[0]
                    for row in conn.execute(
                        "SELECT hltv_match_id FROM matches ORDER BY match_id"
                    )
                ]
                self.assertEqual(result, {"updated": 2, "conflicts": 0})
                self.assertEqual(ids, ["2390001", "2390002"])
            finally:
                conn.close()

    def test_prediction_reference_repair_replaces_placeholder_favorite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            conn = build_db.connect_live_db(Path(temp) / "prediction.db")
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("INSERT INTO events(name) VALUES ('Cup')")
                event_id = conn.execute("SELECT event_id FROM events").fetchone()[0]
                conn.execute("INSERT INTO teams(name,hltv_id) VALUES ('Alpha',1)")
                conn.execute("INSERT INTO teams(name,hltv_id) VALUES ('Beta',2)")
                conn.execute("INSERT INTO teams(name) VALUES ('Alpha/Beta loser')")
                teams = list(
                    conn.execute("SELECT team_id,name FROM teams ORDER BY team_id")
                )
                alpha_id, beta_id, placeholder_id = [row[0] for row in teams]
                conn.execute(
                    """
                    INSERT INTO matches(
                        hltv_match_id,event_id,datetime_utc,datetime_precision,
                        team1_id,team2_id,best_of,status,data_tier
                    ) VALUES ('2390003',?,'2026-01-02T12:00:00Z','exact',
                              ?,?,3,'scheduled','prematch_captured')
                    """,
                    (event_id, alpha_id, beta_id),
                )
                conn.execute(
                    """
                    INSERT INTO predictions(
                        hltv_match_id,model_version,predicted_at_utc,prob_team1,
                        favorite_team_id,favorite_name,decision_favorite_side
                    ) VALUES ('2390003','old','2026-01-02T10:00:00Z',0.2,
                              ?,'Alpha/Beta loser','team2')
                    """,
                    (placeholder_id,),
                )
                result = repair_integrity.repair_prediction_references(conn)
                row = conn.execute(
                    "SELECT match_id,favorite_team_id,favorite_name FROM predictions"
                ).fetchone()
                self.assertEqual(result["prediction_match_links_repaired"], 1)
                self.assertEqual((row[1], row[2]), (beta_id, "Beta"))
                self.assertEqual(
                    repair_integrity.delete_unreferenced_provisional_teams(conn), 1
                )
            finally:
                conn.close()

    def test_wrong_bo3_map_score_is_repaired_then_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            conn = build_db.connect_live_db(Path(temp) / "duplicate.db")
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("INSERT INTO events(name) VALUES ('Cup')")
                conn.execute("INSERT INTO teams(name) VALUES ('Alpha'),('Beta')")
                event_id = conn.execute("SELECT event_id FROM events").fetchone()[0]
                team_ids = [
                    row[0]
                    for row in conn.execute("SELECT team_id FROM teams ORDER BY team_id")
                ]
                conn.execute(
                    """
                    INSERT INTO matches(
                        match_id,hltv_match_id,event_id,datetime_utc,
                        datetime_precision,team1_id,team2_id,best_of,status,
                        data_tier,winner_team_id,score_t1,score_t2
                    ) VALUES
                        (1,NULL,?,'2026-01-02','date_only',?,?,1,'completed',
                         'historical_seed',?,13,11),
                        (2,'2390004',?,'2026-01-02T12:00:00Z','exact',?,?,3,
                         'completed','completed',?,13,11)
                    """,
                    (
                        event_id,
                        *team_ids,
                        team_ids[0],
                        event_id,
                        *team_ids,
                        team_ids[0],
                    ),
                )
                self.assertEqual(
                    repair_integrity.repair_impossible_series_formats(conn), 1
                )
                pairs = deduplicate_matches.duplicate_pairs(conn)
                self.assertEqual(pairs, [(2, 1)])
                conn.commit()
                deduplicate_matches.consolidate(conn, pairs)
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 1
                )
                self.assertEqual(
                    conn.execute("SELECT best_of FROM matches").fetchone()[0], 1
                )
            finally:
                conn.close()


    def test_event_metadata_repair_counts_only_real_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            conn = build_db.connect_live_db(Path(temp) / "metadata.db")
            try:
                conn.execute("INSERT INTO events(name) VALUES ('Cup')")
                event_id = conn.execute(
                    "SELECT event_id FROM events WHERE name='Cup'"
                ).fetchone()[0]
                conn.execute("INSERT INTO teams(name,hltv_id) VALUES ('Alpha',1)")
                conn.execute("INSERT INTO teams(name,hltv_id) VALUES ('Beta',2)")
                teams = [
                    row[0]
                    for row in conn.execute(
                        "SELECT team_id FROM teams ORDER BY team_id"
                    )
                ]
                conn.execute(
                    """
                    INSERT INTO matches(
                        hltv_match_id,event_id,datetime_utc,datetime_precision,
                        team1_id,team2_id,best_of,status,data_tier
                    ) VALUES ('77',?,'2026-01-02T12:00:00Z','exact',?,?,3,
                              'scheduled','prematch_captured')
                    """,
                    (event_id, *teams),
                )
                master = {
                    "77": {
                        "event_metadata": {
                            "hltv_event_id": "900",
                            "prize_pool": 50_000,
                            "teams_competing": 16,
                        }
                    }
                }
                self.assertEqual(
                    repair_integrity.repair_event_metadata(conn, master), 1
                )
                self.assertEqual(
                    repair_integrity.repair_event_metadata(conn, master), 0
                )
                row = conn.execute(
                    """
                    SELECT hltv_event_id,prize_pool,teams_competing
                    FROM events WHERE event_id=?
                    """,
                    (event_id,),
                ).fetchone()
                self.assertEqual(tuple(row), ("900", 50_000.0, 16))
            finally:
                conn.close()


class ClosingOddsTests(unittest.TestCase):
    def test_only_explicit_observed_close_is_inserted_as_closing(self) -> None:
        point = {
            "captured_at": "2026-01-01T11:55:00Z",
            "team1_decimal": 1.8,
            "team2_decimal": 2.1,
            "is_observed_closing": True,
            "seconds_to_start": 300,
            "quality": "closing_observed",
        }
        row = next(build_db.iter_odds_rows(point, "closing"))
        self.assertEqual(row["is_observed_closing"], 1)
        self.assertEqual(row["quality"], "closing_observed")
        legacy = next(build_db.iter_odds_rows(
            {"captured_at": point["captured_at"]}, "closing"
        ))
        self.assertEqual(legacy["is_observed_closing"], 0)


class PredictionLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "ledger.db"
        self.conn = build_db.connect_live_db(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("INSERT INTO events(name) VALUES ('Event')")
        event_id = self.conn.execute("SELECT event_id FROM events").fetchone()[0]
        self.conn.execute("INSERT INTO teams(name,hltv_id) VALUES ('Alpha',1)")
        self.conn.execute("INSERT INTO teams(name,hltv_id) VALUES ('Beta',2)")
        teams = [row[0] for row in self.conn.execute("SELECT team_id FROM teams ORDER BY team_id")]
        self.conn.execute(
            """
            INSERT INTO matches(
                hltv_match_id,event_id,datetime_utc,datetime_precision,
                team1_id,team2_id,best_of,status,data_tier
            ) VALUES ('99',?,'2026-01-02T12:00:00Z','exact',?,?,3,'scheduled','prematch_captured')
            """,
            (event_id, teams[0], teams[1]),
        )
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()
        self.temp.cleanup()

    def _row(self, captured_at: str, probability: float) -> dict:
        return {
            "id": "99",
            "captured_at": captured_at,
            "prediction": {
                "model_prob_team1": probability,
                "decision_prob_team1": probability,
                "reliability_score": 0.8,
            },
            "features": {"elo_prob_centered": probability - 0.5},
            "data_quality": {"real_pre_match_snapshot": True},
            "model_trace": {
                "model_version": "model@v1",
                "artifact_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "feature_policy_sha256": "c" * 64,
                "is_fallback": False,
            },
        }

    def test_frozen_prediction_cannot_be_rewritten_and_is_scored(self) -> None:
        match = self.conn.execute(
            """
            SELECT m.team1_id,m.team2_id,m.datetime_utc,m.datetime_precision,m.status,
                   t1.name,t2.name,t1.hltv_id,t2.hltv_id
            FROM matches m JOIN teams t1 ON t1.team_id=m.team1_id
            JOIN teams t2 ON t2.team_id=m.team2_id WHERE m.hltv_match_id='99'
            """
        ).fetchone()
        match_id = self.conn.execute(
            "SELECT match_id FROM matches WHERE hltv_match_id='99'"
        ).fetchone()[0]
        ingest.upsert_prediction_ledger(
            self.conn,
            self._row("2026-01-02T10:00:00Z", 0.7),
            match_id=match_id,
            match_row=match,
            model_version="model@v1",
        )
        ingest.finalize_prediction_ledger(self.conn, "2026-01-02T12:01:00Z")
        ingest.upsert_prediction_ledger(
            self.conn,
            self._row("2026-01-02T11:00:00Z", 0.2),
            match_id=match_id,
            match_row=match,
            model_version="model@v1",
        )
        frozen = self.conn.execute(
            "SELECT ledger_status,prob_team1 FROM prediction_ledger"
        ).fetchone()
        self.assertEqual((frozen[0], frozen[1]), ("frozen", 0.7))
        team1 = match[0]
        self.conn.execute(
            """
            UPDATE matches SET status='completed',winner_team_id=?,
                score_t1=2,score_t2=0,result_filled_at_utc='2026-01-02T13:00:00Z'
            WHERE match_id=?
            """,
            (team1, match_id),
        )
        ingest.finalize_prediction_ledger(self.conn, "2026-01-02T13:01:00Z")
        evaluated = self.conn.execute(
            """
            SELECT ledger_status,prediction_correct,realized_log_loss
            FROM prediction_ledger
            """
        ).fetchone()
        self.assertEqual((evaluated[0], evaluated[1]), ("evaluated", 1))
        self.assertAlmostEqual(evaluated[2], -np.log(0.7))


class FoldLocalFeatureSelectionTests(unittest.TestCase):
    def _dataset(self):
        rows: list[dict[str, float]] = []
        labels: list[int] = []
        periods: list[int] = []
        rng = np.random.RandomState(7)
        columns = train.candidate_feature_columns("core", {})
        for period in range(16):
            for _ in range(20):
                label = int(rng.rand() >= 0.5)
                row = {column: 0.0 for column in columns}
                row["elo_prob_centered"] = rng.normal(0, 0.01)
                row["glicko_prob_centered"] = rng.normal(0, 0.01)
                row["player_snapshot_available"] = 1.0
                row["player_rating_diff"] = 1.0 if label else -1.0
                rows.append(row)
                labels.append(label)
                periods.append(period)
        return rows, np.asarray(labels), np.asarray(periods), columns

    def test_player_family_requires_past_temporal_log_loss_gain(self) -> None:
        rows, labels, periods, columns = self._dataset()
        thresholds = {
            name: 100_000
            for name, *_ in train.fold_local_family_specs({})
        }
        thresholds["player_snapshots"] = 20
        selected, report = train.select_fold_local_features(
            rows,
            labels,
            periods,
            periods < 14,
            columns,
            ["elo_prob_centered", "glicko_prob_centered"],
            feature_thresholds=thresholds,
            validation_periods=4,
            min_log_loss_gain=0.001,
            min_validation_rows=40,
            min_validation_available_rows=20,
        )
        self.assertTrue(report["families"]["player_snapshots"]["enabled"])
        self.assertIn("player_rating_diff", selected)

        # Labels beyond the outer training mask cannot change the decision.
        future_labels = labels.copy()
        future_labels[periods >= 14] = 1 - future_labels[periods >= 14]
        selected_again, report_again = train.select_fold_local_features(
            rows,
            future_labels,
            periods,
            periods < 14,
            columns,
            ["elo_prob_centered", "glicko_prob_centered"],
            feature_thresholds=thresholds,
            validation_periods=4,
            min_log_loss_gain=0.001,
            min_validation_rows=40,
            min_validation_available_rows=20,
        )
        self.assertEqual(selected, selected_again)
        self.assertEqual(
            report["selected_log_loss"], report_again["selected_log_loss"]
        )


if __name__ == "__main__":
    unittest.main()
