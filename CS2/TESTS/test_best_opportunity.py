from __future__ import annotations

import copy
import json
from pathlib import Path

from BBDD import build_db, ingest
from PIPELINE.opportunity import annotate_opportunities


def _confirmed_rosters() -> dict:
    return {
        side: {
            "players": [{"id": f"{side}-{index}", "name": f"{side} player {index}"} for index in range(1, 6)],
            "announced_lineup_complete": True,
            "integrity": {
                "status": "confirmed",
                "announced_complete": True,
                "announced_matches_rendered": True,
            },
        }
        for side in ("team1", "team2")
    }


def _entry(match_id: str, day: str, confidence: float, reliability: float) -> dict:
    probability = confidence
    return {
        "id": match_id,
        "date": day,
        "rosters": _confirmed_rosters(),
        "prediction": {
            "decision_prob_team1": probability,
            "decision_confidence": confidence,
            "reliability_score": reliability,
        },
    }


def test_annotate_opportunities_applies_gates_and_daily_ranks() -> None:
    entries = [
        _entry("1", "2026-07-01", 0.70, 0.90),
        _entry("2", "2026-07-01", 0.80, 0.60),
        _entry("3", "2026-07-01", 0.90, 0.40),
        _entry("4", "2026-07-02", 0.65, 1.00),
    ]

    annotate_opportunities(entries)

    predictions = {entry["id"]: entry["prediction"] for entry in entries}
    assert abs(predictions["1"]["opportunity_score"] - 0.63) < 1e-12
    assert predictions["1"]["opportunity_rank_day"] == 1
    assert predictions["1"]["is_best_opportunity"] is True
    assert predictions["2"]["opportunity_rank_day"] == 2
    assert predictions["3"]["opportunity_eligible"] is False
    assert predictions["3"]["opportunity_rank"] is None
    assert predictions["4"]["opportunity_rank_day"] == 1
    assert predictions["4"]["is_best_opportunity"] is True


def test_estimate_band_fields_do_not_change_opportunity_order_or_ranks() -> None:
    baseline = [
        _entry("1", "2026-07-01", 0.70, 0.90),
        _entry("2", "2026-07-01", 0.80, 0.60),
        _entry("3", "2026-07-02", 0.65, 1.00),
    ]
    with_bands = copy.deepcopy(baseline)
    for index, entry in enumerate(with_bands):
        entry["prediction"].update(
            {
                "ensemble_disagreement": 0.01 + index * 0.12,
                "estimate_band_half_width": 0.02 + index * 0.10,
                "estimate_confidence_level": ("high", "medium", "low")[index],
                "estimate_history_coverage": (1.0, 0.5, 0.0)[index],
            }
        )

    annotate_opportunities(baseline)
    annotate_opportunities(with_bands)
    ranking_fields = (
        "opportunity_score",
        "opportunity_eligible",
        "opportunity_rank",
        "opportunity_rank_day",
        "is_best_opportunity",
    )
    expected = {entry["id"]: tuple(entry["prediction"][field] for field in ranking_fields) for entry in baseline}
    actual = {entry["id"]: tuple(entry["prediction"][field] for field in ranking_fields) for entry in with_bands}
    assert actual == expected


def test_incomplete_announced_lineup_is_never_a_best_opportunity() -> None:
    incomplete = _entry("incomplete", "2026-07-01", 0.99, 0.99)
    incomplete["rosters"]["team2"]["players"].pop()
    incomplete["rosters"]["team2"]["announced_lineup_complete"] = False
    incomplete["rosters"]["team2"]["integrity"] = {
        "status": "announced_incomplete",
        "announced_complete": False,
        "announced_matches_rendered": None,
    }
    confirmed = _entry("confirmed", "2026-07-01", 0.70, 0.80)

    annotate_opportunities([incomplete, confirmed])

    blocked = incomplete["prediction"]
    assert blocked["opportunity_eligible"] is False
    assert blocked["opportunity_roster_confirmed"] is False
    assert blocked["opportunity_rank_day"] is None
    assert blocked["is_best_opportunity"] is False
    assert "team2_announced_lineup_incomplete" in blocked["opportunity_ineligible_reasons"]
    assert confirmed["prediction"]["is_best_opportunity"] is True


def test_prediction_ingest_persists_best_opportunity_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "cs2.db"
    run_dir = tmp_path / "2026-07-01_080000Z"
    run_dir.mkdir()
    conn = build_db.connect_live_db(db_path)
    conn.execute("INSERT INTO teams(team_id, name, hltv_id) VALUES (1, 'Alpha', 1)")
    conn.execute("INSERT INTO teams(team_id, name, hltv_id) VALUES (2, 'Beta', 2)")
    conn.execute("INSERT INTO events(event_id, name) VALUES (1, 'Test event')")
    conn.execute(
        """
        INSERT INTO matches(
            match_id, hltv_match_id, event_id, datetime_utc, team1_id, team2_id,
            best_of, status, data_tier
        ) VALUES (1, '1001', 1, '2026-07-01T12:00:00Z', 1, 2, 3,
                  'scheduled', 'prematch_captured')
        """
    )
    payload = [
        {
            "id": "1001",
            "captured_at": "2026-07-01T08:00:00Z",
            "team1": {"name": "Alpha"},
            "team2": {"name": "Beta"},
            "prediction": {
                "model_prob_team1": 0.72,
                "decision_prob_team1": 0.72,
                "decision_confidence": 0.72,
                "decision_favorite": "Alpha",
                "decision_favorite_side": "team1",
                "reliability_score": 0.80,
                "opportunity_score": 0.576,
                "opportunity_eligible": True,
                "opportunity_rank": 1,
                "opportunity_rank_day": 1,
                "is_best_opportunity": True,
                "opportunity_policy_version": "best_opportunity_v2_confirmed_lineups",
            },
            "controls": {
                "decision_policy": {"source": "test"},
                "tournament_context": {
                    "environment": "online",
                    "stage": "semi",
                    "high_stakes": True,
                },
            },
            "data_quality": {"real_pre_match_snapshot": True},
        }
    ]
    (run_dir / "predictions_enriched.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    assert ingest.insert_predictions(conn, run_dir) == 1
    row = conn.execute(
        """
        SELECT opportunity_score, opportunity_eligible, opportunity_rank_day,
               is_best_opportunity, favorite_team_id, decision_min_value_odds,
               context_environment, context_stage
        FROM predictions
        """
    ).fetchone()
    conn.close()

    assert row is not None
    assert abs(row[0] - 0.576) < 1e-12
    assert row[1:4] == (1, 1, 1)
    assert row[4] == 1
    assert abs(row[5] - (1.0 / 0.72)) < 1e-12
    assert row[6:] == ("online", "semi")


def test_prediction_reingest_never_deletes_older_prematch_snapshots(tmp_path: Path) -> None:
    db_path = tmp_path / "cs2.db"
    run_dir = tmp_path / "2026-07-01_080000Z"
    run_dir.mkdir()
    conn = build_db.connect_live_db(db_path)
    conn.execute("INSERT INTO teams(team_id, name, hltv_id) VALUES (1, 'Alpha', 1)")
    conn.execute("INSERT INTO teams(team_id, name, hltv_id) VALUES (2, 'Beta', 2)")
    conn.execute("INSERT INTO events(event_id, name) VALUES (1, 'Test event')")
    for match_id in (1001, 1002):
        conn.execute(
            """
            INSERT INTO matches(
                hltv_match_id, event_id, datetime_utc, team1_id, team2_id,
                best_of, status, data_tier
            ) VALUES (?, 1, '2026-07-01T12:00:00Z', 1, 2, 3,
                      'scheduled', 'prematch_captured')
            """,
            (str(match_id),),
        )

    def prediction(match_id: str) -> dict:
        return {
            "id": match_id,
            "captured_at": "2026-07-01T08:00:00Z",
            "team1": {"name": "Alpha"},
            "team2": {"name": "Beta"},
            "prediction": {
                "model_prob_team1": 0.70,
                "decision_prob_team1": 0.70,
                "decision_confidence": 0.70,
                "decision_favorite": "Alpha",
                "decision_favorite_side": "team1",
                "reliability_score": 0.80,
            },
            "data_quality": {"real_pre_match_snapshot": True},
        }

    path = run_dir / "predictions_enriched.json"
    path.write_text(json.dumps([prediction("1001"), prediction("1002")]), encoding="utf-8")
    assert ingest.insert_predictions(conn, run_dir) == 2
    path.write_text(json.dumps([prediction("1002")]), encoding="utf-8")
    assert ingest.insert_predictions(conn, run_dir) == 1

    ids = {row[0] for row in conn.execute("SELECT hltv_match_id FROM predictions ORDER BY hltv_match_id")}
    conn.close()
    assert ids == {"1001", "1002"}


def test_best_opportunity_web_has_no_result_cap() -> None:
    html = (Path(__file__).resolve().parents[2] / "WEB" / "index.html").read_text(encoding="utf-8")
    render_best = html.split("function renderBest(){", 1)[1].split("// ---------- model view ----------", 1)[0]

    assert ".slice(" not in render_best
    assert "x.m.prediction?.opportunity_eligible===true" in render_best
    assert "decisionConfidence(x.m)>=probMin && x.rel>=relMin" in render_best
    assert 'Math.max(0.65, parseFloat($("bestProb")' in render_best
    assert "upsets sufridos" in render_best
    assert "como favorito modelo ≥51%" in render_best

    assert '<option value="0.65">Prob. ganar ≥ 65% (mínimo)</option>' in html
    assert "decisionConfidence(m)>=.65&&reliability(m)>=.45" in html
    assert "prediction_ledger prepartido congelado" in html
    assert "perdio como favorito de mercado" not in html
