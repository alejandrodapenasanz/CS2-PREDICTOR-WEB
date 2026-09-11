"""Opponent adjustment must remain causal through both levels of prediction."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "MODEL"))

from cs2model.features import FEATURE_COLUMNS, build_training_frame
from cs2model.pistol_opponents import (
    CHALLENGER_COLUMNS,
    CivilEloHistory,
    OPPONENT_COLUMNS,
    OPPONENT_DIFF_COLUMNS,
    PistolOpponentHistory,
    PistolSnapshot,
    residual_offset,
)


def sample_history(days: int = 65) -> tuple[list[dict], dict]:
    """Deterministic complete matches and two separately identified pistols each."""
    rng = np.random.default_rng(42)
    rows = []
    teams: dict[str, list[dict]] = defaultdict(list)
    for index in range(days):
        day = datetime(2026, 5, 1) + timedelta(days=index)
        order = rng.permutation(8)
        for a, b in zip(order[::2], order[1::2], strict=True):
            a, b = int(a), int(b)
            mid = str(8000000 + len(rows))
            score = int(rng.random() < 1 / (1 + np.exp((a - b) / 3)))
            rows.append(
                {
                    "id": mid,
                    "date": day.date().isoformat(),
                    "date_obj": day,
                    "datetime_precision": "exact",
                    "team1": f"team{a}",
                    "team2": f"team{b}",
                    "team1_key": f"team{a}",
                    "team2_key": f"team{b}",
                    "team1_id": str(a),
                    "team2_id": str(b),
                    "format": "bo3",
                    "event": "fixture",
                    "team1_win": score,
                    "score1": 2 if score else 0,
                    "score2": 0 if score else 2,
                }
            )
            for half, ct, t in ((1, a, b), (2, b, a)):
                ct_won = int(rng.random() < 1 / (1 + np.exp((ct - t) / 5 - 0.15)))
                for team, rival, side, won in ((ct, t, "ct", ct_won), (t, ct, "t", 1 - ct_won)):
                    teams[str(team)].append(
                        {
                            "match_id": mid,
                            "map_id": int(mid),
                            "half": half,
                            "played_at": day.isoformat(),
                            "captured_at": day.replace(hour=23).isoformat(),
                            "map_name": "Mirage",
                            "side": side,
                            "opponent": str(rival),
                            "win": float(won),
                        }
                    )
    return rows, {"teams": dict(teams), "fingerprint": "fixture-v1"}


def test_same_winrate_against_stronger_rivals_has_higher_adjustment() -> None:
    actual = [1.0, 0.0] * 20
    strong_rivals = residual_offset(actual, [0.3] * 40)
    weak_rivals = residual_offset(actual, [0.7] * 40)
    assert strong_rivals > 0 > weak_rivals
    assert residual_offset([1.0], [0.3]) < strong_rivals
    assert abs(residual_offset([1.0] * 1000, [0.01] * 1000)) <= 1.5


def test_civil_elo_never_observes_another_match_of_the_same_day() -> None:
    rows, _ = sample_history(4)
    extra = {**rows[-4], "id": "9999999", "date_obj": rows[-4]["date_obj"] + timedelta(hours=12)}
    rows.append(extra)
    history = CivilEloHistory(rows)
    key = extra["team1_key"]
    day = extra["date_obj"].date()
    assert history.by_match[extra["id"]]["elo1"] == history.rating(key, day)
    altered = deepcopy(rows)
    altered[-5]["team1_win"] = 1 - altered[-5]["team1_win"]
    assert CivilEloHistory(altered).by_match[extra["id"]] == history.by_match[extra["id"]]


def test_full_frame_optional_day_freeze_does_not_include_current_day() -> None:
    rows, _ = sample_history(4)
    extra = {**rows[-4], "id": "9999999", "date_obj": rows[-4]["date_obj"] + timedelta(hours=12)}
    rows.append(extra)
    frozen, _, _, _ = build_training_frame(rows, freeze_civil_day=True)
    assert frozen[-1]["elo_diff"] == frozen[-5]["elo_diff"]
    assert not set(CHALLENGER_COLUMNS).intersection(FEATURE_COLUMNS)


def test_replay_future_append_late_capture_and_own_result_invariance() -> None:
    rows, store = sample_history()
    target = rows[180]
    history = PistolOpponentHistory(rows, store)
    before = history.match_features(target)
    assert before["pistol_opponent_model_ready"] == 1
    assert before["pistol_opponent_available"] == 1
    history.snapshot(rows[-1]["date"])
    assert history.match_features(target) == before
    prefix_rows = [r for r in rows if r["date"] <= target["date"]]
    prefix_store = deepcopy(store)
    for team in prefix_store["teams"]:
        prefix_store["teams"][team] = [r for r in prefix_store["teams"][team] if r["played_at"][:10] <= target["date"]]
    assert PistolOpponentHistory(prefix_rows, prefix_store).match_features(target) == before
    changed = deepcopy(store)
    for records in changed["teams"].values():
        for event in records:
            if event["played_at"][:10] >= target["date"]:
                event["win"] = 1 - event["win"]
        # Late capture of an old round must not enter the earlier feature row.
        records.append({**records[0], "map_id": 999, "captured_at": rows[-1]["date"]})
    assert PistolOpponentHistory(rows, changed).match_features(target) == before


def test_pistol_rows_not_doubled_and_no_intra_match_training() -> None:
    rows, store = sample_history(30)
    history = PistolOpponentHistory(rows, store)
    assert len(history.events) == 2 * len(rows)
    snapshot = history.snapshot(rows[-1]["date"])
    assert snapshot.fit_n == 2 * (len(rows) - 4)
    by_match: dict[str, list[dict]] = defaultdict(list)
    for prediction in history.predictions:
        by_match[prediction["match_id"]].append(prediction)
    assert all(len(p) == 2 and p[0]["fit_n"] == p[1]["fit_n"] for p in by_match.values())


def test_unknown_map_or_rival_is_not_invented_and_swap_is_symmetric() -> None:
    rows, store = sample_history()
    history = PistolOpponentHistory(rows, store)
    row = rows[-1]
    a = history.match_features(row)
    b = history.features(row["team2_id"], row["team1_id"], row["team2_key"], row["team1_key"], row["date"])
    for column in OPPONENT_COLUMNS:
        assert a[column] == pytest.approx(-b[column] if column in OPPONENT_DIFF_COLUMNS else b[column], abs=1e-12)
    absent = history.features("unknown", row["team2_id"], "unknown", row["team2_key"], row["date"])
    assert absent["pistol_opponent_available"] == 0
    assert absent["pistol_opponent_edge_diff"] == 0
    snapshot = PistolSnapshot(datetime(2026, 1, 1).date(), beta=0.6, ct_intercept=0.2)
    assert snapshot.neutral(200) > snapshot.neutral(0) > snapshot.neutral(-200)
    assert snapshot.neutral(200) + snapshot.neutral(-200) == pytest.approx(1)


def test_reused_holdout_cannot_authorize_new_promotion() -> None:
    from run_pistol_ablation import fresh_predictions

    rows = [{"date": "2026-09-09", "match_id": "old"}, {"date": "2026-09-10", "match_id": "fresh"}]
    assert fresh_predictions(rows) == [rows[1]]


def test_serving_uses_same_columns_symmetry_and_frozen_day_as_evaluation(tmp_path: Path) -> None:
    from cs2model.artifacts import Component, ModelArtifact, ProbabilityColumnEstimator, load_artifact
    from cs2model.pistol_opponents import SymmetricPistolScorer
    from PIPELINE.enrich_predictions import _match_feature_pair, model_probability_team1

    rows, store = sample_history()
    history = PistolOpponentHistory(rows, store)
    row = rows[180]
    extra = history.match_features(row)
    artifact = ModelArtifact(
        feature_columns=["pistol_opponent_probability_centered"],
        components=[Component("fixture", ProbabilityColumnEstimator(0), None)],
        metadata={"pistol_opponent_version": 1, "date_max": "2026-04-30"},
    )
    candidate_path = tmp_path / "candidate.pkl"
    artifact.save(candidate_path)
    restored = load_artifact(candidate_path)
    assert restored is not None
    engine = {"artifact": restored, "state": None, "history_rows": rows, "states_by_day": {}}
    arguments = (engine, row["team1_key"], row["team2_key"], row["date_obj"], row["event"], row["format"])
    forward, reverse = _match_feature_pair(*arguments, extra_features=extra)
    for col in OPPONENT_COLUMNS:
        assert forward[col] == extra[col]
        assert reverse[col] == pytest.approx(-extra[col] if col in OPPONENT_DIFF_COLUMNS else extra[col])
    elo = history.elo.by_match[row["id"]]
    assert forward["elo_diff"] == elo["elo1"] - elo["elo2"]
    served = model_probability_team1(*arguments, extra_features=extra)
    evaluated = SymmetricPistolScorer(restored).predict_proba_team1([forward])[0]
    assert served == pytest.approx(evaluated, abs=1e-12)


def test_candidate_boot_uses_canonical_database_without_legacy_history(tmp_path: Path, monkeypatch) -> None:
    import sqlite3
    from types import SimpleNamespace

    from PIPELINE import enrich_predictions as serving

    rows, store = sample_history(2)
    database = tmp_path / "canonical.sqlite3"
    with sqlite3.connect(database):
        pass
    monkeypatch.setattr(serving, "LIVE_DB", database)
    monkeypatch.setattr(serving, "cs2_load_artifact", lambda: SimpleNamespace(metadata={"pistol_opponent_version": 1}))
    monkeypatch.setattr(serving, "load_pistol_store", lambda connection: store)
    calls = []

    def load_rows(history_path, master_path, **kwargs):
        calls.append((history_path, kwargs))
        return rows if kwargs.get("db_path") == database else []

    monkeypatch.setattr(serving.cs2_dataio, "load_training_rows", load_rows)
    engine = serving.load_model_engine(tmp_path / "missing-legacy-history.json")
    assert engine is not None
    assert engine["history_rows"] == rows
    assert len(calls) == 1 and calls[0][0] is None

    monkeypatch.setattr(serving.cs2_dataio, "load_training_rows", lambda *args, **kwargs: [])
    with pytest.raises(ValueError, match="requires canonical history"):
        serving.load_model_engine(tmp_path / "missing-legacy-history.json")


def test_ingestion_column_contract_does_not_load_model_dependencies() -> None:
    """Scraper-owned ingestion can import column names without NumPy/sklearn."""
    script = """
import importlib.abc
import sys

class BlockModelDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'sklearn'}:
            raise ImportError('Ingestion must not require ' + fullname)
        return None

sys.meta_path.insert(0, BlockModelDependencies())
from BBDD import build_db, ingest
from cs2model.features import EXTENDED_DIFF_COLUMNS
assert 'pistol_opponent_edge_diff' in EXTENDED_DIFF_COLUMNS
assert 'cs2model.pistol_opponents' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
