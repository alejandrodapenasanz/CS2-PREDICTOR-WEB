"""Blocking data/model health gates for training and production promotion."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "MODEL"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from BBDD import build_db, deduplicate_matches
from cs2model import dataio
from cs2model.artifacts import load_artifact
from cs2model.config import get_runtime_config, load_config
from cs2model.identity import is_provisional_team_name
from cs2model.metrics import metric_dict


DEFAULT_REPORT = ROOT / "MODEL" / "results" / "health_gate.json"


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(name: str, passed: bool, actual: Any, expected: Any, detail: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "actual": actual,
        "expected": expected,
        "detail": detail,
    }


def _provisional_completed(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT m.hltv_match_id, t1.name AS team1, t2.name AS team2
        FROM matches m
        JOIN teams t1 ON t1.team_id=m.team1_id
        JOIN teams t2 ON t2.team_id=m.team2_id
        WHERE m.status='completed'
        """
    ).fetchall()
    return [
        {"hltv_match_id": row[0], "team1": row[1], "team2": row[2]}
        for row in rows
        if is_provisional_team_name(row[1]) or is_provisional_team_name(row[2])
    ]


def _unconfirmed_live_matches(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT m.hltv_match_id, m.status,
               t1.name AS team1, t1.hltv_id AS team1_hltv_id,
               t2.name AS team2, t2.hltv_id AS team2_hltv_id
        FROM matches m
        JOIN teams t1 ON t1.team_id=m.team1_id
        JOIN teams t2 ON t2.team_id=m.team2_id
        WHERE m.data_tier <> 'historical_seed'
        """
    ).fetchall()
    return [
        dict(row)
        for row in rows
        if row["team1_hltv_id"] is None
        or row["team2_hltv_id"] is None
        or is_provisional_team_name(row["team1"])
        or is_provisional_team_name(row["team2"])
    ]


def _provisional_teams(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        {"team_id": int(row["team_id"]), "name": str(row["name"])}
        for row in conn.execute("SELECT team_id,name FROM teams")
        if is_provisional_team_name(row["name"])
    ]


def _prediction_reference_errors(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "missing_match": int(
            conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE match_id IS NULL"
            ).fetchone()[0]
        ),
        "favorite_mismatch": int(
            conn.execute(
                """
                SELECT COUNT(*) FROM predictions p
                JOIN matches m ON m.match_id=p.match_id
                WHERE p.favorite_team_id IS NOT NULL
                  AND (
                    (p.decision_favorite_side='team1'
                     AND p.favorite_team_id<>m.team1_id)
                    OR
                    (p.decision_favorite_side='team2'
                     AND p.favorite_team_id<>m.team2_id)
                  )
                """
            ).fetchone()[0]
        ),
    }


def _mergeable_identity_duplicates(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, str, Any]]] = defaultdict(list)
    for team_id, name, hltv_id in conn.execute(
        "SELECT team_id, name, hltv_id FROM teams"
    ):
        key = dataio.clean_team(name)
        if key:
            grouped[key].append((int(team_id), str(name), hltv_id))
    return [
        {"key": key, "rows": values}
        for key, values in grouped.items()
        if len(values) > 1
        and len({str(value[2]) for value in values if value[2] is not None}) <= 1
    ]


def _coverage_degradation(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    max_date = conn.execute(
        """
        SELECT MAX(substr(datetime_utc,1,10))
        FROM matches
        WHERE status='completed' AND data_tier <> 'historical_seed'
        """
    ).fetchone()[0]
    if not max_date:
        return []
    columns = (
        "has_prematch_odds", "has_player_snapshot", "has_ranking_snapshot",
        "has_analytics", "has_context", "has_box_score", "has_veto",
    )
    cfg = get_runtime_config().health_gates
    output: list[dict[str, Any]] = []
    for column in columns:
        row = conn.execute(
            f"""
            SELECT
              SUM(CASE WHEN julianday(?) - julianday(substr(datetime_utc,1,10))
                            BETWEEN 0 AND 29 THEN 1 ELSE 0 END) recent_n,
              SUM(CASE WHEN julianday(?) - julianday(substr(datetime_utc,1,10))
                            BETWEEN 0 AND 29 THEN {column} ELSE 0 END) recent_yes,
              SUM(CASE WHEN julianday(?) - julianday(substr(datetime_utc,1,10))
                            BETWEEN 30 AND 59 THEN 1 ELSE 0 END) previous_n,
              SUM(CASE WHEN julianday(?) - julianday(substr(datetime_utc,1,10))
                            BETWEEN 30 AND 59 THEN {column} ELSE 0 END) previous_yes
            FROM matches
            WHERE status='completed' AND data_tier <> 'historical_seed'
            """,
            (max_date, max_date, max_date, max_date),
        ).fetchone()
        recent_n, recent_yes, previous_n, previous_yes = [int(value or 0) for value in row]
        recent_rate = recent_yes / recent_n if recent_n else None
        previous_rate = previous_yes / previous_n if previous_n else None
        degraded = bool(
            recent_n >= cfg.min_coverage_sample
            and previous_n >= cfg.min_coverage_sample
            and previous_rate
            and recent_rate is not None
            and recent_rate < previous_rate * cfg.min_recent_coverage_ratio
        )
        output.append(
            {
                "feature": column,
                "recent_n": recent_n,
                "recent_rate": recent_rate,
                "previous_n": previous_n,
                "previous_rate": previous_rate,
                "degraded": degraded,
            }
        )
    return output


def validate_data(db_path: Path) -> dict[str, Any]:
    conn = build_db.connect_live_db(db_path)
    conn.row_factory = sqlite3.Row
    conn.commit()
    cfg = get_runtime_config().health_gates
    provisional = _provisional_completed(conn)
    unconfirmed_live = _unconfirmed_live_matches(conn)
    provisional_teams = _provisional_teams(conn)
    prediction_errors = _prediction_reference_errors(conn)
    missing_match_ids = int(
        conn.execute(
            "SELECT COUNT(*) FROM matches WHERE hltv_match_id IS NULL"
        ).fetchone()[0]
    )
    physical_duplicates = deduplicate_matches.duplicate_pairs(conn)
    impossible_formats = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM matches
            WHERE status='completed' AND best_of<>1
              AND (score_t1>5 OR score_t2>5)
            """
        ).fetchone()[0]
    )
    duplicates = _mergeable_identity_duplicates(conn)
    fk_errors = [tuple(row) for row in conn.execute("PRAGMA foreign_key_check")]
    coverage = _coverage_degradation(conn)
    degraded = [row for row in coverage if row["degraded"]]
    rows = dataio.load_training_rows(db_path=db_path)
    checks = [
        _check(
            "completed_participants_resolved",
            len(provisional) <= cfg.max_completed_placeholders,
            len(provisional),
            f"<= {cfg.max_completed_placeholders}",
            json.dumps(provisional[:10], ensure_ascii=False),
        ),
        _check(
            "live_matches_have_two_confirmed_teams",
            not unconfirmed_live,
            len(unconfirmed_live),
            0,
            json.dumps(unconfirmed_live[:10], ensure_ascii=False),
        ),
        _check(
            "no_provisional_team_rows",
            not provisional_teams,
            len(provisional_teams),
            0,
            json.dumps(provisional_teams[:10], ensure_ascii=False),
        ),
        _check(
            "prediction_references_consistent",
            not any(prediction_errors.values()),
            prediction_errors,
            {"missing_match": 0, "favorite_mismatch": 0},
        ),
        _check(
            "all_matches_have_stable_hltv_id",
            missing_match_ids == 0,
            missing_match_ids,
            0,
        ),
        _check(
            "no_physical_match_duplicates",
            not physical_duplicates,
            len(physical_duplicates),
            0,
        ),
        _check(
            "completed_score_format_consistent",
            impossible_formats == 0,
            impossible_formats,
            0,
        ),
        _check("foreign_keys", len(fk_errors) <= cfg.max_foreign_key_errors, len(fk_errors), 0),
        _check("unambiguous_team_duplicates", not duplicates, len(duplicates), 0),
        _check("training_rows_available", len(rows) >= 800, len(rows), ">= 800"),
        _check(
            "recent_feature_coverage",
            not degraded,
            [row["feature"] for row in degraded],
            "no >50% relative degradation",
        ),
    ]
    conn.close()
    return {
        "phase": "data",
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
        "coverage": coverage,
        "training_rows": len(rows),
    }


def _prediction_metrics(csv_path: Path, model: str) -> dict[str, Any]:
    actual: list[int] = []
    probability: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") != model:
                continue
            actual.append(int(row["actual"]))
            probability.append(float(row["prob_team1"]))
    return metric_dict(actual, probability) if actual else {}


def validate_candidate(
    db_path: Path,
    artifact_path: Path,
    result_dir: Path,
) -> dict[str, Any]:
    data_report = validate_data(db_path)
    cfg = get_runtime_config().health_gates
    artifact = load_artifact(artifact_path)
    metrics_path = result_dir / "metrics.json"
    predictions_path = result_dir / "predictions_walkforward.csv"
    checks: list[dict[str, Any]] = [
        _check("data_gate", data_report["status"] == "pass", data_report["status"], "pass"),
        _check("candidate_artifact_exists", artifact is not None, bool(artifact), True),
        _check("metrics_exist", metrics_path.exists(), metrics_path.exists(), True),
        _check("walkforward_predictions_exist", predictions_path.exists(), predictions_path.exists(), True),
    ]
    if artifact is not None and metrics_path.exists() and predictions_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        evaluation_model = str(
            artifact.metadata.get("evaluation_policy")
            or artifact.metadata.get("production_model")
        )
        expected = metrics.get(evaluation_model) or {}
        recalculated = _prediction_metrics(predictions_path, evaluation_model)
        metric_delta = max(
            (
                abs(float(recalculated[key]) - float(expected[key]))
                for key in ("accuracy", "log_loss", "brier")
                if recalculated.get(key) is not None and expected.get(key) is not None
            ),
            default=float("inf"),
        )
        checks.append(
            _check(
                "evaluation_consistency",
                metric_delta <= cfg.max_evaluation_metric_delta,
                metric_delta,
                f"<= {cfg.max_evaluation_metric_delta}",
            )
        )
        production_name = str(artifact.metadata.get("production_model") or "")
        production_metrics = metrics.get(production_name) or expected
        glicko = metrics.get("glicko") or {}
        model_loss = production_metrics.get("log_loss")
        glicko_loss = glicko.get("log_loss")
        checks.append(
            _check(
                "log_loss_beats_glicko",
                model_loss is not None
                and glicko_loss is not None
                and float(model_loss) <= float(glicko_loss) + cfg.max_log_loss_vs_glicko,
                model_loss,
                f"<= {glicko_loss}",
            )
        )
        policies = artifact.metadata.get("feature_policies") or {}
        invalid_activation = [
            name
            for name in ("player_snapshots", "rankings", "analytics", "analytics_extended")
            if (policies.get(name) or {}).get("enabled")
            and (policies.get(name) or {}).get("activation")
            != "fold_local_temporal_log_loss"
        ]
        checks.append(
            _check(
                "optional_families_fold_local",
                not invalid_activation,
                invalid_activation,
                [],
            )
        )
        finite_probe = all(
            math.isfinite(float(value))
            for value in (
                model_loss if model_loss is not None else float("nan"),
                production_metrics.get("accuracy", float("nan")),
            )
        )
        checks.append(_check("finite_candidate_metrics", finite_probe, finite_probe, True))
    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {
        "phase": "candidate",
        "status": status,
        "artifact_sha256": sha256_file(artifact_path) if artifact_path.exists() else None,
        "checks": checks,
        "data_gate": data_report,
    }


def validate_live_ledger(db_path: Path) -> dict[str, Any]:
    conn = build_db.connect_live_db(db_path)
    conn.row_factory = sqlite3.Row
    conn.commit()
    rows = conn.execute(
        """
        SELECT * FROM prediction_ledger
        WHERE ledger_status='evaluated'
        """
    ).fetchall()
    inconsistent = 0
    for row in rows:
        probability = float(row["prob_team1"])
        actual = int(row["actual_team1_win"])
        correct = int((probability >= 0.5) == bool(actual))
        loss = -(actual * math.log(probability) + (1-actual) * math.log(1-probability))
        if (
            correct != int(row["prediction_correct"])
            or abs(loss - float(row["realized_log_loss"])) > 1e-10
        ):
            inconsistent += 1
    late_evaluated = conn.execute(
        """
        SELECT COUNT(*) FROM prediction_ledger
        WHERE ledger_status='evaluated' AND predicted_at_utc >= kickoff_utc
        """
    ).fetchone()[0]
    late_rejected = conn.execute(
        """
        SELECT COUNT(*) FROM prediction_ledger
        WHERE ledger_status='invalid' AND invalid_reason LIKE '%prediction_not_prematch%'
        """
    ).fetchone()[0]
    checks = [
        _check("ledger_evaluation_consistency", inconsistent == 0, inconsistent, 0),
        _check(
            "ledger_has_no_late_evaluations",
            late_evaluated == 0,
            int(late_evaluated),
            0,
        ),
    ]
    conn.close()
    return {
        "phase": "live",
        "status": "pass" if all(item["status"] == "pass" for item in checks) else "fail",
        "checks": checks,
        "evaluated_rows": len(rows),
        "late_predictions_rejected": int(late_rejected),
    }


def store_report(db_path: Path, report: dict[str, Any]) -> None:
    conn = build_db.connect_live_db(db_path)
    conn.execute(
        """
        INSERT INTO health_gate_runs(checked_at_utc, phase, status, report_json)
        VALUES (?,?,?,?)
        """,
        (utcnow(), report["phase"], report["status"], json.dumps(report, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("data", "candidate", "live"), required=True)
    parser.add_argument("--db", default=str(build_db.DEFAULT_DB))
    parser.add_argument("--artifact")
    parser.add_argument("--results", default=str(ROOT / "MODEL" / "results"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    if args.config:
        from cs2model.config import set_runtime_config
        set_runtime_config(load_config(args.config))
    db_path = Path(args.db)
    if args.phase == "data":
        report = validate_data(db_path)
    elif args.phase == "candidate":
        if not args.artifact:
            raise SystemExit("--artifact is required for candidate phase")
        report = validate_candidate(db_path, Path(args.artifact), Path(args.results))
    else:
        report = validate_live_ledger(db_path)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    store_report(db_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
