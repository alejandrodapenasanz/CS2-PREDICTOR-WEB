"""Consolida duplicados fisicos historical_seed + partido HLTV.

La fila con `hltv_match_id` es canonica. Los hijos que solo existan en la fila
legacy se mueven; los duplicados exactos se conservan una sola vez. Los marts
derivados se recalculan al terminar.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_DIR = ROOT / "MODEL"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from BBDD import build_db
from cs2model import dataio


def _clean(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def duplicate_pairs(conn: sqlite3.Connection) -> list[tuple[int, int]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT m.match_id, m.hltv_match_id, m.datetime_utc, m.best_of,
               m.score_t1, m.score_t2, e.name AS event_name,
               t1.name AS team1_name, t2.name AS team2_name
        FROM matches m
        JOIN events e ON e.event_id = m.event_id
        JOIN teams t1 ON t1.team_id = m.team1_id
        JOIN teams t2 ON t2.team_id = m.team2_id
        WHERE m.status = 'completed'
          AND m.score_t1 IS NOT NULL AND m.score_t2 IS NOT NULL
        ORDER BY m.datetime_utc, m.match_id
        """
    ).fetchall()
    groups: dict[tuple[Any, ...], list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        team_score = tuple(
            sorted(
                (
                    (_clean(row["team1_name"]), int(row["score_t1"])),
                    (_clean(row["team2_name"]), int(row["score_t2"])),
                )
            )
        )
        key = (
            str(row["datetime_utc"] or "")[:10],
            _clean(row["event_name"]),
            int(row["best_of"]),
            team_score,
        )
        groups[key].append(row)

    pairs: list[tuple[int, int]] = []
    for group in groups.values():
        canonical = [row for row in group if row["hltv_match_id"]]
        legacy = [row for row in group if not row["hltv_match_id"]]
        if len(canonical) == 1 and len(legacy) == 1:
            pairs.append((int(canonical[0]["match_id"]), int(legacy[0]["match_id"])))
    return pairs


def _copy_children_ignore(
    conn: sqlite3.Connection,
    table: str,
    id_column: str,
    target_id: int,
    source_id: int,
) -> int:
    columns = [
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({table})")
        if not (table == "odds" and str(row[1]) == "odds_id")
    ]
    select_expr = ["?" if column == id_column else column for column in columns]
    before = conn.total_changes
    conn.execute(
        f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) "
        f"SELECT {','.join(select_expr)} FROM {table} WHERE {id_column}=?",
        (target_id, source_id),
    )
    inserted = conn.total_changes - before
    conn.execute(f"DELETE FROM {table} WHERE {id_column}=?", (source_id,))
    return inserted


def _merge_maps(conn: sqlite3.Connection, canonical_id: int, legacy_id: int) -> dict[str, int]:
    stats = defaultdict(int)
    legacy_maps = conn.execute(
        "SELECT map_id, map_number FROM maps WHERE match_id=? ORDER BY map_number",
        (legacy_id,),
    ).fetchall()
    map_columns = [
        str(row[1])
        for row in conn.execute("PRAGMA table_info(maps)")
        if str(row[1]) not in {"map_id", "match_id", "map_number"}
    ]
    for legacy_map_id, map_number in legacy_maps:
        canonical = conn.execute(
            "SELECT map_id FROM maps WHERE match_id=? AND map_number=?",
            (canonical_id, map_number),
        ).fetchone()
        if canonical is None:
            conn.execute("UPDATE maps SET match_id=? WHERE map_id=?", (canonical_id, legacy_map_id))
            stats["maps_moved"] += 1
            continue
        canonical_map_id = int(canonical[0])
        for table in ("map_player_stats", "map_player_side_stats"):
            columns = [str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")]
            select_expr = ["?" if column == "map_id" else column for column in columns]
            before = conn.total_changes
            conn.execute(
                f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) "
                f"SELECT {','.join(select_expr)} FROM {table} WHERE map_id=?",
                (canonical_map_id, legacy_map_id),
            )
            stats[f"{table}_moved"] += conn.total_changes - before
            conn.execute(f"DELETE FROM {table} WHERE map_id=?", (legacy_map_id,))
        assignments = ",".join(
            f"{column}=COALESCE({column},(SELECT {column} FROM maps WHERE map_id=?))"
            for column in map_columns
        )
        conn.execute(
            f"UPDATE maps SET {assignments} WHERE map_id=?",
            (*([legacy_map_id] * len(map_columns)), canonical_map_id),
        )
        conn.execute("DELETE FROM maps WHERE map_id=?", (legacy_map_id,))
        stats["maps_deduplicated"] += 1
    return dict(stats)


def _merge_veto(conn: sqlite3.Connection, canonical_id: int, legacy_id: int) -> dict[str, int]:
    moved = deduplicated = 0
    for veto_id, step_order, team_id in conn.execute(
        "SELECT veto_id, step_order, team_id FROM veto WHERE match_id=? ORDER BY step_order",
        (legacy_id,),
    ).fetchall():
        canonical = conn.execute(
            "SELECT veto_id, team_id FROM veto WHERE match_id=? AND step_order=?",
            (canonical_id, step_order),
        ).fetchone()
        if canonical is None:
            conn.execute("UPDATE veto SET match_id=? WHERE veto_id=?", (canonical_id, veto_id))
            moved += 1
            continue
        conn.execute(
            "UPDATE veto SET team_id=COALESCE(team_id,?) WHERE veto_id=?",
            (team_id, int(canonical[0])),
        )
        conn.execute("DELETE FROM veto WHERE veto_id=?", (veto_id,))
        deduplicated += 1
    return {"veto_moved": moved, "veto_deduplicated": deduplicated}


def _merge_predictions(conn: sqlite3.Connection, canonical_id: int, legacy_id: int) -> int:
    hltv_id = conn.execute(
        "SELECT hltv_match_id FROM matches WHERE match_id=?",
        (canonical_id,),
    ).fetchone()[0]
    moved = 0
    for prediction_id, model_version in conn.execute(
        "SELECT prediction_id, model_version FROM predictions WHERE match_id=?",
        (legacy_id,),
    ).fetchall():
        conflict = conn.execute(
            "SELECT 1 FROM predictions WHERE hltv_match_id=? AND model_version=? AND prediction_id<>?",
            (hltv_id, model_version, prediction_id),
        ).fetchone()
        if conflict:
            conn.execute("DELETE FROM predictions WHERE prediction_id=?", (prediction_id,))
        else:
            conn.execute(
                "UPDATE predictions SET match_id=?, hltv_match_id=COALESCE(hltv_match_id,?) "
                "WHERE prediction_id=?",
                (canonical_id, hltv_id, prediction_id),
            )
            moved += 1
    return moved


def consolidate(conn: sqlite3.Connection, pairs: list[tuple[int, int]]) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    conn.execute("BEGIN IMMEDIATE")
    for canonical_id, legacy_id in pairs:
        for key, value in _merge_maps(conn, canonical_id, legacy_id).items():
            stats[key] += value
        for key, value in _merge_veto(conn, canonical_id, legacy_id).items():
            stats[key] += value
        stats["lineups_moved"] += _copy_children_ignore(
            conn, "match_lineups", "match_id", canonical_id, legacy_id
        )
        stats["odds_moved"] += _copy_children_ignore(
            conn, "odds", "match_id", canonical_id, legacy_id
        )
        stats["predictions_moved"] += _merge_predictions(conn, canonical_id, legacy_id)

        legacy = conn.execute("SELECT * FROM matches WHERE match_id=?", (legacy_id,)).fetchone()
        conn.execute(
            """
            UPDATE matches
            SET prematch_captured_at_utc=COALESCE(prematch_captured_at_utc,?),
                result_filled_at_utc=COALESCE(result_filled_at_utc,?),
                has_prematch_odds=MAX(has_prematch_odds,?),
                has_player_snapshot=MAX(has_player_snapshot,?),
                has_ranking_snapshot=MAX(has_ranking_snapshot,?),
                has_analytics=MAX(has_analytics,?),
                has_context=MAX(has_context,?),
                has_box_score=MAX(has_box_score,?),
                has_veto=MAX(has_veto,?)
            WHERE match_id=?
            """,
            (
                legacy["prematch_captured_at_utc"], legacy["result_filled_at_utc"],
                legacy["has_prematch_odds"], legacy["has_player_snapshot"],
                legacy["has_ranking_snapshot"], legacy["has_analytics"],
                legacy["has_context"], legacy["has_box_score"], legacy["has_veto"],
                canonical_id,
            ),
        )
        conn.execute("DELETE FROM ratings_history WHERE before_match_id IN (?,?)", (canonical_id, legacy_id))
        conn.execute("DELETE FROM match_features WHERE match_id IN (?,?)", (canonical_id, legacy_id))
        conn.execute("DELETE FROM matches WHERE match_id=?", (legacy_id,))
        stats["matches_deleted"] += 1

    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    if violations:
        conn.rollback()
        raise RuntimeError(f"foreign_key_check failed: {violations[:10]}")
    conn.commit()
    return dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consolida duplicados historical_seed + HLTV")
    parser.add_argument("--db", default=str(build_db.DEFAULT_DB))
    parser.add_argument("--apply", action="store_true", help="Ejecuta la consolidacion; sin esto solo audita")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    db_path = Path(args.db)
    conn = build_db.connect_live_db(db_path)
    pairs = duplicate_pairs(conn)
    result: dict[str, Any] = {"db": str(db_path), "duplicate_pairs": len(pairs), "applied": False}
    if not args.apply:
        conn.close()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if not args.no_backup:
        result["backup_before"] = build_db.backup_database(
            db_path, build_db.DEFAULT_BACKUP_DIR, build_db.DEFAULT_MIRROR_BACKUP_DIR
        )
    stats = consolidate(conn, pairs)
    conn.close()

    rows = dataio.load_training_rows_from_db(db_path)
    conn = build_db.connect_live_db(db_path)
    stats.update(build_db._recompute_mart(conn, rows))
    conn.commit()
    violations = conn.execute("PRAGMA foreign_key_check").fetchall()
    remaining = duplicate_pairs(conn)
    conn.close()
    if violations or remaining:
        raise RuntimeError(
            f"post-check failed: foreign_keys={len(violations)} remaining_duplicates={len(remaining)}"
        )
    conn = sqlite3.connect(db_path)
    conn.execute("VACUUM")
    conn.close()
    result.update(
        {
            "applied": True,
            "stats": stats,
            "training_rows": len(rows),
            "remaining_duplicates": 0,
            "foreign_key_violations": 0,
        }
    )
    if not args.no_backup:
        result["backup_after"] = build_db.backup_database(
            db_path, build_db.DEFAULT_BACKUP_DIR, build_db.DEFAULT_MIRROR_BACKUP_DIR
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
