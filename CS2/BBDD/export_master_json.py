"""Exporta un master JSON de compatibilidad desde la BBDD viva.

Durante la migracion, WEB/enrich/modelo aun entienden el formato
PIPELINE/master/matches.json. Este script parte de la BBDD como fuente
de verdad y conserva campos ricos existentes del master cuando ya estaban.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "BBDD" / "cs2.db"
DEFAULT_OUT = ROOT / "PIPELINE" / "master" / "matches.json"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def export_master(db_path: Path, out_path: Path) -> dict[str, Any]:
    existing = read_json(out_path, {})
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 30000;")
    # WAL requiere memoria compartida (-wal/-shm); en carpetas sincronizadas
    # (OneDrive) o de red puede fallar. Si no queda activo, usamos DELETE.
    try:
        _jm = conn.execute("PRAGMA journal_mode = WAL;").fetchone()
    except sqlite3.OperationalError:
        _jm = None
    if not _jm or str(_jm[0]).lower() != "wal":
        try:
            conn.execute("PRAGMA journal_mode = DELETE;")
        except sqlite3.OperationalError:
            pass
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT m.hltv_match_id, m.datetime_utc, m.best_of, m.status, m.data_tier,
               m.score_t1, m.score_t2, m.context_json,
               e.name AS event_name,
               t1.name AS team1_name, t1.hltv_id AS team1_hltv_id,
               t2.name AS team2_name, t2.hltv_id AS team2_hltv_id
        FROM matches m
        JOIN events e ON e.event_id = m.event_id
        JOIN teams t1 ON t1.team_id = m.team1_id
        JOIN teams t2 ON t2.team_id = m.team2_id
        WHERE m.hltv_match_id IS NOT NULL
        ORDER BY m.datetime_utc, m.hltv_match_id
        """
    ).fetchall()
    out = dict(existing)
    for row in rows:
        mid = str(row["hltv_match_id"])
        record = dict(out.get(mid) or {})
        record.setdefault("id", mid)
        record.setdefault("event", row["event_name"])
        record.setdefault("date", str(row["datetime_utc"] or "")[:10])
        if row["datetime_utc"] and "T" in str(row["datetime_utc"]):
            record.setdefault("hour", str(row["datetime_utc"])[11:16])
        record.setdefault("format", f"bo{row['best_of']}")
        record["status"] = row["status"]
        record["data_tier"] = row["data_tier"]
        record.setdefault("team1", {"name": row["team1_name"], "id": str(row["team1_hltv_id"] or "")})
        record.setdefault("team2", {"name": row["team2_name"], "id": str(row["team2_hltv_id"] or "")})
        if row["score_t1"] is not None and row["score_t2"] is not None:
            record["score"] = {"team1": int(row["score_t1"]), "team2": int(row["score_t2"])}
            record["status"] = "completed"
        if row["context_json"] and not record.get("match_context"):
            try:
                record["match_context"] = json.loads(row["context_json"])
            except json.JSONDecodeError:
                pass
        out[mid] = record
    write_json(out_path, out)
    conn.close()
    return {"output": str(out_path), "matches": len(out), "db_rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Exporta master JSON desde cs2.db")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    result = export_master(Path(args.db), Path(args.output))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
