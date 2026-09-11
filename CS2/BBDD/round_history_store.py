"""Additive round-history storage and offline ingestion of existing HLTV captures.

Run ``python BBDD/round_history_store.py --runs PIPELINE/runs --apply``. Without
--apply the database is read-only; no schema or data are changed. No network.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIPELINE.round_history import PARSER_VERSION, parse_round_history


def ensure_schema(connection: sqlite3.Connection) -> None:
    """Apply only the dedicated additive section of the canonical schema."""
    schema = (ROOT / "BBDD/cs2_prediction_schema.sql").read_text(encoding="utf-8")
    section = schema.split("-- BEGIN ROUND HISTORY V1\n", 1)[1].split("-- END ROUND HISTORY V1", 1)[0]
    connection.executescript(section)


def archive_paths(runs: Path) -> list[Path]:
    """Support either the archive root or a single operational run."""
    pattern = "raw_html/mapstats/*.html.gz" if (runs / "raw_html").exists() else "*/raw_html/mapstats/*.html.gz"
    return sorted(runs.glob(pattern))


def ingest_archives(connection: sqlite3.Connection, runs: Path, *, apply: bool = False) -> dict[str, Any]:
    """Validate provenance/hash and persist each capture atomically, never the ledger."""
    if apply:
        ensure_schema(connection)
    maps: dict[str, list[tuple[Any, ...]]] = {}
    for row in connection.execute(
        "SELECT p.map_id,p.hltv_mapstats_id,m.hltv_match_id,t1.hltv_id,t2.hltv_id "
        "FROM maps p JOIN matches m USING(match_id) JOIN teams t1 ON t1.team_id=m.team1_id "
        "JOIN teams t2 ON t2.team_id=m.team2_id"
    ):
        maps.setdefault(str(row[1]), []).append(row)
    has_table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='map_round_sources'"
    ).fetchone()
    known = (
        {
            (str(r[0]), str(r[1]), str(r[2]), int(r[3]))
            for r in connection.execute(
                "SELECT source_file,sha256,captured_at_utc,parser_version FROM map_round_sources"
            )
        }
        if has_table
        else set()
    )
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    examples: list[dict[str, str]] = []
    for path in archive_paths(runs):
        counts["files"] += 1
        source_file = (
            str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path.resolve())
        )
        meta: dict[str, Any] = {}
        mapping = maps.get(path.name.split(".")[0], [])
        if len(mapping) != 1:
            counts["unmapped_or_ambiguous_map"] += 1
            continue
        map_id, mapstats_id, match_id, team1, team2 = mapping[0]
        try:
            meta = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
            captured = datetime.fromisoformat(str(meta["captured_at"]).replace("Z", "+00:00"))
            if captured.tzinfo is None or captured > datetime.now(UTC):
                raise ValueError("invalid_capture_time")
            capture_text = captured.astimezone(UTC).isoformat()
            digest = str(meta["sha256"])
            key = (source_file, digest, capture_text, PARSER_VERSION)
            if key in known:
                counts["already_stored"] += 1
                continue
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                html = handle.read()
            if hashlib.sha256(html.encode("utf-8")).hexdigest() != digest:
                raise ValueError("source_hash_mismatch")
            payload = parse_round_history(html, str(meta["url"]))
            if (
                str(payload["mapstats_id"]) != str(mapstats_id)
                or str(payload["hltv_match_id"]) != str(match_id)
                or {str(t["id"]) for t in payload["teams"]} != {str(team1), str(team2)}
            ):
                raise ValueError("database_identity_mismatch")
            if datetime.fromisoformat(payload["played_at_utc"]) > captured:
                raise ValueError("result_captured_before_match")
        except (OSError, EOFError, OverflowError, ValueError, KeyError, TypeError) as exc:
            reason = str(exc)
            reasons[reason] += 1
            counts["quarantined"] += 1
            if len(examples) < 20:
                examples.append({"source_file": source_file, "reason": reason})
            # Preserve raw HTML unchanged. An invalid provenance is never assigned
            # an invented timestamp or persisted as a model-usable observation.
            continue
        counts["valid_captures"] += 1
        counts["rounds"] += len(payload["rounds"])
        if not apply:
            continue
        with connection:
            cursor = connection.execute(
                "INSERT INTO map_round_sources(map_id,source_file,source_url,sha256,captured_at_utc,"
                "played_at_utc,round_format,map_name,team_left_hltv_id,team_right_hltv_id,parser_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    map_id,
                    source_file,
                    meta["url"],
                    digest,
                    capture_text,
                    payload["played_at_utc"],
                    payload["format"],
                    payload["map_name"],
                    payload["teams"][0]["id"],
                    payload["teams"][1]["id"],
                    PARSER_VERSION,
                ),
            )
            source_id = cursor.lastrowid
            connection.executemany(
                "INSERT INTO map_rounds VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        source_id,
                        r["round_number"],
                        r["half"],
                        r["round_in_half"],
                        int(r["overtime"]),
                        int(r["is_pistol"]),
                        r["winner_hltv_id"],
                        r["team_left_side"],
                        r["outcome"],
                    )
                    for r in payload["rounds"]
                ],
            )
        known.add(key)
    return {
        "counts": dict(counts),
        "quarantine_reasons": dict(reasons),
        "examples": examples,
        "applied": apply,
        "parser_version": PARSER_VERSION,
        "network_requests": 0,
    }


def main() -> int:
    """Ingest archived captures via the dedicated BBDD API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "BBDD/cs2.db")
    parser.add_argument("--runs", type=Path, default=ROOT / "PIPELINE/runs")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    uri = args.db.resolve().as_uri() + ("?mode=rw" if args.apply else "?mode=ro")
    connection = sqlite3.connect(uri, uri=True, timeout=30)
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        report = ingest_archives(connection, args.runs, apply=args.apply)
    finally:
        connection.close()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
