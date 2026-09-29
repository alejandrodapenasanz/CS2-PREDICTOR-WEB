"""Additive storage of ranking badges from already captured match HTML.

    python BBDD/ranking_store.py --runs PATH [--apply]

No network. Default is a read-only preview. --apply only creates/inserts the
dedicated table; it never changes ratings, predictions or historical ledgers.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import UTC, date, datetime
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = ROOT.parent / "VAULT/CS2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIPELINE.ranking_badges import parse_ranking_badges  # noqa: E402

KINDS = ("match_snapshot", "match_page", "pending_match_page", "recovery_match_page")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Execute only the additive canonical section, preserving caller transactions."""
    text = (ROOT / "BBDD/cs2_prediction_schema.sql").read_text(encoding="utf-8")
    section = text.split("-- BEGIN MATCH RANKINGS V1\n", 1)[1].split("-- END MATCH RANKINGS V1", 1)[0]
    for statement in section.split(";"):
        if statement.strip():
            conn.execute(statement)


def ingest_rankings(conn: sqlite3.Connection, runs: Path, *, apply: bool = False) -> dict[str, Any]:
    """Preserve actual capture timestamps and verified source hashes, idempotently."""
    if apply:
        ensure_schema(conn)
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    prefix = "" if (runs / "raw_html").is_dir() else "*/"
    for kind in KINDS:
        for path in sorted(runs.glob(prefix + f"raw_html/{kind}/*.html.gz")):
            counts["files"] += 1
            try:
                meta = json.loads(path.with_suffix(".gz.json").read_text(encoding="utf-8"))
                captured = datetime.fromisoformat(meta["captured_at"].replace("Z", "+00:00"))
                if captured.tzinfo is None or captured > datetime.now(UTC):
                    raise ValueError("invalid_capture_time")
                captured = captured.astimezone(UTC)
                link = urlsplit(meta["url"])
                if link.scheme != "https" or link.netloc != "www.hltv.org":
                    raise ValueError("invalid_match_origin")
                parts = link.path.strip("/").split("/")
                if len(parts) < 2 or parts[0] != "matches" or not parts[1].isdigit():
                    raise ValueError("invalid_match_identity")
                match_id = parts[1]
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    html = handle.read()
                digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
                if digest != meta["sha256"]:
                    raise ValueError("source_hash_mismatch")
                rows = parse_ranking_badges(html)
                for row in rows:
                    if str(row["ranking_date"]) > captured.date().isoformat():
                        reasons["ranking_published_after_capture"] += 1
                        continue
                    counts[str(row["ranking_type"])] += 1
                    if not apply:
                        continue
                    source = path.resolve()
                    source_name = (
                        str(source.relative_to(STATE_ROOT)) if source.is_relative_to(STATE_ROOT) else str(source)
                    )
                    cursor = conn.execute(
                        "INSERT OR IGNORE INTO match_team_ranking_observations "
                        "(match_id,team_id,hltv_match_id,hltv_team_id,ranking_type,position,ranking_date,"
                        "captured_at_utc,ranking_url,match_url,source_file,source_sha256) "
                        "VALUES ((SELECT match_id FROM matches WHERE hltv_match_id=?),"
                        "(SELECT team_id FROM teams WHERE hltv_id=?),?,?,?,?,?,?,?,?,?,?)",
                        (
                            match_id,
                            row["hltv_team_id"],
                            match_id,
                            row["hltv_team_id"],
                            row["ranking_type"],
                            row["position"],
                            row["ranking_date"],
                            captured.isoformat(),
                            row["ranking_url"],
                            meta["url"],
                            source_name,
                            digest,
                        ),
                    )
                    counts["inserted" if cursor.rowcount else "already_stored"] += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                reasons[str(exc) if isinstance(exc, ValueError) else type(exc).__name__] += 1
    return {"counts": dict(counts), "quarantine_reasons": dict(reasons)}


def observations_asof(conn: sqlite3.Connection, day: date) -> list[tuple[Any, ...]]:
    """Research export only: edition AND capture must precede civil day D."""
    return conn.execute(
        "SELECT hltv_match_id,hltv_team_id,ranking_type,position,ranking_date,captured_at_utc "
        "FROM match_team_ranking_observations WHERE ranking_date < ? AND date(captured_at_utc) < ? "
        "ORDER BY ranking_date,captured_at_utc,hltv_match_id,hltv_team_id,ranking_type,source_sha256",
        (day.isoformat(), day.isoformat()),
    ).fetchall()


def main() -> None:
    """Preview by default; apply only through this BBDD-owned additive API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=STATE_ROOT / "PIPELINE/runs")
    parser.add_argument("--db", type=Path, default=STATE_ROOT / "BBDD/cs2.db")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    mode = "rw" if args.apply else "ro"
    with closing(sqlite3.connect(args.db.resolve().as_uri() + f"?mode={mode}", uri=True, timeout=30)) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        report = ingest_rankings(conn, args.runs, apply=args.apply)
        if args.apply:
            conn.commit()
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
