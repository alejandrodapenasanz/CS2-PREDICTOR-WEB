"""Recover acquisition URLs from observed HLTV identities, never guessed names.

References only locate new downloads: their old capture dates are NOT reused as
the availability of newly acquired statistics.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Iterator
from urllib.parse import urlsplit

ROUTES = {
    "player": re.compile(r"/(?:stats/players|player)/(\d+)/([^/]+)/*$"),
    "team": re.compile(r"/(?:stats/teams|team)/(\d+)/([^/]+)/*$"),
}


def verified_reference(value: Any, kind: str, key: str) -> dict[str, str] | None:
    if not isinstance(value, str):
        return None
    parts = urlsplit(value)
    if parts.netloc and (parts.scheme != "https" or parts.netloc != "www.hltv.org"):
        return None
    if parts.scheme and not parts.netloc:
        return None
    match = ROUTES[kind].fullmatch(parts.path)
    if not match or match[1] != key:
        return None
    # Both observed player routes resolve to the same numeric HLTV identity.
    # The stats parser builds the explicit time-window URL from this ID/slug.
    return {"id": key, "slug": match[2], "link": parts.path}


def objects(payload: Any) -> Iterator[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        for value in payload.values():
            if isinstance(value, (dict, list)):
                yield from objects(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from objects(value)


def resolve_references(conn: sqlite3.Connection, needed: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    """One bounded archive scan for all missing player/team references."""
    pending = set(needed)
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT hltv_player_id,player_name,player_link,captured_at_utc "
        "FROM player_stat_snapshots ORDER BY captured_at_utc DESC"
    ):
        key = str(row[0])
        if ("player", key) not in pending:
            continue
        reference = verified_reference(row[2], "player", key)
        if reference:
            found[("player", key)] = {
                **reference,
                "name": row[1] or reference["slug"],
                "provenance": {"kind": "player_stat_snapshots", "captured_at": row[3]},
            }
            pending.remove(("player", key))
    if not pending:
        return found
    # Includes match_assets: often the ONLY reference for a player's first
    # acquisition. Do not require a prior successful player stats download.
    for row in conn.execute(
        "SELECT kind,raw_snapshot_id,captured_at_utc,payload_json FROM raw_snapshots "
        "WHERE kind IN ('match_assets','team_profile','match_snapshot','player_compare_stats','team_ranking') "
        "ORDER BY captured_at_utc DESC,raw_snapshot_id DESC"
    ):
        try:
            payload = json.loads(row[3])
        except (ValueError, TypeError):
            continue
        for node in objects(payload):
            for kind in ("player", "team"):
                keys = (f"{kind}_hltv_id", f"hltv_{kind}_id", f"{kind}_id", "id")
                links = (f"{kind}_link", f"{kind}_url", "link", "url")
                key = str(next((node[k] for k in keys if node.get(k) is not None), ""))
                if (kind, key) not in pending:
                    continue
                source = node.get("source")
                values = [node.get(k) for k in links]
                if isinstance(source, dict):
                    values.append(source.get("link"))
                for value in values:
                    reference = verified_reference(value, kind, key)
                    if reference:
                        found[(kind, key)] = {
                            **reference,
                            "name": node.get(f"{kind}_name") or node.get("name") or reference["slug"],
                            "provenance": {"kind": row[0], "raw_snapshot_id": row[1], "captured_at": row[2]},
                        }
                        pending.remove((kind, key))
                        break
        if not pending:
            break
    return found
