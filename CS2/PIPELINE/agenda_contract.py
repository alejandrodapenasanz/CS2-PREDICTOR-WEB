"""Auditable agenda identities and publication coverage, without database writes.

Only the exact match/side/name in a captured HLTV listing may supply a missing
team ID. This is participant metadata, not a backfill of ratings or statistics.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from parsel import Selector


class AgendaCoverageError(ValueError):
    """A successful HTTP response did not yield a complete, consistent listing."""


def validate_agenda_rows(html: str, rows: list[dict[str, Any]]) -> None:
    """Do not accept a parser returning only part of the downloaded agenda."""
    selector = Selector(text=html)
    sections = selector.css(".matches-list-section")
    if not sections:
        raise AgendaCoverageError("No se reconoce la estructura de la cartelera HLTV.")
    expected = {
        str(node.attrib["data-match-id"])
        for node in selector.css(".matches-list-section .match-wrapper[data-match-id]")
        if node.css(".team1 .match-teamname::text").get()
    }
    actual = [match_id(row) for row in rows]
    if expected and (set(actual) != expected or len(actual) != len(set(actual))):
        raise AgendaCoverageError(f"Cartelera parcial: HTML={len(expected)}, parser={len(actual)}.")


def match_id(row: dict[str, Any]) -> str:
    """Extract the explicit match identity, never guess from participant names."""
    found = re.search(r"/matches/(\d+)(?:/|$)", str(row.get("link") or ""))
    return str(row.get("id") or (found.group(1) if found else ""))


def positive_id(value: Any) -> str | None:
    """Reject empty, provisional and invalid IDs from source attributes."""
    text = str(value or "").strip()
    return text if text.isdigit() and int(text) > 0 else None


def agenda_identities(html: str) -> dict[str, dict[str, Any]]:
    """Read identities from exact match wrappers; conflicting duplicates fail closed."""
    result: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for node in Selector(text=html).css(".matches-list-section .match-wrapper[data-match-id]"):
        identifier = positive_id(node.attrib.get("data-match-id"))
        if not identifier:
            continue
        teams = {}
        for side in ("team1", "team2"):
            teams[side] = {
                "id": positive_id(node.attrib.get(side)),
                "name": " ".join(node.css(f".{side} .match-teamname::text").getall()).strip(),
            }
        if identifier in result and result[identifier] != teams:
            conflicts.add(identifier)
        result[identifier] = teams
    return {identifier: teams for identifier, teams in result.items() if identifier not in conflicts}


def load_identity_evidence(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Verify the original HTML hash and availability before using its metadata."""
    path = run_dir / "raw_html" / "upcoming_page" / "matches.html.gz"
    meta_path = path.with_suffix(path.suffix + ".json")
    if not path.exists() or not meta_path.exists():
        return {}, {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        html = handle.read()
    if (
        meta.get("url") != "https://www.hltv.org/matches"
        or not meta.get("captured_at")
        or hashlib.sha256(html.encode("utf-8")).hexdigest() != meta.get("sha256")
    ):
        raise ValueError("La evidencia de identidades de la cartelera no supera URL/fecha/hash.")
    return agenda_identities(html), meta


def recover_snapshot_identity(
    snapshot: dict[str, Any], evidence: dict[str, dict[str, Any]], meta: dict[str, Any]
) -> dict[str, Any]:
    """Fill IDs in memory only, with original provenance and no future evidence."""
    snapshot = copy.deepcopy(snapshot)
    teams = evidence.get(match_id(snapshot))
    if not teams:
        return snapshot
    try:
        available = datetime.fromisoformat(str(meta["captured_at"]).replace("Z", "+00:00"))
        captured = datetime.fromisoformat(str(snapshot["captured_at"]).replace("Z", "+00:00"))
        if available > captured:
            return snapshot
    except (KeyError, ValueError, TypeError):
        return snapshot
    detail = (snapshot.get("detail") or {}).get("match") or {}
    upcoming = snapshot.get("upcoming_row") or {}
    recovered = []
    for side in ("team1", "team2"):
        observed = teams[side]
        if not observed.get("id") or not observed.get("name"):
            continue
        for target in (upcoming.get(side), detail.get(side)):
            if not isinstance(target, dict) or target.get("name") != observed["name"]:
                continue
            previous = positive_id(target.get("id"))
            if previous and previous != observed["id"]:
                snapshot.setdefault("data_quality", {})["identity_conflict"] = True
                continue
            if not previous:
                target["id"] = observed["id"]
                recovered.append(side)
    if recovered:
        snapshot.setdefault("data_quality", {})["identity_recovered_from_agenda"] = {
            "sides": sorted(set(recovered)),
            "captured_at": meta["captured_at"],
            "sha256": meta["sha256"],
            "source_file": meta.get("source_file"),
        }
    return snapshot


def unavailable_entry(snapshot: dict[str, Any], reason: str) -> dict[str, Any]:
    """Keep an agenda row visible without fabricating a prediction or favorite."""
    detail = (snapshot.get("detail") or {}).get("match") or {}
    upcoming = snapshot.get("upcoming_row") or {}
    return {
        **{key: snapshot.get(key) for key in ("id", "date", "hour", "event", "format", "link", "captured_at")},
        "team1": detail.get("team1") or upcoming.get("team1") or {},
        "team2": detail.get("team2") or upcoming.get("team2") or {},
        "prediction": {"status": "unavailable", "reason": reason, "opportunity_eligible": False},
        "flags": [
            {
                "level": "warning",
                "code": "PREDICTION_UNAVAILABLE",
                "message": "Identidad de los participantes pendiente de confirmar; no se calcula ganador ni probabilidad.",
            }
        ],
    }


def coverage_report(
    source_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    unavailable: list[dict[str, Any]],
    excluded: dict[str, str],
) -> dict[str, Any]:
    """Every acquired match must be predicted, visibly unavailable, or explicitly expired."""
    source_ids = [match_id(row) for row in source_rows]
    predicted_ids = [match_id(row) for row in predictions]
    unavailable_ids = [match_id(row) for row in unavailable]
    accounted = predicted_ids + unavailable_ids + list(excluded)
    complete = (
        bool(all(source_ids))
        and len(set(source_ids)) == len(source_ids)
        and len(set(accounted)) == len(accounted)
        and set(source_ids) == set(accounted)
        and all(reason in {"completed", "completed_detail", "already_started"} for reason in excluded.values())
    )
    return {
        "schema_version": "cs2-agenda-coverage-v1",
        "complete": complete,
        "source_ids": sorted(source_ids),
        "predicted_ids": sorted(predicted_ids),
        "unavailable_ids": sorted(unavailable_ids),
        "excluded": excluded,
        "missing_ids": sorted(set(source_ids) - set(accounted)),
        "unexpected_ids": sorted(set(accounted) - set(source_ids)),
    }
