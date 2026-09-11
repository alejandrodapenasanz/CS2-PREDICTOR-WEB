"""Audit Tennis Abstract history literals without assigning invented match dates.

This is acquisition/coverage evidence, not an Elo or model feature adapter.
The source's first date column labels tournaments, and row widths differ.
Unmapped trailing cells are preserved in the source cache, never guessed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .tennis_abstract_access import ORIGIN, TennisAbstractAcquisitionClient

HISTORY_HEADER = [
    "date",
    "tourn",
    "surf",
    "level",
    "wl",
    "rank",
    "seed",
    "entry",
    "round",
    "score",
    "max",
    "opp",
    "orank",
    "oseed",
    "oentry",
    "ohand",
    "obday",
    "oht",
    "ocountry",
    "oactive",
    "time",
    "aces",
    "dfs",
    "pts",
    "firsts",
    "fwon",
    "swon",
    "games",
    "saved",
    "chances",
    "oaces",
    "odfs",
    "opts",
    "ofirsts",
    "ofwon",
    "oswon",
    "ogames",
    "osaved",
    "ochances",
    "obackhand",
    "chartlink",
    "pslink",
    "whserver",
    "matchid",
]


def literal_array(text: str, variable: str) -> list[Any] | None:
    """Read one bounded JS array literal using a non-executing literal parser."""

    match = re.search(r"\bvar\s+" + re.escape(variable) + r"\s*=\s*", text)
    if match is None:
        return None
    start = match.end()
    if text[start : start + 1] != "[":
        raise ValueError(f"{variable} is not an array literal.")
    quote: str | None = None
    escaped = False
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "[":
            depth += 1
            if depth > 3:
                raise ValueError("Unexpected nested history literal.")
        elif char == "]":
            depth -= 1
            if depth == 0:
                result = ast.literal_eval(text[start : index + 1])
                if not isinstance(result, list):
                    raise ValueError("Expected a history list.")
                return result
    raise ValueError(f"Unterminated {variable} array.")


def partition_history(
    content: bytes, *, quarantine_incomplete: bool = False
) -> tuple[list[list[str]], list[tuple[int, list[str]]]]:
    """Isolate incomplete literal rows without padding or assigning field semantics."""
    text = content.decode("utf-8", errors="strict")
    rows = literal_array(text, "matchmx")
    if rows is None:
        rows = literal_array(text, "morematchmx")
    if not rows or any(
        not isinstance(row, list) or any(not isinstance(cell, str) for cell in row) for row in rows
    ):
        raise ValueError("No valid inspected singles history matrix.")
    incomplete = [(index, row) for index, row in enumerate(rows) if len(row) < 44]
    if incomplete and not quarantine_incomplete:
        raise ValueError("No valid inspected singles history matrix.")
    valid = [row for row in rows if len(row) >= 44]
    if not valid:
        raise ValueError("No complete inspected history rows; player remains pending.")
    for row in valid:
        datetime.strptime(row[0], "%Y%m%d")
    return valid, incomplete


def history_rows(content: bytes) -> list[list[str]]:
    """Strict audit contract: require every row to have the inspected leading fields."""

    return partition_history(content)[0]


def summarize_history(content: bytes) -> dict[str, object]:
    """Measure known leading columns, retaining unknown schema/date caveats."""

    rows = history_rows(content)
    return _summarize_rows(rows, content)


def _summarize_rows(rows: list[list[str]], content: bytes) -> dict[str, object]:
    """Count only complete validated rows while fingerprinting the original body."""

    dates = [datetime.strptime(row[0], "%Y%m%d").date() for row in rows]
    return {
        "rows": len(rows),
        "row_widths": sorted({len(row) for row in rows}),
        "min_source_date": min(dates).isoformat(),
        "max_source_date": max(dates).isoformat(),
        "rows_with_serve_points": sum(bool(row[23]) for row in rows),
        "rows_with_chart_reference": sum(bool(row[40]) for row in rows),
        "date_semantics": "tournament_date_not_verified_match_date",
        "model_integration_status": "quarantined_pending_date_and_schema_contract",
        "sha256": hashlib.sha256(content).hexdigest(),
    }


@dataclass(frozen=True)
class PlayerHistory:
    """Acquisition evidence, deliberately not a dated Elo-event contract."""

    report: dict[str, object]
    header: list[str]
    rows: list[list[str]]
    documents: tuple[tuple[str, bytes], ...]
    quarantined_rows: list[tuple[int, list[str]]] = field(default_factory=list)


def fetch_player_history(
    client: TennisAbstractAcquisitionClient,
    gender: str,
    key: str,
    *,
    quarantine_incomplete: bool = False,
) -> PlayerHistory:
    """Acquire a validated profile matrix and preserve the exact source documents."""

    if gender not in {"M", "F"} or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", key) is None:
        raise ValueError("Expected M/F and an inspected Tennis Abstract player key.")
    page_name = "player" if gender == "M" else "wplayer"
    url = f"{ORIGIN}/cgi-bin/{page_name}-classic.cgi?p={key}"
    response = client.get(url)
    documents = [(url, response.content)]
    body = response.content.decode("utf-8", errors="strict")
    header = literal_array(body, "matchhead")
    if header is None or header[:44] != HISTORY_HEADER:
        raise ValueError("Tennis Abstract history header changed.")
    data_url = url
    content = response.content
    if literal_array(body, "matchmx") is None:
        expected = f"{ORIGIN}/jsmatches/{key}.js"
        links = {
            urljoin(url, str(script.get("src")))
            for script in BeautifulSoup(body, "html.parser").find_all("script", src=True)
        }
        if expected not in links:
            raise ValueError("No explicitly linked singles history data found.")
        data_url = expected
        content = client.get(data_url).content
        documents.append((data_url, content))
    rows, incomplete = partition_history(content, quarantine_incomplete=quarantine_incomplete)
    result = _summarize_rows(rows, content)
    result.update(
        {
            "gender": gender,
            "player_key": key,
            "source_url": data_url,
            "header_columns": len(header),
            "profile_url": url,
            "source_rows": len(rows) + len(incomplete),
            "quarantined_rows": len(incomplete),
            "quarantined_widths": sorted({len(row) for _, row in incomplete}),
        }
    )
    return PlayerHistory(result, header, rows, tuple(documents), incomplete)


def audit_player(
    client: TennisAbstractAcquisitionClient, gender: str, key: str
) -> dict[str, object]:
    """Report coverage through the same parser used by daily acquisition."""

    return fetch_player_history(client, gender, key).report
