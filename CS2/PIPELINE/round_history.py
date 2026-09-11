"""Parse HLTV map round histories without network access or statistical inference.

Contract v1: validate native team IDs, explicit half separators, every cumulative
score, side consistency and the final score. Unknown/incomplete HTML is rejected.
"""

from __future__ import annotations

from datetime import UTC, datetime
import re
from typing import Any
from urllib.parse import urlsplit

from parsel import Selector

PARSER_VERSION = 1
OUTCOMES = {
    "ct_win.svg": "ct",
    "bomb_defused.svg": "ct",
    "stopwatch.svg": "ct",
    "t_win.svg": "t",
    "bomb_exploded.svg": "t",
}


def _required(pattern: str, value: str | None, label: str) -> str:
    """Extract an explicitly published identifier, never a name-based guess."""
    found = re.search(pattern, value or "")
    if not found:
        raise ValueError(f"missing_{label}")
    return found.group(1)


def _segments(row: Selector) -> list[list[dict[str, str]]]:
    """Read explicit half separators and exclude team logos/theme duplicates."""
    segments: list[list[dict[str, str]]] = []
    for child in row.xpath("./*"):
        classes = child.attrib.get("class", "").split()
        if "round-history-bar" in classes:
            segments.append([])
        elif "round-history-outcome" in classes:
            if not segments:
                raise ValueError("missing_half_separator")
            segments[-1].append(
                {
                    "icon": urlsplit(child.attrib.get("src", "")).path.rsplit("/", 1)[-1],
                    "score": child.attrib.get("title", ""),
                }
            )
    return segments


def parse_round_history(html: str, source_url: str) -> dict[str, Any]:
    """Return validated rounds including OT; only regulation half starts are pistols."""
    parsed_url = urlsplit(source_url)
    if parsed_url.hostname != "www.hltv.org" or parsed_url.scheme != "https":
        raise ValueError("invalid_hltv_source")
    mapstats_id = _required(r"/stats/matches/mapstatsid/(\d+)/", parsed_url.path, "mapstats_id")
    page = Selector(text=html)
    box = page.css(".match-info-box")
    if len(box) != 1:
        raise ValueError("missing_match_info")
    teams: list[dict[str, Any]] = []
    for side in ("left", "right"):
        node = box.css(f".team-{side}")
        anchor = node.css("a[href*='/stats/teams/']")
        teams.append(
            {
                "id": _required(r"/stats/teams/(\d+)/", anchor.attrib.get("href"), "team_id"),
                "name": " ".join(anchor.css("::text").getall()).strip(),
                "score": int(node.css(".bold::text").get() or "-1"),
            }
        )
    if teams[0]["id"] == teams[1]["id"] or any(t["score"] < 0 for t in teams):
        raise ValueError("invalid_teams_or_score")
    match_id = _required(r"/matches/(\d+)/", page.css("a.match-page-link::attr(href)").get(), "match_id")
    timestamp = box.css("span[data-unix]::attr(data-unix)").get()
    if not timestamp:
        raise ValueError("missing_match_date")
    played_at = datetime.fromtimestamp(int(timestamp) / 1000, UTC).isoformat()
    map_names = [t.strip() for t in box.xpath("./text()").getall() if t.strip()]
    if len(map_names) != 1:
        raise ValueError("ambiguous_map_name")
    containers = page.css(".round-history-con")
    if not containers:
        raise ValueError("missing_round_history")
    totals = [0, 0]
    rounds: list[dict[str, Any]] = []
    regulation_half_length: int | None = None
    half_offset = 0
    for container_index, container in enumerate(containers):
        overtime = "round-history-overtime" in container.attrib.get("class", "").split()
        if overtime != (container_index > 0):
            raise ValueError("ambiguous_regulation_or_overtime")
        rows = container.css(".round-history-team-row")
        if len(rows) != 2:
            raise ValueError("ambiguous_round_teams")
        for row, team in zip(rows, teams, strict=True):
            names = set(row.css("img.round-history-team::attr(title)").getall())
            if names != {team["name"]}:
                raise ValueError("round_team_orientation_mismatch")
        left, right = (_segments(row) for row in rows)
        if len(left) != len(right) or (not overtime and len(left) != 2) or (overtime and (not left or len(left) % 2)):
            raise ValueError("ambiguous_half_count")
        if not overtime:
            regulation_half_length = len(left[0])
            if regulation_half_length not in (12, 15):
                raise ValueError("unknown_round_format")
        half_sides: list[str] = []
        for half_index, (a, b) in enumerate(zip(left, right, strict=True)):
            if len(a) != len(b):
                raise ValueError("unequal_history_rows")
            if overtime and len(a) != 3:
                raise ValueError("unknown_overtime_format")
            if not overtime and half_index == 0 and len(a) != regulation_half_length:
                raise ValueError("invalid_first_half")
            current_side: str | None = None
            padding_started = False
            for offset, pair in enumerate(zip(a, b, strict=True)):
                occupied = [index for index, icon in enumerate(pair) if icon["icon"] != "emptyHistory.svg"]
                if not occupied:
                    if any(icon["score"] for icon in pair):
                        raise ValueError("empty_round_with_score")
                    padding_started = True
                    continue
                if padding_started or len(occupied) != 1:
                    raise ValueError("invalid_round_gap_or_two_winners")
                winner = occupied[0]
                outcome = pair[winner]
                winner_side = OUTCOMES.get(outcome["icon"])
                if winner_side is None:
                    raise ValueError(f"unknown_outcome:{outcome['icon']}")
                side_left = winner_side if winner == 0 else ("t" if winner_side == "ct" else "ct")
                if current_side is not None and current_side != side_left:
                    raise ValueError("side_changed_inside_half")
                current_side = side_left
                totals[winner] += 1
                if outcome["score"] != f"{totals[0]}-{totals[1]}":
                    raise ValueError("cumulative_score_mismatch")
                rounds.append(
                    {
                        "round_number": len(rounds) + 1,
                        "half": half_offset + half_index + 1,
                        "round_in_half": offset + 1,
                        "overtime": overtime,
                        "is_pistol": not overtime and offset == 0,
                        "winner_hltv_id": teams[winner]["id"],
                        "winner_side": winner_side,
                        "team_left_side": side_left,
                        "outcome": outcome["icon"],
                    }
                )
            if current_side is not None:
                half_sides.append(current_side)
        for pair_index in range(0, len(half_sides) - 1, 2):
            if half_sides[pair_index] == half_sides[pair_index + 1]:
                raise ValueError("missing_side_swap")
        half_offset += len(left)
    if totals != [team["score"] for team in teams]:
        raise ValueError("final_score_mismatch")
    pistols = [r["round_number"] for r in rounds if r["is_pistol"]]
    if pistols != [1, int(regulation_half_length or 0) + 1]:
        raise ValueError("missing_regulation_pistols")
    if max(totals) < int(regulation_half_length or 0) + 1 or totals[0] == totals[1]:
        raise ValueError("unfinished_map")
    return {
        "parser_version": PARSER_VERSION,
        "mapstats_id": mapstats_id,
        "hltv_match_id": match_id,
        "map_name": map_names[0],
        "played_at_utc": played_at,
        "format": f"mr{regulation_half_length}",
        "teams": teams,
        "rounds": rounds,
    }
