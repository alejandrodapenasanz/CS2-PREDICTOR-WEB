"""Parse dated HLTV/VRS ranking badges, without estimating ratings or points."""

from __future__ import annotations

import calendar
from datetime import date
import re
from urllib.parse import parse_qs, urljoin, urlsplit

from parsel import Selector

MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}


def parse_ranking_badges(html: str) -> list[dict[str, str | int]]:
    """Require explicit rank, dated official URL and an unambiguous HLTV team ID.

    A match badge can refer to an older ranking edition. Its publication date
    must not be replaced with the capture date or interpreted as today's rank.
    """
    result: dict[tuple[str, str], dict[str, str | int]] = {}
    conflicts: set[tuple[str, str]] = set()
    for node in Selector(text=html).css("a.ranking-badge"):
        link = urljoin("https://www.hltv.org", node.attrib.get("href", ""))
        url = urlsplit(link)
        if url.scheme != "https" or url.netloc != "www.hltv.org":
            continue
        path = re.fullmatch(r"/(ranking|valve-ranking)/teams/(\d{4})/([a-z]+)/([0-9]{1,2})(?:/([0-9]+))?", url.path)
        if path is None:
            continue
        rank_type = "hltv" if path[1] == "ranking" else "valve"
        label = "HLTV" if rank_type == "hltv" else "VRS"
        text = " ".join(node.xpath(".//text()").getall())
        rank = re.fullmatch(rf"\s*{label}\s*:\s*#\s*([1-9][0-9]*)\s*", text)
        ids = parse_qs(url.query).get("teamId", [])
        if path[5]:
            ids.append(path[5])
        if rank is None or not ids or len(set(ids)) != 1 or not ids[0].isdigit():
            continue
        try:
            published = date(int(path[2]), MONTHS[path[3]], int(path[4])).isoformat()
        except (ValueError, KeyError):
            continue
        row: dict[str, str | int] = {
            "hltv_team_id": ids[0],
            "ranking_type": rank_type,
            "position": int(rank[1]),
            "ranking_date": published,
            "ranking_url": link,
        }
        key = (ids[0], rank_type)
        if key in result and result[key] != row:
            conflicts.add(key)
        result[key] = row
    return [row for key, row in sorted(result.items()) if key not in conflicts]
