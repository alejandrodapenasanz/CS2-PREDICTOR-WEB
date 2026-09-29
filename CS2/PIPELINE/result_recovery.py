"""Daily bounded results acquisition using the existing HLTV transport.

One match page normally supplies all map scores; no mandatory per-map player
statistics downloads. Writes are delegated to BBDD. No model or feature writes.
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import re
import sqlite3
from typing import Any

from parsel import Selector


def parse_result_html(html: str, item: dict[str, Any], obtained_at: str, *, client: Any = None) -> dict[str, Any]:
    """Parse explicit terminal state, participants and map scores, fail closed."""
    if client is None:
        from PIPELINE import start
    else:
        start = client

    selector = Selector(text=html)
    detail = start.parse_match_detail_html(html) or {}
    match = detail.get("match") or {}
    first, second = match.get("team1") or {}, match.get("team2") or {}
    state_text = " ".join(selector.css(".countdown::text,.status::text,.match-status::text").getall()).lower()
    status = "unknown"
    for token, candidate in (
        ("postponed", "postponed"),
        ("cancelled", "cancelled"),
        ("canceled", "cancelled"),
        ("walkover", "walkover"),
        ("match over", "finished"),
        ("finished", "finished"),
        ("live", "live"),
    ):
        if token in state_text:
            status = candidate
            break
    a, b = start.parse_int(first.get("score")), start.parse_int(second.get("score"))
    # Numeric live score alone never proves the match ended.
    if status == "finished" and (a is None or b is None or a == b):
        raise ValueError("Terminal HTML has no coherent final score")
    maps = []
    for number, node in enumerate(selector.css(".mapholder"), 1):
        name = node.css(".mapname::text").get()
        left = start.parse_int(node.css(".results-left .results-team-score::text").get())
        right = start.parse_int(node.css(".results-right .results-team-score::text").get())
        if left is None or right is None or left == right:
            continue
        # HLTV's results-left/right names provide an independent orientation check.
        left_id = node.css('.results-left a[href*="/team/"]::attr(href)').get()
        right_id = node.css('.results-right a[href*="/team/"]::attr(href)').get()
        if left_id and right_id:
            ids = [v.split("/team/", 1)[1].split("/", 1)[0] for v in (left_id, right_id)]
            expected = [str(first.get("id")), str(second.get("id"))]
            if ids == expected[::-1]:
                left, right = right, left
            elif ids != expected:
                raise ValueError("Map participants differ from series participants")
        maps.append(
            {
                "map_number": number,
                "map_name": name.strip() if name else None,
                "rounds_t1": left,
                "rounds_t2": right,
                # Only an explicit OT label establishes overtime. No duration inference.
                "overtime": 1 if re.search(r"\b(?:OT|overtime)\b", node.xpath("string(.)").get() or "", re.I) else None,
            }
        )
    bo_match = re.search(r"Best of\s+([135])", selector.xpath("string(.)").get() or "", re.I)
    best_of = int(bo_match[1]) if bo_match else None
    if (
        best_of == 1
        and status == "finished"
        and len(maps) == 1
        and (a, b) == (maps[0]["rounds_t1"], maps[0]["rounds_t2"])
    ):
        a, b = (1, 0) if a > b else (0, 1)
    return {
        "hltv_match_id": str(item["entity_key"]),
        "source_url": start.hltv_url(item["url"]),
        "obtained_at_utc": obtained_at,
        "team1_hltv_id": first.get("id"),
        "team2_hltv_id": second.get("id"),
        "status": status,
        "score_t1": a if status == "finished" else None,
        "score_t2": b if status == "finished" else None,
        "best_of": best_of,
        "maps": maps if status == "finished" else [],
        "parser_version": 1,
    }


def run_daily(
    database: Path, master: dict[str, Any], run_dir: Path, *, limit: int = 20, client: Any = None
) -> dict[str, Any]:
    """Recover a bounded queue; respect production cooldowns and isolate failures."""
    from BBDD import result_store

    if client is None:
        from PIPELINE import start
    else:
        start = client

    report: dict[str, Any] = {
        "reviewed": 0,
        "new_results": 0,
        "enriched_results": 0,
        "maps_added": 0,
        "errors": 0,
        "stopped_reason": None,
    }
    if limit <= 0:
        return {**report, "stopped_reason": "disabled"}
    if not database.exists():
        return {**report, "stopped_reason": "database_not_initialized"}
    with closing(sqlite3.connect(database, timeout=30)) as conn:
        result_store.ensure_schema(conn)
        conn.commit()
        work = result_store.plan(conn, master, now=start.now_utc(), limit=limit)
        report.update(eligible=work["eligible"], without_url=work["without_url"])
        for item in work["selected"]:
            key = item["entity_key"]
            report["reviewed"] += 1
            now = start.now_utc()
            error = None
            try:
                html = start.fetch_html(item["url"], max_attempts=2, min_interval=max(2.5, start.FETCH_MIN_INTERVAL))
                meta = start.save_raw_html(run_dir, "result_recovery", key, item["url"], html)
                payload = parse_result_html(html, item, meta["captured_at"], client=start)
                start.write_json(run_dir / "result_recovery" / f"{key}.json", payload)
                with conn:
                    delta = result_store.persist(conn, payload)
                for field, value in delta.items():
                    report[field] += value
            except Exception as exc:
                error = str(exc)[:500]
                report["errors"] += 1
                start.log(f"result recovery {key}: {error}", force=True)
                if start.looks_like_cf_or_waf_problem(exc) or isinstance(
                    exc, (start.FetchBudgetExceeded, start.FetchSuppressedError)
                ):
                    report["stopped_reason"] = "source_blocked_or_budget"
            with conn:
                result_store.record_attempt(conn, item["record"]["match_id"], now, error)
            if report["stopped_reason"]:
                break
        report["coverage"] = result_store.coverage(conn, start.now_utc())
    start.write_json(run_dir / "result_recovery_summary.json", report)
    print("[CS2 resultados] " + json.dumps(report, ensure_ascii=False), flush=True)
    return report
