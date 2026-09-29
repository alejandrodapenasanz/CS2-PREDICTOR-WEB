"""Bounded acquisition-only worker, executed by the scraper's Python 3.13.

Reuses the production HTTP client and parsers. Does not write SQLite, change
published runs, reuse historical availability timestamps or invoke training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def safe_url(value: str) -> str:
    from PIPELINE import start

    url = start.hltv_url(value)
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.netloc != "www.hltv.org"
        or not parts.path.startswith(
            (
                "/matches/",
                "/betting/analytics/",
                "/stats/",
                "/team/",
                "/player/",
                "/ranking/teams",
                "/valve-ranking/teams",
            )
        )
    ):
        raise ValueError("URL fuera de los endpoints HLTV permitidos")
    return url


def player_parser() -> Any:
    """Explicit acquisition adapter to the existing player parser, never the model."""
    path = ROOT / "SCRAPER/hltv-scraper-api/scripts/collect_player_compare_stats.py"
    spec = importlib.util.spec_from_file_location("hltv_recovery_player_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se encuentra el parser HLTV de jugadores")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def acquire(item: dict[str, Any], run_dir: Path) -> tuple[str, str]:
    from PIPELINE import start
    from parsel import Selector

    kind, key = item["entity_type"], item["entity_key"]

    def fetch(url: str, category: str, identifier: str) -> tuple[str, dict[str, Any]]:
        html = start.fetch_html(safe_url(url), max_attempts=2, min_interval=max(2.5, start.FETCH_MIN_INTERVAL))
        return html, start.save_raw_html(run_dir, category, identifier, url, html)

    url = item["url"]
    if kind.startswith("match_") and start.match_id_from_link(url) != str(key):
        raise ValueError("La URL no corresponde al ID del partido pendiente")
    if kind == "match_analytics":
        url = start.analytics_link_from_match(url)
        if not url:
            raise ValueError("No se puede resolver la URL de Analytics")
    if kind == "player_stats":
        parser = player_parser()
        player = item["player"]
        reference = parser.PlayerRef(**{k: str(player.get(k) or "") for k in ("id", "name", "slug", "link")})
        url = parser.player_stats_url(reference, "past3months")
    html, provenance = fetch(url, kind, key)
    captured = provenance["captured_at"]
    if kind == "match_detail":
        detail = start.parse_match_detail_html(html)
        match = (detail or {}).get("match") or {}
        if not all((match.get(side) or {}).get("name") for side in ("team1", "team2")):
            raise ValueError("HTML sin detalle de los dos participantes")
        record = item["record"]
        # Do not copy old predictions, odds or timestamps into a newly acquired snapshot.
        payload = {k: record.get(k) for k in ("date", "hour", "event", "format")}
        payload.update(
            id=key,
            link=item["url"],
            captured_at=captured,
            detail=detail,
            raw_html=provenance,
            status="completed" if start.is_completed_detail(detail) else "pending",
            source="maintenance_recovery",
            prematch_lineups=start.parse_prematch_lineups_html(html, detail),
        )
        start.write_json(run_dir / "match_snapshots" / f"{key}.json", payload)
        return "ok", "Detalle real recuperado; disponibilidad actual, sin reescribir predicciones."
    if kind == "match_analytics":
        record = item["record"]
        payload = start.parse_analytics_html(
            html, key, item["url"], start.team_names_for_analytics(record, record.get("detail"))
        )
        payload.update(captured_at=captured, raw_html=provenance)
        start.write_json(run_dir / "analytics" / f"{key}.json", payload)
        return (
            ("ok", "Analytics recuperado") if payload.get("available") else ("partial", "HTML sin Analytics utilizable")
        )
    if kind == "match_assets":
        links = start.extract_mapstats_links(html)
        payload = {
            "match_id": key,
            "match_link": url,
            "captured_at": captured,
            "veto": start.parse_veto_html(html),
            "mapstats_links": links,
            "mapstats": [],
            "raw_html": [provenance],
            "errors": [],
        }
        output = run_dir / "match_assets" / key / "assets.json"
        start.write_json(output, payload)
        for link in links:
            map_id = start.mapstats_id_from_link(link)
            map_html, meta = fetch(link, "mapstats", str(map_id))
            parsed = start.parse_mapstats_html(map_html, link)
            if not parsed.get("player_stats"):
                raise ValueError("Mapa sin estadísticas utilizables; se conserva lo recuperado")
            parsed["raw_html"] = meta
            payload["mapstats"].append(parsed)
            start.write_json(output, payload)
        return (
            ("ok", "Mapas recuperados") if links else ("partial", "HLTV no publica enlaces mapstats para este partido")
        )
    if kind == "player_stats":
        detail = parser.parse_player_profile_page(html, reference, url, "past3months")
        payload = {**player, **detail, "captured_at": captured, "raw_html": provenance, "fetch_origin": "online"}
        start.write_json(
            run_dir / f"player_compare_stats_{key}.json",
            {"year": int(captured[:4]), "captured_at": captured, "results": [{"players": [payload]}]},
        )
        if parser.player_stats_complete(payload.get("stats")):
            return "ok", "Stats de jugador recuperadas"
        return (
            ("not_found", "HLTV publica cero mapas; no se inventa muestra")
            if payload.get("maps") == 0
            else ("partial", "Stats aún incompletas")
        )
    if kind == "team_profile":
        if start.PF is None:
            raise RuntimeError("Falta el parser en el entorno del scraper")
        profile = start.PF.get_parser("team_profile").parse(Selector(text=html))
        if not profile or not profile.get("name"):
            raise ValueError("HTML sin perfil de equipo")
        path = run_dir / "team_profiles.json"
        profiles = start.read_json(path, [])
        profiles.append({"id": key, "source": {"link": url}, "profile": profile, "captured_at": captured})
        start.write_json(path, profiles)
        return "ok", "Perfil recuperado"
    if kind in {"ranking_hltv", "ranking_valve"}:
        payload = start.parse_ranking_html(html, kind.removeprefix("ranking_"))
        if not payload.get("ranking"):
            raise ValueError("HTML sin ranking")
        payload.update(captured_at=captured, raw_html=provenance)
        start.write_json(run_dir / "rankings" / f"{kind.removeprefix('ranking_')}.json", payload)
        return "ok", "Ranking recuperado"
    raise ValueError(f"Entidad no soportada: {kind}")


def run_plan(plan: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    from PIPELINE import start

    start.VERBOSE = True
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {"results": results, "stopped_reason": None, "unattempted": []}
    for index, item in enumerate(plan["selected"], 1):
        print(f"[recuperación {index}/{len(plan['selected'])}] {item['entity_type']} {item['entity_key']}", flush=True)
        result = {k: item[k] for k in ("entity_type", "entity_key")}
        result.update(attempted=True, captured_at=start.now_utc())
        http_before = start._FETCH_STATS["http_attempts"]
        try:
            status, note = acquire(item, run_dir)
            result.update(status=status, note=note, captured_at=start.now_utc())
        except KeyboardInterrupt:
            result.update(status="partial", note="Interrumpido; se conserva evidencia parcial")
            report["stopped_reason"] = "interrupted"
        except Exception as exc:
            blocked = start.looks_like_cf_or_waf_problem(exc) or isinstance(exc, start.FetchSuppressedError)
            result.update(status="blocked" if blocked else "error", note=str(exc)[:500])
            if blocked or isinstance(exc, start.FetchBudgetExceeded):
                report["stopped_reason"] = "budget" if isinstance(exc, start.FetchBudgetExceeded) else "blocked"
            if isinstance(exc, (start.FetchBudgetExceeded, start.FetchSuppressedError)):
                # Do not extend a cooldown if a global budget/quarantine
                # prevented any request. Partial map captures stay on disk.
                result["attempted"] = start._FETCH_STATS["http_attempts"] > http_before
        results.append(result)
        if report["stopped_reason"]:
            pending = plan["selected"][index if result["attempted"] else index - 1 :]
            report["unattempted"] = [
                {"entity_type": x["entity_type"], "entity_key": x["entity_key"], "reason": report["stopped_reason"]}
                for x in pending
            ]
        start.write_json(run_dir / "results.json", report)
        if report["stopped_reason"]:
            break
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker de recuperación HLTV (sin escrituras SQL).")
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Se requiere Python 3.13")
    report = run_plan(json.loads(args.plan.read_text(encoding="utf-8")), args.plan.parent)
    return 2 if report["stopped_reason"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
