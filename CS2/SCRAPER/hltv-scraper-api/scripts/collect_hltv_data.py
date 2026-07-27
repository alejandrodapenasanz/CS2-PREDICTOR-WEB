from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_ROOT = PROJECT_ROOT / "hltv_scraper"
DATA_ROOT = SCRAPY_ROOT / "data"


def scrapy_executable() -> Path:
    exe = PROJECT_ROOT / ".venv" / "Scripts" / "scrapy.exe"
    if exe.exists():
        return exe
    return Path("scrapy")


def load_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_spider(
    spider: str,
    output_path: Path,
    spider_args: list[str] | None = None,
    log_level: str = "ERROR",
) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(scrapy_executable()), "crawl", spider]
    if spider_args:
        cmd.extend(spider_args)
    cmd.extend(["-L", log_level, "-O", str(output_path)])

    process = subprocess.run(
        cmd,
        cwd=SCRAPY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return process.returncode == 0, process.stdout


def collect_upcoming(snapshot_dir: Path) -> dict[str, Any]:
    output = snapshot_dir / "upcoming_matches.json"
    ok, logs = run_spider("hltv_upcoming_matches", output)
    data = load_json(output) if ok else []
    return {"ok": ok, "count": len(data), "file": str(output), "logs": logs[-2000:]}


def collect_ranking(snapshot_dir: Path, spider: str, name: str) -> dict[str, Any]:
    output = snapshot_dir / "rankings" / f"{name}.json"
    ok, logs = run_spider(spider, output)
    data = load_json(output) if ok else []
    ranking_count = 0
    if isinstance(data, list) and data:
        ranking_count = len(data[0].get("ranking", []))
    return {
        "ok": ok,
        "count": ranking_count,
        "file": str(output),
        "logs": logs[-2000:],
    }


def collect_results(snapshot_dir: Path, pages: int, delay: float) -> dict[str, Any]:
    all_results: list[dict[str, Any]] = []
    result_pages: list[dict[str, Any]] = []
    empty_pages = 0

    for page in range(pages):
        offset = page * 100
        output = snapshot_dir / "results_pages" / f"results_offset_{offset}.json"
        ok, logs = run_spider(
            "hltv_results",
            output,
            spider_args=["-a", f"offset={offset}"],
        )
        page_data = load_json(output) if ok else []
        count = len(page_data) if isinstance(page_data, list) else 0
        result_pages.append(
            {
                "offset": offset,
                "ok": ok,
                "count": count,
                "file": str(output),
                "logs": logs[-1200:] if not ok else "",
            }
        )

        if not ok:
            break
        if count == 0:
            empty_pages += 1
            if empty_pages >= 2:
                break
        else:
            empty_pages = 0
            all_results.extend(page_data)

        if delay:
            time.sleep(delay)

    write_json(snapshot_dir / "results_all.json", all_results)
    return {
        "pages_requested": pages,
        "pages_collected": len(result_pages),
        "count": len(all_results),
        "file": str(snapshot_dir / "results_all.json"),
        "pages": result_pages,
    }


def collect_match_details(
    snapshot_dir: Path,
    matches: list[dict[str, Any]],
    folder_name: str,
    limit: int,
    delay: float,
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    detail_pages: list[dict[str, Any]] = []
    aggregate = snapshot_dir / f"{folder_name}.json"

    selected = [match for match in matches if match.get("link")][:limit]
    for match in selected:
        match_path = str(match["link"]).removeprefix("/matches/")
        match_id = match_path.split("/", 1)[0]
        output = snapshot_dir / folder_name / f"{match_id}.json"
        ok, logs = run_spider(
            "hltv_match",
            output,
            spider_args=["-a", f"match={match_path}"],
        )
        data = load_json(output) if ok else []
        item = data[0] if isinstance(data, list) and len(data) == 1 else data
        match_payload = item.get("match", {}) if isinstance(item, dict) else {}
        valid = ok and bool(match_payload.get("team1", {}).get("name")) and bool(
            match_payload.get("team2", {}).get("name")
        )
        if valid:
            details.append({"id": match_id, "source_match": match, "detail": item})
            write_json(aggregate, details)
        detail_pages.append(
            {
                "id": match_id,
                "ok": valid,
                "file": str(output),
                "logs": logs[-1200:] if not valid else "",
            }
        )
        if delay:
            time.sleep(delay)

    write_json(aggregate, details)
    return {
        "requested": len(selected),
        "count": len(details),
        "file": str(aggregate),
        "pages": detail_pages,
    }


def collect_team_profiles(
    snapshot_dir: Path,
    ranking_files: list[Path],
    match_detail_files: list[Path],
    limit: int,
    delay: float,
) -> dict[str, Any]:
    teams: dict[str, dict[str, Any]] = {}
    for ranking_file in ranking_files:
        data = load_json(ranking_file)
        if not isinstance(data, list) or not data:
            continue
        for team in data[0].get("ranking", []):
            team_id = team.get("id")
            link = team.get("link")
            if team_id and link:
                teams.setdefault(team_id, team)

    for detail_file in match_detail_files:
        data = load_json(detail_file)
        if not isinstance(data, list):
            continue
        for item in data:
            match = item.get("detail", {}).get("match", {})
            for key in ("team1", "team2"):
                team = match.get(key, {})
                team_id = team.get("id")
                link = team.get("link")
                if team_id and link:
                    teams.setdefault(
                        team_id,
                        {
                            "id": team_id,
                            "name": team.get("name"),
                            "link": link,
                            "logo": team.get("logo"),
                        },
                    )

    selected = list(teams.values())[:limit]
    profiles: list[dict[str, Any]] = []
    profile_pages: list[dict[str, Any]] = []
    aggregate = snapshot_dir / "team_profiles.json"
    for team in selected:
        team_id = team["id"]
        team_name = str(team.get("link", "")).rstrip("/").split("/")[-1]
        output = snapshot_dir / "team_profiles" / f"{team_id}_{team_name}.json"
        ok, logs = run_spider(
            "hltv_team",
            output,
            spider_args=["-a", f"team={team['link']}"],
        )
        data = load_json(output) if ok else []
        item = data[0] if isinstance(data, list) and len(data) == 1 else data
        valid = ok and isinstance(item, dict) and bool(item.get("name"))
        if valid:
            profiles.append({"id": team_id, "ranking_source": team, "profile": item})
            write_json(aggregate, profiles)
        profile_pages.append(
            {
                "id": team_id,
                "name": team.get("name"),
                "ok": valid,
                "file": str(output),
                "logs": logs[-1200:] if not valid else "",
            }
        )
        if delay:
            time.sleep(delay)

    write_json(aggregate, profiles)
    return {
        "requested": len(selected),
        "count": len(profiles),
        "file": str(aggregate),
        "pages": profile_pages,
    }


def collect_player_profiles(
    snapshot_dir: Path,
    team_profiles_file: Path,
    limit: int,
    delay: float,
) -> dict[str, Any]:
    team_profiles = load_json(team_profiles_file)
    players: dict[str, dict[str, Any]] = {}
    if isinstance(team_profiles, list):
        for team_profile in team_profiles:
            profile = team_profile.get("profile") or {}
            for player in profile.get("squad") or []:
                player_id = player.get("id")
                link = player.get("link")
                if player_id and link:
                    players.setdefault(player_id, player)

    selected = list(players.values())[:limit]
    profiles: list[dict[str, Any]] = []
    profile_pages: list[dict[str, Any]] = []
    aggregate = snapshot_dir / "player_profiles.json"
    for player in selected:
        player_id = player["id"]
        player_name = str(player.get("link", "")).rstrip("/").split("/")[-1]
        output = snapshot_dir / "player_profiles" / f"{player_id}_{player_name}.json"
        ok, logs = run_spider(
            "hltv_player",
            output,
            spider_args=["-a", f"profile={player['link']}"],
        )
        data = load_json(output) if ok else []
        item = data[0] if isinstance(data, list) and len(data) == 1 else data
        valid = ok and isinstance(item, dict) and bool(item.get("nick"))
        if valid:
            profiles.append({"id": player_id, "squad_source": player, "profile": item})
            write_json(aggregate, profiles)
        profile_pages.append(
            {
                "id": player_id,
                "name": player.get("name"),
                "ok": valid,
                "file": str(output),
                "logs": logs[-1200:] if not valid else "",
            }
        )
        if delay:
            time.sleep(delay)

    write_json(aggregate, profiles)
    return {
        "requested": len(selected),
        "count": len(profiles),
        "file": str(aggregate),
        "pages": profile_pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect raw HLTV data through the local Scrapy spiders."
    )
    parser.add_argument("--results-pages", type=int, default=10)
    parser.add_argument("--match-detail-limit", type=int, default=25)
    parser.add_argument("--team-profile-limit", type=int, default=30)
    parser.add_argument("--player-profile-limit", type=int, default=50)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--snapshot-name", default="")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_name = args.snapshot_name or timestamp
    snapshot_dir = DATA_ROOT / "raw" / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = snapshot_dir / "manifest.json"

    manifest: dict[str, Any] = {
        "snapshot": snapshot_name,
        "created_at_utc": timestamp,
        "source": "https://www.hltv.org/",
        "settings": vars(args),
        "collections": {},
    }
    write_json(manifest_path, manifest)

    upcoming_result = collect_upcoming(snapshot_dir)
    manifest["collections"]["upcoming"] = upcoming_result
    write_json(manifest_path, manifest)

    hltv_ranking = collect_ranking(snapshot_dir, "hltv_top30", "hltv_current")
    valve_ranking = collect_ranking(snapshot_dir, "hltv_valve_ranking", "valve_current")
    manifest["collections"]["rankings"] = {
        "hltv": hltv_ranking,
        "valve": valve_ranking,
    }
    write_json(manifest_path, manifest)

    results_result = collect_results(snapshot_dir, args.results_pages, args.delay)
    manifest["collections"]["results"] = results_result
    write_json(manifest_path, manifest)

    upcoming_matches = load_json(Path(upcoming_result["file"])) if upcoming_result["ok"] else []
    results_matches = load_json(Path(results_result["file"]))
    upcoming_details = collect_match_details(
        snapshot_dir,
        upcoming_matches,
        "upcoming_match_details",
        args.match_detail_limit,
        args.delay,
    )
    manifest["collections"]["upcoming_details"] = upcoming_details
    write_json(manifest_path, manifest)

    recent_result_details = collect_match_details(
        snapshot_dir,
        results_matches,
        "recent_result_details",
        args.match_detail_limit,
        args.delay,
    )
    manifest["collections"]["recent_result_details"] = recent_result_details
    write_json(manifest_path, manifest)

    team_profiles = collect_team_profiles(
        snapshot_dir,
        [Path(hltv_ranking["file"]), Path(valve_ranking["file"])],
        [Path(upcoming_details["file"]), Path(recent_result_details["file"])],
        args.team_profile_limit,
        args.delay,
    )
    manifest["collections"]["team_profiles"] = team_profiles
    write_json(manifest_path, manifest)

    manifest["collections"]["player_profiles"] = collect_player_profiles(
        snapshot_dir,
        Path(team_profiles["file"]),
        args.player_profile_limit,
        args.delay,
    )
    write_json(manifest_path, manifest)

    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nSnapshot saved to: {snapshot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
