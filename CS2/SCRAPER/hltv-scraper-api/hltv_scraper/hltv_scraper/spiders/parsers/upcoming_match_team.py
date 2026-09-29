from typing import Any

from .parser import Parser


class UpcomingMatchTeamParser(Parser):
    @staticmethod
    def parse(match, number) -> dict[str, Any]:
        raw_id = match.css(f".match-wrapper::attr(team{number})").get() or ""
        team_id = raw_id if raw_id.isdigit() and int(raw_id) > 0 else None
        return {
            "id": team_id,
            "name": match.css(f"div.team{number} .match-teamname::text").get(),
            "logo": match.css(f"div.team{number} img::attr(src)").get(),
        }
