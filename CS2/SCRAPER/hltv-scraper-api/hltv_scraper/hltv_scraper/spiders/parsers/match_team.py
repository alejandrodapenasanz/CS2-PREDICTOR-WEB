from typing import Any
from .parser import Parser


class MatchTeamParser(Parser):
    @staticmethod
    def parse(teams_box, number: int) -> dict[str, Any]:
        link = teams_box.css(
            f"div.team{number}-gradient a::attr(href), a.dropdownTeam.team{number}::attr(href)"
        ).get()
        return {
            "name": teams_box.css(f"div.team{number}-gradient .teamName::text").get(),
            "id": link.split("/")[2] if link and link.startswith("/team/") else None,
            "link": link,
            "logo": teams_box.css(f"div.team{number}-gradient img::attr(src)").get(),
            "score": teams_box.css(
                f".team{number}-gradient > div:nth-child(2)::text"
            ).get(),
        }
