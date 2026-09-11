from typing import Any

from .parser import Parser


class TeamSquadParser(Parser):
    @staticmethod
    def parse(response) -> list[dict[str, Any]]:
        players_container = response.css(".bodyshot-team.g-grid a.col-custom")
        squad = []
        for player in players_container:
            link = player.css("::attr(href)").get()
            squad.append(
                {
                    "name": player.css(".playerFlagName span.text-ellipsis::text").get(),
                    "id": link.split("/")[2] if link and link.startswith("/player/") else None,
                    "link": link,
                    "img": player.css("img.bodyshot-team-img::attr(src)").get(),
                    "nation": f"https://www.hltv.org{player.css('img.flag::attr(src)').get()}",
                }
            )
        return squad
