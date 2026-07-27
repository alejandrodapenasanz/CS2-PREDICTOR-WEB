from typing import Any, Generator
import scrapy
import cloudscraper
from scrapy.http.response.html import HtmlResponse

from .parsers.date import RankingDateFormatter
from .parsers import ParsersFactory as PF


class HltvValveRankingSpider(scrapy.Spider):
    name = "hltv_valve_ranking"
    allowed_domains = ["www.hltv.org"]

    def __init__(self, year=None, month=None, day=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if year != "" and month != "" and day != 0:
            self.start_urls = [f"https://www.hltv.org/valve-ranking/teams/{year}/{month}/{day}"]
        else:
            self.start_urls = ["https://www.hltv.org/valve-ranking/teams"]

    def start_requests(self):
        scraper = cloudscraper.create_scraper()
        for url in self.start_urls:
            for attempt in range(3):
                response_data = scraper.get(
                    url,
                    headers={
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://www.hltv.org/",
                    },
                )
                if response_data.status_code == 200 and "ranked-team" in response_data.text:
                    response = HtmlResponse(
                        url=url,
                        body=response_data.content,
                        encoding="utf-8",
                    )
                    yield from self.parse(response)
                    return
                self.logger.warning(
                    "Valve ranking fetch attempt %s returned status=%s length=%s",
                    attempt + 1,
                    response_data.status_code,
                    len(response_data.text),
                )

            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response) -> Generator[dict[str, Any], Any, None]:
        ranked_teams = response.css("div.ranked-team.standard-box")
        prev_ranking = response.css("div.ranking-prev-next a.pagination-prev::attr(href)").re_first(r"/valve-ranking/teams/(\d{4}/\w+/\d+)")
        next_ranking = response.css("div.ranking-prev-next a.pagination-next::attr(href)").re_first(r"/valve-ranking/teams/(\d{4}/\w+/\d+)")
        date_text = response.css("div.regional-ranking-header-text::text").get()
        parsed_date = RankingDateFormatter.format(date_text)

        data = {
            "date": parsed_date,
            "prev_ranking": prev_ranking if prev_ranking else None,
            "next_ranking": next_ranking if next_ranking else None,
            "ranking_type": "valve",
            "ranking": [],
        }

        for team in ranked_teams:
            data["ranking"].append(PF.get_parser("team_ranking").parse(team))
            
        yield data
