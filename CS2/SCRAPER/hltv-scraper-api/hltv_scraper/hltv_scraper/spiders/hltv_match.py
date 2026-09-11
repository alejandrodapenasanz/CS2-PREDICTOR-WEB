import json
from pathlib import Path
from typing import Any, Generator

import cloudscraper
import requests
import scrapy
from flask import Request
from scrapy.http.response.html import HtmlResponse

from .parsers import ParsersFactory as PF


class HltvMatchSpider(scrapy.Spider):
    name = "hltv_match"
    allowed_domains = ["www.hltv.org"]

    def __init__(self, match: str, **kwargs: Any) -> None:
        self.start_urls = [f"https://www.hltv.org/matches/{match}"]
        super().__init__(**kwargs)

    def _fetch_with_cf_session(self, url: str):
        session_file = Path(__file__).resolve().parents[2] / "cf_session.json"
        if not session_file.exists():
            return None
        try:
            session = json.loads(session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        headers = {
            "User-Agent": session.get("user_agent") or "Mozilla/5.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.hltv.org/",
        }
        response = requests.get(
            url,
            timeout=30,
            headers=headers,
            cookies={"cf_clearance": session.get("cf_clearance", "")},
        )
        if response.status_code == 200 and "Just a moment" not in response.text[:5000]:
            return response.content
        return None

    def start_requests(self) -> Generator[dict[str, None] | Request, Any, None]:
        scraper = cloudscraper.create_scraper()
        for url in self.start_urls:
            try:
                body = self._fetch_with_cf_session(url)
                if body is None:
                    response_data = scraper.get(url)
                    body = response_data.content
                response = HtmlResponse(url=url, body=body, encoding="utf-8")
                yield from self.parse(response)
            except Exception as e:
                self.logger.error(f"Error fetching {url}: {e}")
                yield scrapy.Request(
                    url=url,
                    callback=self.parse,
                )

    def parse(self, response) -> Generator[dict[str, None], Any, None]:
        teams_box = PF.get_parser("match_teams_box").parse(response.css(".teamsBox"))
        maps_score = PF.get_parser("map_holders").parse(response)
        player_stats = PF.get_parser("table_stats").parse(response.css("#all-content"))

        yield {"match": teams_box, "maps": maps_score, "stats": player_stats}
