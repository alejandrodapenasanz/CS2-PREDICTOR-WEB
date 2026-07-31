# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

import os
import random
import time
import json
from pathlib import Path

from scrapy import signals
from scrapy.downloadermiddlewares.retry import get_retry_request

# useful for handling different item types with a single interface
from itemadapter import is_item, ItemAdapter


# ===========================================================================
# Anti-bloqueo (PROJECT.md §4.5.2): rotación de User-Agent realista, backoff
# exponencial ante 403/429/503 (Cloudflare), y proxy opcional por entorno.
# ===========================================================================

DESKTOP_USER_AGENTS = [
    # Chrome / Edge / Firefox de escritorio recientes y verosímiles.
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


class RotateUserAgentMiddleware:
    """Asigna un User-Agent de escritorio aleatorio a cada petición."""

    def process_request(self, request, spider):
        request.headers["User-Agent"] = random.choice(DESKTOP_USER_AGENTS)
        # Cabeceras de navegador habituales (ayudan con Cloudflare).
        request.headers.setdefault("Accept-Language", "en-US,en;q=0.9")
        request.headers.setdefault(
            "Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        )
        return None


class CloudflareSessionMiddleware:
    """Aplica cf_clearance guardado por grab_cf.py a las peticiones HLTV."""

    def __init__(self):
        self.session_file = Path(__file__).resolve().parents[1] / "cf_session.json"
        self.cf_clearance = None
        self.user_agent = None
        self._load_session()

    def _load_session(self):
        if not self.session_file.exists():
            return
        try:
            payload = json.loads(self.session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self.cf_clearance = payload.get("cf_clearance")
        self.user_agent = payload.get("user_agent")

    def process_request(self, request, spider):
        if "hltv.org" not in request.url:
            return None
        if self.user_agent:
            request.headers["User-Agent"] = self.user_agent
        if self.cf_clearance:
            request.headers["Cookie"] = f"cf_clearance={self.cf_clearance}"
        request.headers.setdefault("Referer", "https://www.hltv.org/")
        return None


class BackoffRetryMiddleware:
    """Reintenta 403/429/503 con backoff exponencial (Cloudflare-friendly).

    Honra `Retry-After` cuando viene. El retardo crece 2^intento (con jitter)
    en lugar de martillear el servidor, tal como exige PROJECT.md §4.5.2.
    """

    BLOCK_CODES = {403, 429, 503}

    @classmethod
    def from_crawler(cls, crawler):
        s = cls()
        s.max_retries = crawler.settings.getint("BLOCK_RETRY_TIMES", 5)
        s.base_delay = crawler.settings.getfloat("BLOCK_RETRY_BASE_DELAY", 3.0)
        s.max_delay = crawler.settings.getfloat("BLOCK_RETRY_MAX_DELAY", 90.0)
        return s

    def process_response(self, request, response, spider):
        if response.status in self.BLOCK_CODES:
            retry_after = response.headers.get("Retry-After")
            attempt = request.meta.get("block_retry", 0) + 1
            if retry_after:
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = self.base_delay * (2 ** attempt)
            else:
                delay = self.base_delay * (2 ** attempt)
            delay = min(delay, self.max_delay) + random.uniform(0, 1.5)
            spider.logger.warning(
                f"[backoff] {response.status} en {request.url} -> espera {delay:.1f}s (intento {attempt})"
            )
            time.sleep(delay)  # simple y suficiente a bajo volumen
            new = get_retry_request(
                request.replace(meta={**request.meta, "block_retry": attempt}),
                spider=spider, reason=f"http_{response.status}", max_retry_times=self.max_retries,
            )
            return new or response
        return response


class ProxyMiddleware:
    """Proxy opcional vía variable de entorno HLTV_PROXY (para backfill grande)."""

    def __init__(self):
        self.proxy = os.environ.get("HLTV_PROXY")

    def process_request(self, request, spider):
        if self.proxy and "proxy" not in request.meta:
            request.meta["proxy"] = self.proxy
        return None


class HltvScraperSpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)


class HltvScraperDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info("Spider opened: %s" % spider.name)
