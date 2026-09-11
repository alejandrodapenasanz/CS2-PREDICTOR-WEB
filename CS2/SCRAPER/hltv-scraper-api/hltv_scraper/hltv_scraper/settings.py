# Scrapy settings for hltv_scraper project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "hltv_scraper"

SPIDER_MODULES = ["hltv_scraper.spiders"]
NEWSPIDER_MODULE = "hltv_scraper.spiders"


# User-Agent: lo rota RotateUserAgentMiddleware (UAs de escritorio realistas).
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Obey robots.txt rules
ROBOTSTXT_OBEY = False

# --- Anti-bloqueo y bajo volumen (PROJECT.md §4.5.2/§4.5.3) ----------------
# Concurrencia baja + ritmo humano para no parecer un bot ni abusar.
CONCURRENT_REQUESTS = 2
CONCURRENT_REQUESTS_PER_DOMAIN = 1
DOWNLOAD_DELAY = 7  # retardo base (segundos)
RANDOMIZE_DOWNLOAD_DELAY = True  # ±50% de jitter
DOWNLOAD_TIMEOUT = 60

# Reintentos: Cloudflare devuelve 403/429/503. El BackoffRetryMiddleware hace
# espera exponencial; además dejamos el retry estándar activo.
RETRY_ENABLED = True
RETRY_TIMES = 6
RETRY_HTTP_CODES = [403, 429, 500, 502, 503, 504, 522, 524]
# Parámetros del backoff exponencial propio (ver middlewares.py):
BLOCK_RETRY_TIMES = 8
BLOCK_RETRY_BASE_DELAY = 3.0
BLOCK_RETRY_MAX_DELAY = 240.0
# The download delay setting will honor only one of:
# CONCURRENT_REQUESTS_PER_DOMAIN = 16
# CONCURRENT_REQUESTS_PER_IP = 16

# Disable cookies (enabled by default)
# COOKIES_ENABLED = False

# Disable Telnet Console (enabled by default)
# TELNETCONSOLE_ENABLED = False

# Override the default request headers:
# DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
# }

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
# SPIDER_MIDDLEWARES = {
#    "hltv_scraper.middlewares.HltvScraperSpiderMiddleware": 543,
# }

# Downloader middlewares: rotación de UA, backoff ante bloqueo y proxy opcional.
# (Se desactiva el UserAgentMiddleware por defecto para que rote el nuestro.)
DOWNLOADER_MIDDLEWARES = {
    "scrapy.downloadermiddlewares.useragent.UserAgentMiddleware": None,
    "hltv_scraper.middlewares.RotateUserAgentMiddleware": 400,
    "hltv_scraper.middlewares.CloudflareSessionMiddleware": 405,
    "hltv_scraper.middlewares.ProxyMiddleware": 410,
    "hltv_scraper.middlewares.BackoffRetryMiddleware": 550,
}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
# EXTENSIONS = {
#    "scrapy.extensions.telnet.TelnetConsole": None,
# }

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
# ITEM_PIPELINES = {
#    "hltv_scraper.pipelines.HltvScraperPipeline": 300,
# }

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 5
AUTOTHROTTLE_MAX_DELAY = 180
AUTOTHROTTLE_TARGET_CONCURRENCY = 0.5  # por debajo de 1 petición en vuelo de media
AUTOTHROTTLE_DEBUG = False

# --- Caché HTTP (clave: NO re-scrapear nunca lo ya descargado) -------------
# El histórico de un partido terminado es inmutable; con la caché activada un
# re-run no vuelve a pedir páginas ya bajadas. PROJECT.md §4.5.3.
HTTPCACHE_ENABLED = True
HTTPCACHE_EXPIRATION_SECS = 0  # 0 = nunca expira (histórico inmutable)
HTTPCACHE_DIR = "httpcache"
HTTPCACHE_IGNORE_HTTP_CODES = [403, 429, 500, 502, 503, 504]  # no cachear bloqueos/errores
HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"
HTTPCACHE_POLICY = "scrapy.extensions.httpcache.DummyPolicy"  # sirve de caché siempre que exista

# Set settings whose default value is deprecated to a future-proof value
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"
FEED_EXPORT_ENCODING = "utf-8"
