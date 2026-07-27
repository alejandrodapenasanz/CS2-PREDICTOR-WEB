@{
    # =========================================================================
    # Configuracion de la pipeline de start.ps1 (config-driven).
    # Fichero de DATOS de PowerShell: solo hashtables/arrays/literales. Se carga
    # con Import-PowerShellDataFile. Editar aqui NO requiere tocar start.ps1.
    # Override en runtime con:  .\start.ps1 -Config <ruta.psd1>
    # =========================================================================

    # Guardas del scraper HLTV. start.ps1 las publica como variables de entorno
    # con Set-DefaultEnv (no pisa lo que ya venga definido en el proceso).
    ScrapeGuards = @{
        HLTV_FETCH_MAX_ATTEMPTS          = '10'
        HLTV_FETCH_BASE_DELAY            = '5.0'
        HLTV_FETCH_MAX_DELAY             = '600.0'
        HLTV_FETCH_MIN_INTERVAL          = '2.5'
        HLTV_FETCH_CACHE_TTL             = '240.0'
        HLTV_BLOCK_COOLDOWN_BASE         = '90.0'
        HLTV_BLOCK_COOLDOWN_MAX          = '1200.0'
        HLTV_BLOCK_STREAK_THRESHOLD      = '3'
        HLTV_CIRCUIT_BREAKER_SLEEP       = '420.0'
        HLTV_FETCH_WARMUP                = '1'
        HLTV_URL_QUARANTINE_SECONDS      = '900.0'
        HLTV_MAX_HTTP_REQUESTS_PER_RUN   = '2500'
        HLTV_USE_SCRAPLING               = '1'
        HLTV_SOLVE_CLOUDFLARE            = '1'
        HLTV_IMPERSONATE                 = 'chrome'
        HLTV_STEALTH_HEADLESS            = '1'
        HLTV_STEALTH_TIMEOUT_MS          = '45000'
        HLTV_STEALTH_MAX_SOLVES_PER_RUN  = '6'
        HLTV_SCRAPLING_TIER1_ATTEMPTS    = '3'
        HLTV_AUTO_REFRESH_CF_ON_BLOCK    = '1'
        HLTV_CF_REFRESH_TIMEOUT_SECONDS  = '240'
        HLTV_PREFER_REQUESTS_AFTER_CF_SECONDS = '900'
        BBDD_TEAM_PROFILE_TTL_DAYS       = '7'
        BBDD_PLAYER_STATS_TTL_DAYS       = '3'
        BBDD_RANKING_TTL_DAYS            = '7'
        BBDD_ASSETS_BACKFILL_LIMIT       = '20'
    }

    # Dependencias que deben poder importarse en el Python del modelo.
    ModelImports = @('numpy', 'pandas', 'sklearn', 'scipy', 'matplotlib', 'lightgbm', 'shap')

    # Paquetes pip a instalar si faltan (nombres de distribucion).
    ModelPipPackages = @('numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'lightgbm', 'shap')

    # Dependencias base y stealth del venv del scraper.
    ScraperBaseImports    = @('scrapy', 'cloudscraper', 'parsel', 'requests')
    ScraperStealthImports = @('scrapling', 'curl_cffi')

    # Codigos de salida por clase de fallo (para diagnostico de CI/automatizacion).
    ExitCodes = @{
        Generic      = 1
        Dependencies = 2
        Scrape       = 3
        IngestPre    = 4
        Train        = 5
        Enrich       = 6
        Context      = 7
        DbPost       = 8
        Drift        = 9
        Blackbox     = 10
        Web          = 11
    }
}
