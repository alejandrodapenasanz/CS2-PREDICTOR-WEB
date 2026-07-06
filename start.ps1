<#
  CS2 Predictor - online pipeline de una sola orden.

  Por defecto (.\start.ps1) ejecuta el flujo online:
     0) Resuelve Python de modelo y Python del scraper.
     1) Instala/repara el venv del scraper si hace falta.
     2) Scrapea HLTV en vivo, actualiza pendientes y reintenta huecos recientes
        de odds/detalle/Analytics. Si el scrape no produce datos, la pipeline falla.
     3) Entrena el modelo si falta el artefacto (o si se pasa -Retrain), ya
        con el master actualizado.
     4) Enriquece predicciones con el modelo calibrado, odds y flags.
     5) Analiza si el contexto HLTV ayuda a calibrar el modelo.
     6) Reconstruye la BBDD SQLite.
     7) Genera WEB\data.js para el dashboard.

  Flags:
     -SkipScrape             No scrapear; usa el ultimo run existente.
     -AllowOfflineFallback   Si el scrape falla, continuar con el ultimo run.
     -Retrain                Forzar reentrenamiento del modelo.
     -NoDb                   No reconstruir la BBDD.
     -MaxMatches N           Limitar numero de partidos a scrapear (debug; no publica master).
     -PlayerDelay S          Retardo entre peticiones de stats de jugador.
     -SkipPlayerStats        No scrapear stats de jugadores.
     -SkipTeamProfiles       No scrapear perfiles de equipo.
     -SkipMatchAssets        No capturar veto/mapstats de completados.
     -MatchAssetsLimit N     Maximo de partidos completados a backfillear (0=todos).
     -MatchAssetsDelay S     Retardo entre peticiones de veto/mapstats.
     -SkipAnalytics          No capturar HLTV betting analytics de upcoming/pendientes.
     -SkipRankings           No capturar ranking HLTV/Valve actual.
     -SkipWarmup             No hacer warm-up inicial de sesion HLTV.
     -SkipSameDayRecovery    No reintentar huecos recientes ya conocidos.
     -RecoveryWindowDays N   Dias hacia atras para recuperar odds/detalle/analytics.
     -RecoveryDelay S        Retardo entre peticiones de recuperacion.
     -RecreateScraperVenv    Recrear el venv del scraper antes de ejecutar.
     -Quiet                  Reduce logs internos del scraper.
#>
param(
    [switch]$SkipScrape,
    [switch]$AllowOfflineFallback,
    [switch]$Retrain,
    [switch]$NoDb,
    [int]$MaxMatches = 0,
    [double]$PlayerDelay = 1.5,
    [switch]$SkipPlayerStats,
    [switch]$SkipTeamProfiles,
    [switch]$SkipMatchAssets,
    [int]$MatchAssetsLimit = 50,
    [double]$MatchAssetsDelay = 1.5,
    [switch]$SkipAnalytics,
    [switch]$SkipRankings,
    [switch]$SkipWarmup,
    [switch]$SkipSameDayRecovery,
    [int]$RecoveryWindowDays = 2,
    [double]$RecoveryDelay = 1.5,
    [switch]$RecreateScraperVenv,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

function Step($n, $total, $msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$ts] [$n/$total] $msg" -ForegroundColor Cyan
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$Description = ""
    )
    & $FilePath @Arguments
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        if ($Description) {
            throw "$Description fallo con exit code $exit."
        }
        throw "Comando fallo con exit code ${exit}: $FilePath $($Arguments -join ' ')"
    }
}

function Get-SystemPython {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "No se encontro Python en PATH."
    }
    return $cmd.Source
}

function Get-ScraperBasePython {
    # Scrapling (navegador stealth) soporta Python 3.10-3.13, NO 3.14.
    # Preferimos 3.13/3.12/3.11 via el 'py' launcher; si no hay, caemos al
    # Python del sistema (scrapling stealth podria no instalar; se degrada).
    foreach ($v in @("3.13", "3.12", "3.11", "3.10")) {
        try {
            $out = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
        } catch { }
    }
    Write-Host "AVISO: no se encontro Python 3.10-3.13; uso el del sistema. El navegador stealth de Scrapling podria no instalar (se usara solo el tier HTTP / requests)." -ForegroundColor Yellow
    return (Get-SystemPython)
}

function Ensure-CaBundle {
    # En redes con inspeccion TLS (proxy corporativo con CA propia), curl_cffi y
    # requests fallan la verificacion. Exportamos el trust store de Windows a un
    # bundle y lo publicamos por variables de entorno. En un PC sin restricciones
    # no hace falta: si el probe pasa, no se genera nada.
    param([string]$ScraperDir)
    $bundle = Join-Path $ScraperDir "corp_ca_bundle.pem"

    # Si ya existe un bundle, publicalo y termina.
    if (Test-Path $bundle) {
        Set-Item -Path "Env:HLTV_CA_BUNDLE" -Value $bundle
        Set-Item -Path "Env:CURL_CA_BUNDLE" -Value $bundle
        Set-Item -Path "Env:SSL_CERT_FILE" -Value $bundle
        Set-Item -Path "Env:REQUESTS_CA_BUNDLE" -Value $bundle
        Write-Host ("CA bundle en uso: " + $bundle) -ForegroundColor DarkGray
        return
    }

    # Probe TLS: si la verificacion por defecto funciona, no hacemos nada.
    $probe = 'try:
    import urllib.request, ssl
    urllib.request.urlopen("https://www.hltv.org/robots.txt", timeout=15)
    print("TLS_OK")
except ssl.SSLCertVerificationError:
    print("TLS_MITM")
except Exception:
    print("TLS_OK")'
    $result = & (Get-SystemPython) -c $probe 2>$null
    if ($result -match "TLS_MITM") {
        Write-Host "Inspeccion TLS detectada; exporto el trust store de Windows a corp_ca_bundle.pem..." -ForegroundColor Yellow
        $stores = @('Cert:\LocalMachine\Root','Cert:\CurrentUser\Root','Cert:\LocalMachine\CA','Cert:\CurrentUser\CA')
        $seen = @{}
        $sb = New-Object System.Text.StringBuilder
        foreach ($s in $stores) {
            Get-ChildItem $s -ErrorAction SilentlyContinue | ForEach-Object {
                if (-not $seen.ContainsKey($_.Thumbprint)) {
                    $seen[$_.Thumbprint] = $true
                    $b64 = [System.Convert]::ToBase64String($_.RawData, 'InsertLineBreaks')
                    [void]$sb.AppendLine("# $($_.Subject)")
                    [void]$sb.AppendLine("-----BEGIN CERTIFICATE-----")
                    [void]$sb.AppendLine($b64)
                    [void]$sb.AppendLine("-----END CERTIFICATE-----")
                }
            }
        }
        [System.IO.File]::WriteAllText($bundle, $sb.ToString())
        Set-Item -Path "Env:HLTV_CA_BUNDLE" -Value $bundle
        Set-Item -Path "Env:CURL_CA_BUNDLE" -Value $bundle
        Set-Item -Path "Env:SSL_CERT_FILE" -Value $bundle
        Set-Item -Path "Env:REQUESTS_CA_BUNDLE" -Value $bundle
        Write-Host ("CA bundle generado: " + $bundle) -ForegroundColor DarkGray
    }
}

function Test-PythonImports($python, $imports) {
    $code = "import " + ($imports -join ", ")
    & $python -c $code 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Set-DefaultEnv($name, $value) {
    if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
        Set-Item -Path ("Env:" + $name) -Value $value
    }
}

function Configure-ScrapeGuards {
    Set-DefaultEnv "HLTV_FETCH_MAX_ATTEMPTS" "10"
    Set-DefaultEnv "HLTV_FETCH_BASE_DELAY" "5.0"
    Set-DefaultEnv "HLTV_FETCH_MAX_DELAY" "600.0"
    Set-DefaultEnv "HLTV_FETCH_MIN_INTERVAL" "2.5"
    Set-DefaultEnv "HLTV_FETCH_CACHE_TTL" "240.0"
    Set-DefaultEnv "HLTV_BLOCK_COOLDOWN_BASE" "90.0"
    Set-DefaultEnv "HLTV_BLOCK_COOLDOWN_MAX" "1200.0"
    Set-DefaultEnv "HLTV_BLOCK_STREAK_THRESHOLD" "3"
    Set-DefaultEnv "HLTV_CIRCUIT_BREAKER_SLEEP" "420.0"
    Set-DefaultEnv "HLTV_FETCH_WARMUP" "1"
    Set-DefaultEnv "HLTV_URL_QUARANTINE_SECONDS" "900.0"
    Set-DefaultEnv "HLTV_MAX_HTTP_REQUESTS_PER_RUN" "2500"
    # Scrapling: tier 1 HTTP impersonation + tier 2 navegador stealth.
    Set-DefaultEnv "HLTV_USE_SCRAPLING" "1"
    Set-DefaultEnv "HLTV_SOLVE_CLOUDFLARE" "1"
    Set-DefaultEnv "HLTV_IMPERSONATE" "chrome"
    Set-DefaultEnv "HLTV_STEALTH_HEADLESS" "1"
    Set-DefaultEnv "HLTV_STEALTH_TIMEOUT_MS" "90000"
    Set-DefaultEnv "HLTV_STEALTH_MAX_SOLVES_PER_RUN" "6"
    Set-DefaultEnv "HLTV_SCRAPLING_TIER1_ATTEMPTS" "3"
    # HLTV_PROXY vacio por defecto (sin proxy). Ej: http://user:pass@host:port
}

function Ensure-ModelPython {
    $python = Get-SystemPython
    $imports = @("numpy", "pandas", "sklearn", "scipy", "matplotlib", "lightgbm", "shap")
    if (-not (Test-PythonImports $python $imports)) {
        Write-Host "Instalando dependencias ML en el Python del sistema..." -ForegroundColor Yellow
        Invoke-Native $python @("-m", "pip", "install", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "lightgbm", "shap") "Instalacion dependencias ML"
    }
    if (-not (Test-PythonImports $python $imports)) {
        throw "No se pudieron cargar las dependencias ML requeridas."
    }
    return $python
}

function Ensure-ScraperPython {
    param([string]$SystemPython)
    $ScraperDir = Join-Path $Root "SCRAPPER\hltv-scraper-api"
    $VenvPython = Join-Path $ScraperDir ".venv\Scripts\python.exe"
    $Requirements = Join-Path $ScraperDir "requirements.txt"

    # El venv debe crearse con un Python 3.10-3.13 para que Scrapling stealth
    # instale. Si el venv ya existe con 3.14 (sin scrapling), se recrea.
    $BasePython = Get-ScraperBasePython
    $baseOk = Test-PythonImports $BasePython @("sys")

    $valid = $false
    if ((Test-Path $VenvPython) -and (-not $RecreateScraperVenv)) {
        # Valido core + scrapling (si el base soporta scrapling exigimos scrapling).
        $valid = Test-PythonImports $VenvPython @("scrapy", "cloudscraper", "parsel", "requests", "scrapling", "curl_cffi")
        if (-not $valid) {
            $valid = Test-PythonImports $VenvPython @("scrapy", "cloudscraper", "parsel", "requests")
            if ($valid) {
                Write-Host "El venv del scraper no tiene Scrapling; se recreara para anadirlo." -ForegroundColor Yellow
                $valid = $false
            }
        }
    }

    if (-not $valid) {
        Write-Host ("Preparando venv online del scraper con: " + $BasePython) -ForegroundColor Yellow
        if (Test-Path (Join-Path $ScraperDir ".venv")) {
            Remove-Item -LiteralPath (Join-Path $ScraperDir ".venv") -Recurse -Force
        }
        Invoke-Native $BasePython @("-m", "venv", (Join-Path $ScraperDir ".venv")) "Creacion venv scraper"
        Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip") "Upgrade pip scraper"
        Invoke-Native $VenvPython @("-m", "pip", "install", "-r", $Requirements) "Instalacion requirements scraper"
        # Descarga de navegadores de Scrapling (patchright/chromium). Best-effort:
        # si falla (p.ej. Python 3.14 sin wheels), seguimos con el tier HTTP.
        try {
            & $VenvPython -c "from scrapling.cli import install; install([], standalone_mode=False)"
            if ($LASTEXITCODE -ne 0) { Write-Host "AVISO: 'scrapling install' devolvio error; el navegador stealth podria no estar disponible." -ForegroundColor Yellow }
        } catch {
            Write-Host ("AVISO: no se pudieron instalar los navegadores de Scrapling: " + $_.Exception.Message) -ForegroundColor Yellow
        }
    }

    if (-not (Test-PythonImports $VenvPython @("scrapy", "cloudscraper", "parsel", "requests"))) {
        throw "El venv del scraper no tiene las dependencias base requeridas."
    }

    # Estado de Scrapling: si no importa, desactivamos su uso y avisamos.
    if (Test-PythonImports $VenvPython @("scrapling", "curl_cffi")) {
        if (Test-PythonImports $VenvPython @("scrapling.fetchers")) {
            Write-Host "Scrapling disponible (tier HTTP impersonation + navegador stealth)." -ForegroundColor DarkGray
        }
    } else {
        Write-Host "AVISO: Scrapling no disponible en el venv; el scraper usara requests/cloudscraper. Instala Python 3.13 para el modo stealth." -ForegroundColor Yellow
        Set-Item -Path "Env:HLTV_USE_SCRAPLING" -Value "0"
    }

    return $VenvPython
}

$ModelPython = Ensure-ModelPython
$ScraperPython = $null
if (-not $SkipScrape) {
    Configure-ScrapeGuards
    Ensure-CaBundle -ScraperDir (Join-Path $Root "SCRAPPER\hltv-scraper-api")
    $ScraperPython = Ensure-ScraperPython -SystemPython $ModelPython
}
Write-Host ("Python modelo:  " + $ModelPython) -ForegroundColor DarkGray
if ($ScraperPython) {
    Write-Host ("Python scraper: " + $ScraperPython) -ForegroundColor DarkGray
    Write-Host (
        "Guardas HLTV: min_interval={0}s attempts={1} base_delay={2}s max_delay={3}s cache_ttl={4}s block_cooldown={5}-{6}s circuit={7}s url_quarantine={8}s max_requests={9}" -f
        $env:HLTV_FETCH_MIN_INTERVAL,
        $env:HLTV_FETCH_MAX_ATTEMPTS,
        $env:HLTV_FETCH_BASE_DELAY,
        $env:HLTV_FETCH_MAX_DELAY,
        $env:HLTV_FETCH_CACHE_TTL,
        $env:HLTV_BLOCK_COOLDOWN_BASE,
        $env:HLTV_BLOCK_COOLDOWN_MAX,
        $env:HLTV_CIRCUIT_BREAKER_SLEEP,
        $env:HLTV_URL_QUARANTINE_SECONDS,
        $env:HLTV_MAX_HTTP_REQUESTS_PER_RUN
    ) -ForegroundColor DarkGray
}

$DailyStart = Join-Path $Root "DAILY_SNAPSHOTS\start.py"
$Enrich = Join-Path $Root "DAILY_SNAPSHOTS\enrich_predictions.py"
$BuildWeb = Join-Path $Root "WEB\build_web.py"
$Train = Join-Path $Root "MODEL\train.py"
$ContextCalibration = Join-Path $Root "MODEL\analyze_context_calibration.py"
$BuildDb = Join-Path $Root "BBDD\build_db.py"
$Artifact = Join-Path $Root "MODEL\artifacts\model.pkl"
$MasterMani = Join-Path $Root "DAILY_SNAPSHOTS\master\manifest.json"

$total = 4
if (-not $NoDb) { $total++ }
$needTrain = $Retrain -or (-not (Test-Path $Artifact))
if ($needTrain) { $total++ }
$n = 0

$n++
if (-not $SkipScrape) {
    Step $n $total "Scrape online de HLTV + actualizacion de pendientes"
    $StartArgs = @($DailyStart, "--player-delay", [string]$PlayerDelay)
    if (-not $Quiet) { $StartArgs += "--verbose" }
    if ($MaxMatches -gt 0) { $StartArgs += @("--max-matches", [string]$MaxMatches) }
    if ($MaxMatches -gt 0) { $StartArgs += "--no-promote" }
    if ($SkipPlayerStats) { $StartArgs += "--skip-player-stats" }
    if ($SkipTeamProfiles) { $StartArgs += "--skip-team-profiles" }
    if ($SkipMatchAssets) { $StartArgs += "--skip-match-assets" }
    if ($SkipAnalytics) { $StartArgs += "--skip-analytics" }
    if ($SkipRankings) { $StartArgs += "--skip-rankings" }
    if ($SkipWarmup) { $StartArgs += "--skip-warmup" }
    if ($SkipSameDayRecovery) { $StartArgs += "--skip-same-day-recovery" }
    $StartArgs += @("--match-assets-limit", [string]$MatchAssetsLimit)
    $StartArgs += @("--match-assets-delay", [string]$MatchAssetsDelay)
    $StartArgs += @("--same-day-recovery-window-days", [string]$RecoveryWindowDays)
    $StartArgs += @("--recovery-delay", [string]$RecoveryDelay)
    if ($AllowOfflineFallback) { $StartArgs += "--allow-empty-scrape" }

    try {
        Invoke-Native $ScraperPython $StartArgs "Scrape online HLTV"
    } catch {
        if ($AllowOfflineFallback) {
            Write-Host ("Scrape fallo; continuo con ultimo run por -AllowOfflineFallback: " + $_.Exception.Message) -ForegroundColor Yellow
        } else {
            throw
        }
    }
} else {
    Step $n $total "Scrape omitido (-SkipScrape): uso el ultimo run existente"
}

if (-not (Test-Path $MasterMani)) {
    throw "No hay master manifest. Ejecuta un scrape online valido primero."
}
$Manifest = Get-Content -LiteralPath $MasterMani -Raw | ConvertFrom-Json
$RunDir = Join-Path $Root ("DAILY_SNAPSHOTS\runs\" + $Manifest.last_run_id)
if (-not (Test-Path $RunDir)) {
    throw "El run $($Manifest.last_run_id) no existe en disco."
}

if ($needTrain) {
    $n++
    Step $n $total "Entrenando modelo (Glicko-2 + calibracion) con master actualizado"
    Invoke-Native $ModelPython @($Train) "Entrenamiento modelo"
}

$n++
Step $n $total "Enriqueciendo predicciones (modelo + odds + flags + calibracion)"
Invoke-Native $ModelPython @($Enrich, "--run-dir", $RunDir) "Enriquecimiento predicciones"

$n++
Step $n $total "Analizando calibracion por contexto HLTV"
Invoke-Native $ModelPython @($ContextCalibration) "Analisis calibracion contexto"

if (-not $NoDb) {
    $n++
    Step $n $total "Reconstruyendo BBDD SQLite (fuente de verdad)"
    Invoke-Native $ModelPython @($BuildDb) "Reconstruccion BBDD"
}

$n++
Step $n $total "Generando WEB\data.js"
Invoke-Native $ModelPython @($BuildWeb, "--run-dir", $RunDir) "Generacion web"

Write-Host ""
Write-Host "Pipeline online completa." -ForegroundColor Green
Write-Host ("  Run:       " + $RunDir)
Write-Host ("  Dashboard: " + (Join-Path $Root "WEB\index.html"))
Write-Host ("  Contexto:  " + (Join-Path $Root "MODEL\results\CONTEXT_CALIBRATION.md"))
Write-Host "  Abrir:     start .\WEB\index.html"
