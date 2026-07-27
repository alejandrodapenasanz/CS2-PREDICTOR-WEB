<#
  CS2 Predictor - online pipeline de una sola orden.

  Por defecto (.\start.ps1) ejecuta el flujo online:
     0) Resuelve Python de modelo y del scraper (venv, CA bundle, guardas HLTV).
     P) BLACKBOX: auto-heal/restore de la BBDD si falta/vacia/corrupta.
     1) Inicializa/siembra la BBDD viva si hace falta.
     2) Scrapea HLTV en vivo y actualiza pendientes. Si no produce datos, falla.
     3) Ingest pre-entreno a la BBDD viva (hechos/odds/assets/snapshots).
     4) Entrena el modelo si falta el artefacto (o si se pasa -Retrain).
     5) Enriquece predicciones con el modelo calibrado, odds y flags.
     6) Analiza si el contexto HLTV ayuda a calibrar el modelo.
     7) Ingest final a la BBDD viva SQLite y export compat JSON.
     8) Monitoriza drift causal sobre predicciones cerradas (log loss + CLV).
     P) BLACKBOX: export de respaldo si se pide -BackupBlackbox.
     9) Genera ..\WEB\data.js para el dashboard compartido.

  La configuracion operativa (guardas HLTV, deps, exit-codes) vive en
  PIPELINE\pipeline.config.psd1. Los helpers en PIPELINE\pipeline.helpers.ps1.

  Flags:
     -SkipScrape             No scrapear; usa el ultimo run existente.
     -AllowOfflineFallback   Si el scrape falla, continuar con el ultimo run.
     -Retrain                Forzar reentrenamiento del modelo.
     -NoDb                   No inicializar ni ingerir en la BBDD.
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
     -BackupBlackbox         Al terminar, exporta la caja negra portatil (BBDD/BLACKBOX).
     -RestoreBlackbox        Al inicio, restaura la BBDD desde BBDD/BLACKBOX (guardian).
     -SkipAutoHeal           No comprobar/curar la BBDD desde BLACKBOX al arrancar.
     -Quiet                  Reduce logs internos del scraper.
     -DryRun / -WhatIf       Muestra que etapas se ejecutarian, sin ejecutarlas.
     -LogLevel LEVEL         DEBUG|INFO|WARN|ERROR (consola). El JSONL guarda todo.
     -Config PATH            Fichero de configuracion .psd1 alternativo.
#>
[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$SkipScrape,
    [switch]$AllowOfflineFallback,
    [switch]$Retrain,
    [switch]$NoDb,
    [ValidateRange(0, [int]::MaxValue)][int]$MaxMatches = 0,
    [ValidateRange(0.0, [double]::MaxValue)][double]$PlayerDelay = 1.5,
    [switch]$SkipPlayerStats,
    [switch]$SkipTeamProfiles,
    [switch]$SkipMatchAssets,
    [ValidateRange(0, [int]::MaxValue)][int]$MatchAssetsLimit = 20,
    [ValidateRange(0.0, [double]::MaxValue)][double]$MatchAssetsDelay = 1.5,
    [switch]$SkipAnalytics,
    [switch]$SkipRankings,
    [switch]$SkipWarmup,
    [switch]$SkipSameDayRecovery,
    [ValidateRange(0, [int]::MaxValue)][int]$RecoveryWindowDays = 2,
    [ValidateRange(0.0, [double]::MaxValue)][double]$RecoveryDelay = 1.5,
    [switch]$RecreateScraperVenv,
    [switch]$BackupBlackbox,
    [switch]$RestoreBlackbox,
    [switch]$SkipAutoHeal,
    [switch]$Quiet,
    [switch]$DryRun,
    [ValidateSet('DEBUG', 'INFO', 'WARN', 'ERROR')][string]$LogLevel = 'INFO',
    [string]$Config
)

$ErrorActionPreference = "Stop"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $Root
$WebRoot = Join-Path $RepoRoot "WEB"
$LogDir = Join-Path $Root "PIPELINE\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# --- Helpers + configuracion -------------------------------------------------
. (Join-Path $Root "PIPELINE\pipeline.helpers.ps1")

if (-not $Config) { $Config = Join-Path $Root "PIPELINE\pipeline.config.psd1" }
$Cfg = Import-PipelineConfig -Path $Config
$script:ExitCode = $Cfg.ExitCodes.Generic
$script:DryRun = [bool]($DryRun -or $WhatIfPreference)

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$script:StartPs1Log = Join-Path $LogDir ("start_" + $stamp + ".log")
$LogJsonl = Join-Path $LogDir ("start_" + $stamp + ".jsonl")
$TimingJson = Join-Path $LogDir ("start_" + $stamp + ".timing.json")
Initialize-PipelineLogging -JsonlPath $LogJsonl -Level $LogLevel

$script:TranscriptStarted = $false
try {
    Start-Transcript -Path $script:StartPs1Log -Force | Out-Null
    $script:TranscriptStarted = $true
    Write-Log ("Log start.ps1: " + $script:StartPs1Log) -Level INFO -Color DarkGray
    Write-Log ("Log estructurado: " + $LogJsonl) -Level DEBUG
} catch {
    Write-Log ("no se pudo iniciar transcript: " + $_.Exception.Message) -Level WARN
}

trap {
    Write-Log ("ERROR start.ps1: " + $_.Exception.Message) -Level ERROR
    Write-Log ("Log completo: " + $script:StartPs1Log) -Level WARN
    if ($_.ScriptStackTrace) { Write-Log $_.ScriptStackTrace -Level DEBUG }
    if ($script:TranscriptStarted) {
        try { Stop-Transcript | Out-Null; $script:TranscriptStarted = $false } catch { }
    }
    exit $script:ExitCode
}

# --- Modelo de etapas: timing + progreso + dry-run + exit codes --------------
$script:StageIndex = 0
$script:StageTotal = 0
$script:StageTimings = @()

function Invoke-Stage {
    <#
      Envuelve una unidad de trabajo: cabecera [n/total], cronometraje, registro
      estructurado, soporte de dry-run y mapeo de exit code al fallar.
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][scriptblock]$Action,
        [int]$FailExit = $Cfg.ExitCodes.Generic,
        [switch]$Auxiliary
    )
    if ($Auxiliary) {
        $header = "[BLACKBOX] $Name"
    } else {
        $script:StageIndex++
        $header = "[{0}/{1}] {2}" -f $script:StageIndex, $script:StageTotal, $Name
    }
    Write-Log $header -Level INFO -Color Cyan -Stage $Name

    if ($script:DryRun) {
        Write-Log ("DRY-RUN: se omitiria '{0}'" -f $Name) -Level INFO -Color DarkYellow -Stage $Name
        $script:StageTimings += [pscustomobject]@{ stage = $Name; seconds = 0; status = 'dry-run' }
        return
    }

    $start = Get-Date
    try {
        & $Action
    } catch {
        $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
        $script:StageTimings += [pscustomobject]@{ stage = $Name; seconds = $elapsed; status = 'FAILED' }
        $script:ExitCode = $FailExit
        throw
    }
    $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
    $script:StageTimings += [pscustomobject]@{ stage = $Name; seconds = $elapsed; status = 'ok' }
    Write-Log ("OK '{0}' ({1}s)" -f $Name, $elapsed) -Level DEBUG -Stage $Name
}

# --- Validacion de combinaciones de parametros -------------------------------
if ($NoDb -and $BackupBlackbox) { Write-Log "-BackupBlackbox se ignora con -NoDb (no se toca la BBDD)." -Level WARN }
if ($NoDb -and $RestoreBlackbox) { Write-Log "-RestoreBlackbox se ignora con -NoDb (no se toca la BBDD)." -Level WARN }
if ($SkipScrape -and $MaxMatches -gt 0) { Write-Log "-MaxMatches se ignora con -SkipScrape (no hay scrape)." -Level WARN }
if ($RestoreBlackbox -and $SkipAutoHeal) { Write-Log "-RestoreBlackbox tiene prioridad; -SkipAutoHeal es redundante." -Level INFO }
if ($script:DryRun) { Write-Log "MODO DRY-RUN: no se ejecutara ninguna etapa (solo se listaran)." -Level WARN }

# --- Rutas de scripts --------------------------------------------------------
$DailyStart = Join-Path $Root "PIPELINE\start.py"
$Enrich = Join-Path $Root "PIPELINE\enrich_predictions.py"
$BuildWeb = Join-Path $WebRoot "build_web.py"
$Train = Join-Path $Root "MODEL\train.py"
$ContextCalibration = Join-Path $Root "MODEL\analyze_context_calibration.py"
$DriftMonitor = Join-Path $Root "MODEL\monitor_drift.py"
$BuildDb = Join-Path $Root "BBDD\build_db.py"
$IngestDb = Join-Path $Root "BBDD\ingest.py"
$ExportMaster = Join-Path $Root "BBDD\export_master_json.py"
$Artifact = Join-Path $Root "MODEL\artifacts\model.pkl"
$MasterMani = Join-Path $Root "PIPELINE\master\manifest.json"
$Blackbox = Join-Path $Root "BBDD\blackbox.py"
$BlackboxDir = Join-Path $Root "BBDD\BLACKBOX"
$DbPath = Join-Path $Root "BBDD\cs2.db"

# --- Fase 0: Python del modelo y del scraper ---------------------------------
$ModelPython = Ensure-ModelPython -Imports $Cfg.ModelImports -PipPackages $Cfg.ModelPipPackages -DryRun:$script:DryRun
$ScraperPython = $null
if (-not $SkipScrape) {
    Set-ScrapeGuardsFromConfig -Guards $Cfg.ScrapeGuards
    Ensure-CaBundle -ScraperDir (Join-Path $Root "SCRAPER\hltv-scraper-api") -DryRun:$script:DryRun
    $ScraperPython = Ensure-ScraperPython -ScraperDir (Join-Path $Root "SCRAPER\hltv-scraper-api") `
        -BaseImports $Cfg.ScraperBaseImports -StealthImports $Cfg.ScraperStealthImports `
        -Recreate:$RecreateScraperVenv -DryRun:$script:DryRun
}
Write-Log ("Python modelo:  " + $ModelPython) -Level DEBUG
if ($ScraperPython) {
    Write-Log ("Python scraper: " + $ScraperPython) -Level DEBUG
    Write-Log ("Guardas HLTV: min_interval={0}s attempts={1} cache_ttl={2}s circuit={3}s max_requests={4} stealth={5}ms" -f `
            $env:HLTV_FETCH_MIN_INTERVAL, $env:HLTV_FETCH_MAX_ATTEMPTS, $env:HLTV_FETCH_CACHE_TTL, `
            $env:HLTV_CIRCUIT_BREAKER_SLEEP, $env:HLTV_MAX_HTTP_REQUESTS_PER_RUN, $env:HLTV_STEALTH_TIMEOUT_MS) -Level DEBUG
}

# --- Total de etapas numeradas (identico al comportamiento previo) -----------
$needTrain = $Retrain -or (-not (Test-Path $Artifact))
$script:StageTotal = 4                       # scrape + enrich + context + web
if (-not $NoDb) { $script:StageTotal += 4 }  # build_db + ingest-pre + ingest-final + drift
if ($needTrain) { $script:StageTotal += 1 }

# --- Fase P: BLACKBOX restore explicito / auto-heal (antes de tocar BBDD) -----
if (-not $NoDb) {
    if ($RestoreBlackbox) {
        Invoke-Stage -Auxiliary -Name "Restauracion explicita desde BLACKBOX (-RestoreBlackbox)" -FailExit $Cfg.ExitCodes.Blackbox -Action {
            Invoke-Native $ModelPython @($Blackbox, "restore", "--db", $DbPath, "--blackbox", $BlackboxDir, "--force") "BLACKBOX restore" | Out-Null
        }
    } elseif (-not $SkipAutoHeal) {
        Invoke-Stage -Auxiliary -Name "Auto-heal: comprobando salud de la BBDD" -FailExit $Cfg.ExitCodes.Blackbox -Action {
            Invoke-Native $ModelPython @($Blackbox, "autoheal", "--db", $DbPath, "--blackbox", $BlackboxDir) "BLACKBOX autoheal" | Out-Null
        }
    }
}

# --- Etapa 1: init/siembra BBDD ----------------------------------------------
if (-not $NoDb) {
    Invoke-Stage -Name "Inicializando/sembrando BBDD viva si hace falta" -FailExit $Cfg.ExitCodes.IngestPre -Action {
        Invoke-Native $ModelPython @($BuildDb) "Semilla BBDD" | Out-Null
    }
}

# --- Etapa 2: scrape ----------------------------------------------------------
if (-not $SkipScrape) {
    Invoke-Stage -Name "Scrape online de HLTV + actualizacion de pendientes" -FailExit $Cfg.ExitCodes.Scrape -Action {
        $StartArgs = @($DailyStart, "--player-delay", [string]$PlayerDelay)
        if (-not $Quiet) { $StartArgs += "--verbose" }
        if ($MaxMatches -gt 0) { $StartArgs += @("--max-matches", [string]$MaxMatches, "--no-promote") }
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
            Invoke-Native $ScraperPython $StartArgs "Scrape online HLTV" | Out-Null
        } catch {
            if ($AllowOfflineFallback) {
                Write-Log ("Scrape fallo; continuo con ultimo run por -AllowOfflineFallback: " + $_.Exception.Message) -Level WARN
            } else {
                throw
            }
        }
    }
} else {
    Invoke-Stage -Name "Scrape omitido (-SkipScrape): uso el ultimo run existente" -Action { }
}

# --- Resolucion del run publicado --------------------------------------------
$RunDir = $null
if (-not (Test-Path $MasterMani)) {
    if ($script:DryRun) {
        Write-Log "DRY-RUN: no hay master manifest; se usaria el ultimo run publicado." -Level WARN
        $RunDir = Join-Path $Root "PIPELINE\runs\<RUN_ID>"
    } else {
        throw "No hay master manifest. Ejecuta un scrape online valido primero."
    }
} else {
    $Manifest = Get-Content -LiteralPath $MasterMani -Raw | ConvertFrom-Json
    $RunDir = Join-Path $Root ("PIPELINE\runs\" + $Manifest.last_run_id)
    if ((-not (Test-Path $RunDir)) -and (-not $script:DryRun)) {
        throw "El run $($Manifest.last_run_id) no existe en disco."
    }
}
Write-Log ("Run activo: " + $RunDir) -Level DEBUG

# --- Etapa 3: ingest pre-entreno (hechos; lo consume enrich y train) ---------
if (-not $NoDb) {
    Invoke-Stage -Name "Ingest pre-entreno a BBDD viva (hechos, odds, assets, snapshots)" -FailExit $Cfg.ExitCodes.IngestPre -Action {
        Invoke-Native $ModelPython @($IngestDb, "--run-dir", $RunDir, "--no-backup", "--no-mirror-backup") "Ingest pre-entreno BBDD" | Out-Null
    }
}

# --- Etapa 4: entrenamiento (solo si -Retrain o falta el artefacto) ----------
if ($needTrain) {
    Invoke-Stage -Name "Entrenando modelo (Glicko-2 + calibracion) con master actualizado" -FailExit $Cfg.ExitCodes.Train -Action {
        $TrainArgs = @($Train)
        if (-not $Quiet) { $TrainArgs += "--verbose" }
        Invoke-Native $ModelPython $TrainArgs "Entrenamiento modelo" | Out-Null
    }
}

# --- Etapa 5: enrich ----------------------------------------------------------
Invoke-Stage -Name "Enriqueciendo predicciones (modelo + odds + flags + calibracion)" -FailExit $Cfg.ExitCodes.Enrich -Action {
    Invoke-Native $ModelPython @($Enrich, "--run-dir", $RunDir) "Enriquecimiento predicciones" | Out-Null
}

# --- Etapa 6: calibracion por contexto (read-only; antes del ingest final) ---
Invoke-Stage -Name "Analizando calibracion por contexto HLTV" -FailExit $Cfg.ExitCodes.Context -Action {
    Invoke-Native $ModelPython @($ContextCalibration) "Analisis calibracion contexto" | Out-Null
}

# --- Etapa 7: ingest final (predicciones) + export master --------------------
if (-not $NoDb) {
    Invoke-Stage -Name "Ingest final a BBDD viva (predicciones) + export master JSON compat" -FailExit $Cfg.ExitCodes.DbPost -Action {
        Invoke-Native $ModelPython @($IngestDb, "--run-dir", $RunDir) "Ingest incremental BBDD" | Out-Null
        Invoke-Native $ModelPython @($ExportMaster) "Export master JSON compat" | Out-Null
    }

    # --- Etapa 8: monitor drift (read-only) ----------------------------------
    Invoke-Stage -Name "Monitorizando drift causal (log loss rodante + CLV)" -FailExit $Cfg.ExitCodes.Drift -Action {
        Invoke-Native $ModelPython @($DriftMonitor) "Monitor drift" | Out-Null
    }
}

# --- Fase P: BLACKBOX export de respaldo -------------------------------------
if ($BackupBlackbox -and (-not $NoDb)) {
    Invoke-Stage -Auxiliary -Name "Export de respaldo portatil (-BackupBlackbox)" -FailExit $Cfg.ExitCodes.Blackbox -Action {
        Invoke-Native $ModelPython @($Blackbox, "export", "--db", $DbPath, "--blackbox", $BlackboxDir) "BLACKBOX export" | Out-Null
    }
}

# --- Etapa 9: web ------------------------------------------------------------
Invoke-Stage -Name "Generando WEB\data.js compartido" -FailExit $Cfg.ExitCodes.Web -Action {
    Invoke-Native $ModelPython @($BuildWeb, "--sport-root", $Root, "--run-dir", $RunDir) "Generacion web" | Out-Null
}

# --- Resumen + tabla de tiempos ----------------------------------------------
$totalSeconds = ($script:StageTimings | Measure-Object -Property seconds -Sum).Sum
try {
    ($script:StageTimings | ConvertTo-Json -Depth 4) | Out-File -LiteralPath $TimingJson -Encoding utf8
} catch { }

Write-Host ""
if ($script:DryRun) {
    Write-Log "DRY-RUN completado: ninguna etapa se ejecuto." -Level WARN
} else {
    Write-Log "Pipeline online completa." -Level INFO -Color Green
}
Write-Host ""
Write-Host ("  Tiempos por etapa (total {0}s):" -f [math]::Round([double]$totalSeconds, 1)) -ForegroundColor Green
$script:StageTimings | Format-Table -AutoSize | Out-String | ForEach-Object { Write-Host $_ }
Write-Host ("  Run:       " + $RunDir)
Write-Host ("  Dashboard: " + (Join-Path $WebRoot "index.html"))
Write-Host ("  Contexto:  " + (Join-Path $Root "MODEL\results\CONTEXT_CALIBRATION.md"))
Write-Host ("  Abrir:     start " + (Join-Path $WebRoot "index.html"))
if ($BackupBlackbox -and (-not $NoDb)) {
    Write-Host ("  Blackbox:  " + $BlackboxDir + "  (copiala a USB/nube)")
}
Write-Host ("  Tiempos:   " + $TimingJson)
Write-Host ("  Log:       " + $script:StartPs1Log)

$script:ExitCode = 0
if ($script:TranscriptStarted) {
    try { Stop-Transcript | Out-Null; $script:TranscriptStarted = $false } catch { }
}
exit 0
