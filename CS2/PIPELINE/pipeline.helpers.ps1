<#
  Helpers de la pipeline (dot-sourced por start.ps1).

  Contiene utilidades reutilizables y sin estado de negocio:
    - Logging estructurado con niveles (Write-Log + JSONL).
    - Carga de configuracion (Import-PipelineConfig).
    - Ejecucion de comandos nativos (Invoke-Native).
    - Resolucion/validacion de Python y venvs (Ensure-*).

  La ORQUESTACION (modelo de etapas, dry-run, timing, exit codes) vive en
  start.ps1, que es quien conoce el flujo concreto.
#>

# --- Estado de logging (script-scope del que hace dot-source) ----------------
$script:PipelineLogLevel = 'INFO'
$script:PipelineLogJsonl = $null
$script:LogLevelRank = @{ DEBUG = 0; INFO = 1; WARN = 2; ERROR = 3 }

function Initialize-PipelineLogging {
    param(
        [string]$JsonlPath,
        [ValidateSet('DEBUG', 'INFO', 'WARN', 'ERROR')][string]$Level = 'INFO'
    )
    $script:PipelineLogLevel = $Level
    $script:PipelineLogJsonl = $JsonlPath
}

function Write-Log {
    <#
      Log con nivel. SIEMPRE registra en el JSONL (registro completo); la consola
      respeta el umbral $script:PipelineLogLevel para reducir ruido.
    #>
    param(
        [Parameter(Mandatory)][string]$Message,
        [ValidateSet('DEBUG', 'INFO', 'WARN', 'ERROR')][string]$Level = 'INFO',
        [string]$Color,
        [string]$Stage
    )
    $ts = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')

    # Registro estructurado (todos los niveles) al JSONL.
    if ($script:PipelineLogJsonl) {
        $record = [ordered]@{ ts = $ts; level = $Level; stage = $Stage; message = $Message }
        try {
            ($record | ConvertTo-Json -Compress -Depth 4) | Out-File -LiteralPath $script:PipelineLogJsonl -Append -Encoding utf8
        } catch { }
    }

    # Consola: solo si el nivel alcanza el umbral.
    if ($script:LogLevelRank[$Level] -ge $script:LogLevelRank[$script:PipelineLogLevel]) {
        $prefix = "[{0}] [{1}]" -f (Get-Date -Format 'HH:mm:ss'), $Level
        $line = "$prefix $Message"
        if (-not $Color) {
            $Color = switch ($Level) {
                'ERROR' { 'Red' }
                'WARN'  { 'Yellow' }
                'DEBUG' { 'DarkGray' }
                default { 'Gray' }
            }
        }
        Write-Host $line -ForegroundColor $Color
    }
}

function Import-PipelineConfig {
    <# Carga el .psd1 de configuracion. Lanza si no existe o es invalido. #>
    param([Parameter(Mandatory)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "No existe el fichero de configuracion: $Path"
    }
    return Import-PowerShellDataFile -LiteralPath $Path
}

function Invoke-Native {
    <#
      Ejecuta un ejecutable nativo, transmite su salida en vivo y LANZA si el
      exit code != 0. Devuelve los segundos de ejecucion.
    #>
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$Description = ""
    )
    $label = if ($Description) { $Description } else { Split-Path -Leaf $FilePath }
    $started = Get-Date
    Write-Log ("ejecutar: {0} {1}" -f $FilePath, ($Arguments -join ' ')) -Level DEBUG -Stage $label

    $stderrFile = New-TemporaryFile
    $oldErrorActionPreference = $ErrorActionPreference
    $oldNativeErrorPreference = $null
    $exit = $null
    if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
        $oldNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $ErrorActionPreference = "Continue"
    try {
        & $FilePath @Arguments 2> $stderrFile.FullName | ForEach-Object { Write-Host $_ }
        $exit = $LASTEXITCODE
        if (($exit -ne 0) -and (Test-Path $stderrFile.FullName)) {
            Get-Content -LiteralPath $stderrFile.FullName -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
        }
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
        if ($null -ne $oldNativeErrorPreference) {
            $PSNativeCommandUseErrorActionPreference = $oldNativeErrorPreference
        }
        Remove-Item -LiteralPath $stderrFile.FullName -Force -ErrorAction SilentlyContinue
    }
    $elapsed = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    if ($exit -ne 0) {
        Write-Log ("{0} FALLO exit={1} elapsed={2}s" -f $label, $exit, $elapsed) -Level ERROR -Stage $label
        if ($Description) { throw "$Description fallo con exit code $exit." }
        throw "Comando fallo con exit code ${exit}: $FilePath $($Arguments -join ' ')"
    }
    Write-Log ("{0} OK exit=0 elapsed={1}s" -f $label, $elapsed) -Level DEBUG -Stage $label
    return $elapsed
}

# --- Resolucion de Python ----------------------------------------------------
function Get-SystemPython {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw "No se encontro Python en PATH." }
    return $cmd.Source
}

function Get-ScraperBasePython {
    # Scrapling (navegador stealth) soporta Python 3.10-3.13, NO 3.14.
    foreach ($v in @("3.13", "3.12", "3.11", "3.10")) {
        try {
            $out = & py "-$v" -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) { return $out.Trim() }
        } catch { }
    }
    Write-Log "No se encontro Python 3.10-3.13; uso el del sistema. El navegador stealth de Scrapling podria no instalar (se usara solo el tier HTTP / requests)." -Level WARN
    return (Get-SystemPython)
}

function Test-PythonImports($python, $imports) {
    $code = "import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in sys.argv[1:]) else 1)"
    & $python -c $code @imports *> $null
    return ($LASTEXITCODE -eq 0)
}

function Set-DefaultEnv($name, $value) {
    if (-not [Environment]::GetEnvironmentVariable($name, "Process")) {
        Set-Item -Path ("Env:" + $name) -Value $value
    }
}

function Set-ScrapeGuardsFromConfig {
    <# Publica las guardas del scraper desde la config como variables de entorno. #>
    param([Parameter(Mandatory)][hashtable]$Guards)
    foreach ($key in $Guards.Keys) {
        Set-DefaultEnv $key ([string]$Guards[$key])
    }
}

function Ensure-CaBundle {
    # En redes con inspeccion TLS (proxy corporativo con CA propia), curl_cffi y
    # requests fallan la verificacion. Exportamos el trust store de Windows a un
    # bundle. Si el probe TLS pasa, no se genera nada.
    param([string]$ScraperDir, [switch]$DryRun)
    $bundle = Join-Path $ScraperDir "corp_ca_bundle.pem"
    if (Test-Path $bundle) {
        Set-Item -Path "Env:HLTV_CA_BUNDLE" -Value $bundle
        Set-Item -Path "Env:CURL_CA_BUNDLE" -Value $bundle
        Set-Item -Path "Env:SSL_CERT_FILE" -Value $bundle
        Set-Item -Path "Env:REQUESTS_CA_BUNDLE" -Value $bundle
        Write-Log ("CA bundle en uso: " + $bundle) -Level DEBUG
        return
    }
    $probe = @'
import urllib.request, ssl
try:
    urllib.request.urlopen('https://www.hltv.org/robots.txt', timeout=15)
    print('TLS_OK')
except ssl.SSLCertVerificationError:
    print('TLS_MITM')
except Exception:
    print('TLS_OK')
'@
    $result = & (Get-SystemPython) -c $probe 2>$null
    if ($result -match "TLS_MITM") {
        if ($DryRun) { Write-Log "DRY-RUN: se generaria corp_ca_bundle.pem (inspeccion TLS detectada)." -Level WARN; return }
        Write-Log "Inspeccion TLS detectada; exporto el trust store de Windows a corp_ca_bundle.pem..." -Level WARN
        $stores = @('Cert:\LocalMachine\Root', 'Cert:\CurrentUser\Root', 'Cert:\LocalMachine\CA', 'Cert:\CurrentUser\CA')
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
        Write-Log ("CA bundle generado: " + $bundle) -Level DEBUG
    }
}

function Ensure-ModelPython {
    <# Resuelve el Python del sistema y asegura las deps ML (instala si faltan). #>
    param([string[]]$Imports, [string[]]$PipPackages, [switch]$DryRun)
    $python = Get-SystemPython
    if (-not (Test-PythonImports $python $Imports)) {
        if ($DryRun) {
            Write-Log "DRY-RUN: faltan deps ML; se instalarian: $($PipPackages -join ', ')" -Level WARN
            return $python
        }
        Write-Log "Instalando dependencias ML en el Python del sistema..." -Level WARN
        Invoke-Native $python (@("-m", "pip", "install") + $PipPackages) "Instalacion dependencias ML" | Out-Null
        if (-not (Test-PythonImports $python $Imports)) {
            throw "No se pudieron cargar las dependencias ML requeridas."
        }
    }
    return $python
}

function Ensure-ScraperPython {
    <# Crea/repara el venv del scraper (Python 3.10-3.13 para Scrapling stealth). #>
    param(
        [Parameter(Mandatory)][string]$ScraperDir,
        [Parameter(Mandatory)][string[]]$BaseImports,
        [Parameter(Mandatory)][string[]]$StealthImports,
        [switch]$Recreate,
        [switch]$DryRun
    )
    $VenvPython = Join-Path $ScraperDir ".venv\Scripts\python.exe"
    $Requirements = Join-Path $ScraperDir "requirements.txt"
    $BasePython = Get-ScraperBasePython

    $valid = $false
    if ((Test-Path $VenvPython) -and (-not $Recreate)) {
        $valid = Test-PythonImports $VenvPython ($BaseImports + $StealthImports + @('scrapling.fetchers'))
        if (-not $valid) {
            $valid = Test-PythonImports $VenvPython $BaseImports
            if ($valid) {
                Write-Log "El venv del scraper no tiene Scrapling; se recreara para anadirlo." -Level WARN
                $valid = $false
            }
        }
    }

    if (-not $valid) {
        if ($DryRun) {
            Write-Log "DRY-RUN: se prepararia el venv del scraper con $BasePython (venv + pip install -r requirements)." -Level WARN
            return $VenvPython
        }
        Write-Log ("Preparando venv online del scraper con: " + $BasePython) -Level WARN
        if (Test-Path (Join-Path $ScraperDir ".venv")) {
            Remove-Item -LiteralPath (Join-Path $ScraperDir ".venv") -Recurse -Force
        }
        Invoke-Native $BasePython @("-m", "venv", (Join-Path $ScraperDir ".venv")) "Creacion venv scraper" | Out-Null
        Invoke-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip") "Upgrade pip scraper" | Out-Null
        Invoke-Native $VenvPython @("-m", "pip", "install", "-r", $Requirements) "Instalacion requirements scraper" | Out-Null
        try {
            & $VenvPython -c "from scrapling.cli import install; install([], standalone_mode=False)"
            if ($LASTEXITCODE -ne 0) { Write-Log "'scrapling install' devolvio error; el navegador stealth podria no estar disponible." -Level WARN }
        } catch {
            Write-Log ("No se pudieron instalar los navegadores de Scrapling: " + $_.Exception.Message) -Level WARN
        }
    }

    if ($DryRun) { return $VenvPython }

    if (-not (Test-PythonImports $VenvPython $BaseImports)) {
        throw "El venv del scraper no tiene las dependencias base requeridas."
    }
    if (Test-PythonImports $VenvPython $StealthImports) {
        if (Test-PythonImports $VenvPython @('scrapling.fetchers')) {
            Write-Log "Scrapling disponible (tier HTTP impersonation + navegador stealth)." -Level DEBUG
        }
    } else {
        Write-Log "Scrapling no disponible en el venv; el scraper usara requests/cloudscraper. Instala Python 3.13 para el modo stealth." -Level WARN
        Set-Item -Path "Env:HLTV_USE_SCRAPLING" -Value "0"
    }
    return $VenvPython
}
