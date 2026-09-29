<#
.SYNOPSIS
Publishes two daily CS2 messages and writes TELEGRAM/daily_report.txt.

.DESCRIPTION
Runs TELEGRAM/scripts/publish_cs2.py from any current working directory. After
a successful or idempotent publication, the Python workflow atomically replaces
daily_report.txt with unseen qualified opportunities from that date onward and
records their match IDs in SQLite. The launcher prefers
VAULT/TELEGRAM/.venv; the launcher reconstructs its declared environment.

.PARAMETER Date
Optional prediction date in YYYY-MM-DD format. When omitted, Python uses the
current date in Europe/Madrid.

.PARAMETER DryRun
Prints both posts and the report preview without network access, database access,
publication state writes, or writing daily_report.txt.

.EXAMPLE
& 'C:\dev\CS2-Predictor\TELEGRAM\run_telegram.ps1' -Date 2026-08-11 -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$Date,

    [Parameter(Mandatory = $false)]
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$TelegramRoot = $PSScriptRoot
. (Join-Path (Split-Path -Parent $TelegramRoot) 'scripts\vault_bootstrap.ps1')
Initialize-ProjectVault -RepositoryRoot (Split-Path -Parent $TelegramRoot) -DryRun:$DryRun
$TelegramStateRoot = Get-VaultEnvironmentRoot -ComponentRoot $TelegramRoot
$PublisherScript = Join-Path $TelegramRoot 'scripts\publish_cs2.py'
$VenvRoot = Join-Path $TelegramStateRoot '.venv'
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'
$RequirementsPath = Join-Path $TelegramRoot 'requirements.txt'
$RequirementsHash = (Get-FileHash -LiteralPath $RequirementsPath -Algorithm SHA256).Hash
$RequirementsStamp = Join-Path $VenvRoot '.requirements.sha256'
$BasePython = Get-VaultBasePython
$EnvironmentOk = $false
if (Test-Path -LiteralPath $VenvPython) {
    & $VenvPython -c "import sys, zoneinfo; zoneinfo.ZoneInfo('Europe/Madrid'); raise SystemExit(0 if sys.version_info[:2] == (3,13) else 1)" *> $null
    $EnvironmentOk = ($LASTEXITCODE -eq 0)
}
if (-not $EnvironmentOk) {
    if ($DryRun) { throw 'DryRun no instala dependencias: prepara primero el entorno con un arranque normal.' }
    New-Item -ItemType Directory -Path $TelegramStateRoot -Force | Out-Null
    & $BasePython -m venv $VenvRoot
    if ($LASTEXITCODE -ne 0) { throw '[VAULT] No se pudo preparar el entorno de Telegram.' }
    & $VenvPython -m pip install --disable-pip-version-check -r $RequirementsPath
    if ($LASTEXITCODE -ne 0) { throw '[VAULT] No se pudieron instalar las dependencias de Telegram.' }
    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw '[VAULT] El entorno de Telegram tiene dependencias incompatibles.' }
}
if (-not $DryRun) {
    $installedHash = if (Test-Path -LiteralPath $RequirementsStamp) { (Get-Content -LiteralPath $RequirementsStamp -Raw).Trim() } else { '' }
    if ($installedHash -ne $RequirementsHash) {
        & $VenvPython -m pip install --disable-pip-version-check -r $RequirementsPath
        if ($LASTEXITCODE -ne 0) { throw '[VAULT] No se pudieron sincronizar las dependencias de Telegram.' }
        [IO.File]::WriteAllText($RequirementsStamp, $RequirementsHash)
    }
    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw '[VAULT] pip check de Telegram ha fallado.' }
}

if (-not (Test-Path -LiteralPath $PublisherScript -PathType Leaf)) {
    throw "Telegram publisher not found: $PublisherScript"
}

$PublisherArguments = @($PublisherScript)
if (-not [string]::IsNullOrWhiteSpace($Date)) {
    $PublisherArguments += @('--date', $Date)
}
if ($DryRun.IsPresent) {
    $PublisherArguments += '--dry-run'
}

Push-Location -LiteralPath $TelegramRoot
try {
    & $VenvPython @PublisherArguments

    if ($LASTEXITCODE -ne 0) {
        throw "Telegram publisher exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
