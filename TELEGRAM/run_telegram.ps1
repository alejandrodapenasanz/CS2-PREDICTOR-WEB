<#
.SYNOPSIS
Publishes two daily CS2 messages and writes TELEGRAM/daily_report.txt.

.DESCRIPTION
Runs TELEGRAM/scripts/publish_cs2.py from any current working directory. After
a successful or idempotent publication, the Python workflow atomically replaces
daily_report.txt with unseen qualified opportunities from that date onward and
records their match IDs in SQLite. The launcher prefers
TELEGRAM/.venv and otherwise falls back to py -3 or python.

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
$PublisherScript = Join-Path $TelegramRoot 'scripts\publish_cs2.py'
$VenvPython = Join-Path $TelegramRoot '.venv\Scripts\python.exe'

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
    if (Test-Path -LiteralPath $VenvPython -PathType Leaf) {
        & $VenvPython @PublisherArguments
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 @PublisherArguments
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python @PublisherArguments
    }
    else {
        throw 'Python was not found. Create TELEGRAM/.venv or install Python 3.'
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Telegram publisher exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
