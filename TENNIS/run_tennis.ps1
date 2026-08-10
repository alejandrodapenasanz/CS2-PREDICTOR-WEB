<#
Qué hace:
    Activa TENNIS/.venv y ejecuta el pipeline operativo del día desde la ruta
    real del lanzador. Con -Retrain actualiza fuentes, regenera la cuarentena
    de identidades, reconstruye Elo/features, reentrena los dos modelos y solo
    entonces predice y concilia una jornada anterior pendiente. Tras un diario
    correcto reconstruye WEB/data.js con el builder existente de la web.

Qué recibe:
    -Date YYYY-MM-DD es opcional; si falta se usa hoy.
    -Retrain activa el ciclo completo semanal antes del diario.

Cómo se ejecuta desde cualquier ruta:
    .\TENNIS\run_tennis.ps1
    .\TENNIS\run_tennis.ps1 -Date 2026-07-30
    .\TENNIS\run_tennis.ps1 -Retrain
    .\TENNIS\run_tennis.ps1 -Retrain -Date 2026-07-30

La conciliación realiza una consulta de una fecha anterior pendiente por cada
invocación. No existe un contador ni un límite diario artificial en código.
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$Date,

    [switch]$Retrain
)

$ErrorActionPreference = 'Stop'
$TennisRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent $TennisRoot
$ActivateScript = Join-Path $TennisRoot '.venv\Scripts\Activate.ps1'
$PythonExecutable = Join-Path $TennisRoot '.venv\Scripts\python.exe'
$DailyScript = Join-Path $TennisRoot 'scripts\daily_predictions.py'
$WebBuildScript = Join-Path $RepositoryRoot 'WEB\build_web.py'

foreach ($RequiredPath in @(
    $ActivateScript,
    $PythonExecutable,
    $DailyScript,
    $WebBuildScript
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Falta un componente requerido de TENNIS: $RequiredPath"
    }
}

. $ActivateScript

if ($Retrain) {
    $RetrainScripts = @(
        'scripts\update_sources.py',
        'scripts\audit_identities.py',
        'scripts\build_elo.py',
        'scripts\build_features.py',
        'scripts\retrain_models.py'
    )
    foreach ($RelativeScript in $RetrainScripts) {
        $ScriptPath = Join-Path $TennisRoot $RelativeScript
        if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
            throw "Falta un script requerido para -Retrain: $ScriptPath"
        }
        Write-Host "`n[TENNIS -Retrain] $RelativeScript"
        $ScriptArguments = @()
        if ($RelativeScript -eq 'scripts\retrain_models.py' -and $Date) {
            $ScriptArguments += @('--training-as-of-date', $Date)
        }
        & $PythonExecutable $ScriptPath @ScriptArguments
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Falló $RelativeScript con código $LASTEXITCODE."
            exit $LASTEXITCODE
        }
    }
}

$DailyArguments = @()
if ($Date) {
    $DailyArguments += @('--date', $Date)
}
if ($Retrain) {
    $DailyArguments += '--retrained'
}

& $PythonExecutable $DailyScript @DailyArguments
$DailyExitCode = $LASTEXITCODE
if ($DailyExitCode -ne 0) {
    exit $DailyExitCode
}

Write-Host "`n[TENNIS] Actualizando WEB/data.js"
& $PythonExecutable $WebBuildScript
$WebExitCode = $LASTEXITCODE
if ($WebExitCode -ne 0) {
    Write-Error "Falló WEB/build_web.py con código $WebExitCode."
    exit $WebExitCode
}
exit 0
