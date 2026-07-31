<#
Qué hace:
    Activa TENNIS/.venv y ejecuta el pipeline diario desde la ubicación real
    de este archivo, por lo que funciona con independencia del directorio
    actual de PowerShell.

Qué recibe:
    Reenvía todos los argumentos a scripts/daily_predictions.py. Por ejemplo:
    - Sin argumentos: predice hoy.
    - --date YYYY-MM-DD: predice la fecha indicada si el modelo es causal.

Cómo se ejecuta:
    .\TENNIS\run_tennis.ps1
    .\TENNIS\run_tennis.ps1 --date 2026-07-30
#>

$ErrorActionPreference = 'Stop'
$TennisRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ActivateScript = Join-Path $TennisRoot '.venv\Scripts\Activate.ps1'
$PythonExecutable = Join-Path $TennisRoot '.venv\Scripts\python.exe'
$PipelineScript = Join-Path $TennisRoot 'scripts\daily_predictions.py'

foreach ($RequiredPath in @(
    $ActivateScript,
    $PythonExecutable,
    $PipelineScript
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Falta un componente requerido de TENNIS: $RequiredPath"
    }
}

. $ActivateScript
& $PythonExecutable $PipelineScript @args
$PipelineExitCode = $LASTEXITCODE
exit $PipelineExitCode
