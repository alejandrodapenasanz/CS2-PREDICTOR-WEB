<#
Lanzador unificado del repositorio.

- Reenvía todos los argumentos al pipeline de CS2.
- Si CS2 termina correctamente, publica Telegram y después ejecuta TENNIS.
- Un fallo de Telegram no impide que TENNIS se ejecute.
- TENNIS entrena o reutiliza sus modelos en todo arranque operativo completo.
- Con -Retrain, CS2 también reentrena y TENNIS acepta el flag por compatibilidad.
- -Retrain no se reenvía al publicador de Telegram.
- Con -DryRun o -WhatIf, solo se ejecuta el dry-run de CS2; Telegram y
  TENNIS se omiten.
- Si TENNIS falla, prevalece su código de salida. Si TENNIS termina bien,
  se devuelve el código de salida de Telegram.

Puede invocarse desde cualquier ruta porque todos los paths parten de
$PSScriptRoot.
#>

$ErrorActionPreference = 'Stop'
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$ForwardedArguments = @($args)

function Test-EnabledSwitchArgument {
    <#
    Devuelve si una colección raw contiene el switch canónico -Name, sin
    distinguir mayúsculas y minúsculas.
    #>
    param(
        [Parameter(Mandatory)][AllowEmptyCollection()][object[]]$Arguments,
        [Parameter(Mandatory)][string]$Name
    )

    $EnabledForm = "-$Name"
    foreach ($Argument in $Arguments) {
        $Text = [string]$Argument
        if ($Text -ieq $EnabledForm) {
            return $true
        }
    }
    return $false
}

$RetrainRequested = Test-EnabledSwitchArgument `
    -Arguments $ForwardedArguments -Name 'Retrain'
$DryRunRequested = (
    (Test-EnabledSwitchArgument -Arguments $ForwardedArguments -Name 'DryRun') -or
    (Test-EnabledSwitchArgument -Arguments $ForwardedArguments -Name 'WhatIf')
)

$Cs2Launcher = Join-Path $PSScriptRoot 'CS2\start.ps1'
$TelegramLauncher = Join-Path $PSScriptRoot 'TELEGRAM\run_telegram.ps1'
$TennisLauncher = Join-Path $PSScriptRoot 'TENNIS\run_tennis.ps1'
$PowerShellExecutable = if ($PSVersionTable.PSEdition -eq 'Core') {
    Join-Path $PSHOME 'pwsh.exe'
} else {
    Join-Path $PSHOME 'powershell.exe'
}

foreach ($Launcher in @(
    $PowerShellExecutable,
    $Cs2Launcher,
    $TelegramLauncher,
    $TennisLauncher
)) {
    if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
        Write-Error "No existe el lanzador requerido: $Launcher"
        exit 1
    }
}

& $PowerShellExecutable -NoProfile -ExecutionPolicy Bypass `
    -File $Cs2Launcher @ForwardedArguments
$Cs2ExitCode = [int]$LASTEXITCODE
if ($Cs2ExitCode -ne 0) {
    exit $Cs2ExitCode
}

if ($DryRunRequested) {
    Write-Host '[TELEGRAM/TENNIS] Omitidos porque el lanzador raíz está en modo DryRun/WhatIf.'
    exit 0
}

& $PowerShellExecutable -NoProfile -ExecutionPolicy Bypass `
    -File $TelegramLauncher
$TelegramExitCode = [int]$LASTEXITCODE
if ($TelegramExitCode -ne 0) {
    Write-Warning (
        "Telegram terminó con código $TelegramExitCode; " +
        'TENNIS se ejecutará igualmente.'
    )
}

$TennisArguments = @()
if ($RetrainRequested) {
    $TennisArguments += '-Retrain'
}

& $PowerShellExecutable -NoProfile -ExecutionPolicy Bypass `
    -File $TennisLauncher @TennisArguments
$TennisExitCode = [int]$LASTEXITCODE
if ($TennisExitCode -ne 0) {
    exit $TennisExitCode
}
exit $TelegramExitCode
