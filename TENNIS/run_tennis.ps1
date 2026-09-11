<#
Que hace:
    Exige CPython 3.13 y un entorno TENNIS/.venv reproducible. Si falta o es
    incompatible, lo crea o repara solo despues de localizar CPython 3.13.
    Instala exclusivamente requirements.lock.txt desde wheels, comprueba pins
    exactos, ejecuta pip check y carga los binarios ML antes del pipeline.
    En cada arranque operativo completo conserva el commit Sackmann base,
    actualiza las fuentes auxiliares, remapea TennisRatio, reconstruye el Elo
    multifuente/features, entrena o reutiliza por fingerprint los modelos y
    solo entonces ejecuta la inferencia. La actualizacion diaria de
    TennisRatio sigue siendo idempotente.

Que recibe:
    -Date YYYY-MM-DD es opcional; si falta se usa hoy.
    -Retrain se conserva como alias compatible; el ciclo completo ya es el
    comportamiento predeterminado de todo arranque operativo.
    -EnvironmentOnly valida/recrea el venv bloqueado y termina sin pipeline.
    -UpdateOnly actualiza Elo/estadisticas Tennis Abstract y TennisRatio, sin entrenar.

Como se ejecuta desde cualquier ruta:
    .\TENNIS\run_tennis.ps1
    .\TENNIS\run_tennis.ps1 -Date 2026-07-30
    .\TENNIS\run_tennis.ps1 -Retrain
    .\TENNIS\run_tennis.ps1 -Retrain -Date 2026-07-30
    .\TENNIS\run_tennis.ps1 -UpdateOnly
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidatePattern('^\d{4}-\d{2}-\d{2}$')]
    [string]$Date,

    [switch]$Retrain,

    [switch]$EnvironmentOnly,

    [switch]$UpdateOnly
)

$ErrorActionPreference = 'Stop'
$ExpectedPythonMajor = 3
$ExpectedPythonMinor = 13
$TennisRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepositoryRoot = Split-Path -Parent $TennisRoot
$VenvRoot = Join-Path $TennisRoot '.venv'
$VenvBuildRoot = Join-Path $TennisRoot '.venv.build'
$VenvPreviousRoot = Join-Path $TennisRoot '.venv.previous'
$PythonExecutable = Join-Path $VenvRoot 'Scripts\python.exe'
$VenvConfig = Join-Path $VenvRoot 'pyvenv.cfg'
$EnvironmentStamp = Join-Path $VenvRoot '.tennis-lock.sha256'
$RequirementsSource = Join-Path $TennisRoot 'requirements.txt'
$RequirementsLock = Join-Path $TennisRoot 'requirements.lock.txt'
$TennisRatioUpdateScript = Join-Path $TennisRoot 'scripts\update_tennisratio.py'
$TennisAbstractUpdateScript = Join-Path $TennisRoot 'scripts\update_tennis_abstract.py'
$DailyScript = Join-Path $TennisRoot 'scripts\daily_predictions.py'
$WebBuildScript = Join-Path $RepositoryRoot 'WEB\build_web.py'
$TennisLogsRoot = Join-Path $TennisRoot 'logs'
$LogRetentionKeep = 2
$script:TennisRunStartedAtUtc = [DateTime]::UtcNow
$TennisLogToken = '{0}_{1}' -f (
    $script:TennisRunStartedAtUtc.ToString('yyyyMMdd_HHmmssfff'),
    $PID
)
$script:TennisLogPath = Join-Path $TennisLogsRoot (
    "run_tennis_$TennisLogToken.log"
)
$script:TennisTranscriptStarted = $false

function Remove-OldTennisLogs {
    <# Conserva unicamente el log de la ejecucion actual y el de la anterior. #>
    try {
        $keep = @(
            Get-ChildItem -LiteralPath $TennisLogsRoot -File -Force -ErrorAction Stop |
                Where-Object { $_.Name -match '^run_tennis_\d{8}_\d{9}_\d+\.log$' } |
                Sort-Object LastWriteTimeUtc, Name -Descending |
                Select-Object -First $LogRetentionKeep
        )
        $keepPaths = @($keep | ForEach-Object { $_.FullName })
        Get-ChildItem -LiteralPath $TennisLogsRoot -File -Force -ErrorAction Stop |
            Where-Object {
                $_.Name -match '^run_tennis_\d{8}_\d{9}_\d+\.log$' -and
                $_.FullName -notin $keepPaths
            } |
            ForEach-Object {
                $item = Get-Item -LiteralPath $_.FullName -Force -ErrorAction Stop
                if (
                    $item.Directory.FullName -ne $TennisLogsRoot -or
                    ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
                ) {
                    throw "Ruta de log no segura para la poda: $($item.FullName)"
                }
                Remove-Item -LiteralPath $item.FullName -Force -ErrorAction Stop
            }
    } catch {
        Write-Warning ('No se pudo completar la rotacion de logs de TENNIS: ' + $_.Exception.Message)
    }
}

function Complete-TennisLauncher {
    <# Cierra el log persistente y termina con el codigo exacto del launcher. #>
    param(
        [Parameter(Mandatory = $true)][int]$ExitCode,
        [Parameter(Mandatory = $true)][string]$Outcome
    )

    $finishedAtUtc = [DateTime]::UtcNow
    $summary = (
        ('[TENNIS] RUN_END status={0} exit_code={1} ' +
        'started_at_utc={2} finished_at_utc={3}') -f
        $Outcome,
        $ExitCode,
        $script:TennisRunStartedAtUtc.ToString('o'),
        $finishedAtUtc.ToString('o')
    )
    Write-Host $summary
    Write-Host ('[TENNIS] Log persistente: ' + $script:TennisLogPath)
    if ($script:TennisTranscriptStarted) {
        try {
            Stop-Transcript | Out-Null
        } catch {
            [Console]::Error.WriteLine(
                '[TENNIS] No se pudo cerrar limpiamente el transcript: ' +
                $_.Exception.Message
            )
        }
        $script:TennisTranscriptStarted = $false
    }
    Remove-OldTennisLogs
    exit $ExitCode
}

trap {
    $failure = $_
    Write-Error (
        '[TENNIS] ERROR no controlado [{0}]: {1}' -f
        $failure.Exception.GetType().FullName,
        $failure.Exception.Message
    ) -ErrorAction Continue
    if ($failure.ScriptStackTrace) {
        Write-Host ('[TENNIS] Stack: ' + $failure.ScriptStackTrace)
    }
    Complete-TennisLauncher -ExitCode 1 -Outcome 'exception'
}

New-Item -ItemType Directory -Path $TennisLogsRoot -Force | Out-Null
Start-Transcript -Path $script:TennisLogPath -Force | Out-Null
$script:TennisTranscriptStarted = $true
Write-Host ('[TENNIS] Log persistente: ' + $script:TennisLogPath)
Remove-OldTennisLogs

if ($UpdateOnly -and ($Retrain -or $EnvironmentOnly -or $Date)) {
    throw '-UpdateOnly no se puede combinar con -Retrain, -EnvironmentOnly ni -Date.'
}
$RunTraining = (
    -not $UpdateOnly -and
    ((-not [bool]$Date) -or $Retrain)
)

$script:PythonRuntimeCheck = @'
import pathlib
import platform
import sys

expected_major = int(sys.argv[1])
expected_minor = int(sys.argv[2])
expected_prefix = sys.argv[3]
valid = (
    platform.python_implementation() == 'CPython'
    and sys.version_info[:2] == (expected_major, expected_minor)
)
if expected_prefix != '-':
    valid = valid and (
        pathlib.Path(sys.prefix).resolve()
        == pathlib.Path(expected_prefix).resolve()
    )
raise SystemExit(0 if valid else 1)
'@

$script:LockCheckCode = @'
from importlib import metadata
from pathlib import Path
import re
import sys

lock_path = Path(sys.argv[1])
try:
    raw_text = lock_path.read_text(encoding='utf-8-sig')
except OSError:
    raise SystemExit(2)

logical_lines = []
pending = ''
for original in raw_text.splitlines():
    stripped = original.strip()
    if not stripped or stripped.startswith('#'):
        continue
    pending = (pending + ' ' + stripped).strip() if pending else stripped
    if pending.endswith("\\"):
        pending = pending[:-1].rstrip()
        continue
    logical_lines.append(pending)
    pending = ''
if pending:
    raise SystemExit(2)

allowed_options = (
    "--index-url ",
    "--extra-index-url ",
    "--trusted-host ",
    "--only-binary ",
    "--only-binary=",
    "--prefer-binary",
    "--require-hashes",
)
pin_pattern = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*)"
    r"(?:\[[A-Za-z0-9_,.-]+\])?"
    r"==([^\s;]+)$"
)
pins = {}
for logical in logical_lines:
    if logical.startswith(allowed_options):
        continue
    if "--hash=" not in logical:
        raise SystemExit(2)
    without_hashes = re.sub(r"\s+--hash=\S+", "", logical).strip()
    match = pin_pattern.fullmatch(without_hashes)
    if match is None:
        raise SystemExit(2)
    name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
    pinned_version = match.group(2)
    previous = pins.get(name)
    if previous is not None and previous != pinned_version:
        raise SystemExit(2)
    pins[name] = pinned_version

if not pins:
    raise SystemExit(2)

for name, pinned_version in pins.items():
    try:
        installed_version = metadata.version(name)
    except metadata.PackageNotFoundError:
        raise SystemExit(4)
    if installed_version != pinned_version:
        raise SystemExit(4)

installed_names = {
    re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
    for distribution in metadata.distributions()
    if distribution.metadata.get("Name")
}
unexpected = installed_names - set(pins) - {"pip"}
if unexpected:
    raise SystemExit(4)
raise SystemExit(0)
'@

$script:ManifestCheckCode = @'
from pathlib import Path
import re
import sys

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ModuleNotFoundError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version

canonical = lambda value: re.sub(r'[-_.]+', '-', value).lower()
direct = [
    Requirement(line.strip())
    for line in Path(sys.argv[1]).read_text(encoding='utf-8-sig').splitlines()
    if line.strip() and not line.lstrip().startswith('#')
]
pins = {}
hashed = set()
current = None
for raw in Path(sys.argv[2]).read_text(encoding='utf-8-sig').splitlines():
    line = raw.strip()
    candidate = line[:-1].rstrip() if line.endswith(chr(92)) else line
    if '==' in candidate and not candidate.startswith(('#', '--')):
        requirement = Requirement(candidate)
        exact = [
            spec.version for spec in requirement.specifier
            if spec.operator == '==' and '*' not in spec.version
        ]
        if len(exact) != 1 or len(list(requirement.specifier)) != 1:
            raise SystemExit(f'non-exact lock pin: {candidate}')
        current = canonical(requirement.name)
        if requirement.marker is None or requirement.marker.evaluate():
            pins[current] = exact[0]
    if current and re.search(r'--hash=sha256:[0-9a-fA-F]{64}', raw):
        hashed.add(current)

active = [item for item in direct if item.marker is None or item.marker.evaluate()]
missing = sorted(canonical(item.name) for item in active if canonical(item.name) not in pins)
unhashed = sorted(set(pins) - hashed)
incompatible = [
    f'{item.name}=={pins[canonical(item.name)]} not in {item.specifier}'
    for item in active
    if item.specifier and not item.specifier.contains(
        Version(pins[canonical(item.name)]), prereleases=True
    )
]
if not direct or not pins or missing or unhashed or incompatible:
    raise SystemExit(
        f'missing={missing}; unhashed={unhashed}; incompatible={incompatible}'
    )
'@

$script:CompiledRuntimeSmokeCode = @'
import importlib

for module_name in (
    'mypy.main',
    'numpy',
    'pandas',
    'pyarrow',
    'pyarrow.parquet',
    'scipy',
    'sklearn',
    'lightgbm',
    'lxml.etree',
    'curl_cffi',
):
    importlib.import_module(module_name)
'@

function ConvertTo-PythonPayload {
    param([Parameter(Mandatory = $true)][string]$Code)
    return [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
}

function Test-CpythonCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$PrefixArguments = @(),
        [string]$ExpectedPrefix = '-'
    )

    $payload = ConvertTo-PythonPayload -Code $script:PythonRuntimeCheck
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Un Python Launcher sin la version pedida escribe a stderr. Es un
        # probe negativo esperado: no debe impedir probar el siguiente
        # candidato de Find-Cpython313.
        $ErrorActionPreference = 'Continue'
        & $Command @PrefixArguments -c `
            'import base64,sys;code=base64.b64decode(sys.argv.pop(1));exec(code)' `
            $payload $ExpectedPythonMajor $ExpectedPythonMinor $ExpectedPrefix *> $null
        $statusCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    return ($statusCode -eq 0)
}

function Find-Cpython313 {
    $launcher = Get-Command py -ErrorAction SilentlyContinue | Select-Object -First 1
    if (
        $launcher -and
        (Test-CpythonCommand -Command $launcher.Source -PrefixArguments @('-3.13'))
    ) {
        return [pscustomobject]@{
            Command = $launcher.Source
            PrefixArguments = @('-3.13')
        }
    }

    foreach ($candidateName in @('python3.13', 'python')) {
        $candidate = Get-Command $candidateName -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($candidate -and (Test-CpythonCommand -Command $candidate.Source)) {
            return [pscustomobject]@{
                Command = $candidate.Source
                PrefixArguments = @()
            }
        }
    }

    throw (
        'TENNIS requiere CPython 3.13. Instale Python 3.13 x64 y asegure ' +
        'que py -3.13 (o python3.13) este disponible. El venv existente ' +
        'no se ha modificado.'
    )
}

function Test-TennisVenv {
    param(
        [string]$Root = $VenvRoot
    )

    $config = Join-Path $Root 'pyvenv.cfg'
    $python = Join-Path $Root 'Scripts\python.exe'
    if (
        -not (Test-Path -LiteralPath $config -PathType Leaf) -or
        -not (Test-Path -LiteralPath $python -PathType Leaf)
    ) {
        return $false
    }
    if (-not (Test-CpythonCommand -Command $python -ExpectedPrefix $Root)) {
        return $false
    }
    & $python -c 'import pip' *> $null
    return ($LASTEXITCODE -eq 0)
}

function Remove-TennisManagedDirectory {
    <# Elimina solo un directorio venv inmediato, validado y no reparse-point. #>
    param(
        [Parameter(Mandatory = $true)][string]$Path
    )
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $root = [IO.Path]::GetFullPath($TennisRoot).TrimEnd('\', '/')
    $resolved = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $name = [IO.Path]::GetFileName($resolved)
    $allowedExact = @('.venv', '.venv.build', '.venv.previous')
    if (
        [IO.Path]::GetDirectoryName($resolved) -ine $root -or
        $name -notin $allowedExact -or
        -not (Test-Path -LiteralPath $resolved -PathType Container)
    ) {
        throw ('Ruta de venv fuera de TENNIS: ' + $resolved)
    }
    $item = Get-Item -LiteralPath $resolved -Force
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw ('Se rechazo un venv enlazado/reparse-point: ' + $resolved)
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}

function New-TennisVenv {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$BootstrapPython,
        [Parameter(Mandatory = $true)]
        [string]$TargetRoot
    )

    $resolvedTennisRoot = [IO.Path]::GetFullPath($TennisRoot).TrimEnd('\', '/')
    $resolvedVenvRoot = [IO.Path]::GetFullPath($TargetRoot).TrimEnd('\', '/')
    if (
        [IO.Path]::GetFileName($resolvedVenvRoot) -ine '.venv.build' -or
        [IO.Path]::GetDirectoryName($resolvedVenvRoot) -ine $resolvedTennisRoot
    ) {
        throw "Ruta de candidato insegura; se rechazo la operacion: $resolvedVenvRoot"
    }

    if (Test-Path -LiteralPath $resolvedVenvRoot) {
        throw ('El candidato ya existe y no se sobrescribira: ' + $resolvedVenvRoot)
    }
    $venvArguments = @('-m', 'venv', $resolvedVenvRoot)

    Write-Host ("[TENNIS] Creando entorno CPython 3.13 desde " + $BootstrapPython.Command)
    & $BootstrapPython.Command @($BootstrapPython.PrefixArguments) @venvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "No se pudo crear TENNIS/.venv con CPython 3.13."
    }
    if (-not (Test-TennisVenv -Root $resolvedVenvRoot)) {
        throw "El TENNIS/.venv creado no es un CPython 3.13 valido."
    }
}

function Get-LockStatus {
    param(
        [Parameter(Mandatory = $true)][string]$Python
    )
    $payload = ConvertTo-PythonPayload -Code $script:LockCheckCode
    & $Python -c 'import base64,sys;code=base64.b64decode(sys.argv.pop(1));exec(code)' `
        $payload $RequirementsLock *> $null
    $statusCode = $LASTEXITCODE
    if ($statusCode -eq 0) {
        return 'satisfied'
    }
    if ($statusCode -eq 4) {
        return 'mismatch'
    }
    throw (
        'requirements.lock.txt es invalido: debe contener exclusivamente ' +
        'pins exactos nombre==version (se admiten extras y hashes pip).'
    )
}

function Install-RequirementsLock {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [switch]$ForceReinstall
    )
    Write-Host '[TENNIS] Sincronizando solo requirements.lock.txt (wheels only)'
    $pipArguments = @(
        '-m', 'pip', 'install',
        '--disable-pip-version-check',
        '--no-input',
        '--no-deps',
        '--require-hashes',
        '--only-binary=:all:',
        '--requirement', $RequirementsLock
    )
    if ($ForceReinstall) { $pipArguments += '--force-reinstall' }
    & $Python @pipArguments
    if ($LASTEXITCODE -ne 0) {
        throw (
            'No se pudo instalar requirements.lock.txt usando solo wheels. ' +
            'No se permite compilar dependencias desde codigo fuente.'
        )
    }
}

function Test-PipConsistency {
    param(
        [Parameter(Mandatory = $true)][string]$Python
    )
    & $Python -m pip check *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-TennisRuntimeImports {
    param(
        [Parameter(Mandatory = $true)][string]$Python
    )
    $payload = ConvertTo-PythonPayload -Code $script:CompiledRuntimeSmokeCode
    & $Python -B -c `
        'import base64,sys;code=base64.b64decode(sys.argv.pop(1));exec(code)' `
        $payload *> $null
    return ($LASTEXITCODE -eq 0)
}

function Initialize-TennisEnvironment {
    $bootstrap = Find-Cpython313
    $manifestPayload = ConvertTo-PythonPayload -Code $script:ManifestCheckCode
    $manifestOutput = & $bootstrap.Command @($bootstrap.PrefixArguments) -c `
        'import base64,sys;code=base64.b64decode(sys.argv.pop(1));exec(code)' `
        $manifestPayload $RequirementsSource $RequirementsLock 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ('requirements.lock.txt no deriva de requirements.txt: ' + ($manifestOutput -join ' '))
    }

    $lockHash = (Get-FileHash -LiteralPath $RequirementsLock -Algorithm SHA256).Hash.ToLowerInvariant()
    $stampOk = $false
    if ((Test-TennisVenv) -and (Test-Path -LiteralPath $EnvironmentStamp -PathType Leaf)) {
        $stampOk = (
            (Get-Content -LiteralPath $EnvironmentStamp -Raw).Trim().ToLowerInvariant() -eq
            $lockHash
        )
    }
    if (
        (Test-TennisVenv) -and $stampOk -and
        ((Get-LockStatus -Python $PythonExecutable) -eq 'satisfied') -and
        (Test-PipConsistency -Python $PythonExecutable) -and
        (Test-TennisRuntimeImports -Python $PythonExecutable)
    ) {
        Write-Host '[TENNIS] CPython 3.13, lock, pip check e imports binarios OK.'
        return
    }

    Write-Host '[TENNIS] Preparando un reemplazo reproducible sin tocar el venv activo.'
    if (Test-Path -LiteralPath $VenvBuildRoot) {
        Remove-TennisManagedDirectory -Path $VenvBuildRoot
        Write-Host '[TENNIS] Candidato incompleto anterior eliminado.'
    }
    New-TennisVenv -BootstrapPython $bootstrap -TargetRoot $VenvBuildRoot
    $buildPython = Join-Path $VenvBuildRoot 'Scripts\python.exe'
    Install-RequirementsLock -Python $buildPython
    if (
        -not (Test-TennisVenv -Root $VenvBuildRoot) -or
        ((Get-LockStatus -Python $buildPython) -ne 'satisfied') -or
        -not (Test-PipConsistency -Python $buildPython) -or
        -not (Test-TennisRuntimeImports -Python $buildPython)
    ) {
        throw 'El candidato no satisface CPython 3.13, lock, pip check o imports binarios; el activo sigue intacto.'
    }
    Set-Content -LiteralPath (Join-Path $VenvBuildRoot '.tennis-lock.sha256') `
        -Value $lockHash -Encoding ascii -NoNewline

    $activeBackedUp = $false
    $candidateActivated = $false
    try {
        if (Test-Path -LiteralPath $VenvPreviousRoot) {
            Remove-TennisManagedDirectory -Path $VenvPreviousRoot
            Write-Host '[TENNIS] Entorno previo más antiguo eliminado.'
        }
        if (Test-Path -LiteralPath $VenvRoot) {
            Move-Item -LiteralPath $VenvRoot -Destination $VenvPreviousRoot
            $activeBackedUp = $true
        }
        Move-Item -LiteralPath $VenvBuildRoot -Destination $VenvRoot
        $candidateActivated = $true

        & $bootstrap.Command @($bootstrap.PrefixArguments) -m venv --upgrade $VenvRoot
        if ($LASTEXITCODE -ne 0) {
            throw 'No se pudieron reparar las rutas del venv despues del swap.'
        }
        Install-RequirementsLock -Python $PythonExecutable -ForceReinstall
        if (
            -not (Test-TennisVenv) -or
            ((Get-LockStatus -Python $PythonExecutable) -ne 'satisfied') -or
            -not (Test-PipConsistency -Python $PythonExecutable) -or
            -not (Test-TennisRuntimeImports -Python $PythonExecutable)
        ) {
            throw 'El venv activado no supero la validacion posterior al swap.'
        }
        Set-Content -LiteralPath $EnvironmentStamp -Value $lockHash `
            -Encoding ascii -NoNewline
    } catch {
        $swapError = $_.Exception.Message
        try {
            if ($candidateActivated -and (Test-Path -LiteralPath $VenvRoot)) {
                Remove-TennisManagedDirectory -Path $VenvRoot
                Write-Host '[TENNIS] Candidato activado fallido eliminado.'
            }
            if ($activeBackedUp -and (Test-Path -LiteralPath $VenvPreviousRoot)) {
                Move-Item -LiteralPath $VenvPreviousRoot -Destination $VenvRoot
            }
        } catch {
            throw ('Fallo el swap (' + $swapError + ') y tambien el rollback: ' + $_.Exception.Message)
        }
        if ($activeBackedUp) {
            throw ('Fallo el swap; el entorno anterior fue restaurado: ' + $swapError)
        }
        throw ('Fallo el swap y no existia un entorno anterior: ' + $swapError)
    }

    Write-Host '[TENNIS] CPython 3.13, lock, pip check e imports binarios OK.'
}

$RequiredPaths = @(
    $RequirementsSource,
    $RequirementsLock,
    $TennisRatioUpdateScript,
    $TennisAbstractUpdateScript
)
if ($RunTraining) {
    $RequiredPaths += @($DailyScript, $WebBuildScript)
}
foreach ($RequiredPath in $RequiredPaths) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) {
        throw "Falta un componente requerido de TENNIS: $RequiredPath"
    }
}

Initialize-TennisEnvironment

if ($EnvironmentOnly) {
    Write-Host '[TENNIS] Entorno reproducible listo; pipeline omitido por -EnvironmentOnly.'
    Complete-TennisLauncher -ExitCode 0 -Outcome 'environment_only'
}

if (-not $EnvironmentOnly) {
    $UpdateSourcesScript = Join-Path $TennisRoot 'scripts\update_sources.py'
    if (-not (Test-Path -LiteralPath $UpdateSourcesScript -PathType Leaf)) {
        throw "Falta un script requerido para entrenar: $UpdateSourcesScript"
    }

    # Sackmann es una base congelada por el handoff Elo. Un re-baseline exige
    # una operacion manual documentada; el arranque diario solo actualiza las
    # fuentes auxiliares y nunca mueve ese commit silenciosamente.
    Write-Host ''
    Write-Host '[TENNIS TRAIN] scripts\update_sources.py --skip-sackmann'
    $SourceArguments = @('--skip-sackmann')
    if ($UpdateOnly) { $SourceArguments += '--skip-match-charting' }
    & $PythonExecutable $UpdateSourcesScript @SourceArguments
    if ($LASTEXITCODE -ne 0) {
        $TrainingExitCode = [int]$LASTEXITCODE
        Write-Error (
            'Fallo scripts\update_sources.py con codigo ' +
            "$TrainingExitCode."
        ) -ErrorAction Continue
        Complete-TennisLauncher `
            -ExitCode $TrainingExitCode `
            -Outcome 'retrain_failed'
    }
}

Write-Host ''
Write-Host '[TENNIS] Actualizando TennisRatio con Scrapling (control automatico diario)'
$TennisRatioAttemptedAtUtc = [DateTime]::UtcNow.ToString('o')
$TennisRatioAttemptStatus = 'success'
& $PythonExecutable $TennisRatioUpdateScript
$TennisRatioUpdateExitCode = [int]$LASTEXITCODE
if ($TennisRatioUpdateExitCode -ne 0) {
    $TennisRatioAttemptStatus = 'failed'
    Write-Error (
        'Fallo scripts\update_tennisratio.py con codigo ' +
        "$TennisRatioUpdateExitCode."
    ) -ErrorAction Continue
    if ($UpdateOnly) {
        Complete-TennisLauncher `
            -ExitCode $TennisRatioUpdateExitCode `
            -Outcome 'tennisratio_update_failed'
    }
    Write-Warning (
        '[TENNIS] La fuente fallo al actualizarse; se continua con el ultimo ' +
        'lote disponible o Tennis Explorer. Las predicciones quedaran marcadas.'
    )
}

Write-Host ''
Write-Host '[TENNIS] Tennis Abstract: solo jugadores de la cartelera (Scrapling; adquisicion, no modelo)'
$TennisAbstractArguments = @()
if ($Date) { $TennisAbstractArguments += @('--date', $Date) }
& $PythonExecutable $TennisAbstractUpdateScript @TennisAbstractArguments
$TennisAbstractUpdateExitCode = [int]$LASTEXITCODE
if ($TennisAbstractUpdateExitCode -ne 0) {
    Write-Warning (
        '[TENNIS] Adquisicion Tennis Abstract incompleta/diferida. Se conserva el progreso ' +
        'y el modelo vigente; revise cartelera, identidades y pausas del resumen anterior. ' +
        'No se recurre al barrido completo.'
    )
}

if ($UpdateOnly) {
    Write-Host '[TENNIS] Actualizacion diaria de fuentes terminada; entrenamiento/publicacion omitidos.'
    if ($TennisAbstractUpdateExitCode -ne 0) {
        Complete-TennisLauncher -ExitCode $TennisAbstractUpdateExitCode -Outcome 'tennis_abstract_update_incomplete'
    }
    Complete-TennisLauncher -ExitCode 0 -Outcome 'sources_update_only'
}

if ($RunTraining) {
    $TrainingScripts = @(
        'scripts\audit_identities.py',
        'scripts\build_elo.py',
        'scripts\build_features.py',
        'scripts\retrain_models.py'
    )
    foreach ($RelativeScript in $TrainingScripts) {
        $ScriptPath = Join-Path $TennisRoot $RelativeScript
        if (-not (Test-Path -LiteralPath $ScriptPath -PathType Leaf)) {
            throw "Falta un script requerido para entrenar: $ScriptPath"
        }
        Write-Host ''
        Write-Host "[TENNIS TRAIN] $RelativeScript"
        $ScriptArguments = @()
        if ($RelativeScript -eq 'scripts\build_elo.py' -and $Date) {
            $ScriptArguments += @('--as-of-date', $Date)
        }
        if ($RelativeScript -eq 'scripts\retrain_models.py' -and $Date) {
            $ScriptArguments += @('--training-as-of-date', $Date)
        }
        & $PythonExecutable $ScriptPath @ScriptArguments
        if ($LASTEXITCODE -ne 0) {
            $TrainingExitCode = [int]$LASTEXITCODE
            Write-Error (
                "Fallo $RelativeScript con codigo $TrainingExitCode."
            ) -ErrorAction Continue
            Complete-TennisLauncher `
                -ExitCode $TrainingExitCode `
                -Outcome 'retrain_failed'
        }
    }
}

$DailyArguments = @()
if ($Date) {
    $DailyArguments += @('--date', $Date)
}
if ($RunTraining) {
    $DailyArguments += '--retrained'
}
$DailyArguments += @(
    '--tennisratio-attempt-status', $TennisRatioAttemptStatus,
    '--tennisratio-attempted-at-utc', $TennisRatioAttemptedAtUtc
)

& $PythonExecutable $DailyScript @DailyArguments
$DailyExitCode = $LASTEXITCODE
if ($DailyExitCode -ne 0) {
    Complete-TennisLauncher `
        -ExitCode ([int]$DailyExitCode) `
        -Outcome 'daily_failed'
}

Write-Host ''
Write-Host '[TENNIS] Actualizando WEB/data.js'
& $PythonExecutable $WebBuildScript
$WebExitCode = $LASTEXITCODE
if ($WebExitCode -ne 0) {
    Write-Error (
        "Fallo WEB/build_web.py con codigo $WebExitCode."
    ) -ErrorAction Continue
    Complete-TennisLauncher `
        -ExitCode ([int]$WebExitCode) `
        -Outcome 'web_build_failed'
}
Complete-TennisLauncher -ExitCode 0 -Outcome 'ok'
