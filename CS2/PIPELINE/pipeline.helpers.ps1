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
    $python = Get-ModelBasePython -DryRun:$DryRun
    $result = & $python -c $probe 2>$null
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

function Test-PythonExact313 {
    <# Devuelve true solo para CPython 3.13; no acepta otra implementacion ni minor. #>
    param([Parameter(Mandatory)][string]$Python)
    $code = "import platform, sys; sys.exit(0 if platform.python_implementation() == 'CPython' and sys.version_info[:2] == (3, 13) else 1)"
    try {
        & $Python -c $code *> $null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Find-ModelPython313 {
    <# Localiza un CPython 3.13 exacto sin aceptar silenciosamente otro Python. #>
    $candidates = New-Object System.Collections.Generic.List[string]
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $resolved = & $launcher.Source -3.13 -c "import sys; print(sys.executable)" 2>$null | Select-Object -First 1
            if (($LASTEXITCODE -eq 0) -and $resolved) {
                [void]$candidates.Add(([string]$resolved).Trim())
            }
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
    }

    $system = Get-Command python -ErrorAction SilentlyContinue
    if ($system) { [void]$candidates.Add($system.Source) }
    if ($env:LOCALAPPDATA) {
        [void]$candidates.Add((Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'))
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if ($candidate -and (Test-PythonExact313 -Python $candidate)) { return $candidate }
    }
    return $null
}

function Get-ModelBasePython {
    <# Obtiene CPython 3.13 y, si es posible, lo aprovisiona con winget una vez. #>
    param([switch]$DryRun)
    $python = Find-ModelPython313
    if ($python) { return $python }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget -and (-not $DryRun)) {
        Write-Log "CPython 3.13 no esta disponible; instalando Python.Python.3.13 con winget..." -Level WARN
        $oldErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $winget.Source install --exact --id Python.Python.3.13 --scope user --silent `
                --accept-package-agreements --accept-source-agreements 2>&1 |
                ForEach-Object { Write-Log ([string]$_) -Level DEBUG }
        } finally {
            $ErrorActionPreference = $oldErrorActionPreference
        }
        $python = Find-ModelPython313
        if ($python) { return $python }
    }

    throw "El modelo exige CPython 3.13 exacto. Instala Python 3.13 (por ejemplo, 'winget install --exact --id Python.Python.3.13 --scope user') y vuelve a ejecutar start.ps1."
}

function Assert-RequirementsLockDerived {
    <# Valida que el lock tenga pins exactos con hashes para cada requisito directo. #>
    param(
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string]$RequirementsPath,
        [Parameter(Mandatory)][string]$LockPath
    )
    if (-not (Test-Path -LiteralPath $RequirementsPath)) {
        throw "No existe el manifiesto de dependencias del modelo: $RequirementsPath"
    }
    if (-not (Test-Path -LiteralPath $LockPath)) {
        throw "No existe el lock de dependencias del modelo: $LockPath"
    }

    $validator = @'
import pathlib
import re
import sys

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ModuleNotFoundError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version

requirements_path = pathlib.Path(sys.argv[1])
lock_path = pathlib.Path(sys.argv[2])
canonical = lambda value: re.sub(r'[-_.]+', '-', value).lower()

direct = []
for raw in requirements_path.read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    try:
        direct.append(Requirement(line))
    except Exception as exc:
        raise SystemExit(f'unsupported direct requirement {line!r}: {exc}') from exc

pin_pattern = re.compile(
    r'^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)(?:\s*;\s*.+)?$'
)
hash_pattern = re.compile(r'--hash=sha256:[0-9a-fA-F]{64}(?:\s|\\|$)')
pins = []
current = None
for raw in lock_path.read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    candidate = line[:-1].rstrip() if line.endswith('\\') else line
    match = pin_pattern.match(candidate)
    if match:
        try:
            locked_requirement = Requirement(candidate)
        except Exception as exc:
            raise SystemExit(f'invalid lock requirement {candidate!r}: {exc}') from exc
        exact = [
            spec.version
            for spec in locked_requirement.specifier
            if spec.operator == '==' and '*' not in spec.version
        ]
        if len(exact) != 1 or len(list(locked_requirement.specifier)) != 1:
            raise SystemExit(f'lock requirement is not one exact pin: {candidate}')
        current = {
            'name': canonical(locked_requirement.name),
            'version': exact[0],
            'active': locked_requirement.marker is None or locked_requirement.marker.evaluate(),
            'hashed': bool(hash_pattern.search(raw)),
        }
        pins.append(current)
        continue
    if current is not None and hash_pattern.search(raw):
        current['hashed'] = True

if not direct:
    raise SystemExit('requirements.txt has no direct requirements')
if not pins:
    raise SystemExit('requirements.lock.txt has no exact pins')
active_pins = {item['name']: item['version'] for item in pins if item['active']}
active_direct = [item for item in direct if item.marker is None or item.marker.evaluate()]
missing = sorted(
    canonical(item.name) for item in active_direct if canonical(item.name) not in active_pins
)
unhashed = sorted({item['name'] for item in pins if not item['hashed']})
if missing:
    raise SystemExit('direct requirements missing from lock: ' + ', '.join(missing))
if unhashed:
    raise SystemExit('lock pins without sha256 hashes: ' + ', '.join(unhashed))
incompatible = []
for requirement in active_direct:
    version = active_pins[canonical(requirement.name)]
    if requirement.specifier and not requirement.specifier.contains(Version(version), prereleases=True):
        incompatible.append(f'{requirement.name} locked={version} requires={requirement.specifier}')
if incompatible:
    raise SystemExit('lock pins incompatible with requirements.txt: ' + ', '.join(incompatible))
'@
    $output = & $Python -c $validator $RequirementsPath $LockPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "requirements.lock.txt no es un lock completo derivado de requirements.txt: $($output -join ' ')"
    }
}

function Test-LockedEnvironment {
    <# Comprueba que cada pin activo del lock coincide exactamente con el venv. #>
    param(
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string]$LockPath
    )
    $validator = @'
import importlib.metadata
import pathlib
import re
import sys

from packaging.requirements import Requirement
from packaging.version import Version

errors = []
canonical = lambda value: re.sub(r'[-_.]+', '-', value).lower()
active_pins = set()
for raw in pathlib.Path(sys.argv[1]).read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if not line or line.startswith(('#', '--')) or '==' not in line:
        continue
    candidate = line[:-1].rstrip() if line.endswith('\\') else line
    try:
        requirement = Requirement(candidate)
    except Exception as exc:
        errors.append(f'invalid lock requirement {candidate!r}: {exc}')
        continue
    if requirement.marker is not None and not requirement.marker.evaluate():
        continue
    exact = [spec.version for spec in requirement.specifier if spec.operator == '==' and '*' not in spec.version]
    if len(exact) != 1 or len(list(requirement.specifier)) != 1:
        errors.append(f'lock requirement is not one exact pin: {candidate}')
        continue
    active_pins.add(canonical(requirement.name))
    try:
        installed = importlib.metadata.version(requirement.name)
    except importlib.metadata.PackageNotFoundError:
        errors.append(f'missing: {requirement.name}=={exact[0]}')
        continue
    if Version(installed) != Version(exact[0]):
        errors.append(f'version mismatch: {requirement.name} installed={installed} locked={exact[0]}')

installed_names = {
    canonical(distribution.metadata['Name'])
    for distribution in importlib.metadata.distributions()
    if distribution.metadata.get('Name')
}
extras = sorted(installed_names - active_pins - {'pip'})
if extras:
    errors.append('unexpected distributions not present in active lock pins: ' + ', '.join(extras))

if errors:
    print('; '.join(errors))
    raise SystemExit(1)
'@
    $output = & $Python -c $validator $LockPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Log ("El entorno Python no satisface exactamente el lock: " + ($output -join ' ')) -Level WARN
        return $false
    }
    return $true
}

function Test-PipCheck {
    <# Ejecuta pip check en el interprete indicado sin alterar el entorno. #>
    param([Parameter(Mandatory)][string]$Python)
    & $Python -m pip check *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-ModelRuntimeImports {
    <# Carga dependencias criticas y sus binarios bajo la politica de Windows. #>
    param([Parameter(Mandatory)][string]$Python)
    & $Python -B -c @'
import catboost
import lightgbm
import matplotlib
import numpy
import optuna
import pandas
import parsel
import scipy
import shap
import sklearn
import xgboost
import yaml
'@ *> $null
    return ($LASTEXITCODE -eq 0)
}

function Test-ScraperRuntimeImports {
    <# Carga el transporte/parser compilado antes de aceptar el venv scraper. #>
    param([Parameter(Mandatory)][string]$Python)
    & $Python -B -c @'
import curl_cffi
import flask
import lxml.etree
import parsel
import scrapling
import scrapy
'@ *> $null
    return ($LASTEXITCODE -eq 0)
}

function Ensure-ModelPython {
    <# Aprovisiona CS2/.venv desde el unico lock con hashes y valida su integridad. #>
    param(
        [Parameter(Mandatory)][string]$ProjectRoot,
        [switch]$DryRun
    )
    $resolvedRoot = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\', '/')
    $venvDir = [System.IO.Path]::GetFullPath((Join-Path $resolvedRoot '.venv'))
    $buildDir = [System.IO.Path]::GetFullPath((Join-Path $resolvedRoot '.venv.build'))
    $previousDir = [System.IO.Path]::GetFullPath((Join-Path $resolvedRoot '.venv.previous'))
    foreach ($managedDir in @($venvDir, $buildDir, $previousDir)) {
        if ([System.IO.Directory]::GetParent($managedDir).FullName.TrimEnd('\', '/') -ne $resolvedRoot) {
            throw "Ruta de venv fuera del proyecto: $managedDir"
        }
    }
    $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    $requirements = Join-Path $resolvedRoot 'requirements.txt'
    $lock = Join-Path $resolvedRoot 'requirements.lock.txt'
    $stamp = Join-Path $venvDir '.requirements-lock.sha256'
    $basePython = Get-ModelBasePython -DryRun:$DryRun

    Assert-RequirementsLockDerived -Python $basePython -RequirementsPath $requirements -LockPath $lock
    $lockHash = (Get-FileHash -LiteralPath $lock -Algorithm SHA256).Hash.ToLowerInvariant()
    $venvVersionOk = (Test-Path -LiteralPath $venvPython) -and (Test-PythonExact313 -Python $venvPython)
    $stampOk = $false
    if ($venvVersionOk -and (Test-Path -LiteralPath $stamp)) {
        $stampOk = ((Get-Content -LiteralPath $stamp -Raw).Trim().ToLowerInvariant() -eq $lockHash)
    }
    if ($venvVersionOk -and $stampOk -and (Test-PipCheck -Python $venvPython) -and
        (Test-LockedEnvironment -Python $venvPython -LockPath $lock) -and
        (Test-ModelRuntimeImports -Python $venvPython)) {
        return $venvPython
    }

    if ($DryRun) {
        Write-Log "DRY-RUN: se crearia/repararia CS2/.venv con CPython 3.13 y requirements.lock.txt." -Level WARN
        return $venvPython
    }

    Write-Log "Construyendo un reemplazo limpio del venv del modelo sin tocar el entorno activo." -Level WARN
    if (Test-Path -LiteralPath $buildDir) {
        Remove-Item -LiteralPath $buildDir -Recurse -Force
    }
    Invoke-Native $basePython @('-m', 'venv', $buildDir) 'Creacion candidato venv modelo CPython 3.13' | Out-Null
    $buildPython = Join-Path $buildDir 'Scripts\python.exe'
    Invoke-Native $buildPython @(
        '-m', 'pip', 'install', '--no-deps', '--only-binary=:all:', '--require-hashes',
        '-r', $lock
    ) 'Instalacion lock candidato modelo' | Out-Null
    if (-not (Test-PythonExact313 -Python $buildPython)) {
        throw "El candidato del modelo no usa CPython 3.13 exacto; el entorno activo no se modifico."
    }
    if (-not (Test-PipCheck -Python $buildPython)) {
        throw "pip check fallo en el candidato del modelo; el entorno activo no se modifico."
    }
    if (-not (Test-LockedEnvironment -Python $buildPython -LockPath $lock)) {
        throw "El candidato del modelo no coincide exactamente con el lock; el entorno activo no se modifico."
    }
    if (-not (Test-ModelRuntimeImports -Python $buildPython)) {
        throw "El candidato del modelo no puede cargar sus binarios; el entorno activo no se modifico."
    }
    Set-Content -LiteralPath (Join-Path $buildDir '.requirements-lock.sha256') -Value $lockHash -Encoding ascii -NoNewline

    $activeBackedUp = $false
    $candidateActivated = $false
    try {
        $activeExists = Test-Path -LiteralPath $venvDir
        if ($activeExists -and (Test-Path -LiteralPath $previousDir)) {
            Remove-Item -LiteralPath $previousDir -Recurse -Force
        }
        if ($activeExists) {
            Move-Item -LiteralPath $venvDir -Destination $previousDir
            $activeBackedUp = $true
        } elseif (Test-Path -LiteralPath $previousDir) {
            # Conserva un backup de una transaccion previa si el activo falta.
            $activeBackedUp = $true
        }
        Move-Item -LiteralPath $buildDir -Destination $venvDir
        $candidateActivated = $true

        # Los venv no son portables: regenera scripts y entrypoints despues del
        # rename, siempre desde el mismo interprete base y el mismo lock.
        Invoke-Native $basePython @('-m', 'venv', '--upgrade', $venvDir) 'Reparacion rutas venv modelo' | Out-Null
        Invoke-Native $venvPython @(
            '-m', 'pip', 'install', '--force-reinstall', '--no-deps', '--only-binary=:all:',
            '--require-hashes', '-r', $lock
        ) 'Reinstalacion lock tras swap modelo' | Out-Null
        if (-not (Test-PythonExact313 -Python $venvPython)) {
            throw "El venv activado del modelo dejo de usar CPython 3.13."
        }
        if (-not (Test-PipCheck -Python $venvPython)) {
            throw "pip check fallo despues del swap del venv del modelo."
        }
        if (-not (Test-LockedEnvironment -Python $venvPython -LockPath $lock)) {
            throw "El venv del modelo no coincide con el lock despues del swap."
        }
        if (-not (Test-ModelRuntimeImports -Python $venvPython)) {
            throw "El venv activado del modelo no puede cargar sus binarios."
        }
        Set-Content -LiteralPath $stamp -Value $lockHash -Encoding ascii -NoNewline
    } catch {
        $swapError = $_.Exception.Message
        try {
            if ($candidateActivated -and (Test-Path -LiteralPath $venvDir)) {
                Move-Item -LiteralPath $venvDir -Destination $buildDir
            }
            if ($activeBackedUp -and (Test-Path -LiteralPath $previousDir)) {
                Move-Item -LiteralPath $previousDir -Destination $venvDir
            }
        } catch {
            throw "Fallo el swap del venv del modelo ($swapError) y tambien su rollback: $($_.Exception.Message)"
        }
        if ($activeBackedUp) {
            throw "Fallo el swap del venv del modelo; el entorno anterior fue restaurado: $swapError"
        }
        if ($activeExists) {
            throw "Fallo el swap del venv del modelo; el entorno activo original permanecio intacto: $swapError"
        }
        throw "Fallo el swap del venv del modelo y no existia un entorno anterior que restaurar: $swapError"
    }
    return $venvPython
}

function Ensure-ScraperPython {
    <# Aprovisiona el venv del scraper desde su lock con hashes y wheels locales. #>
    param(
        [Parameter(Mandatory)][string]$ScraperDir,
        [switch]$Recreate,
        [switch]$DryRun
    )
    $resolvedScraperDir = [System.IO.Path]::GetFullPath($ScraperDir).TrimEnd('\', '/')
    $venvDir = [System.IO.Path]::GetFullPath((Join-Path $resolvedScraperDir '.venv'))
    $buildDir = [System.IO.Path]::GetFullPath((Join-Path $resolvedScraperDir '.venv.build'))
    $previousDir = [System.IO.Path]::GetFullPath((Join-Path $resolvedScraperDir '.venv.previous'))
    foreach ($managedDir in @($venvDir, $buildDir, $previousDir)) {
        if ([System.IO.Directory]::GetParent($managedDir).FullName.TrimEnd('\', '/') -ne $resolvedScraperDir) {
            throw "Ruta de venv del scraper fuera de su componente: $managedDir"
        }
    }
    $venvPython = Join-Path $venvDir 'Scripts\python.exe'
    $requirements = Join-Path $resolvedScraperDir 'requirements.txt'
    $lock = Join-Path $resolvedScraperDir 'requirements.lock.txt'
    $wheelDir = Join-Path $resolvedScraperDir 'wheels'
    $stamp = Join-Path $venvDir '.requirements-lock.sha256'
    $basePython = Get-ModelBasePython -DryRun:$DryRun

    if (-not (Test-Path -LiteralPath $wheelDir -PathType Container)) {
        throw "No existe el directorio de wheels verificados del scraper: $wheelDir"
    }
    Assert-RequirementsLockDerived -Python $basePython -RequirementsPath $requirements -LockPath $lock
    $lockHash = (Get-FileHash -LiteralPath $lock -Algorithm SHA256).Hash.ToLowerInvariant()
    $versionOk = (Test-Path -LiteralPath $venvPython) -and (Test-PythonExact313 -Python $venvPython)
    $stampOk = $false
    if ($versionOk -and (Test-Path -LiteralPath $stamp)) {
        $stampOk = ((Get-Content -LiteralPath $stamp -Raw).Trim().ToLowerInvariant() -eq $lockHash)
    }
    $valid = (-not $Recreate) -and $versionOk -and $stampOk -and
        (Test-PipCheck -Python $venvPython) -and
        (Test-LockedEnvironment -Python $venvPython -LockPath $lock) -and
        (Test-ScraperRuntimeImports -Python $venvPython)
    if ($valid) { return $venvPython }

    if ($DryRun) {
        Write-Log "DRY-RUN: se crearia/repararia el venv del scraper con CPython 3.13 y su requirements.lock.txt." -Level WARN
        return $venvPython
    }

    Write-Log "Construyendo un reemplazo limpio del venv del scraper sin tocar el entorno activo." -Level WARN
    if (Test-Path -LiteralPath $buildDir) {
        Remove-Item -LiteralPath $buildDir -Recurse -Force
    }
    Invoke-Native $basePython @('-m', 'venv', $buildDir) 'Creacion candidato venv scraper CPython 3.13' | Out-Null
    $buildPython = Join-Path $buildDir 'Scripts\python.exe'
    Invoke-Native $buildPython @(
        '-m', 'pip', 'install', '--no-deps', '--only-binary=:all:', '--require-hashes',
        '--find-links', $wheelDir, '-r', $lock
    ) 'Instalacion lock candidato scraper' | Out-Null
    if (-not (Test-PythonExact313 -Python $buildPython)) {
        throw "El candidato del scraper no usa CPython 3.13 exacto; el entorno activo no se modifico."
    }
    if (-not (Test-PipCheck -Python $buildPython)) {
        throw "pip check fallo en el candidato del scraper; el entorno activo no se modifico."
    }
    if (-not (Test-LockedEnvironment -Python $buildPython -LockPath $lock)) {
        throw "El candidato del scraper no coincide exactamente con el lock; el entorno activo no se modifico."
    }
    if (-not (Test-ScraperRuntimeImports -Python $buildPython)) {
        throw "El candidato del scraper no puede cargar sus binarios; el entorno activo no se modifico."
    }
    Set-Content -LiteralPath (Join-Path $buildDir '.requirements-lock.sha256') -Value $lockHash -Encoding ascii -NoNewline

    $activeBackedUp = $false
    $candidateActivated = $false
    try {
        $activeExists = Test-Path -LiteralPath $venvDir
        if ($activeExists -and (Test-Path -LiteralPath $previousDir)) {
            Remove-Item -LiteralPath $previousDir -Recurse -Force
        }
        if ($activeExists) {
            Move-Item -LiteralPath $venvDir -Destination $previousDir
            $activeBackedUp = $true
        } elseif (Test-Path -LiteralPath $previousDir) {
            $activeBackedUp = $true
        }
        Move-Item -LiteralPath $buildDir -Destination $venvDir
        $candidateActivated = $true

        Invoke-Native $basePython @('-m', 'venv', '--upgrade', $venvDir) 'Reparacion rutas venv scraper' | Out-Null
        Invoke-Native $venvPython @(
            '-m', 'pip', 'install', '--force-reinstall', '--no-deps', '--only-binary=:all:',
            '--require-hashes', '--find-links', $wheelDir, '-r', $lock
        ) 'Reinstalacion lock tras swap scraper' | Out-Null
        if (-not (Test-PythonExact313 -Python $venvPython)) {
            throw "El venv activado del scraper dejo de usar CPython 3.13."
        }
        if (-not (Test-PipCheck -Python $venvPython)) {
            throw "pip check fallo despues del swap del venv del scraper."
        }
        if (-not (Test-LockedEnvironment -Python $venvPython -LockPath $lock)) {
            throw "El venv del scraper no coincide con el lock despues del swap."
        }
        if (-not (Test-ScraperRuntimeImports -Python $venvPython)) {
            throw "El venv activado del scraper no puede cargar sus binarios."
        }
        Set-Content -LiteralPath $stamp -Value $lockHash -Encoding ascii -NoNewline
    } catch {
        $swapError = $_.Exception.Message
        try {
            if ($candidateActivated -and (Test-Path -LiteralPath $venvDir)) {
                Move-Item -LiteralPath $venvDir -Destination $buildDir
            }
            if ($activeBackedUp -and (Test-Path -LiteralPath $previousDir)) {
                Move-Item -LiteralPath $previousDir -Destination $venvDir
            }
        } catch {
            throw "Fallo el swap del venv del scraper ($swapError) y tambien su rollback: $($_.Exception.Message)"
        }
        if ($activeBackedUp) {
            throw "Fallo el swap del venv del scraper; el entorno anterior fue restaurado: $swapError"
        }
        if ($activeExists) {
            throw "Fallo el swap del venv del scraper; el entorno activo original permanecio intacto: $swapError"
        }
        throw "Fallo el swap del venv del scraper y no existia un entorno anterior que restaurar: $swapError"
    }

    # Scrapling gestiona los binarios del navegador fuera de pip. Su fallo no
    # altera el entorno bloqueado y queda registrado para diagnostico operativo.
    try {
        & $venvPython -c "from scrapling.cli import install; install([], standalone_mode=False)" `
            2>&1 | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) { Write-Log "'scrapling install' devolvio error; revisa los binarios del navegador." -Level WARN }
    } catch {
        Write-Log ("No se pudieron instalar los navegadores de Scrapling: " + $_.Exception.Message) -Level WARN
    }

    return $venvPython
}
