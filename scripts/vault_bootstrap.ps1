<# Shared storage bootstrap. No sport logic or database writes live here. #>

function Get-VaultBasePython {
    <# Locate the required CPython 3.13 without relying on an activated venv. #>
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $candidate = & py -3.13 -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $candidate) { return ([string]$candidate).Trim() }
        } catch { }
    }
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'),
        (Join-Path $env:ProgramFiles 'Python313\python.exe')
    )
    $command = Get-Command python -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) { continue }
        & $candidate -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3,13) else 1)' *> $null
        if ($LASTEXITCODE -eq 0) { return $candidate }
    }
    throw '[VAULT] Se necesita CPython 3.13 instalado para reconstruir los entornos.'
}

function Initialize-ProjectVault {
    <# Migrate a legacy working tree once; copied vaults need no manual configuration. #>
    param([Parameter(Mandatory)][string]$RepositoryRoot, [switch]$DryRun)
    $resolved = [IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/')
    $vaultRoot = Join-Path $resolved 'VAULT'
    $journal = Join-Path $vaultRoot 'migration.json'
    $needsMigration = -not (Test-Path -LiteralPath $journal)
    if (-not $needsMigration) {
        $state = Get-Content -LiteralPath $journal -Raw -Encoding UTF8 | ConvertFrom-Json
        $needsMigration = $state.status -notin @('migrated', 'migrated_with_pending')
    }
    if (Test-Path -LiteralPath (Join-Path $vaultRoot '.migration.lock')) { $needsMigration = $true }
    if ($needsMigration) {
        if ($DryRun) { return }
        $basePython = Get-VaultBasePython
        Write-Host '[VAULT] Preparando automaticamente el estado privado; no se reescriben las BBDD.'
        & $basePython (Join-Path $resolved 'scripts\vault_migrate.py') --root $resolved --apply
        if ($LASTEXITCODE -notin @(0, 2)) { throw '[VAULT] No se pudo completar la migracion segura.' }
    }
    if (Test-Path -LiteralPath $journal) {
        $state = Get-Content -LiteralPath $journal -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($state.status -notin @('migrated', 'migrated_with_pending')) {
            throw '[VAULT] El inventario no confirma una migracion finalizada.'
        }
        if ($state.pending.Count -gt 0) {
            Write-Warning '[VAULT] Existen residuos pendientes; consulta VAULT/migration.json.'
        }
        foreach ($relative in $state.required) {
            $target = [IO.Path]::GetFullPath((Join-Path $vaultRoot $relative))
            if (-not $target.StartsWith($vaultRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
                -not (Test-Path -LiteralPath $target -PathType Leaf)) {
                throw ('[VAULT] Falta estado necesario. Copia la VAULT completa: ' + $relative)
            }
        }
    }
    $env:PYTHONPYCACHEPREFIX = Join-Path $vaultRoot 'runtime\pycache'
    $env:PIP_CACHE_DIR = Join-Path $vaultRoot 'runtime\pip-cache'
    $env:MPLCONFIGDIR = Join-Path $vaultRoot 'runtime\matplotlib'
    if (-not $DryRun) {
        $locationsPath = Join-Path $vaultRoot 'locations.json'
        $knownRoots = @()
        if (Test-Path -LiteralPath $locationsPath) {
            $knownRoots = @((Get-Content -LiteralPath $locationsPath -Raw -Encoding UTF8 | ConvertFrom-Json).roots)
        }
        if ($resolved -notin $knownRoots) {
            $knownRoots += $resolved
            $locationJson = @{ roots = $knownRoots } | ConvertTo-Json
            $temporary = $locationsPath + '.' + $PID + '.tmp'
            [IO.File]::WriteAllText($temporary, $locationJson, (New-Object Text.UTF8Encoding($false)))
            Move-Item -LiteralPath $temporary -Destination $locationsPath -Force
        }
    }
}

function Get-VaultEnvironmentRoot {
    <# Map a source component to its own private environment namespace. #>
    param([Parameter(Mandatory)][string]$ComponentRoot)
    $source = [IO.Path]::GetFullPath($ComponentRoot).TrimEnd('\', '/')
    if ([IO.Path]::GetFileName($source) -eq 'hltv-scraper-api') {
        $repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $source))
        return Join-Path $repo 'VAULT\CS2\SCRAPER\hltv-scraper-api'
    }
    return Join-Path (Split-Path -Parent $source) ('VAULT\' + [IO.Path]::GetFileName($source))
}
