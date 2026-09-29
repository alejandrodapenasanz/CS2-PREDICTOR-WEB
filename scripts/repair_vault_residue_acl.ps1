<#
Repairs ACLs only for the explicitly reviewed legacy residues. Never deletes or
moves data. Preview is the default; -Apply requires an elevated PowerShell and
the SID of the operator who will run the ordinary migration afterwards.
Example (administrator): .\scripts\repair_vault_residue_acl.ps1 -Apply -OperatorSid S-1-5-21-...
All component Python processes must be stopped. Reparse points are rejected.
#>
[CmdletBinding()]
param([switch]$Apply, [string]$OperatorSid)
$ErrorActionPreference = 'Stop'
$Repository = [IO.Path]::GetFullPath((Split-Path $PSScriptRoot -Parent))
$Rejected = 'VAULT/CS2/MODEL/artifacts/registry/20260824_125852Z'
$Allowed = @(
    '.pytest_cache',
    'TELEGRAM/.pytest-tmp',
    'TELEGRAM/.pytest-vault-check',
    'TELEGRAM/.pytest-vault-final',
    'TELEGRAM/.pytest-vault-finish',
    'CS2/SCRAPER/hltv-scraper-api/.pytest-vault-final',
    'CS2/SCRAPER/hltv-scraper-api/.pytest_cache',
    'TENNIS/data/processed/features/manifest.json',
    'TENNIS/data/processed/features/ranking_conflicts.csv',
    'TENNIS/data/processed/features/training_F.parquet',
    'TENNIS/data/processed/features/training_M.parquet',
    'CS2/TESTS/.cache/pytest',
    'CS2/SCRAPER/hltv-scraper-api/.cache/pytest-tmp'
)

function Assert-ConfinedRegularPath {
    <# Reject a broad target, traversal or any reparse-point ancestor. #>
    param([string]$Path)
    $Full = [IO.Path]::GetFullPath($Path)
    if (-not $Full.StartsWith($Repository + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside repository: $Full"
    }
    $Cursor = $Full
    while ($Cursor -and $Cursor -ne $Repository) {
        $Item = Get-Item -LiteralPath $Cursor -Force
        if ($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Reparse point refused: $Cursor"
        }
        $Cursor = Split-Path $Cursor -Parent
    }
    return $Full
}

$Journal = Get-Content -LiteralPath (Join-Path $Repository 'VAULT/migration.json') -Raw | ConvertFrom-Json
$Targets = @($Journal.pending | ForEach-Object {
    if ($Allowed -cnotcontains $_.source) { throw "Unreviewed residue: $($_.source)" }
    $_.source
}) + @($Rejected)
$Registry = Join-Path $Repository 'VAULT/CS2/MODEL/artifacts/registry'
$Latest = Get-Content -LiteralPath (Join-Path $Registry 'latest.json') -Raw | ConvertFrom-Json
$LastGood = Get-Content -LiteralPath (Join-Path $Registry 'last_good.json') -Raw | ConvertFrom-Json
if (@($Latest.latest, $LastGood.last_good, $LastGood.version) -contains '20260824_125852Z') {
    throw 'The reviewed rejected version is now protected. No ACLs changed.'
}
$Resolved = @($Targets | ForEach-Object {
    $Candidate = Join-Path $Repository $_
    if (Test-Path -LiteralPath $Candidate) { Assert-ConfinedRegularPath $Candidate }
})
$Resolved | ForEach-Object { Write-Host "[ACL preview] $_" }
if (-not $Apply) { exit 0 }
if ($OperatorSid -notmatch '^S-1-5-21-\d+-\d+-\d+-\d+$') {
    throw 'Supply the operator SID, not a broad group or an arbitrary account.'
}
$Identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$Principal = [Security.Principal.WindowsPrincipal]::new($Identity)
if (-not $Principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Administrator rights are required for these inaccessible residues.'
}
$Processes = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object {
    $_.ExecutablePath -and $_.ExecutablePath.StartsWith($Repository + '\', [StringComparison]::OrdinalIgnoreCase)
})
if ($Processes.Count) { throw 'Stop component Python processes before repairing residues.' }
Start-Transcript -LiteralPath (Join-Path $Repository 'VAULT/verification/residue_acl_console.log') -Force | Out-Null
$Report = [Collections.Generic.List[object]]::new()
$Queue = [Collections.Generic.Queue[string]]::new()
foreach ($Target in $Resolved) { $Queue.Enqueue($Target) }
while ($Queue.Count) {
    $Path = Assert-ConfinedRegularPath ($Queue.Dequeue())
    $OldSddl = $null
    try { $OldSddl = (Get-Acl -LiteralPath $Path).Sddl } catch { }
    & takeown.exe /F $Path /A | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Ownership repair failed: $Path" }
    # No /T: inspect and validate each child before granting access to it.
    & icacls.exe $Path /grant:r ('*{0}:(F)' -f $OperatorSid) '*S-1-5-32-544:(F)' /L | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "ACL grant failed: $Path" }
    $Report.Add([pscustomobject]@{path=$Path; previous_sddl=$OldSddl; repaired_at_utc=[DateTime]::UtcNow.ToString('o')})
    $Report | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Repository 'VAULT/verification/residue_acl_repair.json') -Encoding UTF8
    $Item = Get-Item -LiteralPath $Path -Force
    if ($Item.PSIsContainer) {
        foreach ($Child in Get-ChildItem -LiteralPath $Path -Force) { $Queue.Enqueue($Child.FullName) }
    }
}
Write-Host '[ACL] Repair complete. No files moved or deleted; run the hash-verified migration as the ordinary operator.'
Stop-Transcript | Out-Null
