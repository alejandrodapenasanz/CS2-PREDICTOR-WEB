<# Native Python output must reach the host so Start-Transcript records it. #>

function Enter-TennisRun {
    <# Serialize launchers for this exact state directory; Windows releases a crashed owner's lease. #>
    param([Parameter(Mandatory)][string]$StateRoot)
    $hasher = [Security.Cryptography.SHA256]::Create()
    try {
        $canonical = [IO.Path]::GetFullPath($StateRoot).TrimEnd('\').ToUpperInvariant()
        $digest = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($canonical))).Replace('-', '')
    } finally { $hasher.Dispose() }
    $mutex = [Threading.Mutex]::new($false, ('Local\TennisPipeline_' + $digest))
    try {
        while ($true) {
            try {
                if ($mutex.WaitOne(1000)) { break }
            } catch [Threading.AbandonedMutexException] {
                Write-Warning '[TENNIS] El launcher anterior se interrumpio; bloqueo recuperado, se verificaran los artefactos.'
                break  # WaitOne has granted ownership even when it reports abandonment.
            }
            Write-Host '[TENNIS] Otro arranque de tenis sigue activo; esperando sin duplicar adquisicion ni entrenamiento.'
            try {
                if ($mutex.WaitOne(29000)) { break }
            } catch [Threading.AbandonedMutexException] {
                Write-Warning '[TENNIS] Bloqueo abandonado recuperado; se verificaran los artefactos.'
                break
            }
        }
        return $mutex
    } catch {
        $mutex.Dispose()
        throw
    }
}

function Invoke-TennisPython {
    <# Stream stdout/stderr into the transcript and return the real exit code. #>
    param(
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string[]]$Arguments
    )
    $previousPreference = $ErrorActionPreference
    $previousEncoding = [Console]::OutputEncoding
    $previousPythonEncoding = $env:PYTHONIOENCODING
    try {
        # PowerShell 5 treats native stderr as ErrorRecord. A warning written to
        # stderr is not a failed process; only the native exit code decides.
        $ErrorActionPreference = 'Continue'
        [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
        $env:PYTHONIOENCODING = 'utf-8'
        & $Python -u @Arguments 2>&1 | ForEach-Object { Write-Host ([string]$_) }
        $nativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
        [Console]::OutputEncoding = $previousEncoding
        $env:PYTHONIOENCODING = $previousPythonEncoding
    }
    if ($null -eq $nativeExitCode) { throw 'Python did not return an exit code.' }
    return [int]$nativeExitCode
}
