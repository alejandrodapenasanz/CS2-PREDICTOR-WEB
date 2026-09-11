<#
Que hace:
    Instala, consulta o elimina una tarea de Windows que ejecuta diariamente
    los actualizadores Tennis Abstract y TennisRatio. Conserva su nombre legacy.
    La accion usa rutas absolutas y
    llama a run_tennis.ps1 -UpdateOnly, sin duplicar la logica de adquisicion.

Que recibe:
    -Mode Install, Status o Remove. El modo por defecto es Status.
    -DailyTime HH:mm en hora local. El valor por defecto es 06:00.
    -TaskName permite cambiar el nombre estable de la tarea.

Como se ejecuta:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        .\TENNIS\scripts\manage_tennisratio_daily_task.ps1 -Mode Install
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        .\TENNIS\scripts\manage_tennisratio_daily_task.ps1 -Mode Status
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        .\TENNIS\scripts\manage_tennisratio_daily_task.ps1 -Mode Remove

El instalador no solicita ni guarda credenciales. Usa el token interactivo del
usuario actual, StartWhenAvailable y una unica instancia. Por ello la tarea
puede recuperar una ejecucion perdida cuando ese usuario vuelva a iniciar
sesion, pero no puede ejecutarse mientras no haya una sesion de ese usuario.
Un fallo se reintenta tres veces, cada 30 minutos; el actualizador diario es
idempotente y una publicacion correcta no se repite.
El limite de 12 horas permite completar de forma segura el inventario inicial
de perfiles; las ejecuciones incrementales ordinarias terminan mucho antes.
#>

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateSet('Install', 'Status', 'Remove')]
    [string]$Mode = 'Status',

    [ValidatePattern('^(?:[01]\d|2[0-3]):[0-5]\d$')]
    [string]$DailyTime = '06:00',

    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9 ._-]{0,127}$')]
    [string]$TaskName = 'CS2-Predictor-TennisRatio-Daily'
)

$ErrorActionPreference = 'Stop'

function Assert-WindowsScheduledTasksAvailable {
    <# Comprueba Windows y los cmdlets requeridos sin cambiar estado externo. #>

    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw 'La instalacion de la tarea diaria solo esta disponible en Windows.'
    }
    foreach ($CommandName in @(
        'Get-ScheduledTask',
        'Get-ScheduledTaskInfo',
        'New-ScheduledTask',
        'New-ScheduledTaskAction',
        'New-ScheduledTaskPrincipal',
        'New-ScheduledTaskSettingsSet',
        'New-ScheduledTaskTrigger',
        'Register-ScheduledTask',
        'Unregister-ScheduledTask'
    )) {
        if (-not (Get-Command -Name $CommandName -ErrorAction SilentlyContinue)) {
            throw "Falta el cmdlet requerido de Task Scheduler: $CommandName"
        }
    }
}

function Get-TennisRatioTaskConfiguration {
    <# Construye de forma pura las rutas y argumentos canónicos de la tarea. #>
    param(
        [Parameter(Mandatory = $true)][string]$ConfiguredTaskName,
        [Parameter(Mandatory = $true)][string]$ConfiguredDailyTime
    )

    $TennisRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    $LauncherPath = [IO.Path]::GetFullPath((Join-Path $TennisRoot 'run_tennis.ps1'))
    if (-not (Test-Path -LiteralPath $LauncherPath -PathType Leaf)) {
        throw "No existe el launcher TENNIS requerido: $LauncherPath"
    }
    $WindowsRoot = [Environment]::GetEnvironmentVariable('SystemRoot')
    if ([string]::IsNullOrWhiteSpace($WindowsRoot)) {
        throw 'No se pudo resolver SystemRoot para localizar powershell.exe.'
    }
    $PowerShellExecutable = [IO.Path]::GetFullPath(
        (Join-Path $WindowsRoot 'System32\WindowsPowerShell\v1.0\powershell.exe')
    )
    if (-not (Test-Path -LiteralPath $PowerShellExecutable -PathType Leaf)) {
        throw "No existe Windows PowerShell: $PowerShellExecutable"
    }
    $TimeOfDay = [TimeSpan]::ParseExact(
        $ConfiguredDailyTime,
        'hh\:mm',
        [Globalization.CultureInfo]::InvariantCulture
    )
    $TriggerAt = [DateTime]::Today.Add($TimeOfDay)
    $ActionArguments = (
        '-NoProfile -NonInteractive -ExecutionPolicy Bypass ' +
        '-File "{0}" -UpdateOnly' -f $LauncherPath
    )
    return [PSCustomObject]@{
        TaskName = $ConfiguredTaskName
        DailyTime = $ConfiguredDailyTime
        TriggerAt = $TriggerAt
        Execute = $PowerShellExecutable
        Arguments = $ActionArguments
        WorkingDirectory = $TennisRoot
        LauncherPath = $LauncherPath
    }
}

function Get-TennisRatioTaskStatus {
    <# Devuelve estado legible sin registrar, ejecutar ni modificar la tarea. #>
    param([Parameter(Mandatory = $true)][string]$ConfiguredTaskName)

    $Task = Get-ScheduledTask -TaskName $ConfiguredTaskName -ErrorAction SilentlyContinue
    if ($null -eq $Task) {
        return [PSCustomObject]@{
            TaskName = $ConfiguredTaskName
            Installed = $false
            State = 'NotInstalled'
            LastRunTime = $null
            NextRunTime = $null
            LastTaskResult = $null
        }
    }
    $Info = Get-ScheduledTaskInfo -TaskName $ConfiguredTaskName
    return [PSCustomObject]@{
        TaskName = $ConfiguredTaskName
        Installed = $true
        State = [string]$Task.State
        LastRunTime = $Info.LastRunTime
        NextRunTime = $Info.NextRunTime
        LastTaskResult = $Info.LastTaskResult
    }
}

function Install-TennisRatioDailyTask {
    <# Registra o reemplaza una tarea diaria única con la configuración dada. #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param([Parameter(Mandatory = $true)][PSCustomObject]$Configuration)

    $Action = New-ScheduledTaskAction `
        -Execute $Configuration.Execute `
        -Argument $Configuration.Arguments `
        -WorkingDirectory $Configuration.WorkingDirectory
    $Trigger = New-ScheduledTaskTrigger -Daily -At $Configuration.TriggerAt
    $Settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -RunOnlyIfNetworkAvailable `
        -MultipleInstances IgnoreNew `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 30) `
        -ExecutionTimeLimit (New-TimeSpan -Hours 12)
    $CurrentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $Principal = New-ScheduledTaskPrincipal `
        -UserId $CurrentUser `
        -LogonType Interactive `
        -RunLevel Limited
    $Definition = New-ScheduledTask `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Principal $Principal `
        -Description 'Actualizacion diaria de Tennis Abstract y TennisRatio para CS2-Predictor.'
    if ($PSCmdlet.ShouldProcess($Configuration.TaskName, 'Registrar o reemplazar tarea diaria')) {
        Register-ScheduledTask `
            -TaskName $Configuration.TaskName `
            -InputObject $Definition `
            -Force | Out-Null
    }
}

function Remove-TennisRatioDailyTask {
    <# Elimina idempotentemente la tarea indicada, sin tocar datos descargados. #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param([Parameter(Mandatory = $true)][string]$ConfiguredTaskName)

    $Task = Get-ScheduledTask -TaskName $ConfiguredTaskName -ErrorAction SilentlyContinue
    if ($null -eq $Task) {
        Write-Host "[TENNIS] La tarea $ConfiguredTaskName no estaba instalada."
        return
    }
    if ($PSCmdlet.ShouldProcess($ConfiguredTaskName, 'Eliminar tarea diaria')) {
        Unregister-ScheduledTask -TaskName $ConfiguredTaskName -Confirm:$false
    }
}

function Invoke-TennisRatioTaskManager {
    <# Despacha Install, Status o Remove y conserva errores de Task Scheduler. #>
    [CmdletBinding(SupportsShouldProcess = $true)]
    param(
        [Parameter(Mandatory = $true)][string]$RequestedMode,
        [Parameter(Mandatory = $true)][string]$ConfiguredTaskName,
        [Parameter(Mandatory = $true)][string]$ConfiguredDailyTime
    )

    Assert-WindowsScheduledTasksAvailable
    switch ($RequestedMode) {
        'Install' {
            $Configuration = Get-TennisRatioTaskConfiguration `
                -ConfiguredTaskName $ConfiguredTaskName `
                -ConfiguredDailyTime $ConfiguredDailyTime
            Install-TennisRatioDailyTask `
                -Configuration $Configuration `
                -WhatIf:$WhatIfPreference
            if (-not $WhatIfPreference) {
                Get-TennisRatioTaskStatus -ConfiguredTaskName $ConfiguredTaskName
            }
        }
        'Status' {
            Get-TennisRatioTaskStatus -ConfiguredTaskName $ConfiguredTaskName
        }
        'Remove' {
            Remove-TennisRatioDailyTask `
                -ConfiguredTaskName $ConfiguredTaskName `
                -WhatIf:$WhatIfPreference
        }
        default {
            throw "Modo no soportado: $RequestedMode"
        }
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    Invoke-TennisRatioTaskManager `
        -RequestedMode $Mode `
        -ConfiguredTaskName $TaskName `
        -ConfiguredDailyTime $DailyTime `
        -WhatIf:$WhatIfPreference
}
