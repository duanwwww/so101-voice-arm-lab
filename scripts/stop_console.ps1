[CmdletBinding(SupportsShouldProcess)]
param()

$ErrorActionPreference = 'Stop'
$record = Join-Path $PSScriptRoot '../outputs/service.json'
if (-not (Test-Path -LiteralPath $record)) {
    Write-Output 'No background console registered. Use Ctrl+C for a foreground server.'
    return
}
$info = Get-Content -Raw -LiteralPath $record | ConvertFrom-Json
$server = Get-Process -Id ([int]$info.pid) -ErrorAction SilentlyContinue
if (-not $server) { Write-Output 'The registered server already stopped.'; return }
$startTicks = $server.StartTime.ToUniversalTime().Ticks.ToString()
$listener = Get-NetTCPConnection -LocalPort ([int]$info.port) -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalAddress -eq '127.0.0.1' -and $_.OwningProcess -eq $server.Id }
if ($startTicks -ne [string]$info.startUtcTicks -or -not $listener) {
    throw 'PID identity or listener differs from the saved service; no process was stopped.'
}
if ($PSCmdlet.ShouldProcess("SO-101 local console PID $($server.Id) port $($info.port)", 'Stop')) {
    Stop-Process -Id $server.Id
}
