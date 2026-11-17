# Start wake loop + continuous evolution daemon (never kill unless user stops).
param(
    [int] $IntervalMinutes = 30
)
$ErrorActionPreference = "Continue"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$agentsDir = Join-Path $repoRoot ".agents"
New-Item -ItemType Directory -Force -Path $agentsDir | Out-Null

function Test-PidAlive {
    param([string] $PidPath)
    if (-not (Test-Path -LiteralPath $PidPath)) { return $false }
    $pidVal = Get-Content -LiteralPath $PidPath -Raw
    if (-not $pidVal) { return $false }
    $proc = Get-Process -Id ([int]$pidVal) -ErrorAction SilentlyContinue
    return ($null -ne $proc)
}

$wakePid = Join-Path $agentsDir "loop_wake.pid"
$daemonPid = Join-Path $agentsDir "evolution_daemon.pid"
$trajectoryPid = Join-Path $agentsDir "progress_trajectory_watcher.pid"

$env:PYTHONPATH = $repoRoot
if (-not (Test-PidAlive $wakePid)) {
    Write-Host "Starting agent wake loop..."
    Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$scriptDir\agent_loop_wake.ps1`" -IntervalMinutes $IntervalMinutes" `
        -WorkingDirectory $repoRoot `
        -WindowStyle Minimized `
        -PassThru | Out-Null
} else {
    Write-Host "Wake loop already running (pid file exists)."
}

if (-not (Test-PidAlive $daemonPid)) {
    Write-Host "Starting continuous evolution daemon..."
    Start-Process -FilePath "python" `
        -ArgumentList "`"$scriptDir\continuous_evolution_daemon.py`"" `
        -WorkingDirectory $repoRoot `
        -WindowStyle Minimized `
        -PassThru | Out-Null
} else {
    Write-Host "Evolution daemon already running (pid file exists)."
}

if (-not (Test-PidAlive $trajectoryPid)) {
    Write-Host "Starting trajectory watcher (live smooth charts)..."
    Start-Process -FilePath "python" `
        -ArgumentList "`"$scriptDir\progress_trajectory.py`" --watch" `
        -WorkingDirectory $repoRoot `
        -WindowStyle Minimized `
        -PassThru | Out-Null
} else {
    Write-Host "Trajectory watcher already running (pid file exists)."
}

Write-Host "Persistent loop start requested. Re-arm Cursor notify_on_output on AGENT_LOOP_TICK_evolution."
python "$scriptDir\verify_loop_health.py" --repair
