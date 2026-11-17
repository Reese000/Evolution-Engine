# Restart evolution daemon to load latest code.
# Hidden window style kills the process on Windows; use Minimized instead.
# 2026-09-09 12:35 EDT (dispatcher TICK-491 fix): added healthy-daemon guard + restart
# throttle so an unconditional kill-restart cannot thrash fresh batches inside a
# flatline window (observed defect: 10 daemon births in ~70 min, batches killed at
# 0-201 bytes). Pass -Force to bypass both guards (manual deploy of new code).
# NOTE (12:42 EDT): throttle uses lexicographic ISO-8601-UTC compare; Windows PS 5.1
# [datetime] cast chokes on 'Z' suffixes, so no datetime parsing here.
param(
    [switch] $Force = $false,
    [switch] $DryRun = $false
)
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$agentsDir = Join-Path $repoRoot ".agents"
$pidFile = Join-Path $agentsDir "evolution_daemon.pid"
$daemonScript = Join-Path $repoRoot "scripts\continuous_evolution_daemon.py"
$lastRestartFile = Join-Path $agentsDir "last_daemon_restart.json"
$throttleMinutes = 8   # do not auto-restart more often than this; ignores stale-daemon repairs
$graceSeconds = 20     # a daemon younger than this has JUST been started - do not kill it

# Resolve a stable python: prefer the project's miniconda interpreter, fall back to PATH python.
# (Bare 'python' under a caller with a hermes-venv PATH launched the daemon+batch with the WRONG
# interpreter on 2026-09-09 16:40:19Z -> batch exit_code=1. This forces the right one.)
$envPython = $null
foreach ($candidate in @(
    "C:\Users\reese\miniconda3\python.exe",
    (Join-Path $env:USERPROFILE "miniconda3\python.exe")
)) {
    if ($null -ne $candidate -and (Test-Path -LiteralPath $candidate)) {
        $envPython = $candidate
        break
    }
}
if ($null -eq $envPython) {
    $whichPy = (Get-Command python -ErrorAction SilentlyContinue)
    if ($null -ne $whichPy) { $envPython = $whichPy.Source }
}
if ($null -eq $envPython) {
    Write-Host "ERROR: no python interpreter found"
    exit 1
}

function Get-NowUtcIso {
    return (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
}

function Test-PidAlive([int] $ProcessId) {
    return $null -ne (Get-Process -Id $ProcessId -ErrorAction SilentlyContinue)
}

# --- Guard 1: throttle (unless -Force). ---
# ISO-8601 UTC strings of equal width sort lexicographically == chronologically.
$throttled = $false
if (-not $Force -and (Test-Path -LiteralPath $lastRestartFile)) {
    try {
        $rec = Get-Content -LiteralPath $lastRestartFile -Raw | ConvertFrom-Json
        if ($rec.at) {
            $cutoff = (Get-Date).ToUniversalTime().AddMinutes(-$throttleMinutes).ToString("yyyy-MM-ddTHH:mm:ssZ")
            # record.at > cutoff  => restart happened within the throttle window
            $throttled = ($rec.at -gt $cutoff)
        }
    } catch {
        # corrupt/missing throttle record: proceed but record fresh one below
        $throttled = $false
    }
}
if ($throttled) {
    Write-Host "THROTTLED: daemon restarted $($rec.at) (within last $throttleMinutes min); use -Force to deploy now. Skipping."
    if ($DryRun) { exit 0 }
    exit 0
}

# --- Guard 2: do not kill a daemon that is alive and very young (mid-startup). ---
if (Test-Path -LiteralPath $pidFile) {
    $pidVal = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($pidVal -and (Test-PidAlive ([int]$pidVal))) {
        $aliveSec = 999999
        try {
            $started = (Get-Process -Id ([int]$pidVal) -ErrorAction SilentlyContinue).StartTime
            $aliveSec = [int] ((Get-Date) - $started).TotalSeconds
        } catch {
            $aliveSec = 999999
        }
        if (-not $Force -and $aliveSec -lt $graceSeconds) {
            Write-Host "SKIPPED: daemon pid=$pidVal alive only ${aliveSec}s (grace $graceSeconds) - not killed. Use -Force to force-restart."
            if ($DryRun) { exit 0 }
            exit 0
        }
    }
}

if ($DryRun) {
    Write-Host "DRYRUN: no guard blocked; would stop daemon + orphaned batches and start a fresh daemon."
    if (Test-Path -LiteralPath $pidFile) {
        Write-Host ("  - would kill pid: " + (Get-Content -LiteralPath $pidFile -Raw).Trim())
    }
    exit 0
}

if (Test-Path -LiteralPath $pidFile) {
    $pidVal = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($pidVal -and (Test-PidAlive ([int]$pidVal))) {
        Stop-Process -Id ([int]$pidVal) -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped daemon pid=$pidVal"
        Start-Sleep -Seconds 3
    }
}

# Kill orphaned batch subprocesses (daemon restart does not stop run_multi_cycle.py children).
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*run_multi_cycle.py*' } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped orphaned batch pid=$($_.ProcessId)"
    }
Start-Sleep -Seconds 2

$env:PYTHONPATH = $repoRoot
$env:EE_EPI_MOMENTUM_WINDOW = "5000"
$env:EE_EPI_MOMENTUM_TARGET = "0.5"
$env:EE_TERMINAL_VERIFY_PROMOTE = "1"
$env:EE_BATCH_LOG_KEEP = "30"
$env:EE_ARCHIVE_SIDECAR_KEEP = "10"

python (Join-Path $repoRoot "scripts\prune_agent_artifacts.py") --keep 30 --archive-keep 10
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARN: batch log prune failed (exit $LASTEXITCODE)"
}

$proc = Start-Process -FilePath $envPython `
    -ArgumentList "`"$daemonScript`"" `
    -WorkingDirectory $repoRoot `
    -WindowStyle Minimized `
    -PassThru

Start-Sleep -Seconds 3
if ($proc -and (Test-PidAlive $proc.Id)) {
    Set-Content -LiteralPath $pidFile -Value $proc.Id -NoNewline
    $rec = @{ at = (Get-NowUtcIso); pid = $proc.Id } | ConvertTo-Json -Compress
    Set-Content -LiteralPath $lastRestartFile -Value $rec -NoNewline
    Write-Host "Started continuous_evolution_daemon.py pid=$($proc.Id)"
} else {
    Write-Host "ERROR: daemon failed to start"
    exit 1
}