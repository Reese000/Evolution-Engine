# Evolution Engine agent loop - emits wake sentinel for Cursor monitored shell.
# Never exits except manual kill. Heartbeat + pid for certification.
# Usage: powershell -NoProfile -File scripts/agent_loop_wake.ps1 -IntervalMinutes 30
param(
    [int] $IntervalMinutes = 30
)
$ErrorActionPreference = "Continue"
$seconds = $IntervalMinutes * 60
$purpose = "evolution"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$agentsDir = Join-Path $repoRoot ".agents"
$tickStatePath = Join-Path $agentsDir "tick_state.json"
$pidFile = Join-Path $agentsDir "loop_wake.pid"
$heartbeatFile = Join-Path $agentsDir "loop_wake.heartbeat"
$logFile = Join-Path $agentsDir "loop_wake.log"

function Write-LoopLog {
    param([string] $Message)
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $line = "[$ts] $Message"
    Add-Content -LiteralPath $logFile -Value $line -Encoding utf8
}

function Write-Heartbeat {
    param([string] $Status)
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $json = @{
        at = $ts
        status = $Status
        pid = $PID
        interval_minutes = $IntervalMinutes
    } | ConvertTo-Json -Compress
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($heartbeatFile, $json, $utf8NoBom)
}

function Test-UserLoopPaused {
    if (-not (Test-Path -LiteralPath $tickStatePath)) {
        return $false
    }
    try {
        $state = Get-Content -LiteralPath $tickStatePath -Raw | ConvertFrom-Json
        # loop_paused is user-only; flatline uses evolution_paused — never stop wake ticks
        return ($state.loop_paused -eq $true)
    } catch {
        return $false
    }
}

New-Item -ItemType Directory -Force -Path $agentsDir | Out-Null
Set-Content -LiteralPath $pidFile -Value $PID -Encoding ascii
Write-LoopLog "Wake loop started pid=$PID interval=${IntervalMinutes}m"
Write-Heartbeat "started"

# Immediate tick on start (do not wait 30m first)
if (-not (Test-UserLoopPaused)) {
    $payload = '{"prompt":"Evolution Engine tick: read .agents/LOOP_PROMPT.md and .agents/COMPLETION_CRITERIA.md. Execute ONE tick. Never pause loop unless user asks. Update tick_state.json."}'
    Write-Output "AGENT_LOOP_TICK_$purpose $payload"
    Write-Heartbeat "tick_emitted"
    $env:PYTHONPATH = $repoRoot
    python "$scriptDir\verify_loop_health.py" --repair 2>&1 | Out-Null
}

while ($true) {
    try {
        Start-Sleep -Seconds $seconds
        Write-Heartbeat "sleep_ok"
        if (Test-UserLoopPaused) {
            Write-LoopLog "user loop_paused=true - skipping tick emission"
            continue
        }
        $payload = '{"prompt":"Evolution Engine tick: read .agents/LOOP_PROMPT.md and .agents/COMPLETION_CRITERIA.md. Execute ONE tick. Never pause loop unless user asks. Update tick_state.json."}'
        Write-Output "AGENT_LOOP_TICK_$purpose $payload"
        Write-Heartbeat "tick_emitted"
        $env:PYTHONPATH = $repoRoot
        python "$scriptDir\verify_loop_health.py" --repair 2>&1 | Out-Null
    } catch {
        Write-LoopLog "ERROR in wake loop (continuing): $_"
        Write-Heartbeat "error_continuing"
        Start-Sleep -Seconds 60
    }
}
