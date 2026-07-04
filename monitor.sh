#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# monitor.sh — Dashboard-style monitoring for Evolution Engine
#
# Usage:
#   ./monitor.sh              # Single snapshot
#   ./monitor.sh --watch      # Refresh every 10 seconds
#   ./monitor.sh --json       # Raw JSON output (machine-readable)
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Color codes (matching status.sh / stop.sh) ──────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Constants ────────────────────────────────────────────────────────
SERVICE_NAME="evolution-engine"
HEALTH_FILE="$SCRIPT_DIR/health.json"
STATE_FILE="$SCRIPT_DIR/runner_state.json"
WATCH_MODE=false
JSON_MODE=false
REFRESH_INTERVAL=10

# ── Parse flags ──────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --watch|-w)  WATCH_MODE=true ;;
        --json|-j)   JSON_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [--watch] [--json]"
            echo ""
            echo "  --watch    Refresh dashboard every ${REFRESH_INTERVAL}s (Ctrl+C to stop)"
            echo "  --json     Output raw JSON (machine-readable)"
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR ]${NC} Unknown flag: $arg" >&2
            echo "  Run '$0 --help' for usage." >&2
            exit 1
            ;;
    esac
done

# ── Helper: parse a single key from a JSON file using Python ─────────
json_get() {
    local file="$1"
    local key="$2"
    local default="${3:-null}"
    if [ -f "$file" ]; then
        /usr/bin/python3 -c "
import json, sys
try:
    with open('$file') as f:
        d = json.load(f)
    v = d.get('$key', $default)
    print(v if v is not None else '')
except Exception:
    print('')
" 2>/dev/null || echo ""
    else
        echo ""
    fi
}

# ── Helper: format uptime seconds to human-readable ──────────────────
format_uptime() {
    local raw="$1"
    if [ -z "$raw" ] || [ "$raw" = "unknown" ] || [ "$raw" = "None" ]; then
        echo "N/A"
        return
    fi
    /usr/bin/python3 -c "
try:
    seconds = int(float('$raw'))
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    parts = []
    if days > 0: parts.append(f'{days}d')
    if hours > 0: parts.append(f'{hours}h')
    if minutes > 0: parts.append(f'{minutes}m')
    parts.append(f'{secs}s')
    print(' '.join(parts))
except Exception:
    print('N/A')
" 2>/dev/null || echo "N/A"
}

# ── Helper: build a progress bar ─────────────────────────────────────
progress_bar() {
    local current="$1"
    local target="$2"
    local width="${3:-30}"
    /usr/bin/python3 -c "
try:
    cur = float('$current')
    tgt = float('$target')
    if tgt <= 0:
        print('[' + '?' * $width + '] N/A (no target)')
    else:
        ratio = min(cur / tgt, 1.0)
        filled = int(ratio * $width)
        empty = $width - filled
        bar = '#' * filled + '-' * empty
        pct = ratio * 100
        print(f'[{bar}] {pct:.1f}%')
except Exception:
    print('[' + '?' * $width + '] invalid')
" 2>/dev/null || echo "[?] Cannot compute progress"
}

# ── Render the dashboard ─────────────────────────────────────────────
render_dashboard() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S %Z')

    # ── Header ───────────────────────────────────────────────────────
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Evolution Engine — Monitor Dashboard${NC}"
    echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
    echo ""

    # ── 1. Service Status ────────────────────────────────────────────
    echo -e "${BOLD}  Service Status${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────"

    local is_active="false"
    local pid=""
    local since=""

    if systemctl --user is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        is_active="true"
        pid=$(systemctl --user show "$SERVICE_NAME" --property=MainPID --value 2>/dev/null || echo "?")
        since=$(systemctl --user show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value 2>/dev/null || echo "?")
        since="${since:0:24}"
    fi

    if [ "$is_active" = "true" ]; then
        echo -e "  State:    ${GREEN}RUNNING${NC}"
        echo -e "  PID:      $pid"
        echo -e "  Since:    $since"
    else
        echo -e "  State:    ${RED}STOPPED${NC}"
    fi
    echo ""

    # ── 2. Fitness ───────────────────────────────────────────────────
    echo -e "${BOLD}  Fitness${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────"

    if [ -f "$HEALTH_FILE" ]; then
        local fitness target_fitness
        fitness=$(json_get "$HEALTH_FILE" "current_fitness" "0.0")
        target_fitness=$(json_get "$HEALTH_FILE" "target_fitness" "null")

        if [ -n "$fitness" ] && [ "$fitness" != "None" ]; then
            echo -e "  Current:  ${GREEN}${fitness}${NC}"
        else
            echo -e "  Current:  ${YELLOW}N/A${NC}"
        fi

        if [ -n "$target_fitness" ] && [ "$target_fitness" != "None" ] && [ "$target_fitness" != "" ]; then
            echo -e "  Target:   ${CYAN}${target_fitness}${NC}"
            echo ""
            echo -e "  Progress: $(progress_bar "$fitness" "$target_fitness" 30)"
        else
            echo -e "  Target:   ${YELLOW}not set${NC}"
        fi
    else
        echo -e "  ${YELLOW}health.json not found${NC}"
    fi
    echo ""

    # ── 3. Cycle Count ───────────────────────────────────────────────
    echo -e "${BOLD}  Cycle Count${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────"

    local cycles_completed current_cycle
    cycles_completed=""
    current_cycle=""

    if [ -f "$HEALTH_FILE" ]; then
        cycles_completed=$(json_get "$HEALTH_FILE" "cycles_completed" "0")
    fi
    if [ -f "$STATE_FILE" ]; then
        current_cycle=$(json_get "$STATE_FILE" "current_cycle" "0")
    fi

    if [ -n "$cycles_completed" ] && [ "$cycles_completed" != "" ]; then
        echo -e "  Completed: ${CYAN}${cycles_completed}${NC}"
    else
        echo -e "  Completed: ${YELLOW}N/A${NC}"
    fi

    if [ -n "$current_cycle" ] && [ "$current_cycle" != "" ]; then
        echo -e "  Current:   ${CYAN}${current_cycle}${NC}"
    else
        echo -e "  Current:   ${YELLOW}N/A${NC}"
    fi
    echo ""

    # ── 4. Level Progression ─────────────────────────────────────────
    echo -e "${BOLD}  Level Progression${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────"

    local current_level
    current_level=""
    if [ -f "$HEALTH_FILE" ]; then
        current_level=$(json_get "$HEALTH_FILE" "current_level" "0")
    fi

    if [ -n "$current_level" ] && [ "$current_level" != "" ]; then
        echo -e "  Level:     ${GREEN}${current_level}${NC}"
    else
        echo -e "  Level:     ${YELLOW}N/A${NC}"
    fi

    if [ -f "$STATE_FILE" ]; then
        /usr/bin/python3 -c "
import json
try:
    with open('$STATE_FILE') as f:
        state = json.load(f)
    level = state.get('best_level', state.get('current_level', 0))
    stagnation = state.get('stagnation_cycles', 0)
    self_corr = state.get('self_correction_attempts', 0)
    print(f'  Best:      {level}')
    if stagnation > 0:
        print(f'  Stagnation: \033[33m{stagnation} cycles\033[0m')
    if self_corr > 0:
        print(f'  Corrections: {self_corr}')
except Exception:
    pass
" 2>/dev/null || true
    else
        echo -e "  ${YELLOW}runner_state.json not found${NC}"
    fi
    echo ""

    # ── 5. Errors ────────────────────────────────────────────────────
    echo -e "${BOLD}  Errors${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────"

    local error_count
    error_count=""
    if [ -f "$HEALTH_FILE" ]; then
        error_count=$(json_get "$HEALTH_FILE" "errors" "0")
    fi

    if [ -n "$error_count" ] && [ "$error_count" != "" ]; then
        if [ "$error_count" = "0" ] || [ "$error_count" = "0.0" ]; then
            echo -e "  Count:    ${GREEN}0${NC}"
        else
            echo -e "  Count:    ${RED}${error_count}${NC}"
        fi
    else
        echo -e "  Count:    ${YELLOW}N/A${NC}"
    fi

    if [ -f "$STATE_FILE" ]; then
        /usr/bin/python3 -c "
import json
try:
    with open('$STATE_FILE') as f:
        state = json.load(f)
    total = state.get('total_errors', 0)
    consecutive = state.get('consecutive_errors', 0)
    types = state.get('error_types', {})
    print(f'  Total:     {total}')
    if consecutive > 0:
        print(f'  Consecutive: \033[31m{consecutive}\033[0m')
    if types:
        print('  Types:')
        for etype, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f'    {etype}: {count}')
except Exception:
    pass
" 2>/dev/null || true
    else
        echo -e "  ${YELLOW}runner_state.json not found${NC}"
    fi
    echo ""

    # ── 6. Timing ────────────────────────────────────────────────────
    echo -e "${BOLD}  Timing${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────"

    if [ -f "$HEALTH_FILE" ]; then
        local uptime_raw start_timestamp last_cycle
        uptime_raw=$(json_get "$HEALTH_FILE" "uptime_seconds" "null")
        last_cycle=$(json_get "$HEALTH_FILE" "last_cycle" "")

        if [ -n "$uptime_raw" ] && [ "$uptime_raw" != "None" ] && [ "$uptime_raw" != "" ]; then
            echo -e "  Uptime:   $(format_uptime "$uptime_raw")"
        else
            echo -e "  Uptime:   ${YELLOW}N/A${NC}"
        fi

        if [ -n "$last_cycle" ] && [ "$last_cycle" != "" ]; then
            echo -e "  Last cycle: $last_cycle"
        fi
    fi

    if [ -f "$STATE_FILE" ]; then
        start_timestamp=$(json_get "$STATE_FILE" "start_timestamp" "")
        if [ -n "$start_timestamp" ] && [ "$start_timestamp" != "" ]; then
            echo -e "  Started:  $start_timestamp"
        fi
    fi
    echo ""

    # ── 7. Recent Log Entries ────────────────────────────────────────
    echo -e "${BOLD}  Recent Log Entries (last 10)${NC}"
    echo -e "  ─────────────────────────────────────────────────────────────"

    if systemctl --user list-unit-files "$SERVICE_NAME.service" &>/dev/null 2>&1; then
        local logs
        logs=$(journalctl --user -u "$SERVICE_NAME" -n 10 --no-pager 2>/dev/null || true)
        if [ -n "$logs" ]; then
            echo "$logs" | tail -10 | while IFS= read -r line; do
                echo -e "  ${CYAN}${line}${NC}"
            done
        else
            echo -e "  ${YELLOW}No journal entries available${NC}"
        fi
    elif [ -f "$SCRIPT_DIR/evolution.log" ]; then
        tail -10 "$SCRIPT_DIR/evolution.log" | while IFS= read -r line; do
            echo -e "  ${CYAN}${line}${NC}"
        done
    else
        echo -e "  ${YELLOW}No log source available${NC}"
    fi
    echo ""

    # ── Footer ───────────────────────────────────────────────────────
    echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${CYAN}Last updated: ${timestamp}${NC}"
    if [ "$WATCH_MODE" = true ]; then
        echo -e "  ${YELLOW}Refreshing every ${REFRESH_INTERVAL}s — Ctrl+C to stop${NC}"
    fi
    echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# ── JSON mode: aggregate everything into one JSON object ─────────────
render_json() {
    /usr/bin/python3 -c "
import json, os, subprocess, datetime

result = {}

# Service status
try:
    active = subprocess.run(
        ['systemctl', '--user', 'is-active', '--quiet', '$SERVICE_NAME'],
        capture_output=True
    )
    result['service_running'] = (active.returncode == 0)
    if result['service_running']:
        pid_out = subprocess.run(
            ['systemctl', '--user', 'show', '$SERVICE_NAME',
             '--property=MainPID', '--value'],
            capture_output=True, text=True
        )
        result['pid'] = pid_out.stdout.strip()
        since_out = subprocess.run(
            ['systemctl', '--user', 'show', '$SERVICE_NAME',
             '--property=ActiveEnterTimestamp', '--value'],
            capture_output=True, text=True
        )
        result['since'] = since_out.stdout.strip()[:24]
except Exception:
    result['service_running'] = False

# health.json
result['health'] = None
if os.path.isfile('$HEALTH_FILE'):
    try:
        with open('$HEALTH_FILE') as f:
            result['health'] = json.load(f)
    except Exception:
        pass

# runner_state.json
result['runner_state'] = None
if os.path.isfile('$STATE_FILE'):
    try:
        with open('$STATE_FILE') as f:
            result['runner_state'] = json.load(f)
    except Exception:
        pass

# Recent logs
try:
    logs_out = subprocess.run(
        ['journalctl', '--user', '-u', '$SERVICE_NAME', '-n', '10', '--no-pager'],
        capture_output=True, text=True
    )
    result['recent_logs'] = logs_out.stdout.strip().splitlines()
except Exception:
    result['recent_logs'] = []

result['timestamp'] = datetime.datetime.now().isoformat()
print(json.dumps(result, indent=2))
" 2>/dev/null || echo '{"error": "Failed to gather status data"}'
}

# ── Main ─────────────────────────────────────────────────────────────
if [ "$JSON_MODE" = true ]; then
    if [ "$WATCH_MODE" = true ]; then
        while true; do
            clear
            render_json
            sleep "$REFRESH_INTERVAL"
        done
    else
        render_json
    fi
elif [ "$WATCH_MODE" = true ]; then
    while true; do
        clear
        render_dashboard
        sleep "$REFRESH_INTERVAL"
    done
else
    render_dashboard
fi
