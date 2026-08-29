#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# status.sh — Health check and status display for Evolution Engine
#
# Usage:
#   ./status.sh              # Show health status
#   ./status.sh --json       # Output raw JSON
#   ./status.sh --logs       # Show log stats
#   ./status.sh --alerts     # Show recent alerts
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

JSON_MODE=false
LOGS_MODE=false
ALERTS_MODE=false

for arg in "$@"; do
    case "$arg" in
        --json)    JSON_MODE=true ;;
        --logs)    LOGS_MODE=true ;;
        --alerts)  ALERTS_MODE=true ;;
        --help|-h)
            echo "Usage: $0 [--json] [--logs] [--alerts]"
            exit 0
            ;;
    esac
done

# ── Service check (systemd + fallback) ────────────────────────────────
SERVICE_NAME="evolution-engine"
SERVICE_ACTIVE=false
SERVICE_PID=""
RUNTIME_SINCE=""

if systemctl --user is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    SERVICE_ACTIVE=true
    SERVICE_PID=$(systemctl --user show "$SERVICE_NAME" --property=MainPID --value 2>/dev/null || echo "")
    RUNTIME_SINCE=$(systemctl --user show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value 2>/dev/null || echo "")
fi

# Fallback: check for running process if systemd service is not active
if [ "$SERVICE_ACTIVE" = false ]; then
    PID_FILE=".runner.pid"
    PROCESS_RUNNING=false
    RUNNER_PID=""

    if [ -f "$PID_FILE" ]; then
        RUNNER_PID=$(cat "$PID_FILE")
        if kill -0 "$RUNNER_PID" 2>/dev/null; then
            PROCESS_RUNNING=true
        fi
    fi

    if [ "$PROCESS_RUNNING" = false ]; then
        RUNNER_PID=$(pgrep -f "autonomous_runner.py" 2>/dev/null | head -1 || true)
        if [ -n "$RUNNER_PID" ]; then
            PROCESS_RUNNING=true
        fi
    fi
else
    PROCESS_RUNNING=true
    RUNNER_PID="$SERVICE_PID"
fi

# ── Health JSON ─────────────────────────────────────────────────────
HEALTH_FILE="health.json"
HEALTH_DATA=""
if [ -f "$HEALTH_FILE" ]; then
    HEALTH_DATA=$(cat "$HEALTH_FILE")
fi

# ── Log stats ───────────────────────────────────────────────────────
if [ "$LOGS_MODE" = true ]; then
    echo -e "${BOLD}═══ Log Statistics ═══${NC}"
    if [ -d logs ]; then
        LOG_COUNT=$(find logs -name "*.log" -o -name "*.gz" 2>/dev/null | wc -l)
        LOG_SIZE=$(du -sh logs 2>/dev/null | cut -f1)
        echo "  Directory: logs/"
        echo "  Files:     $LOG_COUNT"
        echo "  Size:      $LOG_SIZE"
        echo ""
        echo "  Component files:"
        for f in logs/*.log; do
            if [ -f "$f" ]; then
                SIZE=$(du -h "$f" | cut -f1)
                echo "    $(basename "$f"): $SIZE"
            fi
        done
    else
        echo "  No logs/ directory found"
    fi
    exit 0
fi

# ── Alerts ──────────────────────────────────────────────────────────
if [ "$ALERTS_MODE" = true ]; then
    echo -e "${BOLD}═══ Recent Alerts ═══${NC}"
    python3 -c "
import os, sys
sys.path.insert(0, '.')
from monitoring_alerts import AlertManager
am = AlertManager()
history = am.get_alert_history()
config = am.get_config_summary()
print('  Config:')
for k, v in config.items():
    if isinstance(v, dict):
        print(f'    {k}:')
        for kk, vv in v.items():
            print(f'      {kk}: {vv}')
    else:
        print(f'    {k}: {v}')
print()
print(f'  Alerts this session: {len(history)}')
for a in history[-5:]:
    print(f'    [{a[\"level\"]}] {a[\"title\"]}: {a[\"message\"]}')
" 2>/dev/null || echo "  monitoring_alerts module not available"
    exit 0
fi

# ── JSON output ─────────────────────────────────────────────────────
# Emit pure JSON BEFORE any decorative header so `--json` output is parseable.
# Note: use Python True/False (the shell substitutes the literal), not the
# lowercase JSON true/false, which is invalid Python and raised NameError.
if [ "$JSON_MODE" = true ]; then
    python3 -c "
import json, os
result = {'process_running': $([ "$PROCESS_RUNNING" = true ] && echo 'True' || echo 'False')}
result['pid'] = '$RUNNER_PID'
if os.path.isfile('health.json'):
    with open('health.json') as f:
        result['health'] = json.load(f)
else:
    result['health'] = None
print(json.dumps(result, indent=2))
"
    exit 0
fi

# ── Main status output ──────────────────────────────────────────────
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Evolution Engine — Status${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"

# Process status (systemd or manual)
if [ "$SERVICE_ACTIVE" = true ]; then
    echo -e "  Service:   ${GREEN}RUNNING${NC} (systemd: $SERVICE_NAME)"
    echo -e "  PID:       $SERVICE_PID"
    if [ -n "$RUNTIME_SINCE" ]; then
        echo -e "  Since:     ${RUNTIME_SINCE:0:24}"
    fi
elif [ "$PROCESS_RUNNING" = true ]; then
    echo -e "  Process:   ${GREEN}RUNNING${NC} (PID: $RUNNER_PID)"
else
    echo -e "  Process:   ${RED}STOPPED${NC}"
fi

# Health data
if [ -n "$HEALTH_DATA" ]; then
    echo ""
    python3 -c "
import json
h = json.loads('''$HEALTH_DATA''')
status = h.get('status', 'unknown')
cycles = h.get('cycles_completed', 0)
fitness = h.get('current_fitness', 0.0)
level = h.get('current_level', 0)
errors = h.get('errors', 0)
uptime = h.get('uptime_seconds', 0)
target = h.get('target_fitness', None)

# Color the status
status_colors = {'running': '\033[32m', 'idle': '\033[33m', 'stopped': '\033[31m'}
color = status_colors.get(status, '\033[0m')
reset = '\033[0m'

print(f'  Status:    {color}{status.upper()}{reset}')
print(f'  Cycles:    {cycles}')
print(f'  Fitness:   {fitness:.3f}' + (f' / {target}' if target else ''))
print(f'  Level:     {level}')
print(f'  Errors:    {errors}')
print(f'  Uptime:    {uptime:.0f}s ({uptime/3600:.1f}h)')
" 2>/dev/null || echo "  (Unable to parse health.json)"
else
    echo -e "  Health:    ${YELLOW}NOT AVAILABLE${NC} (no health.json)"
fi

# Stop file
if [ -f stop ]; then
    echo -e "  Stop file: ${YELLOW}PRESENT${NC}"
fi

echo -e "${BOLD}═══════════════════════════════════════════════════════${NC}"
echo ""
