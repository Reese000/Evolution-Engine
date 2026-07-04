#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# start_autonomous.sh — Start the Evolution Engine daemon
#
# Usage:
#   ./start_autonomous.sh              # Start the daemon
#   ./start_autonomous.sh --force      # Stop existing, then start fresh
#   ./start_autonomous.sh --dry-run    # Validate config without starting
#   ./start_autonomous.sh --log        # Start and tail the journal logs
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

log()  { echo -e "${CYAN}[ start ]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
err()  { echo -e "${RED}[ERROR ]${NC} $*"; }

# ── Constants ────────────────────────────────────────────────────────
SERVICE_NAME="evolution-engine"
SERVICE_DIR="$HOME/.config/systemd/user"
SERVICE_FILE="${SERVICE_DIR}/${SERVICE_NAME}.service"

# ── Parse flags ──────────────────────────────────────────────────────
FORCE=false
DRY_RUN=false
TAIL_LOG=false

for arg in "$@"; do
    case "$arg" in
        --force)    FORCE=true ;;
        --dry-run)  DRY_RUN=true ;;
        --log)      TAIL_LOG=true ;;
        --help|-h)
            echo "Usage: $0 [--force] [--dry-run] [--log]"
            echo ""
            echo "  --force     Stop any existing process first, then start fresh"
            echo "  --dry-run   Validate config without starting the daemon"
            echo "  --log       Tail the journal logs after starting"
            exit 0
            ;;
        *)
            err "Unknown flag: $arg"
            echo "  Run '$0 --help' for usage."
            exit 1
            ;;
    esac
done

# ── Banner ───────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Evolution Engine — Start${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
echo ""

# ── Dry-run mode ─────────────────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    log "Validating configuration (dry-run)..."
    echo ""
    if /usr/bin/python3 autonomous_runner.py --dry-run --config config/engine_config.json; then
        ok "Configuration is valid"
    else
        err "Configuration validation failed"
        exit 1
    fi
    exit 0
fi

# ── Check if already running ─────────────────────────────────────────
is_running() {
    systemctl --user is-active --quiet "$SERVICE_NAME" 2>/dev/null
}

if is_running; then
    warn "Evolution Engine is already running"
    echo ""
    systemctl --user status "$SERVICE_NAME" --no-pager || true
    echo ""
    ok "Use '$0 --force' to restart, or './stop.sh' to stop first."
    exit 0
fi

# ── Handle --force: stop any existing process first ──────────────────
if [ "$FORCE" = true ]; then
    # Stop systemd service if it exists (even if not "active")
    if systemctl --user list-unit-files "$SERVICE_NAME.service" &>/dev/null; then
        log "Stopping existing systemd service..."
        systemctl --user stop "$SERVICE_NAME" 2>/dev/null || true
    fi

    # Also check for orphaned runner processes
    ORPHAN_PIDS=$(pgrep -f "autonomous_runner.py" 2>/dev/null || true)
    if [ -n "$ORPHAN_PIDS" ]; then
        log "Killing orphaned runner processes: $ORPHAN_PIDS"
        for pid in $ORPHAN_PIDS; do
            kill -TERM "$pid" 2>/dev/null || true
        done
        # Wait briefly for graceful shutdown
        sleep 3
        # Force-kill any survivors
        for pid in $ORPHAN_PIDS; do
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi

    # Clean up stale PID file and stop file
    rm -f ".runner.pid" stop

    ok "Existing processes cleaned up"
    echo ""
fi

# ── Check if service file is installed ───────────────────────────────
if [ ! -f "$SERVICE_FILE" ]; then
    log "Service file not found at: $SERVICE_FILE"
    log "Installing service via install_service.sh..."
    echo ""

    if [ ! -f "$SCRIPT_DIR/install_service.sh" ]; then
        err "install_service.sh not found in $SCRIPT_DIR"
        err "Cannot install the service."
        exit 1
    fi

    bash "$SCRIPT_DIR/install_service.sh"
    echo ""
    ok "Service installed and started"
else
    # ── Service file exists but not running — start it ───────────────
    log "Starting $SERVICE_NAME..."
    systemctl --user start "$SERVICE_NAME"
    ok "Service started"
fi

# ── Wait for startup and show status ─────────────────────────────────
log "Waiting 3 seconds for daemon to initialize..."
sleep 3

echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Service Status${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
echo ""

if is_running; then
    echo -e "  Status:  ${GREEN}RUNNING${NC}"
    PID=$(systemctl --user show "$SERVICE_NAME" --property=MainPID --value 2>/dev/null || echo "?")
    echo -e "  PID:     $PID"
    SINCE=$(systemctl --user show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value 2>/dev/null || echo "?")
    echo -e "  Since:   ${SINCE:0:24}"
else
    echo -e "  Status:  ${RED}NOT RUNNING${NC}"
    warn "The service may have failed to start. Checking journal..."
    echo ""
    journalctl --user -u "$SERVICE_NAME" -n 15 --no-pager 2>/dev/null || true
    echo ""
    err "Service is not running. Review the logs above for details."
    exit 1
fi

# ── Check health.json for additional info ─────────────────────────────
if [ -f "$SCRIPT_DIR/health.json" ]; then
    echo ""
    /usr/bin/python3 -c "
import json
try:
    with open('health.json') as f:
        h = json.load(f)
    fitness = h.get('current_fitness', 0.0)
    level = h.get('current_level', 0)
    cycles = h.get('cycles_completed', 0)
    errors = h.get('errors', 0)
    target = h.get('target_fitness', None)
    print(f'  Fitness: {fitness:.3f}' + (f' / {target}' if target else ''))
    print(f'  Level:   {level}')
    print(f'  Cycles:  {cycles}')
    print(f'  Errors:  {errors}')
except Exception:
    pass
" 2>/dev/null || true
fi

# ── Tail logs if requested ───────────────────────────────────────────
if [ "$TAIL_LOG" = true ]; then
    echo ""
    echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}  Tailing Logs (Ctrl+C to stop)${NC}"
    echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
    echo ""
    journalctl --user -u "$SERVICE_NAME" -f --no-pager 2>/dev/null || {
        warn "Cannot tail journal — falling back to log file"
        tail -f "$SCRIPT_DIR/evolution.log" 2>/dev/null || {
            err "No log output available"
        }
    }
    exit 0
fi

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Useful Commands${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}Monitor:${NC}      ./monitor.sh"
echo -e "  ${CYAN}Status:${NC}       ./status.sh"
echo -e "  ${CYAN}Logs:${NC}         journalctl --user -u $SERVICE_NAME -f"
echo -e "  ${CYAN}Stop:${NC}         ./stop.sh"
echo -e "  ${CYAN}Restart:${NC}      systemctl --user restart $SERVICE_NAME"
echo -e "  ${CYAN}Force start:${NC}  $0 --force"
echo -e "  ${CYAN}Dry run:${NC}      $0 --dry-run"
echo ""
