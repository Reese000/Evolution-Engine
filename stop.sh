#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# stop.sh — Graceful shutdown of Evolution Engine
#
# Usage:
#   ./stop.sh              # Graceful shutdown (wait for current cycle)
#   ./stop.sh --force      # Immediate kill
#   ./stop.sh --docker     # Stop Docker container
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[ stop ]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

FORCE=false
DOCKER_STOP=false

for arg in "$@"; do
    case "$arg" in
        --force)   FORCE=true ;;
        --docker)  DOCKER_STOP=true ;;
        --help|-h)
            echo "Usage: $0 [--force] [--docker]"
            exit 0
            ;;
    esac
done

# ── Docker stop ──────────────────────────────────────────────────────
if [ "$DOCKER_STOP" = true ]; then
    log "Stopping Docker containers..."
    docker compose down 2>/dev/null || docker-compose down 2>/dev/null || {
        err "Failed to stop Docker containers"
        exit 1
    }
    ok "Docker containers stopped"
    exit 0
fi

# ── Find runner PID ──────────────────────────────────────────────────
PID_FILE=".runner.pid"
RUNNER_PID=""

if [ -f "$PID_FILE" ]; then
    RUNNER_PID=$(cat "$PID_FILE")
fi

# Also check for any autonomous_runner processes
if [ -z "$RUNNER_PID" ] || ! kill -0 "$RUNNER_PID" 2>/dev/null; then
    RUNNER_PID=$(pgrep -f "autonomous_runner.py" 2>/dev/null | head -1 || true)
fi

if [ -z "$RUNNER_PID" ]; then
    log "No running Evolution Engine found"
    # Clean up stop file if present
    rm -f "$PID_FILE" stop
    ok "Clean"
    exit 0
fi

log "Found runner process: PID $RUNNER_PID"

if [ "$FORCE" = true ]; then
    log "Force killing..."
    kill -9 "$RUNNER_PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    ok "Process killed (force)"
    exit 0
fi

# ── Graceful shutdown ───────────────────────────────────────────────
log "Sending SIGTERM for graceful shutdown..."
kill -TERM "$RUNNER_PID" 2>/dev/null || true

# Also create stop file as a fallback
touch stop

# Wait up to 120 seconds for graceful exit
TIMEOUT=120
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
        ok "Runner stopped gracefully after ${ELAPSED}s"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    if [ $((ELAPSED % 10)) -eq 0 ]; then
        log "Waiting for graceful shutdown... (${ELAPSED}s / ${TIMEOUT}s)"
    fi
done

# Still alive after timeout — force kill
warn "Timeout reached — force killing"
kill -9 "$RUNNER_PID" 2>/dev/null || true
rm -f "$PID_FILE"
ok "Process killed after timeout"
