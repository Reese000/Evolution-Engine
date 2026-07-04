#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════
# deploy.sh — One-command deployment for Evolution Engine
#
# Usage:
#   ./deploy.sh              # Start with defaults
#   ./deploy.sh --gpu        # Start with GPU support
#   ./deploy.sh --dry-run    # Validate config only
#   ./deploy.sh --status     # Show health status
# ══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[deploy]${NC} $*"; }
ok()   { echo -e "${GREEN}[  OK  ]${NC} $*"; }
warn() { echo -e "${YELLOW}[ WARN ]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Parse arguments ──────────────────────────────────────────────────
USE_GPU=false
DRY_RUN=false
STATUS_ONLY=false
USE_DOCKER=false

for arg in "$@"; do
    case "$arg" in
        --gpu)      USE_GPU=true ;;
        --dry-run)  DRY_RUN=true ;;
        --status)   STATUS_ONLY=true ;;
        --docker)   USE_DOCKER=true ;;
        --help|-h)
            echo "Usage: $0 [--gpu] [--dry-run] [--status] [--docker]"
            exit 0
            ;;
    esac
done

# ── Status mode ──────────────────────────────────────────────────────
if [ "$STATUS_ONLY" = true ]; then
    log "Checking health status..."
    if [ -f health.json ]; then
        python3 -c "
import json
with open('health.json') as f:
    h = json.load(f)
print('  Status:           ', h.get('status', 'unknown'))
print('  Cycles completed: ', h.get('cycles_completed', 0))
print('  Current fitness:  ', h.get('current_fitness', 0))
print('  Current level:    ', h.get('current_level', 0))
print('  Errors:           ', h.get('errors', 0))
print('  Uptime:           ', h.get('uptime_seconds', 0), 'seconds')
"
        exit 0
    else
        err "No health.json found. Runner may not be started."
        exit 1
    fi
fi

# ── Docker mode ──────────────────────────────────────────────────────
if [ "$USE_DOCKER" = true ]; then
    log "Deploying via Docker Compose..."
    if [ ! -f docker-compose.yml ]; then
        err "docker-compose.yml not found"
        exit 1
    fi
    if [ "$USE_GPU" = true ]; then
        export EE_USE_GPU=true
        docker compose --profile gpu up -d --build
    else
        docker compose up -d --build
    fi
    ok "Container started. Run '$0 --status' to check health."
    exit 0
fi

# ── Pre-flight checks ───────────────────────────────────────────────
log "Pre-flight checks..."

if ! command -v python3 &>/dev/null; then
    err "python3 not found in PATH"
    exit 1
fi

if [ ! -f config/engine_config.json ]; then
    err "config/engine_config.json not found"
    exit 1
fi

# Check Python deps
python3 -c "import torch; import transformers" 2>/dev/null || {
    warn "Some dependencies may be missing. Install: pip install -r requirements.txt"
}

# Create required directories
mkdir -p logs metrics cycle_data archive

ok "Pre-flight passed"

# ── Dry run ──────────────────────────────────────────────────────────
if [ "$DRY_RUN" = true ]; then
    log "Running dry-run validation..."
    python3 autonomous_runner.py --dry-run
    ok "Dry-run passed"
    exit 0
fi

# ── Check if already running ────────────────────────────────────────
PID_FILE=".runner.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        warn "Runner already running (PID $OLD_PID)"
        warn "Use ./stop.sh to stop first, or kill $OLD_PID"
        exit 1
    else
        log "Removing stale PID file"
        rm -f "$PID_FILE"
    fi
fi

# ── Start runner ─────────────────────────────────────────────────────
log "Starting Evolution Engine..."
RESUME_FLAG=""
if [ -f runner_state.json ]; then
    RESUME_FLAG="--resume"
    log "Found saved state — resuming"
fi

nohup python3 autonomous_runner.py $RESUME_FLAG \
    > logs/runner_stdout.log 2>&1 &
RUNNER_PID=$!
echo "$RUNNER_PID" > "$PID_FILE"

# Wait for health.json to appear
log "Waiting for startup (PID: $RUNNER_PID)..."
for i in $(seq 1 15); do
    if [ -f health.json ]; then
        ok "Runner started successfully (PID: $RUNNER_PID)"
        ok "Health: $(python3 -c "import json; print(json.load(open('health.json'))['status'])" 2>/dev/null || echo 'starting')"
        exit 0
    fi
    sleep 1
done

# Check if process is still alive
if kill -0 "$RUNNER_PID" 2>/dev/null; then
    ok "Runner started (PID: $RUNNER_PID) — health.json pending"
    exit 0
else
    err "Runner failed to start. Check logs/runner_stdout.log"
    exit 1
fi
