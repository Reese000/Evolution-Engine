#!/usr/bin/env bash
# v10_1_runner.sh — Self-healing launcher for v10.1
# Designed to be run from cron. Detects if v10.1 is already running,
# and if not, starts a new instance.
set -euo pipefail

PID_FILE="/root/continuous_iteration/v10_1.pid"
LOG_DIR="/root/continuous_iteration"
SCRIPT_DIR="/root/milestone2"

# Check if already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "v10.1 already running (PID $OLD_PID)"
        exit 0
    fi
    echo "Stale PID $OLD_PID — cleaning up"
    rm -f "$PID_FILE"
fi

# Clear pycache
rm -rf "${SCRIPT_DIR}/__pycache__"

# Launch with nohup
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_mc_v101_${TIMESTAMP}.log"

nohup python3 -u "${SCRIPT_DIR}/run_multi_cycle.py" \
    --cycles 48 --gens 50 \
    --retrain-model-size medium \
    --retrain-freeze-layers 2 \
    --retrain-lr 3e-5 --retrain-patience 5 --retrain-max-epochs 50 \
    > "${LOG_FILE}" 2>&1 &

NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "v10.1 launched (PID $NEW_PID, log: $LOG_FILE)"
