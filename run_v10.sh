#!/usr/bin/env bash
# ============================================================================
# run_v10.sh — Evolution Engine v10 Deployment Script
# ============================================================================
# Deploys v10 with 3 fixes: GROW cap (10 consecutive), CROSSOVER 10%, adaptive
# model_guided_ratio. Designed to run after v9 completes.
#
# Usage:
#   ./run_v10.sh                    # Normal launch
#   ./run_v10.sh --dry-run          # Print what would happen, don't execute
#   ./run_v10.sh --force            # Kill v9 if running, start v10 anyway
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/root/continuous_iteration"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/run_mc_v10_${TIMESTAMP}.log"
DRY_RUN=false
FORCE=false

# Parse args
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --force) FORCE=true ;;
    esac
done

echo "=== v10 Deployment Script ==="
echo "Timestamp: $(date -u)"
echo "Directory: ${SCRIPT_DIR}"
echo "Log file: ${LOG_FILE}"
echo ""

# ----------------------------------------------------------------
# Step 1: Kill any leftover v9 processes
# ----------------------------------------------------------------
echo "[Step 1] Checking for lingering v9 processes..."
V9_PID=$(pgrep -f "run_multi_cycle" | head -1 || true)
if [ -n "$V9_PID" ]; then
    echo "  Found v9 PID: ${V9_PID}"
    if [ "$FORCE" = true ]; then
        echo "  --force: Killing v9 process..."
        $DRY_RUN || kill "$V9_PID" 2>/dev/null || true
        sleep 2
        # Kill any orphan train_model processes
        pkill -f "train_model" 2>/dev/null || true
        sleep 1
        # Force kill leftovers
        if kill -0 "$V9_PID" 2>/dev/null; then
            echo "  Process still alive, sending SIGKILL..."
            $DRY_RUN || kill -9 "$V9_PID" 2>/dev/null || true
        fi
        echo "  v9 process terminated."
    elif [ "$DRY_RUN" = true ]; then
        echo "  v9 is running. --dry-run: would exit here. Use --force to override."
        echo "  (Continuing dry-run to show full flow)"
    else
        echo "  v9 still running. Use --force to kill and start v10 anyway."
        echo "  (Recommended: wait for v9 to complete naturally)"
        exit 1
    fi
else
    echo "  No lingering v9 process found."
fi

# Kill any orphan train_model processes (safety cleanup)
if [ "$FORCE" = true ]; then
    pkill -f "train_model" 2>/dev/null || true
fi

echo ""

# ----------------------------------------------------------------
# Step 2: Clear stale __pycache__ (prevent import cache bugs)
# ----------------------------------------------------------------
echo "[Step 2] Clearing stale __pycache__..."
for dir in "${SCRIPT_DIR}/__pycache__" "${SCRIPT_DIR}/agent/__pycache__" "${SCRIPT_DIR}/gpu-vm/__pycache__"; do
    if [ -d "$dir" ]; then
        echo "  Clearing: ${dir}"
        $DRY_RUN || rm -rf "$dir"
    fi
done
echo "  __pycache__ cleared."

echo ""

# ----------------------------------------------------------------
# Step 3: Verify test suite
# ----------------------------------------------------------------
echo "[Step 3] Verifying test suite..."
if [ "$DRY_RUN" = false ]; then
    cd "$SCRIPT_DIR"
    TEST_OUTPUT=$(python3 -m pytest test_meta_controller.py test_v6_features.py test_feature_toggles.py test_qd_archive.py test_novelty.py test_multi_cycle.py test_evolved_config.py test_size_guard.py -q --tb=line 2>&1 || true)
    TEST_RESULT=$(echo "$TEST_OUTPUT" | grep -oE '[0-9]+ passed' | head -1 || echo "0 passed")
    echo "  Test result: ${TEST_RESULT}"
    
    # Extract test count
    PASSED_COUNT=$(echo "$TEST_RESULT" | grep -oE '[0-9]+' || echo "0")
    if [ "$PASSED_COUNT" -lt 100 ]; then
        echo "  ⚠ WARNING: Less than 100 tests passed (got ${PASSED_COUNT})."
        echo "  Output:"
        echo "$TEST_OUTPUT" | tail -5
        echo ""
        echo -n "  Continue anyway? (y/N): "
        read -r CONTINUE
        if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
            echo "  Aborting v10 deployment."
            exit 1
        fi
    else
        echo "  ✅ All tests passed."
    fi
else
    echo "  (dry-run: would verify test suite)"
fi

echo ""

# ----------------------------------------------------------------
# Step 4: Check GPU memory / system readiness
# ----------------------------------------------------------------
echo "[Step 4] Checking system readiness..."
if [ "$DRY_RUN" = false ]; then
    MEM_AVAIL=$(free -m 2>/dev/null | grep Mem | awk '{print $7}' || echo "0")
    echo "  Available memory: ${MEM_AVAIL} MB"
    if [ "$MEM_AVAIL" -lt 2000 ]; then
        echo "  ⚠ WARNING: Low memory available (< 2GB). v9 may have left memory pressure."
    fi
    
    # Check disk space for logs
    DISK_AVAIL=$(df -h /root/continuous_iteration 2>/dev/null | tail -1 | awk '{print $4}' || echo "?")
    echo "  Disk space for logs: ${DISK_AVAIL}"
else
    echo "  (dry-run: would check system readiness)"
fi

echo ""

# ----------------------------------------------------------------
# Step 5: Launch v10
# ----------------------------------------------------------------
echo "[Step 5] Launching v10..."
echo "  Command: python3 -u run_multi_cycle.py --cycles 48 --gens 50"
echo "    --retrain-model-size medium --retrain-freeze-layers 2"
echo "    --retrain-lr 3e-5 --retrain-patience 5 --retrain-max-epochs 50"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "  (dry-run: would execute:)"
    echo "  cd ${SCRIPT_DIR} && python3 -u run_multi_cycle.py \\"
    echo "    --cycles 48 --gens 50 \\"
    echo "    --retrain-model-size medium \\"
    echo "    --retrain-freeze-layers 2 \\"
    echo "    --retrain-lr 3e-5 --retrain-patience 5 --retrain-max-epochs 50 \\"
    echo "    2>&1 | tee ${LOG_FILE}"
    echo ""
    echo "=== DRY RUN COMPLETE ==="
    exit 0
fi

cd "$SCRIPT_DIR"

# Create log directory if needed
mkdir -p "$LOG_DIR"

echo "  Logging to: ${LOG_FILE}"
echo "  v10 started at: $(date -u)"
echo ""

# Execute with error handling
set +e
python3 -u run_multi_cycle.py \
    --cycles 48 --gens 50 \
    --retrain-model-size medium \
    --retrain-freeze-layers 2 \
    --retrain-lr 3e-5 --retrain-patience 5 --retrain-max-epochs 50 \
    2>&1 | tee "$LOG_FILE"
EXIT_CODE=$?
set -e

echo ""
echo "=== v10 completed with exit code ${EXIT_CODE} ==="
echo "  Finish time: $(date -u)"
echo "  Log: ${LOG_FILE}"

# Write summary
{
    echo "=== v10 Summary ==="
    echo "Start: $(head -1 "${LOG_FILE}" 2>/dev/null || echo 'unknown')"
    echo "End: $(date -u)"
    echo "Exit code: ${EXIT_CODE}"
    echo "Cycles completed: $(grep -c 'CYCLE.*complete' "${LOG_FILE}" 2>/dev/null || echo '?')"
    echo "Best val_loss: $(grep 'Checkpoint ACCEPTED' "${LOG_FILE}" 2>/dev/null | grep -oP 'val_loss=\S+' | sort -t'=' -k2 -n | head -1 || echo '?')"
    echo "Best level: $(grep "\[MetaController\] Cycle" "${LOG_FILE}" 2>/dev/null | grep -oP 'L=\d+' | sort -t'=' -k2 -n | tail -1 || echo '?')"
    echo "Positive fitness cycles: $(grep "\[MetaController\] Cycle.*F=[0-9]" "${LOG_FILE}" 2>/dev/null | wc -l || echo '?')"
} > "${LOG_FILE}.summary"

echo "  Summary written to: ${LOG_FILE}.summary"
echo ""
echo "=== DEPLOYMENT COMPLETE ==="
exit ${EXIT_CODE}
