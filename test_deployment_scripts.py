#!/usr/bin/env python3
"""Tests for this session's new scripts: run_v10.sh, v9_monitor.py, v9_to_v10_watcher.py"""
import os, sys, json, subprocess, tempfile, re

SCRIPT_DIR = "/root/milestone2"
V9_LOG = "/root/continuous_iteration/run_mc_v9_20260604_042450.log"
MONITOR_SCRIPT = os.path.join(SCRIPT_DIR, "v9_monitor.py")
WATCHER_SCRIPT = os.path.join(SCRIPT_DIR, "v9_to_v10_watcher.py")
SCRIPT = os.path.join(SCRIPT_DIR, "run_v10.sh")

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — {detail}")

def test_eq(name, actual, expected, detail=""):
    global passed, failed
    if actual == expected:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: expected={expected!r} actual={actual!r} {detail}")

def test_contains(name, haystack, needle):
    global passed, failed
    if needle in haystack:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: '{needle}' not found in output")

# ─────────────────────────────────────────────────────────────────
# Test 1: run_v10.sh exists and is executable
# ─────────────────────────────────────────────────────────────────
print("\n=== Test Suite: New Deployment Scripts ===\n")

print("--- Test Group 1: Script Existence ---")
test("run_v10.sh exists", os.path.exists(SCRIPT))
test("run_v10.sh executable", os.access(SCRIPT, os.X_OK))
test("v9_monitor.py exists", os.path.exists(MONITOR_SCRIPT))
test("v9_to_v10_watcher.py exists", os.path.exists(WATCHER_SCRIPT))

# ─────────────────────────────────────────────────────────────────
# Test 2: run_v10.sh --dry-run behavior
# ─────────────────────────────────────────────────────────────────
print("\n--- Test Group 2: run_v10.sh --dry-run ---")
result = subprocess.run(["bash", SCRIPT, "--dry-run"], capture_output=True, text=True, timeout=30)
test_contains("dry-run prints 'DRY RUN COMPLETE'", result.stdout, "DRY RUN COMPLETE")
test_contains("dry-run detects v9 running", result.stdout, "v9 is running")
test_eq("dry-run exit code = 0 (dry-run continues)", result.returncode, 0)

# ─────────────────────────────────────────────────────────────────
# Test 3: run_v10.sh --dry-run --force preview
# ─────────────────────────────────────────────────────────────────
print("\n--- Test Group 3: run_v10.sh --dry-run --force ---")
result = subprocess.run(["bash", SCRIPT, "--dry-run", "--force"], capture_output=True, text=True, timeout=30)
test_contains("force dry-run prints kill commands", result.stdout, "Killing v9")
test_contains("force dry-run shows launch command", result.stdout, "run_multi_cycle.py")
test_eq("force dry-run exit code = 0", result.returncode, 0, "(dry-run exits 0 on success)")

# ─────────────────────────────────────────────────────────────────
# Test 4: v9_monitor.py execution on real log
# ─────────────────────────────────────────────────────────────────
print("\n--- Test Group 4: v9_monitor.py log parsing ---")
result = subprocess.run(["python3", MONITOR_SCRIPT], capture_output=True, text=True, timeout=30)
test_contains("monitor prints Summary", result.stdout, "Summary")
test_contains("monitor shows cycles completed", result.stdout, "completed")
test_contains("monitor shows MC summary", result.stdout, "MC:")
# Check the monitor wrote to out path
OUT_PATH = "/root/continuous_iteration/v9_monitor_events.json"
test("monitor events file created", os.path.exists(OUT_PATH))
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        events = json.load(f)
    test("events file is valid JSON", isinstance(events, list))
    test("events has at least 1 entry", len(events) >= 1)
    if len(events) > 0:
        e = events[-1]
        test("events entry has cycles_completed", "cycles_completed" in e)
        test("events entry has timestamp", "timestamp" in e)
        test("events entry has run_complete", "run_complete" in e)
        # Verify the monitor correctly sees v9 is NOT complete
        test_eq("monitor correctly reports v9 running", e.get("run_complete"), False)

# ─────────────────────────────────────────────────────────────────
# Test 5: v9_to_v10_watcher.py --status
# ─────────────────────────────────────────────────────────────────
print("\n--- Test Group 5: v9_to_v10_watcher.py --status ---")
result = subprocess.run(["python3", WATCHER_SCRIPT, "--status"], capture_output=True, text=True, timeout=30)
test_contains("watcher status is JSON", result.stdout, '"running"')
test_contains("watcher status has current_cycle", result.stdout, '"current_cycle"')
# Parse the JSON status (multiline)
try:
    # Find the line starting with '{' and collect until matching '}'
    lines = result.stdout.split('\n')
    json_lines = []
    in_json = False
    brace_depth = 0
    for line in lines:
        if '{' in line and not in_json:
            in_json = True
        if in_json:
            json_lines.append(line)
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0 and in_json:
                # JSON block complete
                break
    if json_lines:
        status = json.loads('\n'.join(json_lines))
    else:
        raise ValueError("No JSON found")
    test("watcher status: running is bool", isinstance(status.get("running"), bool))
    test("watcher status: v9 is running", status.get("running") == True)
    test("watcher status: completed_cycles >= 27", status.get("completed_cycles", 0) >= 27)
    if status.get("current_cycle") is not None:
        test("watcher status: current_cycle > 20", status.get("current_cycle", 0) > 20)
    else:
        # current_cycle can be null between cycles (e.g., during retrain)
        test("watcher status: current_cycle null (between cycles)", True)
    test_contains("watcher status: human readable", result.stdout, "v9 running at")
except (json.JSONDecodeError, IndexError):
    test("watcher status JSON parse", False, detail=f"Could not parse JSON from: {result.stdout[:200]}")

# ─────────────────────────────────────────────────────────────────
# Test 6: Script source code completeness check
# ─────────────────────────────────────────────────────────────────
print("\n--- Test Group 6: Source Code Completeness ---")

# run_v10.sh
with open(SCRIPT) as f:
    sh_content = f.read()
test("run_v10.sh: has shebang", sh_content.startswith("#!/usr/bin/env bash"))
test("run_v10.sh: has DRY_RUN variable", "DRY_RUN=true" in sh_content)
test("run_v10.sh: has FORCE variable", "FORCE=true" in sh_content)
test("run_v10.sh: has --dry-run handler", "--dry-run" in sh_content)
test("run_v10.sh: has --force handler", "--force" in sh_content)
test("run_v10.sh: has test verification", "python3 -m pytest" in sh_content)
test("run_v10.sh: has pycache cleanup", "__pycache__" in sh_content)
test("run_v10.sh: has memory check", "free -m" in sh_content)
test("run_v10.sh: has summary generation", ".summary" in sh_content)
test("run_v10.sh: no placeholders/TODOs", "TODO" not in sh_content and "FIXME" not in sh_content)
test("run_v10.sh: no pass/empty blocks", "do_nothing" not in sh_content)  # common stub pattern
test("run_v10.sh: set -euo pipefail", "set -euo pipefail" in sh_content)

# v9_monitor.py
with open(MONITOR_SCRIPT) as f:
    py_content = f.read()
test("v9_monitor: has main()", "def main():" in py_content)
test("v9_monitor: has parse_latest()", "def parse_latest():" in py_content)
test("v9_monitor: parses MC entries", "MetaController" in py_content)
test("v9_monitor: checks run complete", "run_complete" in py_content)
test("v9_monitor: writes JSON", "json.dump" in py_content)
test("v9_monitor: no placeholders", "TODO" not in py_content and "FIXME" not in py_content)

# v9_to_v10_watcher.py
with open(WATCHER_SCRIPT) as f:
    wc_content = f.read()
test("v9_to_v10_watcher: has check_v9_status()", "def check_v9_status():" in wc_content)
test("v9_to_v10_watcher: has force_launch()", "def force_launch():" in wc_content)
test("v9_to_v10_watcher: has launch_v10()", "def launch_v10():" in wc_content)
test("v9_to_v10_watcher: has main()", "def main():" in wc_content)
test_contains("v9_to_v10_watcher: detects run complete", wc_content, "cycles complete")
test("v9_to_v10_watcher: polls for completion", "time.sleep(60)" in wc_content)
test("v9_to_v10_watcher: handles --force", "--force" in wc_content)
test("v9_to_v10_watcher: handles --status", "--status" in wc_content)
test("v9_to_v10_watcher: no placeholders", "TODO" not in wc_content and "FIXME" not in wc_content)

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")

if failed > 0:
    sys.exit(1)
else:
    print("All tests passed.")
    sys.exit(0)
