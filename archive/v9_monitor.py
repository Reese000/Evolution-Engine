#!/usr/bin/env python3
"""
v9_finish_monitor.py — Monitors v9 log for cycle completions.
Logs key events to a JSON file for later analysis.
Designed to run in background during session.
"""
import json
import os
import re
import time

LOG_PATH = "/root/continuous_iteration/run_mc_v9_20260604_042450.log"
OUT_PATH = "/root/continuous_iteration/v9_monitor_events.json"
POLL_INTERVAL = 60  # seconds


def parse_latest():
    """Parse the log and return latest metrics."""
    if not os.path.exists(LOG_PATH):
        return None

    with open(LOG_PATH) as f:
        content = f.read()

    lines = content.split("\n")

    # Get last seen events
    result = {}

    # Last MetaController entry
    for line in reversed(lines):
        m = re.match(r"\[MetaController\] Cycle (\d+): F=([-\d.]+) L=(\d+) val=([-\d.]+) div=([-\d.]+)", line)
        if m:
            result["last_mc"] = {
                "cycle": int(m.group(1)),
                "fitness": float(m.group(2)),
                "level": int(m.group(3)),
                "val_loss": float(m.group(4)),
                "diversity": float(m.group(5)),
            }
            break

    # Last CYCLE header
    for line in reversed(lines):
        m = re.match(r"CYCLE (\d+)/48", line)
        if m:
            result["current_cycle"] = int(m.group(1))
            break

    # Last complete message
    for line in reversed(lines):
        m = re.match(r".*Cycle (\d+) complete: gen=(\d+), fitness=([-\d.]+), level=(\d+)", line)
        if m:
            result["last_complete"] = {
                "cycle": int(m.group(1)),
                "gen": int(m.group(2)),
                "fitness": float(m.group(3)),
                "level": int(m.group(4)),
            }
            break

    # Count cycles completed
    result["cycles_completed"] = len(re.findall(r"Cycle \d+ complete:", content))

    # Check if run finished
    if re.search(r"All \d+ cycles complete", content) or re.search(r"Run complete", content):
        result["run_complete"] = True
    else:
        result["run_complete"] = False

    result["timestamp"] = time.time()

    return result


def main():
    # Load previous state
    events = []
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH) as f:
                events = json.load(f)
        except Exception:
            events = []

    # Poll once and record
    state = parse_latest()
    if state:
        # Only record if new cycle completed since last check
        prev_cycles = events[-1].get("cycles_completed", 0) if events else 0
        if state["cycles_completed"] > prev_cycles or not events:
            events.append(state)
            with open(OUT_PATH, "w") as f:
                json.dump(events, f, indent=2)
            print(
                f"[{time.strftime('%H:%M:%S')}] v9: C{state.get('current_cycle','?')}/48 running, {state.get('cycles_completed',0)} completed"
            )
            if state.get("last_mc"):
                mc = state["last_mc"]
                flag = " ★" if mc["fitness"] > 0 else ""
                print(f"  MC: F={mc['fitness']:+.3f} L={mc['level']:>2} val={mc['val_loss']:.4f}{flag}")
            if state.get("run_complete"):
                print("  *** v9 RUN COMPLETE ***")
                print("  Launch v10: /root/milestone2/run_v10.sh")

    # Print summary
    print(f"\nSummary from {len(events)} polls:")
    if events:
        latest = events[-1]
        print(f"  Cycles completed: {latest.get('cycles_completed', 0)}")
        print(f"  Run complete: {latest.get('run_complete', False)}")
        if latest.get("last_mc"):
            mc = latest["last_mc"]
            print(f"  MC: C{mc['cycle']} F={mc['fitness']:+.3f} L={mc['level']} val={mc['val_loss']:.4f}")
        print(f"  Time: {time.strftime('%H:%M:%S', time.localtime(latest['timestamp']))}")

    # Save events
    with open(OUT_PATH, "w") as f:
        json.dump(events, f, indent=2)


if __name__ == "__main__":
    main()
