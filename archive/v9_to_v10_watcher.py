#!/usr/bin/env python3
"""
v9_to_v10_watcher.py — Watches v9 log and auto-launches v10 when v9 completes.
Can also force-launch v10 immediately with --force flag.

Usage:
  python3 v9_to_v10_watcher.py            # Watch mode: launch v10 when v9 finishes
  python3 v9_to_v10_watcher.py --force    # Force: kill v9 and launch v10 NOW
  python3 v9_to_v10_watcher.py --status   # Just print status, don't act
"""
import re, os, sys, time, subprocess, json

LOG_PATH = "/root/continuous_iteration/run_mc_v9_20260604_042450.log"
SCRIPT_DIR = "/root/milestone2"
SCRIPT = os.path.join(SCRIPT_DIR, "run_v10.sh")

def check_v9_status():
    """Check if v9 is still running and its completion status."""
    if not os.path.exists(LOG_PATH):
        return {"running": False, "cycles": 0, "complete": False, "error": "no log"}
    
    with open(LOG_PATH) as f:
        content = f.read()
    
    # Check if run is complete
    if re.search(r'All \d+ cycles complete', content) or re.search(r'Run complete', content):
        return {"running": False, "cycles": 48, "complete": True, "error": None}
    
    # Count completed cycles
    completed = len(re.findall(r'CYCLE \d+/48\n', content))
    
    # Check if process is still running
    result = subprocess.run(
        ["pgrep", "-f", "run_multi_cycle"],
        capture_output=True, text=True, timeout=5
    )
    running = bool(result.stdout.strip())
    
    # Get latest cycle (look backwards through content for CYCLE X/48)
    latest = None
    for m in re.finditer(r'CYCLE (\d+)/48', content):
        latest = int(m.group(1))
    
    # Get latest MetaController entry
    mc = None
    for line in reversed(content.split('\n')):
        m = re.match(r'\[MetaController\] Cycle (\d+): F=([-\d.]+) L=(\d+)', line)
        if m:
            mc = {"cycle": int(m.group(1)), "fitness": float(m.group(2)), "level": int(m.group(3))}
            break
    
    # Log file still growing?
    last_size = os.path.getsize(LOG_PATH) if os.path.exists(LOG_PATH) else 0
    still_writing = len(re.findall(r'CYCLE \d+/48', content)) > 1  # multiple cycle headers
    
    return {
        "running": running,
        "completed_cycles": completed,
        "current_cycle": latest,
        "complete": False,
        "mc": mc,
        "log_size": last_size,
        "error": None
    }

def launch_v10():
    """Launch v10 run."""
    print("=" * 60)
    print("v9 COMPLETE — Launching v10...")
    print("=" * 60)
    print(f"Command: {SCRIPT}")
    print(f"Time: {time.strftime('%H:%M:%S UTC')}")
    print()
    
    result = subprocess.run(
        [SCRIPT], 
        capture_output=True, text=True, timeout=30
    )
    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    print(f"Exit code: {result.returncode}")
    return result.returncode == 0

def force_launch():
    """Force-launch v10, killing v9 if running."""
    print("=" * 60)
    print("Force-launching v10 (killing v9)...")
    print("=" * 60)
    
    # Kill v9
    subprocess.run(["pkill", "-f", "run_multi_cycle"], timeout=5)
    time.sleep(2)
    subprocess.run(["pkill", "-9", "-f", "train_model"], timeout=5)
    time.sleep(1)
    
    # Clear pycache
    for d in ["/root/milestone2/__pycache__"]:
        if os.path.exists(d):
            subprocess.run(["rm", "-rf", d], timeout=5)
    
    return launch_v10()

def main():
    if "--force" in sys.argv:
        force_launch()
        return
    
    if "--status" in sys.argv:
        status = check_v9_status()
        print(json.dumps(status, indent=2))
        if status.get("complete"):
            print("\nv9 is COMPLETE. Run: python3 v9_to_v10_watcher.py")
        elif not status.get("running") and status.get("current_cycle"):
            print(f"\nv9 process dead at C{status['current_cycle']}/48. Run: python3 v9_to_v10_watcher.py --force")
        else:
            print(f"\nv9 running at C{status.get('current_cycle','?')}/48, {status.get('completed_cycles',0)} completed.")
        return
    
    # Watch mode: poll every 60s until v9 completes
    print("v9→v10 Watcher — Polling every 60s")
    print(f"Log: {LOG_PATH}")
    print("Press Ctrl+C to exit.")
    print()
    
    # Check current state
    if not os.path.exists(LOG_PATH):
        print(f"ERROR: Log not found at {LOG_PATH}")
        sys.exit(1)
    
    last_completed = 0
    stall_count = 0
    
    try:
        while True:
            status = check_v9_status()
            now = time.strftime('%H:%M:%S')
            
            if status.get("complete"):
                print(f"[{now}] v9 COMPLETE — all 48 cycles done!") 
                launch_v10()
                return
            
            if not status.get("running") and status.get("completed_cycles", 0) > 10:
                # v9 process died but didn't finish normally
                print(f"[{now}] v9 process dead at C{status.get('current_cycle','?')} ({status.get('completed_cycles',0)} completed)")
                # Check if log is still being written
                current_size = status.get("log_size", 0)
                if stall_count > 5:  # 5 minutes of no change = dead
                    print(f"[{now}] Log stopped growing — forcing v10 launch...")
                    force_launch()
                    return
                stall_count += 1
            else:
                stall_count = 0
                mc = status.get("mc")
                if mc:
                    flag = " ★" if mc["fitness"] > 0 else ""
                    print(f"[{now}] v9 C{mc['cycle']}/48: F={mc['fitness']:+.3f} L={mc['level']}{flag}")
                else:
                    print(f"[{now}] v9 C{status.get('current_cycle','?')}/48")
            
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nWatcher stopped.")
        print("Run 'python3 v9_to_v10_watcher.py --status' to check status.")
        print("Run 'python3 v9_to_v10_watcher.py --force' to launch v10 now.")

if __name__ == '__main__':
    main()
