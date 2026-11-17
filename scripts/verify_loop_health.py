#!/usr/bin/env python3
"""Verify and repair the persistent agent/evolution loop infrastructure.

Run every agent tick and from start_persistent_loop.ps1:
  python scripts/verify_loop_health.py          # check only, exit 1 if unhealthy
  python scripts/verify_loop_health.py --repair # restart dead/stale components
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

AGENTS = os.path.join(PROJECT_ROOT, ".agents")
SCRIPTS = os.path.join(PROJECT_ROOT, "scripts")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _heartbeat_age_sec(path: str) -> float | None:
    if not os.path.isfile(path):
        return None
    try:
        data = _read_json(path)
        at = data.get("at", "")
        if not at:
            return None
        ts = datetime.fromisoformat(at.replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _repair_pid_from_heartbeat(pid_path: str, hb_path: str) -> bool:
    """Recreate a missing pid file when heartbeat reports a live process."""
    if os.path.isfile(pid_path):
        return False
    if not os.path.isfile(hb_path):
        return False
    try:
        data = _read_json(hb_path)
        pid = int(data.get("pid", 0))
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return False
    if pid <= 0 or not _pid_alive(pid):
        return False
    os.makedirs(os.path.dirname(pid_path), exist_ok=True)
    with open(pid_path, "w", encoding="utf-8") as f:
        f.write(str(pid))
    return True


def _check_component(name: str, pid_path: str, hb_path: str, max_age: float) -> tuple[bool, str]:
    if not os.path.isfile(pid_path):
        return False, f"{name}: missing pid file"
    try:
        with open(pid_path, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False, f"{name}: invalid pid file"
    if not _pid_alive(pid):
        return False, f"{name}: pid {pid} not running"
    age = _heartbeat_age_sec(hb_path)
    if age is None:
        return False, f"{name}: missing/stale heartbeat"
    if age > max_age:
        return False, f"{name}: heartbeat {age:.0f}s old (max {max_age:.0f}s)"
    return True, f"{name}: pid {pid} ok (hb {age:.0f}s)"


def _test_flatline_import() -> tuple[bool, str]:
    try:
        from evolution.flatline_guard import assess_flatline, is_evolution_paused

        return True, "flatline_guard import ok"
    except Exception as exc:  # noqa: BLE001
        return False, f"flatline_guard import failed: {exc!r}"


def _run_powershell(script: str, *args: str) -> int:
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        os.path.join(SCRIPTS, script),
        *args,
    ]
    return subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode


def verify(repair: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "at": _utc_now(),
        "ok": True,
        "issues": [],
        "actions": [],
        "components": {},
    }

    tick_path = os.path.join(AGENTS, "tick_state.json")
    tick: dict[str, Any] = _read_json(tick_path) if os.path.isfile(tick_path) else {}

    if tick.get("loop_paused"):
        report["issues"].append("tick_state.loop_paused=true (user pause — wake ticks skipped)")

    component_specs = [
        ("wake_loop", os.path.join(AGENTS, "loop_wake.pid"), os.path.join(AGENTS, "loop_wake.heartbeat"), 50 * 60),
        (
            "evolution_daemon",
            os.path.join(AGENTS, "evolution_daemon.pid"),
            os.path.join(AGENTS, "evolution_daemon.heartbeat"),
            180,
        ),
        (
            "trajectory_watcher",
            os.path.join(AGENTS, "progress_trajectory_watcher.pid"),
            os.path.join(AGENTS, "progress_trajectory.heartbeat"),
            120,
        ),
    ]
    for name, pid_path, hb_path, _max_age in component_specs:
        if _repair_pid_from_heartbeat(pid_path, hb_path):
            report["actions"].append(f"repaired_pid:{name}")

    ok, msg = _check_component(
        "wake_loop",
        os.path.join(AGENTS, "loop_wake.pid"),
        os.path.join(AGENTS, "loop_wake.heartbeat"),
        max_age=50 * 60,
    )
    report["components"]["wake_loop"] = msg
    if not ok:
        report["ok"] = False
        report["issues"].append(msg)

    ok, msg = _check_component(
        "evolution_daemon",
        os.path.join(AGENTS, "evolution_daemon.pid"),
        os.path.join(AGENTS, "evolution_daemon.heartbeat"),
        max_age=180,  # must beat flatline poll (120s) + margin
    )
    report["components"]["evolution_daemon"] = msg
    if not ok:
        report["ok"] = False
        report["issues"].append(msg)

    ok, msg = _check_component(
        "trajectory_watcher",
        os.path.join(AGENTS, "progress_trajectory_watcher.pid"),
        os.path.join(AGENTS, "progress_trajectory.heartbeat"),
        max_age=120,
    )
    report["components"]["trajectory_watcher"] = msg
    if not ok:
        report["issues"].append(msg)  # non-fatal for agent loop

    ok, msg = _test_flatline_import()
    report["components"]["flatline_guard"] = msg
    if not ok:
        report["ok"] = False
        report["issues"].append(msg)

    if repair and not report["ok"]:
        if any("wake_loop" in i for i in report["issues"]):
            report["actions"].append("restart_wake_loop")
            _run_powershell("start_persistent_loop.ps1")
        if any("evolution_daemon" in i and "missing pid file" not in i for i in report["issues"]):
            report["actions"].append("restart_evolution_daemon")
            _run_powershell("restart_evolution_daemon.ps1")
        if any("trajectory_watcher" in i for i in report["issues"]):
            report["actions"].append("restart_trajectory_watcher")
            _run_powershell("start_persistent_loop.ps1")
        time.sleep(4)
        return verify(repair=False)

    # Persist health snapshot
    os.makedirs(AGENTS, exist_ok=True)
    with open(os.path.join(AGENTS, "loop_health.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if os.path.isfile(tick_path):
        tick["last_loop_health_at"] = report["at"]
        tick["last_loop_health_ok"] = report["ok"]
        if not tick.get("loop_paused"):
            tick["loop_wake_running"] = report["components"].get("wake_loop", "").startswith("wake_loop: pid")
        try:
            from evolution.epi_momentum_governor import assess_epi_momentum

            epi = assess_epi_momentum(PROJECT_ROOT)
            tick["epi_momentum_avg"] = epi.avg_momentum
            tick["epi_momentum_recent_100_avg"] = epi.recent_avg
            tick["epi_momentum_target"] = epi.target_avg
            tick["epi_momentum_window"] = epi.window
            tick["goal_met"] = epi.goal_met
            tick["needs_intervention"] = epi.needs_intervention
        except Exception:  # noqa: BLE001, S110 - best-effort tick bookkeeping, never fail health check
            pass
        with open(tick_path, "w", encoding="utf-8") as f:
            json.dump(tick, f, indent=2)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify/repair persistent loop")
    parser.add_argument("--repair", action="store_true", help="Restart failed components")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    args = parser.parse_args()

    report = verify(repair=args.repair)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"loop_health ok={report['ok']} at={report['at']}")
        for k, v in report["components"].items():
            print(f"  {k}: {v}")
        for issue in report["issues"]:
            print(f"  ISSUE: {issue}")
        for action in report.get("actions", []):
            print(f"  ACTION: {action}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
