#!/usr/bin/env python3
"""Check for metric flatline and halt evolution daemon when progress has stopped."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evolution.flatline_guard import (
    apply_flatline_halt,
    assess_flatline,
    clear_flatline_halt,
    is_flatline_halted,
)


def _stop_daemon() -> None:
    pid_path = os.path.join(PROJECT_ROOT, ".agents", "evolution_daemon.pid")
    if not os.path.isfile(pid_path):
        return
    try:
        with open(pid_path, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return
    import subprocess

    subprocess.run(
        ["taskkill", "/PID", str(pid), "/F"],
        capture_output=True,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Flatline detection and halt")
    parser.add_argument("--check", action="store_true", help="Check only; exit 2 if flatline")
    parser.add_argument("--halt", action="store_true", help="Halt daemon if flatline detected")
    parser.add_argument("--resume", action="store_true", help="Clear flatline halt flag")
    parser.add_argument("--force-halt", action="store_true", help="Halt immediately (user-directed)")
    args = parser.parse_args()

    if args.resume:
        clear_flatline_halt(PROJECT_ROOT)
        print("Flatline halt cleared. Restart daemon manually when ready.")
        return 0

    report = assess_flatline(PROJECT_ROOT)
    print(json.dumps(report.to_dict(), indent=2))

    if args.force_halt:
        report.halted = True
        if not report.reasons:
            report.reasons.append("user-directed force halt")

    if report.halted and (args.halt or args.force_halt):
        apply_flatline_halt(PROJECT_ROOT, report)
        print("FLATLINE: evolution_paused=true (agent wake loop continues)")
        print("Development required — see .agents/flatline_state.json")
        print("Resume: python scripts/flatline_guard.py --resume")
        return 2

    if report.halted and args.check:
        return 2

    if is_flatline_halted(PROJECT_ROOT):
        print("Flatline halt already active")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
