#!/usr/bin/env python3
"""Deploy certified restore: test, queue rollback, refresh carry-forward, restart."""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd: list[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})


def main() -> int:
    steps = [
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_certified_metrics.py",
            "tests/unit/test_champion_rollback.py",
            "tests/unit/test_finalize_run_summary.py",
            "-q",
        ],
        [sys.executable, "scripts/smoke_restore.py"],
        [sys.executable, "scripts/backfill_l30_certified_metrics.py"],
        [sys.executable, "scripts/queue_c3511_rollback.py"],
        [sys.executable, "scripts/refresh_l30_carry_forward.py"],
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", "scripts/restart_evolution_daemon.ps1"],
        [sys.executable, "scripts/verify_epi_momentum_goal.py", "--json"],
    ]
    for cmd in steps:
        if run(cmd) != 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
