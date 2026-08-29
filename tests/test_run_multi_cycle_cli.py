#!/usr/bin/env python3
"""Regression test for run_multi_cycle CLI output visibility.

run_multi_cycle logged via _logger_rmc without calling setup_logging(), so all
its progress/FATAL messages were dropped (no root handler). Assert the input
validation now actually prints and exits non-zero.
"""
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_invalid_cycles_prints_fatal_and_exits_nonzero():
    env = dict(os.environ)
    env["EE_USE_GPU"] = "0"
    proc = subprocess.run(
        [sys.executable, os.path.join(_REPO, "run_multi_cycle.py"), "--cycles", "0"],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_REPO,
        env=env,
    )
    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "FATAL" in combined and "--cycles" in combined
