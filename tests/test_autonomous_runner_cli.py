#!/usr/bin/env python3
"""Regression tests for autonomous_runner CLI output.

--dry-run and --status are documented to show the validated config and health
status, but produced NO output because the runner used _logger.info(...) without
ever calling setup_logging() (root logger had no handler, so INFO was dropped).
These assert the documented commands actually print something.
"""
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args):
    env = dict(os.environ)
    env["EE_USE_GPU"] = "0"
    return subprocess.run(
        [sys.executable, os.path.join(_REPO, "autonomous_runner.py"), *args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=_REPO,
        env=env,
    )


def test_status_produces_output():
    proc = _run(["--status"])
    assert proc.returncode == 0, proc.stderr[-500:]
    combined = proc.stdout + proc.stderr
    assert "Health Status" in combined or "No health.json" in combined


def test_dry_run_produces_output():
    proc = _run(["--dry-run"])
    assert proc.returncode == 0, proc.stderr[-500:]
    combined = proc.stdout + proc.stderr
    assert "Config validated OK" in combined
