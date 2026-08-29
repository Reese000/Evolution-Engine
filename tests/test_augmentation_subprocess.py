#!/usr/bin/env python3
"""Regression test for augmentation_subprocess.py's stdout JSON contract.

milestone1/training/loop.py runs this script as a subprocess and parses its
ENTIRE stdout with json.loads. A bug let data.augmentation "[Train] ..." prints
leak to stdout, corrupting that JSON so loop.py's json.loads raised and it
silently fell back to in-process augmentation every time (defeating R51's
CUDA-crash isolation and doing the work twice). stdout must be pure JSON.
"""
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_stdout_is_pure_json():
    script = os.path.join(_REPO, "augmentation_subprocess.py")
    env = dict(os.environ)
    env["EE_USE_GPU"] = "0"
    proc = subprocess.run(
        [sys.executable, script],
        input=json.dumps({"seed": 42, "max_sequences": 2, "data_path": ""}),
        capture_output=True,
        text=True,
        timeout=180,
        cwd=_REPO,
        env=env,
    )
    assert proc.returncode == 0, f"nonzero exit; stderr tail:\n{proc.stderr[-500:]}"
    # The whole stdout must parse as JSON (exactly what loop.py does).
    data = json.loads(proc.stdout)
    assert "sequences" in data and "count" in data
    assert data["count"] == len(data["sequences"])
    # Progress/log lines must go to stderr, never stdout.
    assert "[Train]" not in proc.stdout
