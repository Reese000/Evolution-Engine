#!/usr/bin/env python3
"""Regression test for status.sh --json output.

`./status.sh --json` is documented to emit JSON status. It had two bugs: it
printed a decorative header before the JSON (unparseable), and its embedded
Python used the shell's lowercase `false` (invalid Python -> NameError). This
asserts the --json output is now pure, parseable JSON.
"""
import json
import os
import shutil
import subprocess

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_status_json_is_parseable():
    proc = subprocess.run(
        ["bash", os.path.join(_REPO, "status.sh"), "--json"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_REPO,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr[-500:]}"
    data = json.loads(proc.stdout)  # must be pure JSON
    assert {"process_running", "pid", "health"}.issubset(data)
    assert isinstance(data["process_running"], bool)
