"""Read-only: list processes whose CommandLine matches patterns (no killing)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

PAT = sys.argv[1] if len(sys.argv) > 1 else "multi_cycle|meta_controller|evolution|run_multi"
HERE = os.path.dirname(os.path.abspath(__file__))

out = subprocess.run(
    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(HERE, "_probe_procs5.ps1")],
    capture_output=True,
    text=True,
    timeout=180,
)
if out.returncode != 0:
    print("ERR", out.returncode, out.stderr[:2000])
    sys.exit(1)

raw = out.stdout.strip()
if not raw:
    print("EMPTY stdout; stderr:", out.stderr[:500])
    sys.exit(1)

try:
    rows = json.loads(raw)
except json.JSONDecodeError:
    print("parse fail:", raw[:800])
    sys.exit(1)

if isinstance(rows, dict):
    rows = [rows]

rx = re.compile(PAT, re.I)
for r in rows:
    cl = r.get("cl") or ""
    if rx.search(cl):
        print(f"{r['pid']:>8} ppid={r['ppid']:<8} {cl[:300]}")
