"""List python.exe processes with command lines (psutil if present)."""
from __future__ import annotations

import sys

try:
    import psutil
except ImportError:
    print("psutil NOT available in", sys.executable)
    raise SystemExit(0)

me = psutil.Process()
for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
    try:
        if (p.info["name"] or "").lower() not in {"python.exe", "pythonw.exe", "powershell.exe"}:
            continue
        cmd = " ".join(p.info["cmdline"] or [])
        if not cmd:
            continue
        print(p.info["pid"], "|", cmd[:220])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue
