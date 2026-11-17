"""List python/evolution processes: psutil if available, else PowerShell CIM via subprocess."""

from __future__ import annotations

import subprocess
import sys

hits = 0
try:
    import psutil  # type: ignore

    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            cl = p.info.get("cmdline") or []
        except Exception:
            continue
        if not cl:
            continue
        joined = " ".join(cl)
        low = joined.lower()
        if any(k in low for k in ("multi_cycle", "meta_controller", "evolution_engine", "continuous_evolution")):
            print(f"PID {p.info['pid']}: {joined[:300]}")
            hits += 1
    print(f"--- psutil matched {hits}")
except Exception as exc:  # noqa: BLE001
    print(f"psutil unavailable: {exc}")

if hits:
    sys.exit(0)

ps = (
    "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
    "ForEach-Object { $c = $_.CommandLine; if ($c -eq $null) { $c = '' }; "
    "if ($c.Length -gt 260) { $c = $c.Substring(0,260) }; "
    '"$($_.ProcessId)|$($_.CreationDate)|$c" }'
)
try:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        capture_output=True,
        text=True,
        timeout=120,
    )
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
    print(f"--- powershell rows: {len(lines)}")
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in ("multi_cycle", "meta_controller", "evolution", "continuous")):
            print("MATCH:", ln)
    if out.stderr.strip():
        print("STDERR:", out.stderr.strip()[:400])
except Exception as exc:  # noqa: BLE001
    print(f"powershell probe failed: {exc}")
