"""Inspect daemon log tail + health/progress snapshots (read-only)."""

from __future__ import annotations

import json
import os

os.chdir(r"C:/Users/reese/Projects/Evolution Engine")

print("=== health.json ===")
try:
    print(open("health.json", encoding="utf-8", errors="replace").read()[:2000])
except Exception as exc:
    print("err", exc)

print("=== progress_report_extended.json (top keys + selected) ===")
try:
    d = json.load(open("progress_report_extended.json", encoding="utf-8", errors="replace"))
    print("keys:", sorted(d.keys())[:60])
    for k in ("updated_at", "generated_at", "last_cycle", "cycle", "daemon", "status", "elapsed_seconds"):
        if k in d:
            print(k, "=", json.dumps(d[k], default=str)[:500])
except Exception as exc:
    print("err", exc)

print("=== meta_controller.log tail ===")
try:
    with open("meta_controller.log", "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - 6000))
        tail = fh.read().decode("utf-8", errors="replace")
    print(tail[-5000:])
except Exception as exc:
    print("err", exc)
