"""Print command lines of all python processes (psutil if present)."""
from __future__ import annotations

import sys

try:
    import psutil  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"no psutil: {exc}")
    sys.exit(0)

hits = 0
for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
    try:
        cl = p.info.get("cmdline") or []
    except Exception:
        continue
    if not cl:
        continue
    joined = " ".join(cl)
    if any(k in joined.lower() for k in ("multi_cycle", "meta_controller", "evolution")):
        print(f"PID {p.info['pid']}: {joined}")
        hits += 1
print(f"--- matched {hits}")


def list_py() -> None:
    seen = set()
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cl = p.info.get("cmdline") or []
        except Exception:
            continue
        if cl and "python" in (p.info.get("name") or "").lower():
            j = " ".join(cl)
            if j not in seen:
                seen.add(j)
    for j in sorted(seen):
        print("PY:", j[:400])


list_py()
