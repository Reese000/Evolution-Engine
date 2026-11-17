"""List top-level/method defs in evolution_engine.py within an optional line range."""
from __future__ import annotations

import re
import sys

lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
hi = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
pat = re.compile(r"^(\s*)(def |class )")
with open("evolution_engine.py", encoding="utf-8", errors="replace") as fh:
    for i, line in enumerate(fh, 1):
        m = pat.match(line)
        if not m or not (lo <= i <= hi):
            continue
        indent = len(m.group(1))
        if indent > 8:
            continue
        print(f"{i:6d} {'  ' * (indent // 4)}{line.rstrip()[:140]}")
