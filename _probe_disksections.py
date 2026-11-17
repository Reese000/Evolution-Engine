"""Report section names / sizes of the VM score disk cache (read-only)."""

from __future__ import annotations

import json
import os

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agents", "vm_score_disk_cache.json")
data = json.load(open(p, encoding="utf-8"))
print("size:", os.path.getsize(p))
for sec, bucket in data.items():
    print(f"  {sec}: {len(bucket)} entries")
