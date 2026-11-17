"""Report interpreter + torch availability (read-only)."""

from __future__ import annotations

import importlib.util
import sys

print("exe:", sys.executable)
print("version:", sys.version.replace("\n", " "))
spec = importlib.util.find_spec("torch")
print("torch spec:", spec.origin if spec else "NOT FOUND")
if spec:
    try:
        import torch

        print("torch version:", torch.__version__)
    except Exception as exc:  # noqa: BLE001
        print("torch import FAILED:", type(exc).__name__, exc)
for mod in ("pytest", "numpy", "hypothesis"):
    s = importlib.util.find_spec(mod)
    print(mod, "->", "ok" if s else "MISSING")
