"""Report interpreter + torch availability for an arbitrary argv[1] interpreter is not
possible from inside the interpreter; instead this script is executed BY the target
interpreter, so it reports whichever python runs it."""
from __future__ import annotations

import importlib.util
import sys

print("exe:", sys.executable)
print("version:", sys.version.replace("\n", " "))
for mod in ("torch", "numpy", "pytest", "hypothesis", "psutil"):
    s = importlib.util.find_spec(mod)
    print(f"  {mod}: {'ok ' + str(getattr(__import__(mod), '__version__', '')) if s else 'MISSING'}"
          if s and mod != "torch" else f"  {mod}: {'present' if s else 'MISSING'}")
try:
    import torch

    print("torch version:", torch.__version__)
except Exception as exc:  # noqa: BLE001
    print("torch import FAILED:", type(exc).__name__, exc)
