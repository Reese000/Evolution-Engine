"""Report interpreter + torch availability (no -c flags)."""
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
        print("torch import failed:", repr(exc))
