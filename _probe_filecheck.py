"""Check line endings + syntax of a source file (read-only)."""

from __future__ import annotations

import py_compile
import sys
import tempfile

paths = sys.argv[1:] or ["evolution_engine.py"]
for p in paths:
    data = open(p, "rb").read()
    crlf = data.count(b"\r\n")
    lf_only = data.count(b"\n") - crlf
    cr_only = data.count(b"\r") - crlf
    print(f"{p}: bytes={len(data)} crlf={crlf} lf_only={lf_only} cr_only={cr_only}")
    try:
        compile(data.decode("utf-8"), p, "exec")
        print("   syntax: OK")
    except SyntaxError as exc:
        print(f"   syntax: FAIL {exc}")

# show whether a specific region is double spaced
src = open(paths[0], encoding="utf-8", errors="replace").read()
idx = src.find("for _ in range(pop_size):")
if idx != -1:
    print("region repr:", repr(src[idx : idx + 180]))
