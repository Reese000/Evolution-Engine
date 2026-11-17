"""Report physical memory pressure (read-only)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


st = MEMORYSTATUSEX()
st.dwLength = ctypes.sizeof(st)
ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
print(f"memory load: {st.dwMemoryLoad}%")
print(f"physical: {st.ullAvailPhys / 1e9:.1f} GB free of {st.ullTotalPhys / 1e9:.1f} GB")
