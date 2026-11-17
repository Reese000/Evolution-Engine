"""Enumerate processes whose command line mentions Evolution Engine."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt

TH32CS_SNAPPROCESS = 0x00000002
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wt.DWORD),
        ("cntUsage", wt.DWORD),
        ("th32ProcessID", wt.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wt.DWORD),
        ("cntThreads", wt.DWORD),
        ("th32ParentProcessID", wt.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wt.DWORD),
        ("szExeFile", ctypes.c_char * 260),
    ]


def cmdline(pid: int) -> str:
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        n = wt.DWORD(0)
        buf = ctypes.create_unicode_buffer(32768)
        sz = wt.DWORD(32768)
        ok = ctypes.windll.kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(sz))
        img = buf.value if ok else ""
    finally:
        k32.CloseHandle(h)
    return img


def main() -> None:
    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE:
        print("snapshot failed")
        return
    entry = PROCESSENTRY32()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
    rows = []
    if k32.Process32First(snap, ctypes.byref(entry)):
        while True:
            pid = entry.th32ProcessID
            exe = entry.szExeFile.decode("mbcs", "replace")
            rows.append((pid, exe))
            if not k32.Process32Next(snap, ctypes.byref(entry)):
                break
    k32.CloseHandle(snap)

    for pid, exe in rows:
        if "python" not in exe.lower():
            continue
        img = cmdline(pid)
        if not img:
            continue
        low = img.lower()
        if "evolution" in low or "multi_cycle" in low or "meta_controller" in low or "run_multi" in low:
            print(f"PID {pid}: {img}")
    print(f"--- total processes scanned: {len(rows)}")


if __name__ == "__main__":
    main()
