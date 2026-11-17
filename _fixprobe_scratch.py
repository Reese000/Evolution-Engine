"""Scratch probe: validate the review's prescribed fix for test :1252 without
touching the reviewed test file. Run from repo root with the venv python.

Steps:
  1. Copy tests/unit/test_scan_floor_throughput.py -> test_scan_floor_fixprobe.py
  2. Apply the 3-component fix to the copy (exact-match, fail loudly if absent):
     a) monkeypatch.setenv("EE_EQ_REFINE_WARM_SKIP_PERTURB", "0")
     b) _eq_refine_late_merged_cache.clear() + patch load_eq_refine_late_merged_cache -> None
     c) monkeypatch.setenv("EE_EQ_REFINE_LATE_MERGED_DISK", "0")  (stop repo .agents write)
  3. Run the FULL copied file: expect 90 passed / 0 failed.
  4. Check .agents/scan_floor_eq_refine_late_merged_cache.json: after a clean-slate
     run the last writer among :605/:1125/:1203 is test :1203 (cycle 1); a cycle==7
     entry would mean test :1252 still writes to the repo (fix incomplete).
  5. Re-run the ORIGINAL full file: expect the same 1 failed / 89 passed (determinism).
  6. Restore the .agents cache file to its pre-run bytes; delete the probe copy.
"""
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "tests" / "unit" / "test_scan_floor_throughput.py"
DST = ROOT / "tests" / "unit" / "test_scan_floor_fixprobe.py"
AGENTS_CACHE = ROOT / ".agents" / "scan_floor_eq_refine_late_merged_cache.json"

results = {}


def run_pytest(target: str) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--color=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def summary_line(out: str) -> str:
    for line in reversed(out.splitlines()):
        line = line.strip("= \t\r\n")
        if "passed" in line or "failed" in line or "error" in line:
            return line
    return "<no summary line>"


# --- 1. copy -------------------------------------------------------------
text = SRC.read_text(encoding="utf-8")

# --- 2. apply fix (exact-match with counts) ------------------------------
OLD = '    monkeypatch.setenv("EE_EQ_REFINE_WARM_SKIP_LOCUS", "0")'
count = text.count(OLD)
assert count == 1, f"expected exactly 1 WARM_SKIP_LOCUS setenv, found {count}"
NEW = OLD + """
    # FIXPROBE copy of the review-prescribed fix: the warm-skip perturb guard
    # (default "1") also zeroes perturb_budget, and the module-level merged
    # cache is pre-occupied by an earlier test under the same plateau key.
    monkeypatch.setenv("EE_EQ_REFINE_WARM_SKIP_PERTURB", "0")
    monkeypatch.setenv("EE_EQ_REFINE_LATE_MERGED_DISK", "0")
    from evolution.mutation_helpers import _eq_refine_late_merged_cache

    _eq_refine_late_merged_cache.clear()
    monkeypatch.setattr(
        "evolution.mutation_helpers.load_eq_refine_late_merged_cache",
        lambda *_a, **_k: None,
    )"""
text = text.replace(OLD, NEW)

assert text != SRC.read_text(encoding="utf-8"), "fix did not change the source"
DST.write_text(text, encoding="utf-8")

# --- 3. clean-slate .agents state so the cycle check is unambiguous ------
backup = None
if AGENTS_CACHE.exists():
    backup = AGENTS_CACHE.read_bytes()
    AGENTS_CACHE.unlink()

try:
    rc_fix, out_fix = run_pytest("tests/unit/test_scan_floor_fixprobe.py")
    results["fixprobe"] = (rc_fix, summary_line(out_fix))

    # --- 4. who wrote the .agents late-merged cache during the fixprobe? --
    if AGENTS_CACHE.exists():
        payload = json.loads(AGENTS_CACHE.read_text(encoding="utf-8"))
        results["agents_cache_after_fixprobe"] = {
            "cycle": payload.get("cycle"),
            "late_cache_key": payload.get("late_cache_key"),
            "verdict": (
                "test :1252 STILL writes repo state (cycle 7)"
                if payload.get("cycle") == 7
                else "test :1252 did NOT write (last writer is an earlier test; hermetic fix holds)"
            ),
        }
    else:
        results["agents_cache_after_fixprobe"] = {
            "verdict": "no file created — no test wrote repo state"
        }

    # --- 5. determinism re-run of the ORIGINAL file -----------------------
    rc_orig, out_orig = run_pytest("tests/unit/test_scan_floor_throughput.py")
    results["original_rerun"] = (rc_orig, summary_line(out_orig))
finally:
    # --- 6. cleanup + restore --------------------------------------------
    if DST.exists():
        DST.unlink()
    pyc = ROOT / "tests" / "unit" / "__pycache__"
    if pyc.is_dir():
        for f in pyc.glob("test_scan_floor_fixprobe*"):
            f.unlink()
    if backup is not None:
        AGENTS_CACHE.write_bytes(backup)

print(json.dumps(
    {k: (list(v) if isinstance(v, tuple) else v) for k, v in results.items()},
    indent=2,
))
# Exit non-zero unless BOTH runs matched expectations.
ok = (
    results["fixprobe"][1].startswith("90 passed")
    and results["original_rerun"][1].startswith("1 failed, 89 passed")
    and "cycle 7" not in results["agents_cache_after_fixprobe"]["verdict"]
)
print("PROBE_VERDICT:", "ALL EXPECTATIONS MET" if ok else "UNEXPECTED RESULT")
sys.exit(0 if ok else 1)
