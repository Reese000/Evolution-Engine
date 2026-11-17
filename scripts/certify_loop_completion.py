#!/usr/bin/env python3
"""Verify all COMPLETION_CRITERIA.md gates. Exit 0 only when fully certified."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
AGENTS = os.path.join(PROJECT_ROOT, ".agents")
RESULTS = os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl")
SUMMARY = os.path.join(PROJECT_ROOT, "multi_cycle_summary.json")
MIN_CYCLE_SECONDS = float(os.environ.get("EE_MIN_CYCLE_SECONDS", "30"))
GENS = int(os.environ.get("EE_DAEMON_GENS", "50"))


def _cycle_verified(row: dict) -> bool:
    """Champion quality gate — uses same curriculum verify threshold as advancement."""
    if row.get("champion_verified"):
        return True
    score = row.get("champion_verify_score", 0)
    level = row.get("final_level", 0)
    from evolution.progress_controller import ProgressController

    return ProgressController.passes_certify_verify(int(level), float(score))


def _record_trajectory() -> None:
    script = os.path.join(PROJECT_ROOT, "scripts", "progress_trajectory.py")
    try:
        subprocess.run(
            [sys.executable, script, "--record"],
            cwd=PROJECT_ROOT,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  (trajectory record skipped: {exc})")


def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _load_results() -> list[dict]:
    if not os.path.isfile(RESULTS):
        return []
    rows: list[dict] = []
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _valid_results(rows: list[dict]) -> list[dict]:
    out = []
    min_gens = min(10, max(1, GENS - 1))
    for r in rows:
        if "error" in r:
            continue
        elapsed = r.get("elapsed_seconds", r.get("phase_total_seconds", 0))
        gens = r.get("generations_completed", 0)
        if isinstance(gens, (int, float)) and gens >= min_gens:
            out.append(r)
            continue
        if isinstance(elapsed, (int, float)) and elapsed >= MIN_CYCLE_SECONDS:
            out.append(r)
    return out


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _heartbeat_fresh(path: str, max_age_sec: float) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        data = _read_json(path)
        at = data.get("at", "")
        if not at:
            return False
        ts = datetime.fromisoformat(at.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return age <= max_age_sec
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _check_pid_file(path: str, max_hb_age: float) -> tuple[bool, str]:
    hb = path.replace(".pid", ".heartbeat")
    if path.endswith("loop_wake.pid"):
        hb = os.path.join(AGENTS, "loop_wake.heartbeat")
    if not os.path.isfile(path):
        return False, f"missing {os.path.basename(path)}"
    try:
        with open(path, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return False, "invalid pid file"
    if not _pid_alive(pid):
        return False, f"pid {pid} not running"
    if not _heartbeat_fresh(hb, max_hb_age):
        return False, f"stale heartbeat ({os.path.basename(hb)})"
    return True, f"pid {pid} ok"


def main() -> int:
    failures: list[str] = []
    tick_path = os.path.join(AGENTS, "tick_state.json")
    tick = _read_json(tick_path) if os.path.isfile(tick_path) else {}

    if tick.get("loop_paused"):
        failures.append("A2: loop_paused is true")

    ok, msg = _check_pid_file(os.path.join(AGENTS, "loop_wake.pid"), 45 * 60)
    if not ok:
        failures.append(f"A1: wake loop - {msg}")

    ok, msg = _check_pid_file(os.path.join(AGENTS, "evolution_daemon.pid"), 120 * 60)
    if not ok:
        failures.append(f"A3: evolution daemon - {msg}")

    if not tick.get("loop_wake_monitored"):
        failures.append("A4: loop_wake_monitored false")

    valid = _valid_results(_load_results())
    if len(valid) < 20:
        failures.append(f"B1: need >=20 valid results, have {len(valid)}")

    retrained = [r for r in valid if r.get("retrained") and r.get("val_loss") is not None]
    if len(retrained) >= 10:
        last10 = [float(r["val_loss"]) for r in retrained[-10:]]
        if last10[-1] > last10[0] + 0.01:
            failures.append(f"B2: val_loss worsening (first={last10[0]:.4f} last={last10[-1]:.4f})")
    else:
        failures.append(f"B2: need 10 retrained cycles with val_loss, have {len(retrained)}")

    if os.path.isfile(SUMMARY):
        summary = _read_json(SUMMARY)
        recent5 = [r for r in _load_results() if "error" not in r][-5:]
        gains = []
        for i, row in enumerate(recent5):
            fg = row.get("fitness_gain", 0)
            if isinstance(fg, (int, float)) and fg > 0:
                gains.append(1)
            elif i > 0:
                prev_f = recent5[i - 1].get("final_fitness", 0)
                cur_f = row.get("final_fitness", 0)
                if isinstance(prev_f, (int, float)) and isinstance(cur_f, (int, float)) and cur_f >= prev_f:
                    gains.append(1)
        if not gains or sum(gains) < max(1, len(recent5) // 2):
            failures.append("B3: fitness not improving on majority of last 5 cycles")
        strat_src = summary if len(summary) >= 3 else valid
        recent3 = [r for r in _load_results() if "error" not in r][-3:]
        if len(recent3) >= 3:
            strat = [m.get("strategy_archive_records", 0) for m in recent3]
        else:
            strat = [m.get("strategy_archive_records", 0) for m in strat_src[-3:]]
        at_cap = len(strat) >= 3 and strat[-1] >= 10000
        growing = len(strat) >= 3 and strat[-1] > strat[0]
        active = len(strat) >= 3 and max(strat) >= 5000
        if len(strat) < 3 or strat[-1] < 500 or (not growing and not at_cap and not active):
            failures.append(f"B4: strategy archive not healthy (last3={strat})")
        last5_gpu = summary[-5:] if len(summary) >= 5 else valid[-5:]
        for m in last5_gpu:
            if m.get("vm_gpu_batches", 0) <= 0 or m.get("vm_cpu_fallbacks", 0) != 0:
                failures.append("B5: GPU batch eval not healthy in last 5 cycles")
                break
    else:
        failures.append("B3-B5: missing multi_cycle_summary.json")

    last10 = valid[-10:]
    completed_ok = sum(1 for r in last10 if r.get("generations_completed", 0) >= GENS - 1)
    if len(last10) < 10 or completed_ok < 7:
        failures.append(f"B6: generations not completing ({completed_ok}/10)")

    if valid:
        last_level = valid[-1].get("final_level", 0)
        levels = [r.get("final_level", 0) for r in valid[-20:]]
        if last_level < 4 and levels == sorted(levels):
            failures.append(f"B7: curriculum stagnant at L{last_level}")

    last_pytest = tick.get("last_full_pytest_at")
    if not last_pytest:
        failures.append("C1: no last_full_pytest_at in tick_state")
    else:
        try:
            ts = datetime.fromisoformat(last_pytest.replace("Z", "+00:00"))
            if (datetime.now(timezone.utc) - ts).total_seconds() > 48 * 3600:
                failures.append("C1: full pytest older than 48h")
        except ValueError:
            failures.append("C1: invalid last_full_pytest_at")

    hb_path = os.path.join(AGENTS, "evolution_daemon.heartbeat")
    if os.path.isfile(hb_path):
        hb = _read_json(hb_path)
        if hb.get("status") == "error" and hb.get("detail"):
            failures.append(f"C2: daemon last status error ({hb.get('detail')[:80]})")

    last10v = [r for r in _load_results() if "error" not in r][-10:]
    verified = sum(1 for r in last10v if _cycle_verified(r))
    if len(last10v) < 10 or verified < 8:
        failures.append(f"C3: champion verify rate {verified}/10 (full or partial-by-level)")

    if failures:
        print("NOT CERTIFIED - failures:")
        for f in failures:
            print(f"  - {f}")
        _record_trajectory()
        return 1

    cert = {
        "certified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "valid_results": len(valid),
        "criteria": "COMPLETION_CRITERIA.md all gates",
    }
    out = os.path.join(AGENTS, "completion_certified.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cert, f, indent=2)
    print("CERTIFIED - all completion criteria satisfied.")
    print(json.dumps(cert, indent=2))
    _record_trajectory()
    return 0


if __name__ == "__main__":
    sys.exit(main())
