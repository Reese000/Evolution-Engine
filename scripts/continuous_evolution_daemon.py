#!/usr/bin/env python3
"""Run multi-cycle evolution indefinitely with resume and crash recovery.

Writes heartbeat to .agents/evolution_daemon.heartbeat for certification.
Never exits except on unhandled fatal error (logged); outer loop restarts batches.
On flatline: logs advisory only — evolution batches never pause.
"""

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
HEARTBEAT = os.path.join(AGENTS, "evolution_daemon.heartbeat")
PID_FILE = os.path.join(AGENTS, "evolution_daemon.pid")
LOG_FILE = os.path.join(AGENTS, "evolution_daemon.log")
BATCH_CYCLES = int(os.environ.get("EE_DAEMON_BATCH_CYCLES", "100"))
GENS = int(os.environ.get("EE_DAEMON_GENS", "50"))
SLEEP_BETWEEN_BATCHES = int(os.environ.get("EE_DAEMON_SLEEP_SEC", "15"))
FLATLINE_POLL_SEC = int(os.environ.get("EE_FLATLINE_POLL_SEC", "120"))
BATCH_LOG_KEEP = int(os.environ.get("EE_BATCH_LOG_KEEP", "30"))
ARCHIVE_SIDECAR_KEEP = int(os.environ.get("EE_ARCHIVE_SIDECAR_KEEP", "10"))
RESULTS_JSONL = os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl")
# Per-cycle budget for extended L30 paths (~450s observed); auto-scale with batch size.
_CYCLE_BUDGET_SEC = int(os.environ.get("EE_CYCLE_MAX_SEC", "600"))
_BATCH_BUFFER_SEC = int(os.environ.get("EE_BATCH_BUFFER_SEC", "1800"))
# Only kill when jsonl has not advanced for this long (progress-aware hang detection).
_JSONL_STALE_SEC = int(os.environ.get("EE_BATCH_JSONL_STALE_SEC", "900"))


def compute_batch_max_sec(
    batch_cycles: int = BATCH_CYCLES,
    cycle_budget_sec: int = _CYCLE_BUDGET_SEC,
    buffer_sec: int = _BATCH_BUFFER_SEC,
) -> int:
    """Wall-clock budget for a full daemon batch (100 cycles @ 600s ≈ 17h)."""
    return max(3600, batch_cycles * cycle_budget_sec + buffer_sec)


def _resolve_batch_max_sec() -> int:
    raw = os.environ.get("EE_BATCH_MAX_SEC", "").strip()
    if raw and raw != "0":
        return int(raw)
    return compute_batch_max_sec()


BATCH_MAX_SEC = _resolve_batch_max_sec()


def _jsonl_age_sec() -> float | None:
    try:
        return time.time() - os.path.getmtime(RESULTS_JSONL)
    except OSError:
        return None


def should_terminate_batch(
    elapsed_sec: float,
    jsonl_age_sec: float | None,
    batch_max_sec: int = BATCH_MAX_SEC,
    stale_sec: int = _JSONL_STALE_SEC,
) -> bool:
    """Kill only when over budget AND results jsonl is stale (cycles not landing)."""
    if elapsed_sec <= batch_max_sec:
        return False
    if jsonl_age_sec is None:
        return True
    return jsonl_age_sec > stale_sec


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}\n"
    os.makedirs(AGENTS, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.rstrip(), flush=True)


def _heartbeat(status: str, detail: str = "") -> None:
    os.makedirs(AGENTS, exist_ok=True)
    _write_pid()
    with open(HEARTBEAT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "status": status,
                "detail": detail,
                "batch_cycles": BATCH_CYCLES,
                "gens": GENS,
                "pid": os.getpid(),
            },
            f,
            indent=2,
        )


def _write_pid() -> None:
    os.makedirs(AGENTS, exist_ok=True)
    with open(PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def _prune_artifacts() -> dict[str, int]:
    """Drop stale batch logs and archive sidecars to avoid EMFILE on long runs."""
    try:
        from scripts.prune_agent_artifacts import prune_all

        counts = prune_all(
            AGENTS,
            batch_keep=BATCH_LOG_KEEP,
            archive_keep=ARCHIVE_SIDECAR_KEEP,
        )
        if counts["batch_logs"] or counts["archive_sidecars"]:
            _log(
                "Pruned artifacts: "
                f"batch_logs={counts['batch_logs']} (keep={BATCH_LOG_KEEP}), "
                f"archive_sidecars={counts['archive_sidecars']} (keep={ARCHIVE_SIDECAR_KEEP})"
            )
        return counts
    except Exception as exc:  # noqa: BLE001
        _log(f"Artifact prune failed: {exc!r}")
        return {"batch_logs": 0, "archive_sidecars": 0}


def _run_batch() -> int:
    cmd = [
        sys.executable,
        "-u",
        os.path.join(PROJECT_ROOT, "run_multi_cycle.py"),
        "--cycles",
        str(BATCH_CYCLES),
        "--gens",
        str(GENS),
        "--resume",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    _log(f"Starting batch: {' '.join(cmd)}")
    _heartbeat("running_batch", " ".join(cmd[-4:]))
    batch_log = os.path.join(AGENTS, f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log")
    hb_interval = int(os.environ.get("EE_DAEMON_HEARTBEAT_SEC", "600"))
    with open(batch_log, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        _log(f"Batch subprocess pid={proc.pid} log={batch_log}")
        batch_started = time.monotonic()
        while proc.poll() is None:
            time.sleep(min(hb_interval, 60))
            elapsed = time.monotonic() - batch_started
            jsonl_age = _jsonl_age_sec()
            if should_terminate_batch(elapsed, jsonl_age):
                stale_note = f"jsonl_age={jsonl_age:.0f}s" if jsonl_age is not None else "jsonl missing"
                _log(
                    f"Batch hang detected: pid={proc.pid} elapsed={elapsed:.0f}s "
                    f"> max={BATCH_MAX_SEC}s and {stale_note} > {_JSONL_STALE_SEC}s — terminating"
                )
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                _heartbeat("batch_hang_killed", f"elapsed={elapsed:.0f}s")
                return 124
            if proc.poll() is None:
                _heartbeat("running_batch", " ".join(cmd[-4:]))
        proc.wait()
    returncode = proc.returncode
    _log(f"Batch finished exit_code={returncode} log={batch_log}")
    if returncode != 0:
        try:
            with open(batch_log, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-4000:]
            _log(f"Batch tail: {tail[-1200:]}")
        except OSError:
            pass
    return returncode


def _maybe_apply_flatline_halt() -> bool:
    """Detect flatline and record halt state. Returns True when the daemon
    must NOT spawn a new batch (flatline halted OR user paused)."""
    if os.environ.get("EE_FLATLINE_GUARD", "1") != "1":
        return False
    try:
        from evolution.flatline_guard import apply_flatline_halt, assess_flatline, is_evolution_paused

        report = None
        try:
            from concurrent.futures import ThreadPoolExecutor
            from concurrent.futures import TimeoutError as FuturesTimeout

            with ThreadPoolExecutor(max_workers=1) as pool:
                report = pool.submit(assess_flatline, PROJECT_ROOT).result(timeout=8)
        except FuturesTimeout:
            _log("Flatline assess timed out (8s); starting batch anyway")
        if report is not None and report.halted:
            apply_flatline_halt(PROJECT_ROOT, report)
            _log(f"FLATLINE advisory: {'; '.join(report.reasons[:2])}")
        return is_evolution_paused(PROJECT_ROOT)
    except Exception as exc:  # noqa: BLE001
        _log(f"Flatline guard check failed: {exc!r}")
        return False


def _wait_while_evolution_paused_or_halted() -> None:
    """Block until evolution may resume — daemon process never exits."""
    from evolution.flatline_guard import is_evolution_paused, is_flatline_halted

    poll = 0
    while is_flatline_halted(PROJECT_ROOT) or is_evolution_paused(PROJECT_ROOT):
        _heartbeat("evolution_paused", "flatline halt or user pause active")
        poll += 1
        if poll % 5 == 0:
            _log(f"Evolution halted (poll {poll}) — waiting for resume (.agents/flatline_state.json advice)")
        time.sleep(FLATLINE_POLL_SEC)


def _startup_integrity() -> None:
    """Dedupe cycle results and sync trajectory before the first batch."""
    _prune_artifacts()
    try:
        from evolution.loop_integrity import ensure_metrics_integrity

        ensure_metrics_integrity(PROJECT_ROOT, log=_log)
    except Exception as exc:  # noqa: BLE001
        _log(f"Startup integrity check failed: {exc!r}")


def main() -> None:
    _write_pid()
    _log(
        f"Continuous evolution daemon started pid={os.getpid()} (never exits) "
        f"batch_max_sec={BATCH_MAX_SEC} cycles={BATCH_CYCLES}"
    )
    _heartbeat("started", f"batch={BATCH_CYCLES} gens={GENS} max_sec={BATCH_MAX_SEC}")
    _startup_integrity()
    batch_num = 0
    while True:
        if _maybe_apply_flatline_halt():
            _wait_while_evolution_paused_or_halted()
        batch_num += 1
        try:
            code = _run_batch()
            _heartbeat("idle_between_batches", f"last_exit={code} batch={batch_num}")
            _maybe_apply_flatline_halt()
            if code == 0:
                _prune_artifacts()
                try:
                    from evolution.loop_integrity import ensure_metrics_integrity

                    ensure_metrics_integrity(PROJECT_ROOT, log=_log)
                except Exception as exc:  # noqa: BLE001
                    _log(f"Post-batch integrity check failed: {exc!r}")
            if code != 0:
                _log(f"Batch {batch_num} failed (exit {code}); sleeping before retry")
                time.sleep(60)
            else:
                time.sleep(SLEEP_BETWEEN_BATCHES)
        except KeyboardInterrupt:
            _log("KeyboardInterrupt — daemon stopping")
            _heartbeat("stopped", "keyboard_interrupt")
            raise
        except Exception as exc:  # noqa: BLE001
            _log(f"Batch {batch_num} exception: {exc!r}; sleeping 120s")
            _heartbeat("error", str(exc))
            time.sleep(120)


if __name__ == "__main__":
    main()
