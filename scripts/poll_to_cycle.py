#!/usr/bin/env python3
"""Poll multi_cycle_results.jsonl until target cycle; run verifiers on each landing."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "multi_cycle_results.jsonl"
TRAJ = PROJECT_ROOT / ".agents" / "progress_trajectory.jsonl"
STATUS = PROJECT_ROOT / ".agents" / "status" / "agent-evolution.json"
DONE_DIR = PROJECT_ROOT / ".agents" / "tasks" / "agent-evolution" / "done"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tail_record(path: Path) -> dict:
    with open(path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        chunk = b""
        while pos > 0 and b"\n" not in chunk:
            read = min(8192, pos)
            pos -= read
            f.seek(pos)
            chunk = f.read(read) + chunk
        return json.loads(chunk.decode().strip().split("\n")[-1])


def run_json(cmd: list[str]) -> tuple[dict, int]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, check=False)
    out = (r.stdout or r.stderr or "").strip()
    try:
        start = out.index("{")
        end = out.rindex("}") + 1
        return json.loads(out[start:end]), r.returncode
    except Exception:  # noqa: BLE001 - subprocess stdout may be non-JSON; fall back to raw text
        return {"raw": out, "returncode": r.returncode}, r.returncode


def update_status(rec: dict, epi: dict, health: dict, parity: dict, interventions: list) -> None:
    status = {
        "agent": "agent-evolution",
        "status": "working",
        "task": "EPI momentum goal monitoring",
        "updated_at": utc(),
        "notes": (
            f"C{rec['cycle']} landed fit {rec['final_fitness']:.2f} "
            f"({rec['elapsed_seconds']:.1f}s). "
            f"avg_momentum {epi.get('avg_momentum', 0):.4f}. "
            f"recent_100 {epi.get('recent_100_avg', 0):.4f}."
        ),
        "goal_met": epi.get("goal_met", False),
        "last_cycle": rec["cycle"],
        "last_fitness": rec["final_fitness"],
        "last_elapsed_seconds": rec["elapsed_seconds"],
        "in_flight_cycle": rec["cycle"] + 1,
        "avg_momentum": epi.get("avg_momentum"),
        "recent_100_avg": epi.get("recent_100_avg"),
        "sample_count": epi.get("sample_count"),
        "samples_until_full_window": epi.get("samples_until_full_window"),
        "growth_gate_met": epi.get("growth_gate_met"),
        "next_check_triggers": ["C4000", "avg_momentum > 0.30", "sample_count >= 4000"],
        "recommendation": "GOAL_MET" if epi.get("goal_met") else "WAIT",
        "loop_health": {"ok": health.get("ok"), "components": health.get("components", {})},
        "trajectory_parity": parity,
        "interventions_taken": interventions,
    }
    STATUS.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")


def on_new_cycle(rec: dict, last_cycle: int, interventions: list, log: list) -> tuple[int, dict]:
    c = rec["cycle"]
    log.append(f"{utc()} C{c} fit={rec['final_fitness']:.4f} elapsed={rec['elapsed_seconds']:.1f}")
    subprocess.run(
        [sys.executable, "scripts/progress_trajectory.py", "--record"],
        capture_output=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    epi, _ = run_json([sys.executable, "scripts/verify_epi_momentum_goal.py", "--json"])
    health, _ = run_json([sys.executable, "scripts/verify_loop_health.py", "--json"])
    if not health.get("ok"):
        health, _ = run_json([sys.executable, "scripts/verify_loop_health.py", "--json", "--repair"])
        interventions.append({"at": utc(), "action": "loop_health_repair", "health": health})

    rc = tail_record(RESULTS)["cycle"]
    tc = tail_record(TRAJ)["cycle"] if TRAJ.exists() else None
    drift = tc != rc
    if drift:
        subprocess.run(
            [sys.executable, "scripts/progress_trajectory.py", "--rebuild"],
            capture_output=True,
            cwd=PROJECT_ROOT,
            check=False,
        )
        tc = tail_record(TRAJ)["cycle"] if TRAJ.exists() else None
        drift = tc != rc
        interventions.append(
            {
                "at": utc(),
                "action": "trajectory_rebuild",
                "results_cycle": rc,
                "trajectory_cycle": tc,
            }
        )

    parity = {"drift": drift, "results_cycle": rc, "trajectory_cycle": tc}
    if epi.get("needs_intervention"):
        interventions.append({"at": utc(), "action": "epi_needs_intervention", "epi": epi})
    if epi.get("growth_gate_met") is False and epi.get("growth_reason") != "ok":
        interventions.append({"at": utc(), "action": "growth_gate_at_risk", "epi": epi})

    update_status(rec, epi, health, parity, interventions)
    log.append(f"  epi avg={epi.get('avg_momentum')} recent100={epi.get('recent_100_avg')} goal={epi.get('goal_met')}")
    return c, epi


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    start_from = int(sys.argv[2]) if len(sys.argv) > 2 else tail_record(RESULTS)["cycle"]

    last_cycle = start_from
    interventions: list = []
    log: list = []
    start = time.time()

    while last_cycle < target:
        age = time.time() - RESULTS.stat().st_mtime
        rec = tail_record(RESULTS)
        c = rec["cycle"]
        if c > last_cycle:
            last_cycle, epi = on_new_cycle(rec, last_cycle, interventions, log)
            if epi.get("goal_met"):
                break
        elif age > 600:
            health, _ = run_json([sys.executable, "scripts/verify_loop_health.py", "--json"])
            if not health.get("ok"):
                health, _ = run_json([sys.executable, "scripts/verify_loop_health.py", "--json", "--repair"])
            interventions.append(
                {
                    "at": utc(),
                    "action": "stale_jsonl_600s",
                    "age_s": round(age, 1),
                    "in_flight": c + 1,
                    "health": health,
                }
            )
            log.append(f"{utc()} STALE jsonl age={age:.0f}s in_flight C{c + 1}")
        time.sleep(30)

    final_rec = tail_record(RESULTS)
    final_epi, _ = run_json([sys.executable, "scripts/verify_epi_momentum_goal.py", "--json"])
    final_health, _ = run_json([sys.executable, "scripts/verify_loop_health.py", "--json"])
    rc = final_rec["cycle"]
    tc = tail_record(TRAJ)["cycle"] if TRAJ.exists() else None
    summary = {
        "reached_target": rc >= target,
        "target_cycle": target,
        "last_cycle": rc,
        "last_fitness": final_rec["final_fitness"],
        "last_elapsed": final_rec["elapsed_seconds"],
        "goal_met": final_epi.get("goal_met"),
        "epi": final_epi,
        "loop_health_ok": final_health.get("ok"),
        "parity": {"drift": tc != rc, "results_cycle": rc, "trajectory_cycle": tc},
        "interventions": interventions,
        "poll_seconds": round(time.time() - start, 1),
        "log": log,
    }
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DONE_DIR / f"poll_c{target}_summary.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["reached_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
