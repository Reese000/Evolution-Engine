#!/usr/bin/env python3
"""One-shot EPI momentum goal progress report (loop health + ETA)."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evolution.epi_momentum_governor import DEFAULT_TARGET, DEFAULT_WINDOW, load_recent_momentum


def _max_cycle(results_path: str) -> int:
    if not os.path.isfile(results_path):
        return 0
    best = 0
    with open(results_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            best = max(best, int(row.get("cycle", 0)))
    return best


def _estimate_cycles_to_goal(
    window: list[float],
    target: float,
    *,
    latest: float | None = None,
    recent_100: float | None = None,
) -> int:
    if not window:
        return -1
    incoming_n = min(20, len(window))
    incoming_avg = sum(window[-incoming_n:]) / incoming_n
    tail_n = min(5, len(window))
    tail_avg = sum(window[-tail_n:]) / tail_n
    if latest is not None:
        incoming_avg = max(incoming_avg, tail_avg, float(latest))
    elif recent_100 is not None and recent_100 > target:
        incoming_avg = max(incoming_avg, tail_avg)
    if incoming_avg <= target:
        return -1
    win = list(window)
    cycles = 0
    while len(win) >= len(window) and sum(win) / len(win) <= target and cycles < 5000:
        win.pop(0)
        win.append(incoming_avg)
        cycles += 1
    return cycles if sum(win) / len(win) > target else -1


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor EPI momentum goal progress")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from evolution.epi_momentum_governor import assess_epi_momentum

    report = assess_epi_momentum(PROJECT_ROOT, window=args.window, target_avg=args.target)
    samples = load_recent_momentum(PROJECT_ROOT, args.window)
    eta = _estimate_cycles_to_goal(
        samples,
        args.target,
        latest=report.latest_momentum,
        recent_100=report.recent_avg,
    )
    max_cycle = _max_cycle(os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl"))

    loop_ok = False
    loop_detail: dict[str, object] = {}
    try:
        from scripts.verify_loop_health import verify

        loop_detail = verify(repair=False)
        loop_ok = bool(loop_detail.get("ok"))
    except Exception as exc:  # noqa: BLE001
        loop_detail = {"error": str(exc)}

    payload = {
        "goal_met": report.goal_met,
        "max_cycle": max_cycle,
        "window": report.window,
        "avg_momentum": report.avg_momentum,
        "recent_avg": report.recent_avg,
        "incoming_avg_20": round(sum(samples[-min(20, len(samples)) :]) / min(20, len(samples)), 4) if samples else 0.0,
        "latest_momentum": report.latest_momentum,
        "target_avg": report.target_avg,
        "needs_intervention": report.needs_intervention,
        "estimated_cycles_remaining": eta,
        "loop_ok": loop_ok,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        status = "PASS" if report.goal_met else "IN PROGRESS"
        eta_str = f"~{eta} cycles" if eta >= 0 else "n/a (incoming below target)"
        print(
            f"{status} | cycle {max_cycle} | "
            f"1000-avg={report.avg_momentum:.4f} recent-100={report.recent_avg:.4f} "
            f"latest={report.latest_momentum:.2f} "
            f"(target >{args.target}) | ETA {eta_str} | loop={'OK' if loop_ok else 'ISSUE'}"
        )
    return 0 if report.goal_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
