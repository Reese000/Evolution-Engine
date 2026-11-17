#!/usr/bin/env python3
"""Verify EPI momentum goal: avg > target over window consecutive cycles."""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from evolution.epi_momentum_governor import (
    DEFAULT_TARGET,
    DEFAULT_WINDOW,
    assess_epi_momentum,
    assess_exponential_growth,
)


def _project_cycles_to_goal(
    samples: list[float],
    window: int,
    target: float,
    steady_rate: float,
) -> int | None:
    """Estimate additional cycles until a full window averages above target."""
    if not samples:
        return None
    if len(samples) >= window:
        tail = samples[-window:]
        if sum(tail) / len(tail) > target:
            return 0
    extended = list(samples)
    for added in range(1, 25000):
        extended.append(steady_rate)
        tail = extended[-window:]
        if len(tail) >= window and sum(tail) / len(tail) > target:
            return added
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify EPI momentum goal")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = assess_epi_momentum(PROJECT_ROOT, window=args.window, target_avg=args.target)
    growth = assess_exponential_growth(PROJECT_ROOT, window=args.window)
    recent_n = min(100, report.sample_count)
    from evolution.epi_momentum_governor import load_recent_momentum

    all_mom = load_recent_momentum(PROJECT_ROOT, args.window)
    recent_avg = round(sum(all_mom[-recent_n:]) / recent_n, 4) if recent_n else 0.0
    recent_50 = all_mom[-50:] if len(all_mom) >= 50 else all_mom
    steady = recent_avg if recent_avg > args.target else 0.55
    projected_add = _project_cycles_to_goal(all_mom, args.window, args.target, steady)
    above_target = sum(1 for m in all_mom if m > args.target)
    payload = {
        "goal_met": report.goal_met,
        "momentum_gate_met": report.momentum_gate_met,
        "growth_gate_met": report.growth_gate_met,
        "growth_fitness_gain": growth.fitness_gain,
        "growth_epi_gain": growth.epi_gain,
        "growth_lookback": growth.lookback,
        "growth_reason": growth.reason,
        "window": report.window,
        "target_avg": report.target_avg,
        "sample_count": report.sample_count,
        "avg_momentum": report.avg_momentum,
        "recent_100_avg": recent_avg,
        "recent_50_avg": round(sum(recent_50) / len(recent_50), 4) if recent_50 else 0.0,
        "samples_above_target": above_target,
        "min_momentum": report.min_momentum,
        "max_momentum": report.max_momentum,
        "latest_momentum": report.latest_momentum,
        "needs_intervention": report.needs_intervention,
        "samples_until_full_window": max(0, args.window - report.sample_count),
        "cycles_to_flush_bad_history": max(0, args.window - report.sample_count),
        "projected_cycles_to_goal": projected_add,
        "projected_goal_cycle": (report.sample_count + projected_add if projected_add is not None else None),
        "steady_momentum_assumption": round(steady, 4),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        status = "PASS" if report.goal_met else "FAIL"
        print(
            f"{status}: epi_momentum avg={report.avg_momentum:.4f} "
            f"(target >{report.target_avg}, n={report.sample_count}/{report.window})"
        )
    return 0 if report.goal_met else 1


if __name__ == "__main__":
    raise SystemExit(main())
