#!/usr/bin/env python3
"""Evolution Progress Index (EPI) — trajectory tracking and visualization data.

Streams multi_cycle_results.jsonl (large lines with embedded archives) via regex
extraction, computes composite progress metrics, writes timeline artifacts, and
can refresh the progress canvas.

Usage:
  python scripts/progress_trajectory.py --rebuild    # full trajectory from results
  python scripts/progress_trajectory.py --append       # incremental (per new result line)
  python scripts/progress_trajectory.py --watch        # live tail + smooth canvas refresh
  python scripts/progress_trajectory.py --record       # append or rebuild if needed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

_IN_WATCH_LOOP = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
AGENTS = os.path.join(PROJECT_ROOT, ".agents")
RESULTS = os.path.join(PROJECT_ROOT, "multi_cycle_results.jsonl")
TRAJECTORY = os.path.join(AGENTS, "progress_trajectory.jsonl")
SUMMARY = os.path.join(AGENTS, "progress_trajectory_summary.json")
STATE_FILE = os.path.join(AGENTS, "progress_trajectory_state.json")
HEARTBEAT = os.path.join(AGENTS, "progress_trajectory.heartbeat")
WATCHER_PID_FILE = os.path.join(AGENTS, "progress_trajectory_watcher.pid")
METRIC_DOC = os.path.join(AGENTS, "PROGRESS_METRIC.md")
CANVAS_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cursor",
    "projects",
    "c-Users-reese-Projects-Evolution-Engine",
    "canvases",
)
CANVAS_PATH = os.path.join(CANVAS_DIR, "evolution-progress-trajectory.canvas.tsx")
CANVAS_TEMPLATE_VERSION = "terminal-ops-v3"
EPI_FORMULA_VERSION = "terminal-loop-fitness-v1"
PROGRESS_CONTROLLER_STATE = os.path.join(AGENTS, "progress_controller_state.json")
AGENT_STATUS = os.path.join(AGENTS, "status", "agent-evolution.json")

MAX_LEVEL = float(os.environ.get("EE_MAX_LEVEL", "30"))
PLATEAU_FIX_DATE = "2026-09-03"  # curriculum partial-verify gate


def _terminal_depth() -> int:
    """Canonical terminal depth — not EE_MAX_LEVEL clamp recovery caps (e.g. 31)."""
    try:
        from evolution.terminal_stagnation import terminal_depth

        return terminal_depth()
    except ImportError:
        return int(
            os.environ.get(
                "EE_TERMINAL_DEPTH",
                os.environ.get("EE_CURRICULUM_MAX_LEVEL", "30"),
            )
        )


SMOOTH_WINDOW = max(3, int(os.environ.get("EE_TRAJECTORY_SMOOTH_WINDOW", "25")))
RECENT_WINDOW = max(40, int(os.environ.get("EE_TRAJECTORY_RECENT_WINDOW", "250")))
WATCH_INTERVAL = float(os.environ.get("EE_TRAJECTORY_WATCH_SEC", "12"))
TERMINAL_MOMENTUM_FLOOR = float(os.environ.get("EE_TERMINAL_MOMENTUM_FLOOR", "0.55"))
TERMINAL_MOMENTUM_EPI_MIN = float(os.environ.get("EE_TERMINAL_MOMENTUM_EPI_MIN", "90.0"))
TERMINAL_STREAK_BONUS_START = int(os.environ.get("EE_TERMINAL_STREAK_BONUS_START", "20"))
TERMINAL_STREAK_BONUS_STEP = float(os.environ.get("EE_TERMINAL_STREAK_BONUS_STEP", "0.02"))
TERMINAL_STREAK_BONUS_CAP = float(os.environ.get("EE_TERMINAL_STREAK_BONUS_CAP", "1.0"))
TERMINAL_FITNESS_STAGNATION_CYCLES = int(os.environ.get("EE_TERMINAL_FITNESS_STAGNATION_CYCLES", "30"))

_FIELD_PATTERNS: dict[str, tuple[str, type]] = {
    "cycle": (r'"cycle":\s*(\d+)', int),
    "final_level": (r'"final_level":\s*(\d+)', int),
    "final_fitness": (r'"final_fitness":\s*([-\d.eE+]+)', float),
    "champion_verify_score": (r'"champion_verify_score":\s*([-\d.eE+]+)', float),
    "max_pop_verify": (r'"max_pop_verify":\s*([-\d.eE+]+)', float),
    "generations_completed": (r'"generations_completed":\s*(\d+)', int),
    "elapsed_seconds": (r'"elapsed_seconds":\s*([-\d.eE+]+)', float),
    "strategy_archive_records": (r'"strategy_archive_records":\s*(\d+)', int),
    "vm_gpu_batches": (r'"vm_gpu_batches":\s*(\d+)', int),
    "vm_cpu_fallbacks": (r'"vm_cpu_fallbacks":\s*(\d+)', int),
    "fitness_gain": (r'"fitness_gain":\s*([-\d.eE+]+)', float),
    "fitness_end": (r'"fitness_end":\s*([-\d.eE+]+)', float),
    "champion_words": (r'"champion_words":\s*(\d+)', int),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fast_extract(line: str) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, (pat, cast) in _FIELD_PATTERNS.items():
        m = re.search(pat, line)
        if m:
            try:
                row[key] = cast(m.group(1))
            except (ValueError, TypeError):
                pass
    if "error" in line and '"error":' in line and '"error": null' not in line:
        err_m = re.search(r'"error":\s*"([^"]*)"', line)
        if err_m and err_m.group(1):
            row["error"] = err_m.group(1)
    return row


def _verify_effective(level: int, verify: float) -> float:
    """Raw verify quality 0–1; partial-verify cap is informational only (not flattened to 1.0)."""
    return min(1.0, max(0.0, verify))


def _fitness_vs_level_best(fitness: float, level_best: float) -> float:
    """0–1 progress vs best fitness seen at this curriculum level."""
    if level_best <= 0:
        return 0.5
    ratio = fitness / level_best
    return min(1.0, max(0.0, ratio))


def _terminal_fitness_growth_bonus(
    level: int,
    fitness: float,
    prev_level_best: float | None,
) -> float:
    """Extra 0–0.25 EPI component credit when fitness rises at max depth."""
    if level < _terminal_depth() or prev_level_best is None or prev_level_best <= 0:
        return 0.0
    if fitness <= prev_level_best:
        return 0.0
    growth = (fitness - prev_level_best) / max(abs(prev_level_best), 1.0)
    return min(0.25, growth * 0.5)


def _length_efficiency_score(
    level: int,
    words: int,
    prev_best_words: float | None = None,
) -> float:
    """0–1 compactness signal at terminal depth; neutral 0.5 elsewhere.

    Rewards shorter verified programs vs running best and EE_TERMINAL_REF_WORDS.
    Anti-exploit: requires words > 0; no credit when level < terminal depth.
    """
    if level < _terminal_depth() or words <= 0:
        return 0.5
    ref = int(os.environ.get("EE_TERMINAL_REF_WORDS", "73"))
    if prev_best_words is not None and prev_best_words > 0:
        ref = max(ref, int(prev_best_words))
    ratio = ref / max(words, 1)
    return min(1.0, max(0.0, 0.5 + (ratio - 1.0) * 0.25))


def _terminal_verify_growth_bonus(
    level: int,
    verify: float,
    prev_level_verify: float | None,
) -> float:
    """Extra 0–0.25 verify-eff credit when partial verify rises at max depth."""
    if level < _terminal_depth() or prev_level_verify is None or prev_level_verify <= 0:
        return 0.0
    if verify <= prev_level_verify + 1e-6:
        return 0.0
    headroom = max(0.05, 1.0 - prev_level_verify)
    growth = (verify - prev_level_verify) / headroom
    return min(0.25, growth * 0.5)


def _certified_terminal_epi_row(level: int, verify: float) -> bool:
    """L30 + saturated verify — core EPI components plateau; fitness must still move EPI."""
    return level >= _terminal_depth() and verify >= 0.99 - 1e-6


_LOOP_FITNESS_MIN = 200.0
_LOOP_FITNESS_FLOOR_SLACK = 120.0


def _fitness_for_epi(row: dict[str, Any]) -> float:
    """In-loop champion fitness, unclamped from the certified reporting floor.

    ``final_fitness`` can freeze at a reporting floor while ``fitness_end``
    (best_fitness_history[-1]) keeps moving +0.3/cycle. Growth and momentum
    gates must see that live loop signal, not the clamp.
    """
    final = float(row.get("final_fitness") or 0.0)
    level = int(row.get("final_level") or 0)
    end_raw = row.get("fitness_end")
    if end_raw is None or level < _terminal_depth():
        return final
    try:
        end = float(end_raw)
    except (TypeError, ValueError):
        return final
    if end >= final - 1e-9:
        return end
    if end >= _LOOP_FITNESS_MIN and (final - end) <= _LOOP_FITNESS_FLOOR_SLACK:
        return end
    return final


def _terminal_fitness_epi_bump(fitness: float) -> float:
    """Monotonic EPI credit from absolute fitness at certified terminal depth.

    Linear in (fitness - base) so growth gate tracks sustained plateau gains without
    saturating when verify/curriculum components are already maxed.
    """
    base = float(os.environ.get("EE_TERMINAL_FITNESS_EPI_BASE", "250"))
    scale = float(os.environ.get("EE_TERMINAL_FITNESS_EPI_SCALE", "0.055"))
    bump_max = float(os.environ.get("EE_TERMINAL_FITNESS_EPI_BUMP_MAX", "25.0"))
    if fitness <= base:
        return 0.0
    return round(min(bump_max, (fitness - base) * scale), 4)


def _epi_verify_certify_floor(level: int) -> float:
    """Certify bar for EPI partial-verify penalty — relaxed at terminal depth."""
    from evolution.progress_controller import ProgressController

    if level >= _terminal_depth():
        from evolution.curriculum_advance import depth_carry_verify_floor

        return depth_carry_verify_floor(level)
    return ProgressController.certify_verify_threshold(level)


def _gens_ratio(gens: int, *, level: int | None = None) -> float:
    """Throughput credit. Intentional terminal gens cap is full credit, not a stall."""
    if level is not None and level >= _terminal_depth():
        cap = int(os.environ.get("EE_TERMINAL_MAX_GENS", "30"))
        if gens >= cap:
            return 1.0
    if gens >= 49:
        return 1.0
    if gens >= 10:
        return min(1.0, gens / 49.0)
    return max(0.0, gens / 10.0)


def _compute_epi(
    row: dict[str, Any],
    level_median_fitness: dict[int, float],
    level_best_fitness: dict[int, float],
    prev_level_best: float | None = None,
    prev_level_verify: float | None = None,
    prev_best_words: float | None = None,
) -> dict[str, Any]:
    level = int(row.get("final_level") or 0)
    champion_verify = float(row.get("champion_verify_score") or 0.0)
    max_pop_verify = float(row.get("max_pop_verify") or 0.0)
    verify = champion_verify
    if level >= _terminal_depth() and max_pop_verify > champion_verify:
        verify = max_pop_verify
    fitness = _fitness_for_epi(row)
    gens = int(row.get("generations_completed") or 0)
    strat = int(row.get("strategy_archive_records") or 0)
    gpu_batches = int(row.get("vm_gpu_batches") or 0)
    cpu_fb = int(row.get("vm_cpu_fallbacks") or 0)

    level_score = min(1.0, level / MAX_LEVEL)
    verify_eff = _verify_effective(level, verify)
    verify_eff = min(
        1.0,
        verify_eff + _terminal_verify_growth_bonus(level, verify, prev_level_verify),
    )
    gens_score = _gens_ratio(gens, level=level)
    archive_score = min(1.0, strat / 10000.0) if strat > 0 else 0.3

    level_best = level_best_fitness.get(level, fitness)
    compare_best = prev_level_best if prev_level_best is not None and prev_level_best > 0 else level_best
    level_best_score = _fitness_vs_level_best(fitness, compare_best)
    level_best_score = min(
        1.0,
        level_best_score + _terminal_fitness_growth_bonus(level, fitness, prev_level_best),
    )

    median = level_median_fitness.get(level, fitness)
    if median > 0:
        momentum = min(1.0, max(0.0, 0.5 + (fitness - median) / (median * 2)))
    else:
        momentum = 0.5

    # Penalize timeout oscillation and hollow partial-verify at high levels
    if fitness <= -20.0:
        momentum = 0.0
        level_best_score = min(level_best_score, 0.15)
    if level >= 17:
        certify = _epi_verify_certify_floor(level)
        if verify < certify:
            verify_eff = verify_eff * max(0.2, verify / certify)
            momentum = min(momentum, 0.25)
            level_best_score = min(level_best_score, 0.35)

    gpu_score = 1.0 if gpu_batches > 0 and cpu_fb == 0 else (0.5 if gpu_batches > 0 else 0.2)
    words = int(row.get("champion_words") or 0)
    length_eff = _length_efficiency_score(level, words, prev_best_words)

    # Plateau-aware blend: when curriculum stalls, within-level fitness carries more signal.
    level_score_w = 0.38
    verify_w = 0.22
    momentum_w = 0.15
    level_best_w = 0.0
    gens_w = 0.10
    archive_w = 0.10
    gpu_w = 0.05
    if level >= 8:
        level_score_w = 0.22
        verify_w = 0.18
        level_best_w = 0.28
        momentum_w = 0.12
        gens_w = 0.08
        archive_w = 0.06
        gpu_w = 0.04

    components = {
        "curriculum_depth": round(level_score * 100, 2),
        "verify_quality": round(verify_eff * 100, 2),
        "fitness_momentum": round(momentum * 100, 2),
        "fitness_at_level": round(level_best_score * 100, 2),
        "generation_throughput": round(gens_score * 100, 2),
        "archive_learning": round(archive_score * 100, 2),
        "gpu_health": round(gpu_score * 100, 2),
        "length_efficiency": round(length_eff * 100, 2),
    }

    epi = (
        level_score_w * level_score
        + verify_w * verify_eff
        + momentum_w * momentum
        + level_best_w * level_best_score
        + gens_w * gens_score
        + archive_w * archive_score
        + gpu_w * gpu_score
    ) * 100

    terminal_bump = 0.0
    if _certified_terminal_epi_row(level, verify):
        terminal_bump = _terminal_fitness_epi_bump(fitness)
        components["terminal_fitness_bump"] = round(terminal_bump, 2)

    return {
        "epi": round(epi + terminal_bump, 2),
        "components": components,
        "level": level,
        "fitness": round(fitness, 3),
        "verify_score": round(verify, 4),
        "champion_words": words,
        "generations_completed": gens,
    }


def _carry_terminal_archive_learning(
    metrics: dict[str, Any],
    prev_archive: float | None,
) -> dict[str, Any]:
    """Sidecar prune must not drop terminal EPI via archive_learning collapse."""
    if prev_archive is None:
        return metrics
    if int(metrics.get("level") or 0) < _terminal_depth():
        return metrics
    comps = metrics.setdefault("components", {})
    current = float(comps.get("archive_learning") or 0.0)
    if float(prev_archive) <= current + 1e-9:
        return metrics
    delta = (float(prev_archive) - current) * 0.06
    comps["archive_learning"] = round(float(prev_archive), 2)
    metrics["epi"] = round(float(metrics["epi"]) + delta, 2)
    return metrics


def _stream_result_rows() -> list[dict[str, Any]]:
    if not os.path.isfile(RESULTS):
        return []
    rows: list[dict[str, Any]] = []
    with open(RESULTS, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = _fast_extract(line)
            if row.get("cycle") is None:
                continue
            rows.append(row)
    return rows


def _dedupe_rows_by_cycle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep best row per cycle (highest level, then verify, then fitness)."""
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        cycle = int(row.get("cycle", -1))
        if cycle < 0:
            continue
        rank = (
            int(row.get("final_level", 0)),
            float(row.get("champion_verify_score", 0)),
            float(row.get("final_fitness", -1e9)),
        )
        prev = best.get(cycle)
        if prev is None or rank > (
            int(prev.get("final_level", 0)),
            float(prev.get("champion_verify_score", 0)),
            float(prev.get("final_fitness", -1e9)),
        ):
            best[cycle] = row
    return [best[c] for c in sorted(best)]


def _build_trajectory_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _dedupe_rows_by_cycle(rows)
    level_fitness: dict[int, list[float]] = {}
    for row in rows:
        lvl = int(row.get("final_level") or 0)
        fit = _fitness_for_epi(row)
        level_fitness.setdefault(lvl, []).append(fit)

    level_median = {lvl: sorted(vals)[len(vals) // 2] for lvl, vals in level_fitness.items() if vals}

    cumulative_elapsed = 0.0
    level_running_best: dict[int, float] = {}
    level_running_best_verify: dict[int, float] = {}
    level_running_best_words: dict[int, int] = {}
    points: list[dict[str, Any]] = []
    for row in rows:
        cumulative_elapsed += float(row.get("elapsed_seconds") or 0)
        lvl = int(row.get("final_level") or 0)
        fit = _fitness_for_epi(row)
        champ_v = float(row.get("champion_verify_score") or 0.0)
        pop_v = float(row.get("max_pop_verify") or 0.0)
        row_verify = max(champ_v, pop_v) if lvl >= _terminal_depth() else champ_v
        row_words = int(row.get("champion_words") or 0)
        prev_best = level_running_best.get(lvl, 0.0)
        prev_verify = level_running_best_verify.get(lvl, 0.0)
        prev_words = level_running_best_words.get(lvl, 0)
        level_best_snapshot = max(prev_best, fit)
        verify_snapshot = max(prev_verify, row_verify)
        words_snapshot = row_words if row_words > 0 else prev_words
        if prev_words > 0 and row_words > 0:
            words_snapshot = min(prev_words, row_words)
        metrics = _compute_epi(
            row,
            level_median,
            {lvl: level_best_snapshot},
            prev_level_best=prev_best if prev_best > 0 else None,
            prev_level_verify=prev_verify if prev_verify > 0 else None,
            prev_best_words=float(prev_words) if prev_words > 0 else None,
        )
        prev_arch = None
        if points:
            prev_arch = float((points[-1].get("components") or {}).get("archive_learning") or 0.0)
        metrics = _carry_terminal_archive_learning(metrics, prev_arch)
        level_running_best[lvl] = level_best_snapshot
        level_running_best_verify[lvl] = verify_snapshot
        if words_snapshot > 0:
            level_running_best_words[lvl] = words_snapshot
        points.append(
            {
                "at": _utc_now(),  # replaced on rebuild with synthetic ordering
                "cycle": int(row["cycle"]),
                "cumulative_hours": round(cumulative_elapsed / 3600.0, 3),
                **metrics,
            }
        )

    epi_series = [p["epi"] for p in points]
    window = min(50, len(epi_series))
    rolling = _rolling_mean(epi_series, window)
    for i, p in enumerate(points):
        p["epi_momentum"] = _momentum_for_point(points, i, epi_series, rolling)
    return points


def _downsample(points: list[dict[str, Any]], max_points: int = 400) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    out: list[dict[str, Any]] = []
    i = 0.0
    while int(i) < len(points):
        out.append(points[int(i)])
        i += step
    if points[-1] not in out:
        out.append(points[-1])
    return out


def _is_certified_terminal_point(point: dict[str, Any]) -> bool:
    """L30 + near-perfect verify — do not gate on instantaneous EPI.

    EPI can dip at plateau when fitness_momentum tracks median (e.g. 100→50)
    even though the champion remains certified. Requiring EPI>=90 broke terminal
    momentum and injected large negative classic deltas into the 5000-window.
    """
    level = int(point.get("level", 0))
    if level < _terminal_depth():
        return False
    verify = float(point.get("verify_score", 0.0))
    return verify >= 0.99 - 1e-6


def _certified_terminal_streak(points: list[dict[str, Any]], index: int) -> int:
    streak = 0
    for j in range(index, -1, -1):
        if not _is_certified_terminal_point(points[j]):
            break
        streak += 1
    return streak


def _streak_growth_bonus(points: list[dict[str, Any]], index: int) -> float:
    """Reward compounding stability on long certified L30 runs."""
    streak = _certified_terminal_streak(points, index)
    if streak < TERMINAL_STREAK_BONUS_START:
        return 0.0
    return min(
        TERMINAL_STREAK_BONUS_CAP,
        (streak - TERMINAL_STREAK_BONUS_START + 1) * TERMINAL_STREAK_BONUS_STEP,
    )


def _fitness_unchanged_streak(points: list[dict[str, Any]], index: int, epsilon: float = 1e-3) -> int:
    """Consecutive cycles ending at index with unchanged champion fitness."""
    fit = float(points[index].get("fitness", 0.0))
    streak = 1
    for j in range(index - 1, -1, -1):
        if abs(float(points[j].get("fitness", 0.0)) - fit) > epsilon:
            break
        streak += 1
    return streak


def _terminal_growth_momentum(points: list[dict[str, Any]], index: int) -> float:
    """Credit terminal progress when classic EPI delta flattens — not frozen fitness."""
    p = points[index]
    if not _is_certified_terminal_point(p):
        return 0.0
    epi = float(p.get("epi", 0.0))
    fit_stall = _fitness_unchanged_streak(points, index)
    if fit_stall > TERMINAL_FITNESS_STAGNATION_CYCLES:
        bonus = 0.0
    else:
        bonus = TERMINAL_MOMENTUM_FLOOR + _streak_growth_bonus(points, index)
    if index <= 0:
        return bonus
    prev = points[index - 1]
    prev_fit = float(prev.get("fitness", 0.0))
    fit = float(p.get("fitness", 0.0))
    if fit > prev_fit + 1e-3:
        rel = (fit - prev_fit) / max(abs(prev_fit), 1.0)
        bonus += min(3.0, rel * 50.0)
    prev_epi = float(prev.get("epi", 0.0))
    if epi > prev_epi + 0.01:
        bonus += min(2.0, (epi - prev_epi) * 2.0)
    words = int(p.get("champion_words", 0))
    prev_words = int(prev.get("champion_words", 0))
    if words > 0 and prev_words > 0 and words < prev_words:
        rel = (prev_words - words) / max(prev_words, 1)
        bonus += min(2.5, rel * 25.0)
    comps = p.get("components") or {}
    prev_comps = prev.get("components") or {}
    arch = float(comps.get("archive_learning", 0.0))
    prev_arch = float(prev_comps.get("archive_learning", 0.0))
    if arch > prev_arch + 0.05:
        bonus += min(1.5, (arch - prev_arch) * 0.05)
    length_eff = float(comps.get("length_efficiency", 50.0))
    prev_length = float(prev_comps.get("length_efficiency", 50.0))
    if length_eff > prev_length + 0.5:
        bonus += min(1.5, (length_eff - prev_length) * 0.1)
    return bonus


def _momentum_for_point(
    points: list[dict[str, Any]],
    index: int,
    epi_series: list[float],
    rolling: list[float],
) -> float:
    classic = round(epi_series[index] - rolling[index], 2)
    terminal = round(_terminal_growth_momentum(points, index), 2)
    if terminal > 0.0:
        return round(max(classic, terminal), 2)
    return classic


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    out: list[float] = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start : i + 1]
        out.append(round(sum(chunk) / len(chunk), 2))
    return out


def _smooth(values: list[float]) -> list[float]:
    """Rolling mean for chart display."""
    return _rolling_mean(values, min(SMOOTH_WINDOW, len(values)))


def _load_json_file(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * max(0.0, min(100.0, p)) / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo]) * (1.0 - frac) + float(ordered[hi]) * frac


def _robust_bounds(
    values: list[float],
    *,
    lo_p: float = 5.0,
    hi_p: float = 95.0,
    pad_frac: float = 0.12,
    allow_negative: bool = True,
    include_latest: bool = True,
) -> tuple[float, float]:
    """Y-axis that keeps typical movement readable when a few spikes exist."""
    if not values:
        return 0.0, 1.0
    lo = _percentile(values, lo_p)
    hi = _percentile(values, hi_p)
    if hi <= lo:
        lo, hi = float(min(values)), float(max(values))
    span = max(hi - lo, abs(hi) * 0.04, 1.0)
    pad = span * pad_frac
    y_min = lo - pad
    y_max = hi + pad
    if include_latest:
        last = float(values[-1])
        if last <= hi + span:
            y_max = max(y_max, last + pad)
        if last >= lo - span:
            y_min = min(y_min, last - pad)
    if not allow_negative:
        y_min = max(0.0, y_min)
    if y_max <= y_min:
        y_max = y_min + max(pad, 1.0)
    return round(y_min, 1), round(y_max, 1)


def _cycle_seconds(points: list[dict[str, Any]]) -> list[float]:
    seconds: list[float] = []
    prev_hours: float | None = None
    for point in points:
        hours = float(point.get("cumulative_hours") or 0.0)
        if prev_hours is None:
            seconds.append(0.0)
        else:
            seconds.append(round(max(0.0, (hours - prev_hours) * 3600.0), 1))
        prev_hours = hours
    return seconds


def _median(values: list[float]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    ordered = sorted(finite)
    return float(ordered[len(ordered) // 2])


def _live_ops(latest: dict[str, Any]) -> dict[str, Any]:
    """Overlay scan-floor / promotion-raw from live controller + agent status."""
    ctrl = _load_json_file(PROGRESS_CONTROLLER_STATE)
    status = _load_json_file(AGENT_STATUS)
    heldout = ctrl.get("terminal_efficiency_scan_floor_heldout")
    prom = status.get("champion_promotion_raw")
    if prom is None:
        prom = ctrl.get("raw_fitness")
    scan_words = ctrl.get("terminal_efficiency_scan_floor_words")
    if scan_words is None:
        scan_words = ctrl.get("terminal_efficiency_certified_words")
    ops: dict[str, Any] = {
        "prom_raw": round(float(prom), 3) if prom is not None else None,
        "heldout_pct": round(float(heldout) * 100.0, 2) if heldout is not None else None,
        "cycles_at_level": ctrl.get("cycles_at_level"),
        "scan_floor_words": scan_words,
        "status_task": status.get("current_task"),
        "words": int(latest.get("champion_words") or 0),
        "verify": float(latest.get("verify_score") or 0.0),
        "level": int(latest.get("level") or 0),
    }
    return ops


def _load_state() -> dict[str, Any]:
    if not os.path.isfile(STATE_FILE):
        return {"results_byte_offset": 0, "results_lines": 0, "level_running_best": {}}
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"results_byte_offset": 0, "results_lines": 0, "level_running_best": {}}


def _save_state(state: dict[str, Any]) -> None:
    os.makedirs(AGENTS, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _read_heartbeat_data() -> dict[str, Any]:
    if not os.path.isfile(HEARTBEAT):
        return {}
    try:
        with open(HEARTBEAT, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _read_watcher_pid() -> int | None:
    if not os.path.isfile(WATCHER_PID_FILE):
        return None
    try:
        with open(WATCHER_PID_FILE, encoding="utf-8") as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_pid(pid: int) -> bool:
    """Terminate another process (Windows-safe)."""
    if pid <= 0 or pid == os.getpid():
        return False
    if sys.platform == "win32":
        result = subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except OSError:
        return False


def _canvas_on_disk_version() -> str | None:
    if not os.path.isfile(CANVAS_PATH):
        return None
    try:
        with open(CANVAS_PATH, encoding="utf-8") as f:
            head = f.read(256)
    except OSError:
        return None
    match = re.search(r"canvas-template:(\S+)", head)
    if match:
        return match.group(1)
    if "<H1>" in head or "Evolution Progress Trajectory" in head:
        return "legacy"
    return "unknown"


def _watcher_pid_candidates() -> list[int]:
    pids: list[int] = []
    hb = _read_heartbeat_data()
    hb_pid = hb.get("pid")
    if isinstance(hb_pid, int) and hb_pid > 0:
        pids.append(hb_pid)
    file_pid = _read_watcher_pid()
    if file_pid is not None and file_pid not in pids:
        pids.append(file_pid)
    return pids


def _is_stale_watcher_pid(pid: int, hb: dict[str, Any]) -> bool:
    if pid == os.getpid() or not _pid_alive(pid):
        return False
    hb_pid = hb.get("pid")
    hb_version = hb.get("canvas_template_version")
    hb_epi_version = hb.get("epi_formula_version")
    hb_status = str(hb.get("status", ""))
    watcher_status = hb_status.startswith("watch") or hb_status in ("idle", "live", "error")
    if hb_pid == pid and watcher_status and hb_version != CANVAS_TEMPLATE_VERSION:
        return True
    if hb_pid == pid and watcher_status and hb_epi_version != EPI_FORMULA_VERSION:
        return True
    return _canvas_on_disk_version() == "legacy"


def _stop_stale_watcher() -> bool:
    """Stop a background watcher still running an older canvas template."""
    hb = _read_heartbeat_data()
    stopped = False
    for pid in _watcher_pid_candidates():
        if not _is_stale_watcher_pid(pid, hb):
            continue
        if _terminate_pid(pid):
            stopped = True
            time.sleep(0.2)
    if stopped and os.path.isfile(WATCHER_PID_FILE):
        try:
            os.remove(WATCHER_PID_FILE)
        except OSError:
            pass
    return stopped


def _write_heartbeat(status: str, detail: str = "", *, watcher: bool = False) -> None:
    os.makedirs(AGENTS, exist_ok=True)
    if watcher or status.startswith("watch"):
        with open(WATCHER_PID_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
    with open(HEARTBEAT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "at": _utc_now(),
                "status": status,
                "detail": detail,
                "smooth_window": SMOOTH_WINDOW,
                "canvas_template_version": CANVAS_TEMPLATE_VERSION,
                "epi_formula_version": EPI_FORMULA_VERSION,
                "pid": os.getpid(),
            },
            f,
            indent=2,
        )


def _load_trajectory_points() -> list[dict[str, Any]]:
    if not os.path.isfile(TRAJECTORY):
        return []
    points: list[dict[str, Any]] = []
    with open(TRAJECTORY, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                points.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return points


def _level_median_from_points(points: list[dict[str, Any]]) -> dict[int, float]:
    level_fitness: dict[int, list[float]] = {}
    for p in points:
        lvl = int(p.get("level") or 0)
        fit = float(p.get("fitness") or 0)
        level_fitness.setdefault(lvl, []).append(fit)
    return {lvl: sorted(vals)[len(vals) // 2] for lvl, vals in level_fitness.items() if vals}


def _apply_epi_momentum(points: list[dict[str, Any]]) -> None:
    deduped = _dedupe_trajectory_points(points)
    points.clear()
    points.extend(deduped)
    epi_series = [float(p["epi"]) for p in points]
    window = min(50, len(epi_series))
    rolling = _rolling_mean(epi_series, window)
    for i, p in enumerate(points):
        p["epi_momentum"] = _momentum_for_point(points, i, epi_series, rolling)


def _read_new_result_rows(byte_offset: int) -> tuple[list[dict[str, Any]], int]:
    if not os.path.isfile(RESULTS):
        return [], byte_offset
    rows: list[dict[str, Any]] = []
    with open(RESULTS, encoding="utf-8", errors="replace") as f:
        f.seek(byte_offset)
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            row = _fast_extract(line)
            if row.get("cycle") is None:
                continue
            rows.append(row)
        new_offset = f.tell()
    return rows, new_offset


def _dedupe_trajectory_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One point per cycle — last write wins (matches upserted results)."""
    by_cycle: dict[int, dict[str, Any]] = {}
    for p in points:
        cycle = int(p.get("cycle", -1))
        if cycle >= 0:
            by_cycle[cycle] = p
    return [by_cycle[c] for c in sorted(by_cycle)]


def _extend_points(
    existing: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
    level_median: dict[int, float],
    level_running_best: dict[int, float],
    level_running_best_verify: dict[int, float],
    level_running_best_words: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    if level_running_best_words is None:
        level_running_best_words = {}
    cumulative_elapsed = float(existing[-1].get("cumulative_hours", 0) * 3600) if existing else 0.0
    out = list(existing)
    for row in new_rows:
        cumulative_elapsed += float(row.get("elapsed_seconds") or 0)
        lvl = int(row.get("final_level") or 0)
        fit = _fitness_for_epi(row)
        prev_best = level_running_best.get(lvl, 0.0)
        prev_verify = level_running_best_verify.get(lvl, 0.0)
        prev_words = level_running_best_words.get(lvl, 0)
        champ_v = float(row.get("champion_verify_score") or 0.0)
        pop_v = float(row.get("max_pop_verify") or 0.0)
        row_verify = max(champ_v, pop_v) if lvl >= _terminal_depth() else champ_v
        row_words = int(row.get("champion_words") or 0)
        level_best_snapshot = max(prev_best, fit)
        verify_snapshot = max(prev_verify, row_verify)
        words_snapshot = row_words if row_words > 0 else prev_words
        if prev_words > 0 and row_words > 0:
            words_snapshot = min(prev_words, row_words)
        metrics = _compute_epi(
            row,
            level_median,
            {lvl: level_best_snapshot},
            prev_level_best=prev_best if prev_best > 0 else None,
            prev_level_verify=prev_verify if prev_verify > 0 else None,
            prev_best_words=float(prev_words) if prev_words > 0 else None,
        )
        prev_arch = None
        if out:
            prev_arch = float((out[-1].get("components") or {}).get("archive_learning") or 0.0)
        metrics = _carry_terminal_archive_learning(metrics, prev_arch)
        level_running_best[lvl] = level_best_snapshot
        level_running_best_verify[lvl] = verify_snapshot
        if words_snapshot > 0:
            level_running_best_words[lvl] = words_snapshot
        out.append(
            {
                "at": _utc_now(),
                "cycle": int(row["cycle"]),
                "cumulative_hours": round(cumulative_elapsed / 3600.0, 3),
                **metrics,
            }
        )
        level_median = _level_median_from_points(out)
    return _dedupe_trajectory_points(out)


def _write_trajectory_points(points: list[dict[str, Any]]) -> None:
    """Rewrite trajectory so recomputed momentum is persisted for every cycle."""
    os.makedirs(AGENTS, exist_ok=True)
    with open(TRAJECTORY, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in points)


def _append_trajectory_lines(points: list[dict[str, Any]], start_index: int) -> None:
    _write_trajectory_points(points)


def _chart_indices(length: int, max_points: int = 300) -> list[int]:
    if length <= max_points:
        return list(range(length))
    step = length / max_points
    indices: list[int] = []
    i = 0.0
    while int(i) < length:
        indices.append(int(i))
        i += step
    if length - 1 not in indices:
        indices.append(length - 1)
    return indices


def _series_at_indices(values: list[float], indices: list[int]) -> list[float]:
    return [values[i] for i in indices]


def _prepare_chart_data(points: list[dict[str, Any]]) -> dict[str, Any]:
    empty = {
        "categories": [],
        "fitness": [],
        "level": [],
        "fitness_at_level": [],
        "epi": [],
        "momentum": [],
        "recent_categories": [],
        "recent_words": [],
        "recent_fitness": [],
        "recent_seconds": [],
        "recent_length_eff": [],
        "recent_epi": [],
        "recent_median_fitness": None,
        "median_cycle_s": None,
        "last_cycle_s": None,
        "at_terminal": False,
        "level_varies": False,
        "show_words": False,
        "recent_window": 0,
        "recent_start_cycle": None,
        "recent_end_cycle": None,
        "historical_level_best": None,
        "lbest_contaminated": False,
    }
    if not points:
        return empty
    indices = _chart_indices(len(points))
    fitness_raw = [float(p["fitness"]) for p in points]
    level_raw = [float(p["level"]) for p in points]
    fal_raw = [float((p.get("components") or {}).get("fitness_at_level", 0)) for p in points]
    epi_raw = [float(p["epi"]) for p in points]
    mom_raw = [float(p.get("epi_momentum", 0)) for p in points]
    words_raw = [float(p.get("champion_words") or 0) for p in points]
    length_raw = [float((p.get("components") or {}).get("length_efficiency", 0)) for p in points]
    seconds_raw = _cycle_seconds(points)

    fitness_sm = _smooth(fitness_raw)
    fal_sm = _smooth(fal_raw)
    epi_sm = _smooth(epi_raw)
    mom_sm = _smooth(mom_raw)

    recent_n = min(RECENT_WINDOW, len(points))
    recent = points[-recent_n:]
    recent_fitness = _smooth([float(p["fitness"]) for p in recent])
    recent_words = [float(p.get("champion_words") or 0) for p in recent]
    recent_seconds = seconds_raw[-recent_n:]
    recent_length = [float((p.get("components") or {}).get("length_efficiency", 0)) for p in recent]
    recent_epi = _smooth([float(p["epi"]) for p in recent])
    recent_fit_raw = [float(p["fitness"]) for p in recent]
    recent_median = _median(recent_fit_raw)
    positive_secs = [s for s in recent_seconds if s >= 1.0]
    median_cycle = _median(positive_secs)
    last_cycle_s = recent_seconds[-1] if recent_seconds else None
    latest = points[-1]
    at_terminal = _is_certified_terminal_point(latest)
    recent_levels = [float(p["level"]) for p in recent]
    level_varies = (max(recent_levels) - min(recent_levels)) > 0.5 if recent_levels else False
    hist_best = None
    if at_terminal:
        term_fits = [float(p["fitness"]) for p in points if int(p.get("level") or 0) >= _terminal_depth()]
        if term_fits:
            hist_best = max(term_fits)
    lbest_contaminated = bool(
        hist_best is not None
        and recent_median is not None
        and hist_best > max(recent_median * 2.5, recent_median + 200)
    )
    return {
        "categories": [str(points[i]["cycle"]) for i in indices],
        "fitness": _series_at_indices(fitness_sm, indices),
        "level": _series_at_indices(level_raw, indices),
        "fitness_at_level": _series_at_indices(fal_sm, indices),
        "epi": _series_at_indices(epi_sm, indices),
        "momentum": _series_at_indices(mom_sm, indices),
        "words_full": _series_at_indices(words_raw, indices),
        "recent_categories": [str(p["cycle"]) for p in recent],
        "recent_words": [round(w, 1) for w in recent_words],
        "recent_fitness": [round(v, 2) for v in recent_fitness],
        "recent_seconds": [round(v, 1) for v in recent_seconds],
        "recent_length_eff": [round(v, 2) for v in recent_length],
        "recent_epi": [round(v, 2) for v in recent_epi],
        "recent_median_fitness": round(recent_median, 2) if recent_median is not None else None,
        "median_cycle_s": round(median_cycle, 1) if median_cycle is not None else None,
        "last_cycle_s": round(last_cycle_s, 1) if last_cycle_s is not None else None,
        "at_terminal": at_terminal,
        "level_varies": level_varies,
        "show_words": any(w > 0 for w in recent_words),
        "recent_window": recent_n,
        "recent_start_cycle": recent[0]["cycle"],
        "recent_end_cycle": recent[-1]["cycle"],
        "historical_level_best": round(hist_best, 1) if hist_best is not None else None,
        "lbest_contaminated": lbest_contaminated,
        "tail_rows": [
            {
                "cycle": int(p["cycle"]),
                "words": int(p.get("champion_words") or 0),
                "fitness": round(float(p["fitness"]), 2),
                "epi": round(float(p["epi"]), 2),
                "seconds": recent_seconds[len(recent_seconds) - len(recent[-8:]) + i],
            }
            for i, p in enumerate(recent[-8:])
        ],
    }


def _canvas_fingerprint(summary: dict[str, Any]) -> str:
    latest = summary.get("latest") or {}
    comps = latest.get("components") or {}
    return json.dumps(
        {
            "count": summary.get("count"),
            "latest_epi": latest.get("epi"),
            "latest_fitness": latest.get("fitness"),
            "latest_words": latest.get("champion_words"),
            "peak_level": summary.get("peak_level"),
            "length_efficiency": comps.get("length_efficiency"),
            "prom_raw": summary.get("prom_raw"),
            "heldout_pct": summary.get("heldout_pct"),
            "template": CANVAS_TEMPLATE_VERSION,
        },
        sort_keys=True,
    )


def _canvas_write_needed(summary: dict[str, Any], *, force: bool = False) -> bool:
    if force:
        return True
    if not os.path.isfile(CANVAS_PATH):
        return True
    if _canvas_on_disk_version() != CANVAS_TEMPLATE_VERSION:
        return True
    state = _load_state()
    fingerprint = _canvas_fingerprint(summary)
    return state.get("last_canvas_fingerprint") != fingerprint


def append_live(quiet: bool = False) -> int:
    """Incremental update from new bytes in multi_cycle_results.jsonl."""
    state = _load_state()
    offset = int(state.get("results_byte_offset", 0))
    new_rows, new_offset = _read_new_result_rows(offset)
    if not new_rows:
        _write_heartbeat("idle", "no new rows", watcher=_IN_WATCH_LOOP)
        return 0

    existing = _load_trajectory_points()
    level_median = _level_median_from_points(existing)
    level_running_best = {int(k): float(v) for k, v in (state.get("level_running_best") or {}).items()}
    level_running_best_verify = {int(k): float(v) for k, v in (state.get("level_running_best_verify") or {}).items()}
    level_running_best_words = {int(k): int(v) for k, v in (state.get("level_running_best_words") or {}).items()}
    start_len = len(existing)
    points = _extend_points(
        existing,
        new_rows,
        level_median,
        level_running_best,
        level_running_best_verify,
        level_running_best_words,
    )
    _apply_epi_momentum(points)
    _append_trajectory_lines(points, start_len)

    state["results_byte_offset"] = new_offset
    state["results_lines"] = int(state.get("results_lines", 0)) + len(new_rows)
    state["level_running_best"] = level_running_best
    state["level_running_best_verify"] = level_running_best_verify
    state["level_running_best_words"] = level_running_best_words
    state["last_cycle"] = points[-1]["cycle"]
    _save_state(state)

    summary = _write_summary(points)
    _write_metric_doc()
    _write_canvas(points, summary)
    _write_heartbeat("live", f"+{len(new_rows)} cycles", watcher=_IN_WATCH_LOOP)
    if not quiet:
        print(f"  Appended {len(new_rows)} cycles -> {len(points)} total (EPI {points[-1]['epi']})")
    return len(new_rows)


def _trajectory_needs_epi_rebuild(points: list[dict[str, Any]], *, tail: int = 40) -> bool:
    """Detect append rows written by an older EPI formula (missing/wrong terminal bump)."""
    if not points:
        return False
    for p in points[-tail:]:
        level = int(p.get("level") or 0)
        verify = float(p.get("verify_score") or 0.0)
        if not _certified_terminal_epi_row(level, verify):
            continue
        expected = _terminal_fitness_epi_bump(float(p.get("fitness") or 0.0))
        if expected <= 0:
            continue
        bump = (p.get("components") or {}).get("terminal_fitness_bump")
        if bump is None or abs(float(bump) - expected) > 0.5:
            return True
    return False


def watch_loop() -> None:
    global _IN_WATCH_LOOP
    _stop_stale_watcher()
    if _trajectory_needs_epi_rebuild(_load_trajectory_points()):
        rebuild()
    os.makedirs(AGENTS, exist_ok=True)
    with open(WATCHER_PID_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    _IN_WATCH_LOOP = True
    try:
        _write_heartbeat("watch_started", watcher=True)
        script_mtime_at_start = os.path.getmtime(__file__)
        print(
            f"Trajectory watcher pid={os.getpid()} interval={WATCH_INTERVAL}s "
            f"smooth={SMOOTH_WINDOW} canvas={CANVAS_TEMPLATE_VERSION}"
        )
        while True:
            if os.path.getmtime(__file__) != script_mtime_at_start:
                _write_heartbeat(
                    "watch_stale_exit",
                    "progress_trajectory.py changed — restart watcher",
                    watcher=True,
                )
                break
            try:
                append_live(quiet=True)
                if _trajectory_needs_epi_rebuild(_load_trajectory_points()):
                    rebuild()
            except Exception as extra:  # noqa: BLE001
                _write_heartbeat("error", str(extra), watcher=True)
            time.sleep(WATCH_INTERVAL)
    finally:
        _IN_WATCH_LOOP = False


def _write_summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        summary = {"updated_at": _utc_now(), "count": 0}
    else:
        epi_series = [p["epi"] for p in points]
        level_series = [p["level"] for p in points]
        window = min(50, len(epi_series))
        sustained_peak = 0
        for p in points:
            if float(p.get("fitness", 0)) >= 80 and float(p.get("verify_score", 0)) >= 0.7:
                sustained_peak = max(sustained_peak, int(p.get("level", 0)))
        summary = {
            "updated_at": _utc_now(),
            "count": len(points),
            "latest": points[-1],
            "peak_epi": max(epi_series),
            "peak_level": max(level_series),
            "sustained_peak_level": sustained_peak,
            "epi_start": epi_series[0],
            "epi_end": epi_series[-1],
            "epi_delta": round(epi_series[-1] - epi_series[0], 2),
            "epi_rolling_50": _rolling_mean(epi_series, window)[-1],
            "fitness_start": points[0]["fitness"],
            "fitness_end": points[-1]["fitness"],
            "fitness_at_level_end": points[-1]["components"].get("fitness_at_level"),
            "epi_momentum_end": points[-1].get("epi_momentum"),
            "level_at_start": level_series[0],
            "level_at_end": level_series[-1],
            "plateau_fix_date": PLATEAU_FIX_DATE,
            "metric": "Evolution Progress Index (EPI)",
            "weights": {
                "curriculum_depth": 0.38,
                "verify_quality": 0.22,
                "fitness_momentum": 0.15,
                "generation_throughput": 0.10,
                "archive_learning": 0.10,
                "gpu_health": 0.05,
            },
        }
        ops = _live_ops(points[-1])
        summary["prom_raw"] = ops.get("prom_raw")
        summary["heldout_pct"] = ops.get("heldout_pct")
        summary["scan_floor_words"] = ops.get("scan_floor_words")
        summary["cycles_at_level"] = ops.get("cycles_at_level")
    os.makedirs(AGENTS, exist_ok=True)
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _write_trajectory_jsonl(points: list[dict[str, Any]]) -> None:
    os.makedirs(AGENTS, exist_ok=True)
    with open(TRAJECTORY, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in points)


def _write_metric_doc() -> None:
    text = f"""# Evolution Progress Index (EPI)

Single composite score (0–100) for uncapped self-improvement trajectory.

## Formula

```
EPI = 100 × (
  0.38 × curriculum_depth     # final_level / EE_MAX_LEVEL (default 30)
+ 0.22 × verify_quality       # champion_verify_score; L10+ normalized to 0.875 cap
+ 0.28 × fitness_at_level     # vs running best at current level (plateau signal)
+ 0.12 × fitness_momentum     # vs median fitness at level
+ 0.08 × generation_throughput
+ 0.06 × archive_learning
+ 0.04 × gpu_health
)
```

At levels **&lt; 8**, weights favor curriculum depth (38%) and standard verify/momentum split.

At certified L30 the canvas is an **operations monitor**, not a curriculum-progress chart:
- Primary series are champion **words**, **loop fitness**, **seconds/cycle**, and **held-out / promotion-raw**.
- Recent window is `{RECENT_WINDOW}` cycles at full resolution (`EE_TRAJECTORY_RECENT_WINDOW`).
- Fitness Y-axis uses 5th–95th percentile bounds so a historical spike (e.g. 6741) cannot flatten live ~300–360 movement.
- `fitness_at_level` vs all-time L-best is omitted when that best is an outlier relative to the recent median.

## Artifacts

| File | Purpose |
|------|---------|
| `.agents/progress_trajectory.jsonl` | One row per evolution cycle |
| `.agents/progress_trajectory_summary.json` | Latest snapshot + deltas |
| `canvases/evolution-progress-trajectory.canvas.tsx` | Visual trajectory |

## Commands

```bash
python scripts/progress_trajectory.py --rebuild   # after bulk evolution
python scripts/progress_trajectory.py --append    # after each cycle (auto from run_multi_cycle)
python scripts/progress_trajectory.py --watch     # live background watcher
python scripts/progress_trajectory.py --record    # tick sync (append or rebuild)
```

Charts use a **{SMOOTH_WINDOW}-cycle rolling mean** (`EE_TRAJECTORY_SMOOTH_WINDOW`, default 25) on fitness/EPI.
Watcher interval: `EE_TRAJECTORY_WATCH_SEC` (default 12s).
"""
    with open(METRIC_DOC, "w", encoding="utf-8") as f:
        f.write(text)


def _chart_bounds(
    values: list[float],
    pad: float = 2.0,
    cap_max: float | None = 100.0,
    *,
    allow_negative: bool = False,
) -> tuple[float, float]:
    if not values:
        return 0.0, 100.0
    lo = min(values)
    hi = max(values)
    y_min = lo - pad
    if not allow_negative:
        y_min = max(0.0, y_min)
    y_max = hi + pad
    if cap_max is not None:
        y_max = min(cap_max, y_max)
    if y_max <= y_min:
        y_max = y_min + pad
    return round(y_min, 1), round(y_max, 1)


def _jsx_line_card(
    title: str,
    categories_expr: str,
    series_name: str,
    data_expr: str,
    tone: str,
    height: int,
    y_min: float,
    y_max: float,
    *,
    suffix: str | None = None,
    trailing: str | None = None,
    reference_zero: bool = False,
) -> str:
    trailing_xml = f" trailing={{{trailing}}}" if trailing else ""
    suffix_attr = f'\n            valueSuffix="{suffix}"' if suffix else ""
    refs = ""
    if reference_zero:
        refs = '\n            referenceLines={[{ value: 0, label: "baseline", tone: "neutral" }]}'
    return f"""      <Card>
        <CardHeader{trailing_xml}>{title}</CardHeader>
        <CardBody style={{{{ paddingTop: 4, paddingBottom: 4 }}}}>
          <LineChart
            categories={{{categories_expr}}}
            series={{[{{ name: "{series_name}", data: {data_expr}, tone: "{tone}" }}]}}
            height={{{height}}}
            beginAtZero={{false}}
            yMin={{{y_min}}}
            yMax={{{y_max}}}{suffix_attr}{refs}
          />
        </CardBody>
      </Card>"""


def _write_canvas(points: list[dict[str, Any]], summary: dict[str, Any], *, force: bool = False) -> None:
    _stop_stale_watcher()
    latest = summary.get("latest") or {}
    ops = _live_ops(latest)
    summary = {
        **summary,
        "prom_raw": ops.get("prom_raw"),
        "heldout_pct": ops.get("heldout_pct"),
    }
    if not _canvas_write_needed(summary, force=force):
        return
    if not points:
        return
    chart = _prepare_chart_data(points)
    categories = chart["categories"]
    epi_data = chart["epi"]
    momentum_data = chart["momentum"]
    level_data = chart["level"]
    fitness_at_level = chart["fitness_at_level"]
    recent_words = chart["recent_words"]
    recent_fitness = chart["recent_fitness"]
    recent_seconds = chart["recent_seconds"]
    recent_length = chart["recent_length_eff"]

    epi_y_min, epi_y_max = _chart_bounds(epi_data, pad=3.0, cap_max=None)
    mom_y_min, mom_y_max = _chart_bounds(momentum_data, pad=1.0, cap_max=None, allow_negative=True)
    word_y_min = max(0.0, min(recent_words) - 1) if recent_words else 0.0
    word_y_max = (max(recent_words) + 1) if recent_words else 2.0
    if word_y_max <= word_y_min:
        word_y_max = word_y_min + 2.0
    fit_y_min, fit_y_max = (
        _robust_bounds(recent_fitness, allow_negative=False, include_latest=True) if recent_fitness else (0.0, 1.0)
    )
    sec_y_min, sec_y_max = (
        _robust_bounds(
            [s for s in recent_seconds if s >= 1.0] or recent_seconds,
            allow_negative=False,
            include_latest=False,
        )
        if recent_seconds
        else (0.0, 1.0)
    )
    len_y_min, len_y_max = _chart_bounds(recent_length, pad=2.0, cap_max=100.0) if recent_length else (0.0, 100.0)
    level_y_min, level_y_max = (
        _chart_bounds([float(x) for x in level_data], pad=1.0, cap_max=None) if level_data else (0.0, 32.0)
    )
    fal_y_min, fal_y_max = _chart_bounds(fitness_at_level, pad=5.0) if fitness_at_level else (0.0, 100.0)

    latest_fit = latest.get("fitness")
    vs_recent = None
    if chart.get("recent_median_fitness") and latest_fit and chart["recent_median_fitness"] > 0:
        vs_recent = round(100.0 * float(latest_fit) / float(chart["recent_median_fitness"]), 1)
    length_eff = (latest.get("components") or {}).get("length_efficiency")

    meta = {
        "updated_at": summary.get("updated_at", _utc_now()),
        "live_at": _utc_now(),
        "count": summary.get("count", len(points)),
        "latest_epi": latest.get("epi"),
        "latest_fitness": latest_fit,
        "latest_words": ops.get("words"),
        "peak_level": summary.get("peak_level"),
        "prom_raw": ops.get("prom_raw"),
        "heldout_pct": ops.get("heldout_pct"),
        "vs_recent_median_pct": vs_recent,
        "recent_median_fitness": chart.get("recent_median_fitness"),
        "median_cycle_s": chart.get("median_cycle_s"),
        "last_cycle_s": chart.get("last_cycle_s"),
        "cycles_at_level": ops.get("cycles_at_level"),
        "scan_floor_words": ops.get("scan_floor_words"),
        "length_efficiency": length_eff,
        "smooth_window": SMOOTH_WINDOW,
        "recent_window": chart.get("recent_window"),
        "at_terminal": chart.get("at_terminal"),
        "lbest_contaminated": chart.get("lbest_contaminated"),
        "historical_level_best": chart.get("historical_level_best"),
        "status_task": ops.get("status_task"),
        "epi_y_min": epi_y_min,
        "epi_y_max": epi_y_max,
    }

    pills = []
    if chart.get("at_terminal"):
        pills.append('<Pill active size="sm">L30 certified</Pill>')
    if ops.get("scan_floor_words"):
        pills.append(f'<Pill size="sm">{ops["scan_floor_words"]}w scan floor</Pill>')
    if ops.get("cycles_at_level"):
        pills.append(f'<Pill size="sm">{ops["cycles_at_level"]} cycles at L{ops.get("level") or "—"}</Pill>')
    pills_jsx = "\n      ".join(pills)

    primary = [f'<Stat value="{ops.get("words") or "—"}" label="Champion words" tone="info" />']
    if ops.get("prom_raw") is not None:
        primary.append(f'<Stat value="{ops["prom_raw"]}" label="Promotion raw" />')
    if ops.get("heldout_pct") is not None:
        held_tone = ' tone="success"' if float(ops["heldout_pct"]) >= 99 else ""
        primary.append(f'<Stat value="{ops["heldout_pct"]}%" label="Held-out L30"{held_tone} />')
    if chart.get("median_cycle_s") is not None:
        primary.append(f'<Stat value="{chart["median_cycle_s"]}s" label="Median cycle" />')
    secondary = [f'<Stat value="{latest.get("epi") if latest.get("epi") is not None else "—"}" label="EPI" />']
    if latest_fit is not None:
        secondary.append(f'<Stat value="{latest_fit}" label="Loop fitness" />')
    if vs_recent is not None:
        secondary.append(f'<Stat value="{vs_recent}%" label="vs recent median" tone="success" />')
    if length_eff is not None:
        secondary.append(f'<Stat value="{length_eff}%" label="Length efficiency" />')

    callout_jsx = ""
    if chart.get("lbest_contaminated"):
        callout_jsx = (
            f'    <Callout tone="warning" title="Historical L-best is a spike, not the champion">\n'
            f"      All-time L30 fitness is {chart['historical_level_best']} from a short-program spike. "
            f"Recent-window median is {chart['recent_median_fitness']}. "
            f"The live compact champion is {ops.get('words')} words; vs-L-best is omitted because it reads as a false regression.\n"
            f"    </Callout>"
        )

    recent_cards: list[str] = []
    if chart.get("show_words") and recent_words:
        recent_cards.append(
            _jsx_line_card(
                "Champion length (words, unsmoothed)",
                "RECENT_CYCLES",
                "Words",
                "RECENT_WORDS",
                "info",
                130,
                word_y_min,
                word_y_max,
                trailing=f'"{ops.get("words")}w now"',
            )
        )
    if recent_fitness:
        recent_cards.append(
            _jsx_line_card(
                f"Loop fitness ({SMOOTH_WINDOW}-cycle mean, robust Y)",
                "RECENT_CYCLES",
                "Fitness",
                "RECENT_FITNESS",
                "success",
                130,
                fit_y_min,
                fit_y_max,
            )
        )
    if recent_seconds:
        recent_cards.append(
            _jsx_line_card(
                "Seconds per cycle",
                "RECENT_CYCLES",
                "Seconds",
                "RECENT_SECONDS",
                "warning",
                110,
                sec_y_min,
                sec_y_max,
                suffix=" s",
            )
        )
    if chart.get("at_terminal") and recent_length:
        recent_cards.append(
            _jsx_line_card(
                "Length efficiency (%)",
                "RECENT_CYCLES",
                "Length efficiency",
                "RECENT_LENGTH",
                "neutral",
                110,
                len_y_min,
                len_y_max,
                suffix="%",
            )
        )
    elif chart.get("level_varies") and level_data:
        recent_cards.append(
            _jsx_line_card(
                "Curriculum level by cycle",
                "CATEGORIES",
                "Level",
                "LEVEL_SERIES",
                "warning",
                110,
                level_y_min,
                level_y_max,
            )
        )
    elif fitness_at_level:
        recent_cards.append(
            _jsx_line_card(
                "Fitness vs level best (% smoothed)",
                "CATEGORIES",
                "% of L-best",
                "FITNESS_AT_LEVEL_SERIES",
                "info",
                110,
                fal_y_min,
                fal_y_max,
                suffix="%",
            )
        )

    history_cards = [
        _jsx_line_card(
            "EPI (full run, zoomed Y)",
            "CATEGORIES",
            "EPI",
            "EPI_SERIES",
            "info",
            115,
            epi_y_min,
            epi_y_max,
        ),
        _jsx_line_card(
            "EPI momentum (Δ vs 50-cycle mean)",
            "CATEGORIES",
            "Δ vs rolling",
            "MOMENTUM_SERIES",
            "neutral",
            115,
            mom_y_min,
            mom_y_max,
            reference_zero=True,
        ),
    ]

    def _grid(cards: list[str]) -> str:
        if not cards:
            return ""
        inner = "\n".join(cards)
        return f"    <Grid columns={{2}} gap={{8}}>\n{inner}\n    </Grid>"

    recent_grid = _grid(recent_cards)
    history_grid = _grid(history_cards)
    recent_heading = ""
    if recent_cards:
        start_c = chart.get("recent_start_cycle")
        end_c = chart.get("recent_end_cycle")
        recent_heading = (
            f"    <H3 style={{{{ margin: 0 }}}}>Last {chart.get('recent_window')} cycles (C{start_c}–C{end_c})</H3>"
        )
    table_jsx = ""
    tail_rows = chart.get("tail_rows") or []
    if tail_rows:
        js_rows = json.dumps(
            [
                [
                    str(r["cycle"]),
                    str(r["words"]),
                    f"{r['fitness']:.2f}",
                    f"{r['epi']:.2f}",
                    f"{r['seconds']:.0f}",
                ]
                for r in tail_rows
            ]
        )
        table_jsx = f"""    <H3 style={{{{ margin: 0 }}}}>Latest cycles</H3>
    <Table
      headers={{["Cycle", "Words", "Fitness", "EPI", "s/cycle"]}}
      rows={{{js_rows}}}
      columnAlign={{["right", "right", "right", "right", "right"]}}
      striped
    />"""

    imports = [
        "Callout",
        "Card",
        "CardBody",
        "CardHeader",
        "Grid",
        "H2",
        "H3",
        "LineChart",
        "Pill",
        "Row",
        "Stack",
        "Stat",
        "Table",
        "Text",
    ]
    if callout_jsx == "":
        imports = [name for name in imports if name != "Callout"]
    if not pills:
        imports = [name for name in imports if name != "Pill"]
    if not table_jsx:
        imports = [name for name in imports if name != "Table"]

    used_jsx = f"{recent_grid}\n{history_grid}\n{table_jsx}\n{callout_jsx}\n{pills_jsx}"
    const_decls = [
        f"const META = {json.dumps(meta)};",
        f"const CATEGORIES = {json.dumps(categories)};",
        f"const EPI_SERIES = {json.dumps(epi_data)};",
        f"const MOMENTUM_SERIES = {json.dumps(momentum_data)};",
    ]
    optional_consts = [
        ("LEVEL_SERIES", level_data),
        ("FITNESS_AT_LEVEL_SERIES", fitness_at_level),
        ("RECENT_CYCLES", chart["recent_categories"]),
        ("RECENT_WORDS", recent_words),
        ("RECENT_FITNESS", recent_fitness),
        ("RECENT_SECONDS", recent_seconds),
        ("RECENT_LENGTH", recent_length),
    ]
    for name, values in optional_consts:
        if name in used_jsx:
            const_decls.append(f"const {name} = {json.dumps(values)};")
    consts_block = "\n".join(const_decls)

    canvas = f"""// @generated evolution-progress-trajectory canvas-template:{CANVAS_TEMPLATE_VERSION}
import {{
  {", ".join(imports)},
}} from "cursor/canvas";

{consts_block}
const LIVE_SHORT = META.live_at ? META.live_at.slice(5, 16).replace("T", " ") : "—";
const CYCLE_RANGE = CATEGORIES.length > 0
  ? `cycles ${{CATEGORIES[0]}}–${{CATEGORIES[CATEGORIES.length - 1]}}`
  : "";

export default function EvolutionProgressTrajectory() {{
  return (
  <Stack gap={{10}} style={{{{ padding: 8 }}}}>
    <Row gap={{8}} align="center" wrap>
      <H2 style={{{{ margin: 0 }}}}>Evolution Progress</H2>
      {pills_jsx}
      <Text size="small" tone="secondary">
        {{META.count}} cycles · recent {{META.recent_window}} · {{LIVE_SHORT}}
      </Text>
    </Row>
    <Card variant="borderless">
      <CardBody style={{{{ padding: "2px 0" }}}}>
        <Row gap={{16}} wrap>
            {" ".join(primary)}
        </Row>
        <Row gap={{16}} wrap style={{{{ marginTop: 8 }}}}>
            {" ".join(secondary)}
        </Row>
      </CardBody>
    </Card>
{callout_jsx}
{recent_heading}
{recent_grid}
    <H3 style={{{{ margin: 0 }}}}>Full run</H3>
{history_grid}
{table_jsx}
    <Text size="small" tone="secondary">
      Source: .agents/progress_trajectory.jsonl + progress_controller_state.json · {{CYCLE_RANGE}} · words/held-out/prom-raw are the live L30 axes. Fitness Y uses 5th–95th percentile so spike cycles cannot flatten the recent band. See .agents/PROGRESS_METRIC.md.
    </Text>
  </Stack>
  );
}}
"""
    os.makedirs(CANVAS_DIR, exist_ok=True)
    tmp_path = f"{CANVAS_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(canvas)
    os.replace(tmp_path, CANVAS_PATH)
    state = _load_state()
    state["last_canvas_fingerprint"] = _canvas_fingerprint(summary)
    _save_state(state)


def rebuild() -> int:
    if _stop_stale_watcher():
        print("  Stopped stale trajectory watcher (old canvas template)")
    print("Streaming results (regex extract)...")
    rows = _stream_result_rows()
    print(f"  Parsed {len(rows)} cycle rows")
    points = _build_trajectory_points(rows)
    _write_trajectory_jsonl(points)
    summary = _write_summary(points)
    _write_metric_doc()
    _write_canvas(points, summary, force=True)

    level_running_best: dict[int, float] = {}
    level_running_best_verify: dict[int, float] = {}
    level_running_best_words: dict[int, int] = {}
    for p in points:
        lvl = int(p["level"])
        fit = float(p["fitness"])
        verify = float(p.get("verify_score") or 0.0)
        words = int(p.get("champion_words") or 0)
        level_running_best[lvl] = max(level_running_best.get(lvl, fit), fit)
        level_running_best_verify[lvl] = max(level_running_best_verify.get(lvl, verify), verify)
        if words > 0:
            prev_w = level_running_best_words.get(lvl, words)
            level_running_best_words[lvl] = min(prev_w, words) if prev_w > 0 else words
    results_size = os.path.getsize(RESULTS) if os.path.isfile(RESULTS) else 0
    _save_state(
        {
            "results_byte_offset": results_size,
            "results_lines": len(rows),
            "level_running_best": level_running_best,
            "level_running_best_verify": level_running_best_verify,
            "level_running_best_words": level_running_best_words,
            "last_cycle": points[-1]["cycle"] if points else 0,
        }
    )
    _write_heartbeat("rebuilt", f"{len(points)} points", watcher=False)
    print(f"  Wrote {TRAJECTORY}")
    print(f"  Wrote {SUMMARY}")
    print(f"  Wrote {CANVAS_PATH}")
    if points:
        print(f"  EPI: {points[0]['epi']} -> {points[-1]['epi']} (level {points[-1]['level']})")
    return 0


def sync_trajectory() -> int:
    """Fast path: incremental append; full rebuild if state missing or stale."""
    state = _load_state()
    results_size = os.path.getsize(RESULTS) if os.path.isfile(RESULTS) else 0
    trajectory_count = len(_load_trajectory_points())
    state_lines = int(state.get("results_lines", 0))
    if (
        state.get("results_byte_offset", 0) > results_size
        or trajectory_count == 0
        or state_lines < trajectory_count - 5
    ):
        return rebuild()
    added = append_live()
    if added == 0 and trajectory_count == 0:
        return rebuild()
    if _trajectory_needs_epi_rebuild(_load_trajectory_points()):
        return rebuild()
    return 0


def record_tick() -> int:
    return sync_trajectory()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evolution Progress Index trajectory")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild trajectory from results")
    parser.add_argument("--append", action="store_true", help="Append new result lines only")
    parser.add_argument("--watch", action="store_true", help="Live watch loop (smooth canvas refresh)")
    parser.add_argument("--record", action="store_true", help="Sync trajectory (append or rebuild)")
    args = parser.parse_args()
    if args.watch:
        watch_loop()
        return 0
    if args.rebuild:
        return rebuild()
    if args.append:
        return 0 if append_live() >= 0 else 1
    if args.record:
        return sync_trajectory()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
