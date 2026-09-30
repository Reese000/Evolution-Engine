"""Detect metric flatlines and halt pointless evolution cycles."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from evolution.progress_controller import ProgressController

DEVELOPMENT_BLOCKERS = [
    "Champion programs timeout (fit=-25) or partial-verify without task completion at L17+",
    "Mutation/selection cannot fix register-level failures (e.g. L17 x20, L19 memory ops)",
    "Retrain loop not improving seed model for high-curriculum tasks",
    "End-of-run level inflation outpaces real verify quality (hollow curriculum)",
    "C3 certify gate requires sustained verify — evolution metrics do not correlate",
]


@dataclass
class FlatlineReport:
    halted: bool
    window: int
    reasons: list[str] = field(default_factory=list)
    epi_range: float = 0.0
    fitness_range: float = 0.0
    verify_range: float = 0.0
    level_range: int = 0
    c3_passes: int = 0
    cycle_start: int = 0
    cycle_end: int = 0
    development_blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_trajectory_points(agents_dir: str, window: int) -> list[dict[str, Any]]:
    traj = os.path.join(agents_dir, "progress_trajectory.jsonl")
    if os.path.isfile(traj):
        rows: list[dict[str, Any]] = []
        with open(traj, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows[-window:]

    results = os.path.join(os.path.dirname(agents_dir), "multi_cycle_results.jsonl")
    if not os.path.isfile(results):
        return []

    from scripts.progress_trajectory import _fast_extract

    rows = []
    with open(results, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = _fast_extract(line)
            if row and "error" not in row and row.get("cycle") is not None:
                rows.append(row)
    return rows[-window:]


def assess_flatline(
    project_root: str,
    window: int | None = None,
    min_window: int | None = None,
) -> FlatlineReport:
    """Return halted=True when progress metrics are flat across the window."""
    window = window or int(os.environ.get("EE_FLATLINE_WINDOW", "40"))
    min_window = min_window or int(os.environ.get("EE_FLATLINE_MIN_WINDOW", "30"))
    agents = os.path.join(project_root, ".agents")

    points = _load_trajectory_points(agents, window)
    report = FlatlineReport(halted=False, window=window, development_blockers=list(DEVELOPMENT_BLOCKERS))

    if len(points) < min_window:
        report.reasons.append(f"insufficient history ({len(points)} < {min_window})")
        return report

    epis = [float(p.get("epi", 0)) for p in points]
    fitnesses = [float(p.get("fitness", p.get("final_fitness", 0))) for p in points]
    verifies = [float(p.get("verify_score", p.get("champion_verify_score", 0))) for p in points]
    levels = [int(p.get("level", p.get("final_level", 0))) for p in points]
    cycles = [int(p.get("cycle", 0)) for p in points]

    report.epi_range = max(epis) - min(epis) if epis else 0.0
    report.fitness_range = max(fitnesses) - min(fitnesses) if fitnesses else 0.0
    report.verify_range = max(verifies) - min(verifies) if verifies else 0.0
    report.level_range = (max(levels) - min(levels)) if levels else 0
    report.cycle_start = min(cycles) if cycles else 0
    report.cycle_end = max(cycles) if cycles else 0

    last10 = points[-10:]
    report.c3_passes = sum(
        1
        for p in last10
        if ProgressController.passes_certify_verify(
            int(p.get("level", p.get("final_level", 0))),
            float(p.get("verify_score", p.get("champion_verify_score", 0))),
        )
    )

    epi_delta = epis[-1] - epis[0] if len(epis) >= 2 else 0.0
    timeout_frac = sum(1 for f in fitnesses if f <= -25.0) / len(fitnesses)

    state_keys = {(lv, round(v, 2)) for lv, v in zip(levels, verifies)}
    unique_states = len(state_keys)

    tail_n = int(os.environ.get("EE_FLATLINE_TAIL", "15"))
    tail = points[-tail_n:] if len(points) >= tail_n else points
    tail_levels = [int(p.get("level", p.get("final_level", 0))) for p in tail]
    tail_verifies = [float(p.get("verify_score", p.get("champion_verify_score", 0))) for p in tail]
    tail_plateau = (
        len(tail) >= tail_n
        and (max(tail_levels) - min(tail_levels)) == 0
        and (max(tail_verifies) - min(tail_verifies)) < 0.06
    )

    last30 = points[-30:] if len(points) >= 30 else points
    c3_last30 = sum(
        1
        for p in last30
        if ProgressController.passes_certify_verify(
            int(p.get("level", p.get("final_level", 0))),
            float(p.get("verify_score", p.get("champion_verify_score", 0))),
        )
    )

    epi_flat = report.epi_range < float(os.environ.get("EE_FLATLINE_EPI_RANGE", "8"))
    verify_flat = report.verify_range < float(os.environ.get("EE_FLATLINE_VERIFY_RANGE", "0.15"))
    level_flat = report.level_range <= int(os.environ.get("EE_FLATLINE_LEVEL_RANGE", "2"))
    c3_stuck = report.c3_passes < int(os.environ.get("EE_FLATLINE_C3_MIN", "2"))
    epi_stalled = epi_delta < float(os.environ.get("EE_FLATLINE_EPI_DELTA", "3"))
    timeout_stuck = timeout_frac >= float(os.environ.get("EE_FLATLINE_TIMEOUT_FRAC", "0.6"))
    oscillation_trap = unique_states <= int(os.environ.get("EE_FLATLINE_MAX_STATES", "4"))

    if tail_plateau and c3_stuck:
        report.halted = True
        report.reasons.append(
            f"tail plateau: L{tail_levels[-1]} verify={tail_verifies[-1]:.3f} " f"unchanged for {len(tail)} cycles"
        )
        report.reasons.append(f"C3 stuck ({report.c3_passes}/10)")

    if len(last30) >= 30 and c3_last30 == 0:
        report.halted = True
        report.reasons.append(f"C3 0/{len(last30)} over last {len(last30)} cycles")

    if oscillation_trap and c3_stuck and level_flat and len(points) >= min_window:
        report.halted = True
        report.reasons.append(f"oscillation trap: {unique_states} unique (level,verify) states in {len(points)} cycles")

    if epi_flat and verify_flat and level_flat and c3_stuck and epi_stalled:
        report.halted = True
        report.reasons.append(
            f"EPI flat (range={report.epi_range:.1f}, delta={epi_delta:.1f}) over {len(points)} cycles"
        )
        report.reasons.append(f"verify flat (range={report.verify_range:.3f})")
        report.reasons.append(f"level flat (range={report.level_range})")
        report.reasons.append(f"C3 stuck ({report.c3_passes}/10)")

    if timeout_stuck and c3_stuck:
        report.halted = True
        report.reasons.append(f"timeout fitness in {timeout_frac:.0%} of window; C3 {report.c3_passes}/10")

    if report.halted and points:
        from evolution.terminal_stagnation import terminal_certified_plateau

        last = points[-1]
        level = int(last.get("level", last.get("final_level", 0)))
        verify = float(last.get("verify_score", last.get("champion_verify_score", 0)))
        fitness = float(last.get("fitness", last.get("final_fitness", 0)))
        if terminal_certified_plateau(level, verify, fitness):
            report.halted = False
            report.reasons.append("certified L30 terminal plateau: flat metrics expected; evolution continues")

    return report


def write_flatline_state(project_root: str, report: FlatlineReport) -> str:
    from datetime import datetime, timezone

    agents = os.path.join(project_root, ".agents")
    os.makedirs(agents, exist_ok=True)
    path = os.path.join(agents, "flatline_state.json")
    payload = {
        "halted_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "halted": report.halted,
        "report": report.to_dict(),
        "resume_requires": "development work on blockers; run scripts/flatline_guard.py --resume",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def apply_flatline_halt(project_root: str, report: FlatlineReport) -> None:
    """Record flatline advisory — does NOT pause evolution or agent loop."""
    write_flatline_state(project_root, report)
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    tick: dict[str, Any] = {}
    if os.path.isfile(tick_path):
        try:
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
        except (json.JSONDecodeError, OSError):
            tick = {}
    tick["flatline_halted"] = True
    tick["flatline_advisory"] = True
    tick["evolution_paused"] = False
    tick["loop_paused"] = False
    tick["status"] = "in_progress"
    tick["current_task"] = tick.get("current_task") or "flatline-advisory"
    tick["notes"] = "; ".join(report.reasons[:3])
    with open(tick_path, "w", encoding="utf-8") as f:
        json.dump(tick, f, indent=2)


def is_evolution_paused(project_root: str) -> bool:
    """True only on explicit user evolution pause — flatline never pauses evolution."""
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    if not os.path.isfile(tick_path):
        return False
    try:
        with open(tick_path, encoding="utf-8") as f:
            tick = json.load(f)
        return bool(tick.get("evolution_paused_user"))
    except (json.JSONDecodeError, OSError):
        return False


def _certified_terminal_plateau_active(project_root: str) -> bool:
    """True when max-depth champion is certified — flatline is expected, not failure."""
    state_path = os.path.join(project_root, ".agents", "progress_controller_state.json")
    try:
        if not os.path.isfile(state_path):
            return False
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    from evolution.terminal_stagnation import terminal_certified_plateau

    return terminal_certified_plateau(
        int(state.get("level", 0)),
        float(state.get("verify_score", 0.0)),
        float(state.get("raw_fitness", 0.0)),
    )


def is_flatline_halted(project_root: str) -> bool:
    path = os.path.join(project_root, ".agents", "flatline_state.json")
    if not os.path.isfile(path):
        return False
    if _certified_terminal_plateau_active(project_root):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("halted"))
    except (json.JSONDecodeError, OSError):
        return False


def clear_flatline_halt(project_root: str) -> None:
    path = os.path.join(project_root, ".agents", "flatline_state.json")
    if os.path.isfile(path):
        os.remove(path)
    tick_path = os.path.join(project_root, ".agents", "tick_state.json")
    if os.path.isfile(tick_path):
        try:
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
            tick["loop_paused"] = False
            tick["evolution_paused"] = False
            tick["evolution_paused_user"] = False
            tick["flatline_halted"] = False
            tick["flatline_advisory"] = False
            tick["status"] = "in_progress"
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump(tick, f, indent=2)
        except (json.JSONDecodeError, OSError):
            pass
