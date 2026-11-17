"""Cross-cycle level stall tracking — nudge start level when stuck."""

from __future__ import annotations

import json
import os
from typing import Any

from evolution.curriculum_advance import curriculum_verify_threshold, nudge_fitness_floor, nudge_fitness_passes


def _state_path(project_root: str) -> str:
    return os.path.join(project_root, ".agents", "level_stall_state.json")


def load_state(project_root: str) -> dict[str, Any]:
    path = _state_path(project_root)
    if not os.path.isfile(path):
        return {"level": -1, "cycles": 0, "verify": 0.0}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"level": -1, "cycles": 0, "verify": 0.0}


def save_state(project_root: str, state: dict[str, Any]) -> None:
    agents = os.path.join(project_root, ".agents")
    os.makedirs(agents, exist_ok=True)
    with open(_state_path(project_root), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def update_and_maybe_nudge_start_level(
    project_root: str,
    final_level: int,
    verify_score: float,
    final_fitness: float = 0.0,
    max_level: int = 30,
    stall_cycles_before_nudge: int = 3,
    min_fitness: float | None = None,
) -> int | None:
    """Return start_level+1 when stuck at level with adequate verify and fitness."""
    state = load_state(project_root)
    prev_level = int(state.get("level", -1))
    if final_level != prev_level:
        state = {"level": final_level, "cycles": 1, "verify": verify_score}
    else:
        state["cycles"] = int(state.get("cycles", 0)) + 1
        state["verify"] = verify_score
    save_state(project_root, state)

    cycles = int(state["cycles"])
    if cycles < stall_cycles_before_nudge or final_level >= max_level:
        return None

    stall = max(3, cycles)
    threshold = curriculum_verify_threshold(final_level, stall, 3)
    fitness_floor = min_fitness if min_fitness is not None else nudge_fitness_floor(final_level)
    fitness_ok = final_fitness >= fitness_floor or nudge_fitness_passes(final_level, final_fitness, verify_score)
    if verify_score >= threshold and fitness_ok:
        return final_level + 1
    return None
