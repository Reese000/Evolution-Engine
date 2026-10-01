"""Unified progress controller — single authority for curriculum advancement and anti-plateau."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from evolution.curriculum_advance import (
    curriculum_fitness_gate,
    curriculum_verify_threshold,
    forced_advance_verify_floor,
    nudge_fitness_floor,
    nudge_fitness_passes,
    nudge_verify_passes,
)


class StallTier(Enum):
    NORMAL = auto()
    SOFT = auto()  # >= stall_threshold gens
    DEEP = auto()  # >= 2x stall_threshold
    ESCAPE = auto()  # >= 4x stall_threshold


class ProgressAction(Enum):
    ADVANCE_LEVEL = auto()
    REFRESH_TASK = auto()
    QD_RESET = auto()
    NUDGE_START_LEVEL = auto()


@dataclass
class ProgressState:
    level: int = 0
    generations_stalled: int = 0
    cycles_at_level: int = 0
    verify_score: float = 0.0
    raw_fitness: float = 0.0
    champion_fitness: float = 0.0
    stall_threshold: int = 2
    max_level: int = 30


@dataclass
class ActionPlan:
    actions: list[ProgressAction] = field(default_factory=list)
    nudge_start_level: int | None = None
    verify_threshold: float = 1.0
    fitness_gate_min: float = 110.0
    log_lines: list[str] = field(default_factory=list)


class ProgressController:
    """Owns all curriculum / plateau decisions for within-cycle and cross-cycle horizons."""

    STATE_FILE = "progress_controller_state.json"

    def __init__(self, project_root: str, stall_threshold: int = 2, max_level: int = 30):
        self.project_root = project_root
        self.state = ProgressState(stall_threshold=stall_threshold, max_level=max_level)
        self._log_fn: Callable[[str], None] | None = None
        self._load()

    def set_logger(self, log_fn: Callable[[str], None]) -> None:
        self._log_fn = log_fn

    def _log(self, msg: str) -> None:
        if self._log_fn:
            self._log_fn(msg)

    def _state_path(self) -> str:
        return os.path.join(self.project_root, ".agents", self.STATE_FILE)

    def _load(self) -> None:
        path = self._state_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.state.level = int(data.get("level", 0))
            self.state.cycles_at_level = int(data.get("cycles_at_level", 0))
            self.state.verify_score = float(data.get("verify_score", 0))
            self.state.raw_fitness = float(data.get("raw_fitness", 0))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    def save(self) -> None:
        agents = os.path.join(self.project_root, ".agents")
        os.makedirs(agents, exist_ok=True)
        path = self._state_path()
        merged: dict[str, Any] = {}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    merged = json.load(f)
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                merged = {}
        merged.update(
            {
                "level": self.state.level,
                "cycles_at_level": self.state.cycles_at_level,
                "verify_score": self.state.verify_score,
                "raw_fitness": self.state.raw_fitness,
                "generations_stalled": self.state.generations_stalled,
            }
        )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

    def reconcile_from_cycle_result(self, final_level: int, verify: float, fitness: float) -> None:
        """Persist completed cycle level — repairs stale writes from collapse recovery."""
        lvl = int(final_level)
        if lvl != self.state.level:
            self.state.level = lvl
            self.state.cycles_at_level = 1
        self.state.verify_score = float(verify)
        self.state.raw_fitness = float(fitness)
        self.save()

    def stall_tier(self) -> StallTier:
        s = self.state.generations_stalled
        t = self.state.stall_threshold
        if s >= 4 * t:
            return StallTier.ESCAPE
        if s >= 2 * t:
            return StallTier.DEEP
        if s >= t:
            return StallTier.SOFT
        return StallTier.NORMAL

    def verify_threshold(self, level: int | None = None) -> float:
        lvl = level if level is not None else self.state.level
        return curriculum_verify_threshold(lvl, self.state.generations_stalled, self.state.stall_threshold)

    def fitness_gate_passes(self, raw_score: float | None = None, level: int | None = None) -> bool:
        lvl = level if level is not None else self.state.level
        score = raw_score if raw_score is not None else self.state.raw_fitness
        return curriculum_fitness_gate(lvl, self.state.generations_stalled, score, self.state.stall_threshold)

    def _advance_verify_threshold(self, level: int) -> float:
        """Verify bar for curriculum advance — aligned with C3 certify gate at L8+."""
        lvl = int(level)
        if lvl >= 8 and os.environ.get("EE_CERTIFY_STRICT_ADVANCE", "1") != "0":
            return self.certify_verify_threshold(lvl)
        eff_stall = max(self.state.generations_stalled, self.state.stall_threshold)
        return curriculum_verify_threshold(lvl, eff_stall, self.state.stall_threshold)

    def champion_verify_accepts(self, verify_score: float, level: int | None = None) -> bool:
        if verify_score >= 1.0:
            return True
        lvl = level if level is not None else self.state.level
        if lvl >= 17:
            return verify_score >= self.certify_verify_threshold(lvl)
        return verify_score >= self._advance_verify_threshold(lvl)

    def can_advance_level(
        self,
        verify_score: float,
        raw_fitness: float,
        level: int | None = None,
    ) -> bool:
        lvl = level if level is not None else self.state.level
        if lvl >= self.state.max_level:
            return False
        if verify_score < self._advance_verify_threshold(lvl):
            return False
        return curriculum_fitness_gate(lvl, self.state.generations_stalled, raw_fitness, self.state.stall_threshold)

    def can_forced_escape(self, verify_score: float, raw_fitness: float, level: int | None = None) -> bool:
        if os.environ.get("EE_CERTIFY_STRICT_ADVANCE", "1") != "0":
            return False
        if self.stall_tier() != StallTier.ESCAPE:
            return False
        lvl = level if level is not None else self.state.level
        floor = forced_advance_verify_floor(lvl, self.state.generations_stalled, self.state.stall_threshold)
        if verify_score < floor:
            return False
        return self.fitness_gate_passes(raw_fitness, lvl)

    def sync_generation(self, generations_stalled: int, level: int, verify: float, raw: float, fitness: float) -> None:
        self.state.generations_stalled = generations_stalled
        self.state.level = level
        self.state.verify_score = verify
        self.state.raw_fitness = raw
        self.state.champion_fitness = fitness

    def on_generation_end(
        self, generations_stalled: int, level: int, verify: float, raw: float, fitness: float
    ) -> ActionPlan:
        """Within-cycle: decide per-generation anti-plateau actions."""
        self.sync_generation(generations_stalled, level, verify, raw, fitness)
        plan = ActionPlan(verify_threshold=self.verify_threshold(level))

        if generations_stalled >= self.state.stall_threshold:
            if self.can_advance_level(verify, raw, level):
                plan.actions.append(ProgressAction.ADVANCE_LEVEL)
            elif self.can_forced_escape(verify, raw, level):
                plan.actions.append(ProgressAction.ADVANCE_LEVEL)
                plan.log_lines.append(f"[Progress] Forced escape at stall={generations_stalled}")

        if generations_stalled >= 25 and generations_stalled % 10 == 0:
            plan.actions.append(ProgressAction.REFRESH_TASK)

        if generations_stalled >= 35 and ProgressAction.QD_RESET not in plan.actions:
            plan.actions.append(ProgressAction.QD_RESET)

        return plan

    def on_run_end(self, level: int, verify: float, raw: float, fitness: float) -> ActionPlan:
        """End of evolution run: quality-gated chain advance (max 3 steps)."""
        self.sync_generation(
            max(self.state.generations_stalled, self.state.stall_threshold), level, verify, raw, fitness
        )
        plan = ActionPlan()
        min_fitness = float(os.environ.get("EE_END_RUN_MIN_FITNESS", "80"))
        max_steps = int(os.environ.get("EE_END_RUN_MAX_ADVANCE", "3"))

        cur_level = level
        cur_verify, cur_raw, cur_fit = verify, raw, fitness
        for step in range(max_steps):
            if not curriculum_fitness_gate(
                cur_level, self.state.generations_stalled, cur_raw, self.state.stall_threshold
            ):
                break
            if not self.passes_certify_verify(cur_level, cur_verify):
                plan.log_lines.append(
                    f"[Progress] Run-end chain halt at L{cur_level}: verify {cur_verify:.3f} < certify bar"
                )
                break
            if cur_fit < min_fitness:
                plan.log_lines.append(f"[Progress] Run-end chain halt: fitness {cur_fit:.1f} < {min_fitness}")
                break
            plan.actions.append(ProgressAction.ADVANCE_LEVEL)
            plan.log_lines.append(f"[Progress] Run-end advance step {step + 1} L{cur_level}->L{cur_level + 1}")
            cur_level += 1
            if cur_level >= self.state.max_level:
                break
        return plan

    def on_cycle_end(self, final_level: int, verify: float, fitness: float) -> ActionPlan:
        """Cross-cycle: update counters and maybe nudge start level for next cycle."""
        plan = ActionPlan()
        if final_level != self.state.level:
            self.state.level = final_level
            self.state.cycles_at_level = 1
        else:
            self.state.cycles_at_level += 1
        self.state.verify_score = verify
        self.state.raw_fitness = fitness

        nudge_after = int(os.environ.get("EE_NUDGE_CYCLES", "3"))
        if final_level >= 21:
            nudge_after = max(
                nudge_after,
                int(os.environ.get("EE_DEPTH_ESCAPE_CYCLES", "8")),
            )

        if (
            self.state.cycles_at_level >= nudge_after
            and final_level < self.state.max_level
            and nudge_verify_passes(final_level, verify, self.state.cycles_at_level)
            and nudge_fitness_passes(final_level, fitness, verify)
        ):
            next_level = final_level + 1
            plan.actions.append(ProgressAction.NUDGE_START_LEVEL)
            plan.nudge_start_level = next_level
            plan.log_lines.append(
                f"[Progress] Cross-cycle nudge L{final_level}->L{next_level} "
                f"({self.state.cycles_at_level} cycles at level)"
            )

        self.save()
        return plan

    @staticmethod
    def certify_verify_threshold(level: int, stall: int = 2, stall_threshold: int = 2) -> float:
        """Verify bar used by certify_loop_completion C3 (and carry-forward clamp)."""
        return curriculum_verify_threshold(int(level), stall, stall_threshold)

    @staticmethod
    def passes_certify_verify(level: int, verify: float, stall: int = 2, stall_threshold: int = 2) -> bool:
        if not isinstance(verify, (int, float)) or not isinstance(level, (int, float)):
            return False
        return float(verify) >= ProgressController.certify_verify_threshold(int(level), stall, stall_threshold)

    @staticmethod
    def carry_forward_start_level(
        final_level: int,
        verify: float,
        sustained_cap: int,
    ) -> int:
        """Next-cycle start level — reject hollow end-of-run spikes below C3 verify bar."""
        lvl = int(final_level)
        ver = float(verify)
        if ProgressController.passes_certify_verify(lvl, ver):
            return lvl
        from evolution.curriculum_advance import depth_carry_verify_floor

        if lvl >= 20 and ver >= depth_carry_verify_floor(lvl):
            return lvl
        if sustained_cap > 0:
            return sustained_cap
        return max(3, lvl - 1)

    @staticmethod
    def sustained_resume_level(
        results: list[dict[str, Any]],
        min_fitness: float = 80.0,
        min_verify: float = 0.65,
    ) -> int:
        """Peak level with quality gates — rejects hollow spikes (negative fitness, low verify at depth)."""
        best = 0
        for r in results:
            if "error" in r:
                continue
            lvl = int(r.get("final_level", 0))
            fit = float(r.get("final_fitness", 0))
            ver = float(r.get("champion_verify_score", 0))
            if fit <= 0:
                continue
            eff_min_fit = min_fitness
            eff_min_ver = min_verify
            from evolution.curriculum_advance import depth_carry_verify_floor

            if lvl >= 23:
                eff_min_fit = 50.0
                eff_min_ver = depth_carry_verify_floor(lvl)
            elif lvl >= 22:
                eff_min_fit = max(min_fitness, 62.0)
                eff_min_ver = depth_carry_verify_floor(lvl)
            elif lvl >= 20:
                eff_min_fit = max(min_fitness, 95.0)
                eff_min_ver = depth_carry_verify_floor(lvl)
            elif lvl >= 18:
                eff_min_fit = max(min_fitness, 95.0)
                eff_min_ver = max(min_verify, 0.70)
            if fit >= eff_min_fit and ver >= eff_min_ver:
                best = max(best, lvl)
        return best
