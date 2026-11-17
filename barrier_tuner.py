"""Barrier detection and self-tuning for evolution cycles.

Monitors cycle metrics to detect fitness plateaus, level stalls,
diversity collapse, and validation loss regressions. Provides
tuning recommendations for evolution and retrain parameters.

Wired into run_multi_cycle.py to enable adaptive parameter
adjustment based on detected barriers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BarrierType(Enum):
    """Types of evolutionary barriers that can be detected."""

    NORMAL = "normal"
    FITNESS_PLATEAU = "fitness_plateau"
    LEVEL_STALL = "level_stall"
    DIVERSITY_COLLAPSE = "diversity_collapse"
    VAL_LOSS_REGRESSION = "val_loss_regression"


@dataclass
class CycleMetrics:
    """Metrics collected from a single evolution cycle."""

    cycle: int
    fitness: float
    level: int
    champion_words: int = 0
    verify_score: float = 0.0
    diversity: float = 0.3
    qd_occupied: int = 0
    qd_total: int = 100
    novelty_size: int = 0
    traces_count: int = 0
    retrained: bool = False
    val_loss: float | None = None
    generations_completed: int = 0
    mutation_rate: float = 0.5
    # Training visibility (set post-construction by diagnostic wiring)
    memorization_score: float | None = None
    generalization_score: float | None = None


@dataclass
class BarrierInfo:
    """Result of barrier analysis for a cycle."""

    barrier: BarrierType = BarrierType.NORMAL
    consecutive_count: int = 0
    severity: float = 0.0
    recommendations: dict = field(default_factory=dict)


class BarrierTuner:
    """Centralized barrier detection and self-tuning.

    Tracks cycle metrics over time, detects evolutionary barriers,
    and provides tuning recommendations for evolution and retrain
    parameters. Persists state across runs via JSON.

    Attributes used by run_multi_cycle.py:
        level_stall_cycles: Consecutive cycles without level advancement.
        fitness_plateau_cycles: Consecutive cycles with stagnant fitness.
        diversity_collapse_threshold: Diversity below this triggers alerts.
        history: List of CycleMetrics for all completed cycles.
        _tracker: Optional ProgressionTracker instance.
    """

    def __init__(
        self,
        state_path: str = "",
        model_n_layers: int = 8,
    ):
        self.state_path = state_path
        self.model_n_layers = model_n_layers

        # Barrier tracking
        self.level_stall_cycles: int = 0
        self.fitness_plateau_cycles: int = 0
        self.diversity_collapse_cycles: int = 0
        self.val_loss_regression_cycles: int = 0
        self.diversity_collapse_threshold: float = 0.15

        # History
        self.history: list[CycleMetrics] = []
        self.barriers_detected: list[BarrierType] = []
        self.tuning_history: list[dict] = []

        # Last barrier info (from most recent update)
        self._last_barrier_info: BarrierInfo = BarrierInfo()

        # Stagnation tracking for auto-termination
        self._stagnation_start_cycle: int = 0
        self._max_stagnation_cycles: int = 15
        self._termination_threshold: int = 10

        # ProgressionTracker (optional, for compatibility with TuningPolicy)
        self._tracker: Any = None

        # Baseline tracking for comparison
        self._baseline_fitness: float = 0.0
        self._baseline_level: int = 0

    def load_state(self, state_path: str) -> bool:
        """Load persisted state from JSON. Returns True on success."""
        if not os.path.isfile(state_path):
            return False
        try:
            with open(state_path) as f:
                data = json.load(f)
            self.level_stall_cycles = data.get("level_stall_cycles", 0)
            self.fitness_plateau_cycles = data.get("fitness_plateau_cycles", 0)
            self.diversity_collapse_cycles = data.get("diversity_collapse_cycles", 0)
            self.val_loss_regression_cycles = data.get("val_loss_regression_cycles", 0)
            self._stagnation_start_cycle = data.get("stagnation_start_cycle", 0)
            self._baseline_fitness = data.get("baseline_fitness", 0.0)
            self._baseline_level = data.get("baseline_level", 0)
            # Restore history (limited to last 50 cycles to prevent unbounded growth)
            raw_history = data.get("history", [])
            self.history = [CycleMetrics(**entry) for entry in raw_history[-50:]]
            return True
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def save_state(self, state_path: str) -> None:
        """Persist state to JSON using atomic write."""
        import tempfile

        data = {
            "level_stall_cycles": self.level_stall_cycles,
            "fitness_plateau_cycles": self.fitness_plateau_cycles,
            "diversity_collapse_cycles": self.diversity_collapse_cycles,
            "val_loss_regression_cycles": self.val_loss_regression_cycles,
            "stagnation_start_cycle": self._stagnation_start_cycle,
            "baseline_fitness": self._baseline_fitness,
            "baseline_level": self._baseline_level,
            "history": [
                {
                    "cycle": m.cycle,
                    "fitness": m.fitness,
                    "level": m.level,
                    "champion_words": m.champion_words,
                    "verify_score": m.verify_score,
                    "diversity": m.diversity,
                    "qd_occupied": m.qd_occupied,
                    "qd_total": m.qd_total,
                    "novelty_size": m.novelty_size,
                    "traces_count": m.traces_count,
                    "retrained": m.retrained,
                    "val_loss": m.val_loss,
                    "generations_completed": m.generations_completed,
                    "mutation_rate": m.mutation_rate,
                }
                for m in self.history[-50:]
            ],
        }
        dir_name = os.path.dirname(state_path) or "."
        os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, state_path)
        except (OSError, PermissionError):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _detect_level_stall(self, metrics: CycleMetrics) -> tuple[BarrierType, int, float]:
        """Detect level stall from history."""
        barrier = BarrierType.NORMAL
        consecutive = 0
        severity = 0.0
        if len(self.history) >= 2:
            prev = self.history[-2]
            if metrics.level > prev.level:
                self.level_stall_cycles = 0
            else:
                self.level_stall_cycles += 1
            if self.level_stall_cycles >= 3:
                barrier = BarrierType.LEVEL_STALL
                consecutive = self.level_stall_cycles
                severity = min(1.0, self.level_stall_cycles / 10.0)
        return barrier, consecutive, severity

    def _detect_fitness_plateau(self, metrics: CycleMetrics) -> tuple[BarrierType, int, float]:
        """Detect fitness plateau from history."""
        barrier = BarrierType.NORMAL
        consecutive = 0
        severity = 0.0
        if len(self.history) >= 3:
            recent = self.history[-3:]
            fitnesses = [m.fitness for m in recent]
            if max(fitnesses) - min(fitnesses) < 1.0:
                self.fitness_plateau_cycles += 1
                if self.fitness_plateau_cycles >= 3:
                    barrier = BarrierType.FITNESS_PLATEAU
                    consecutive = self.fitness_plateau_cycles
                    severity = min(1.0, self.fitness_plateau_cycles / 8.0)
            else:
                self.fitness_plateau_cycles = 0
        return barrier, consecutive, severity

    def _detect_diversity_collapse(self, metrics: CycleMetrics) -> tuple[BarrierType, int, float]:
        """Detect diversity collapse."""
        barrier = BarrierType.NORMAL
        consecutive = 0
        severity = 0.0
        if metrics.diversity < self.diversity_collapse_threshold:
            self.diversity_collapse_cycles += 1
            if self.diversity_collapse_cycles >= 2:
                barrier = BarrierType.DIVERSITY_COLLAPSE
                consecutive = self.diversity_collapse_cycles
                severity = min(1.0, self.diversity_collapse_cycles / 5.0)
        else:
            self.diversity_collapse_cycles = 0
        return barrier, consecutive, severity

    def _detect_val_loss_regression(self, metrics: CycleMetrics) -> tuple[BarrierType, int, float]:
        """Detect validation loss regression."""
        barrier = BarrierType.NORMAL
        consecutive = 0
        severity = 0.0
        if len(self.history) >= 2 and metrics.val_loss is not None:
            prev_vals = [h.val_loss for h in self.history[:-1] if h.val_loss is not None]
            if prev_vals:
                best_prev = min(prev_vals)
                if metrics.val_loss > best_prev * 1.1:
                    self.val_loss_regression_cycles += 1
                    if self.val_loss_regression_cycles >= 2:
                        barrier = BarrierType.VAL_LOSS_REGRESSION
                        consecutive = self.val_loss_regression_cycles
                        severity = min(1.0, self.val_loss_regression_cycles / 5.0)
                else:
                    self.val_loss_regression_cycles = 0
        return barrier, consecutive, severity

    def update(self, metrics: CycleMetrics) -> BarrierInfo:
        """Update with new cycle metrics and return barrier analysis.

        Detects fitness plateaus, level stalls, diversity collapse,
        and validation loss regressions. Updates internal counters
        and returns a BarrierInfo with the detected barrier type.
        """
        self.history.append(metrics)
        barrier = BarrierType.NORMAL
        consecutive = 0
        severity = 0.0
        for detector in (
            self._detect_level_stall,
            self._detect_fitness_plateau,
            self._detect_diversity_collapse,
            self._detect_val_loss_regression,
        ):
            b, c, s = detector(metrics)
            if b != BarrierType.NORMAL:
                if barrier == BarrierType.NORMAL:
                    barrier = b
                consecutive = max(consecutive, c)
                severity = max(severity, s)
        if barrier != BarrierType.NORMAL:
            self.barriers_detected.append(barrier)
        recommendations = self._generate_recommendations(barrier, metrics)
        barrier_info = BarrierInfo(
            barrier=barrier, consecutive_count=consecutive, severity=severity, recommendations=recommendations
        )
        self._last_barrier_info = barrier_info
        self.tuning_history.append(
            {
                "cycle": metrics.cycle,
                "barrier": barrier.value,
                "severity": severity,
                "recommendations": recommendations,
            }
        )
        return barrier_info

    def _generate_recommendations(self, barrier: BarrierType, metrics: CycleMetrics) -> dict:
        """Generate parameter tuning recommendations based on barrier type."""
        recs: dict[str, Any] = {}

        if barrier == BarrierType.FITNESS_PLATEAU:
            recs["mutation_rate"] = "increase"
            recs["diversity_inject_ratio"] = "increase"
        elif barrier == BarrierType.LEVEL_STALL:
            recs["diversity_inject_ratio"] = "increase"
            recs["novelty_weight"] = "increase"
        elif barrier == BarrierType.DIVERSITY_COLLAPSE:
            recs["diversity_inject_ratio"] = "increase"
            recs["mutation_rate"] = "increase"
        elif barrier == BarrierType.VAL_LOSS_REGRESSION:
            recs["retrain_lr"] = "decrease"
            recs["retrain_freeze_layers"] = "increase"

        return recs

    def tune_evolution_params(self) -> dict[str, Any]:
        """Return evolution parameter overrides based on barrier analysis."""
        params: dict[str, Any] = {}
        info = self._last_barrier_info

        if info.barrier == BarrierType.FITNESS_PLATEAU:
            params["mutation_rate"] = min(0.6, 0.3 + 0.1 * info.severity)
        elif info.barrier == BarrierType.LEVEL_STALL:
            params["diversity_inject_ratio"] = min(0.3, 0.1 + 0.05 * info.severity)
            params["novelty_weight"] = min(0.5, 0.3 + 0.1 * info.severity)
        elif info.barrier == BarrierType.DIVERSITY_COLLAPSE:
            params["diversity_inject_ratio"] = min(0.4, 0.15 + 0.1 * info.severity)
            params["mutation_rate"] = min(0.6, 0.4 + 0.1 * info.severity)

        return params

    def tune_retrain_params(self) -> dict[str, Any]:
        """Return retrain parameter overrides based on barrier analysis."""
        params: dict[str, Any] = {}
        info = self._last_barrier_info

        if info.barrier == BarrierType.VAL_LOSS_REGRESSION:
            params["retrain_lr"] = 1e-5
            params["retrain_freeze_layers"] = min(6, self.model_n_layers // 2)
        elif info.barrier == BarrierType.LEVEL_STALL and info.severity > 0.5:
            params["retrain_max_epochs"] = 30

        return params

    def tune_archive_params(self) -> dict[str, Any]:
        """Return archive parameter overrides based on barrier analysis."""
        params: dict[str, Any] = {}
        info = self._last_barrier_info

        # Certified L30+ plateau is expected progress — do not inflate gens and
        # slow the EPI momentum window flush (daemon stays at --gens 50).
        recent = self.history[-3:] if self.history else []
        if self._certified_terminal_plateau(recent):
            return params

        if info.barrier == BarrierType.LEVEL_STALL and info.severity > 0.5:
            params["max_generations"] = 80  # Extend evolution time
        elif info.barrier == BarrierType.DIVERSITY_COLLAPSE:
            params["max_generations"] = 60

        return params

    def notify_rollback(self) -> None:
        """Reset stagnation tracking after a rollback event."""
        self.level_stall_cycles = 0
        self.fitness_plateau_cycles = 0
        self.diversity_collapse_cycles = 0
        self.val_loss_regression_cycles = 0
        self._stagnation_start_cycle = 0
        if self.history:
            last = self.history[-1]
            self._baseline_fitness = last.fitness
            self._baseline_level = last.level

    def _certified_terminal_plateau(self, recent: list[CycleMetrics]) -> bool:
        """Sustained L30 + high fitness is goal progress, not batch stagnation."""
        if not recent:
            return False
        min_level = int(os.environ.get("EE_TERMINAL_PLATEAU_MIN_LEVEL", "30"))
        min_fitness = float(os.environ.get("EE_TERMINAL_PLATEAU_MIN_FITNESS", "250"))
        min_verify = float(os.environ.get("EE_TERMINAL_PLATEAU_MIN_VERIFY", "0.99"))
        for m in recent:
            if m.level < min_level or m.fitness < min_fitness:
                return False
            verify = float(getattr(m, "verify_score", 0.0) or 0.0)
            if verify > 0.0 and verify < min_verify:
                return False
        return True

    def should_terminate(self) -> bool:
        """Check if the run should auto-terminate due to prolonged stagnation.

        Terminates when no meaningful progress (fitness improvement or
        level advancement) has been made for _max_stagnation_cycles.
        """
        if os.environ.get("EE_BARRIER_AUTO_TERMINATE", "1") == "0":
            return False
        if len(self.history) < 3:
            return False

        recent = self.history[-self._max_stagnation_cycles :]
        if len(recent) < self._max_stagnation_cycles:
            return False

        if self._certified_terminal_plateau(recent):
            return False

        fitness_range = max(m.fitness for m in recent) - min(m.fitness for m in recent)
        level_range = max(m.level for m in recent) - min(m.level for m in recent)

        # No meaningful progress if fitness range < 2.0 and no level advancement
        return fitness_range < 2.0 and level_range == 0

    def get_termination_reason(self) -> str:
        """Return human-readable reason for auto-termination."""
        if not self.should_terminate():
            return ""
        len(self.history)
        recent = self.history[-self._max_stagnation_cycles :]
        fitness_range = max(m.fitness for m in recent) - min(m.fitness for m in recent)
        return (
            f"Auto-terminating after {self._max_stagnation_cycles} cycles of stagnation: "
            f"fitness range={fitness_range:.2f}, no level advancement "
            f"(cycles {recent[0].cycle}-{recent[-1].cycle})"
        )

    def analyze(self, metrics: CycleMetrics) -> dict:
        """Analyze cycle metrics and return tuning recommendations.

        Compatibility method used by simpler callers.
        """
        if len(self.history) < 3:
            return {}
        recent = self.history[-3:]
        recommendations: dict[str, Any] = {}

        fitnesses = [m.fitness for m in recent]
        if max(fitnesses) - min(fitnesses) < 1.0:
            recommendations["mutation_rate"] = "increase"

        levels = [m.level for m in recent]
        if len(set(levels)) == 1 and levels[0] > 0:
            recommendations["diversity_inject_ratio"] = "increase"

        return recommendations
