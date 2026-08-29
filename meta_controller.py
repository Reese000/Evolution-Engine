#!/usr/bin/env python3
"""
MetaController — Dynamic Parameterization Orchestrator.

Hybrid architecture: centralized cross-cycle orchestration with
component-local self-tuning. Responsibilities:

  1. Checkpoint gating — only accept models that improve (or match) best val loss
  2. Telemetry tracking — fitness, level, val loss, diversity over cycles
  3. Stagnation detection — classify plateau vs. regression vs. stall
  4. Cross-cycle parameter tuning — retrain LR, freeze layers, evolution params
  5. Circuit breaker — roll back to best checkpoint on regression
  6. Recovery mode — when stagnation detected, apply countermeasures

Usage:
    from meta_controller import MetaController

    mc = MetaController(checkpoint_dir="/path/to/checkpoints")
    mc.update_cycle(cycle=5, fitness=-1.5, level=11, val_loss=2.1)
    if mc.should_rollback():
        checkpoint = mc.get_best_checkpoint()
    params = mc.tune_retrain_params(cycle=6)
"""

from __future__ import annotations

import collections
import json
import os
import re as _re
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Any

from logging_config import get_logger

_logger = get_logger("MetaController")


# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════


class StagnationResult(Enum):
    """Classifications of system state based on telemetry."""

    NORMAL = "normal"
    FITNESS_PLATEAU = "fitness_plateau"  # fitness flat for N cycles
    LEVEL_STALLED = "level_stalled"  # same level for N cycles
    VAL_LOSS_REGRESSION = "val_loss_regression"  # val loss spiked
    DIVERSITY_COLLAPSE = "diversity_collapse"  # diversity below threshold
    COMPOUND_STALL = "compound_stall"  # multiple indicators


# Pre-built set of stagnation state values for O(1) lookup in _record_tuning.
# Avoids iterating StagnationResult members on every tuning log entry.
_STAGNATION_STATE_VALUES: frozenset[str] = frozenset(s.value for s in StagnationResult)


# ══════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════


@dataclass
class TelemetrySnapshot:
    """Immutable snapshot of system state at the end of a cycle.

    R37: Uses __slots__ to reduce per-instance memory overhead by ~40%.
    Over a 200-cycle multi-cycle run, this saves ~40KB compared to
    regular dict-based __dict__ instances.
    """

    __slots__ = (
        "champion_words",
        "cycle",
        "diversity",
        "fitness",
        "generations_completed",
        "level",
        "novelty_size",
        "qd_occupied",
        "qd_total",
        "retrained",
        "traces_count",
        "val_loss",
    )
    cycle: int
    fitness: float
    level: int
    champion_words: int
    val_loss: float | None  # None = retrain didn't run or wasn't captured
    diversity: float  # population diversity at end of cycle
    novelty_size: int
    qd_occupied: int
    qd_total: int
    traces_count: int
    retrained: bool  # whether retrain ran this cycle
    generations_completed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "fitness": self.fitness,
            "level": self.level,
            "champion_words": self.champion_words,
            "val_loss": self.val_loss,
            "diversity": self.diversity,
            "novelty_size": self.novelty_size,
            "qd_occupied": self.qd_occupied,
            "qd_total": self.qd_total,
            "traces_count": self.traces_count,
            "retrained": self.retrained,
            "generations_completed": self.generations_completed,
        }


class TuningAction:
    """Record of a tuning decision made by the MetaController."""

    def __init__(self, domain: str, param: str, old_value: Any, new_value: Any, reason: str, cycle: int = 0):
        self.domain = domain  # "retrain", "evolution", "recovery"
        self.param = param  # e.g. "retrain_lr"
        self.old_value = old_value
        self.new_value = new_value
        self.reason = reason  # e.g. "stagnation: fitness_plateau"
        self.cycle = cycle  # which cycle this tuning was applied in

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "param": self.param,
            "old": self.old_value,
            "new": self.new_value,
            "reason": self.reason,
            "cycle": self.cycle,
        }


# ══════════════════════════════════════════════════════════════════════
# MetaController
# ══════════════════════════════════════════════════════════════════════


class MetaController:
    """
    Cross-cycle dynamic parameterization orchestrator.

    Tracks telemetry and tunes hyperparameters to:
    - Prevent catastrophic regression (checkpoint gating)
    - Break plateaus (stagnation detection → parameter adjustment)
    - Maintain diversity (adaptive thresholds)
    - Recover from failures (circuit breaker, rollback)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        best_checkpoint_name: str = "checkpoint_best.pt",
        initial_val_loss: float = float("inf"),
        # ── Tuning bounds ──────────────────────────────────────────────
        lr_min: float = 1e-6,
        lr_max: float = 3e-4,
        freeze_min: int = 0,
        freeze_max: int = 8,
        patience_min: int = 2,
        patience_max: int = 10,
        # ── Stagnation thresholds ──────────────────────────────────────
        fitness_plateau_cycles: int = 3,
        level_stall_cycles: int = 5,
        diversity_collapse_threshold: float = 0.05,
        val_loss_regression_ratio: float = 1.5,
        # ── Checkpoint gating ──────────────────────────────────────────
        checkpoint_acceptance_tolerance: float = 1.0,
        # ── Logging ────────────────────────────────────────────────────
        log_file: str | None = None,
    ):
        # Paths
        self.checkpoint_dir = checkpoint_dir
        self.best_checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint_name)
        self._best_val_loss = initial_val_loss
        self._best_checkpoint_backup = ""

        # Tuning bounds
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.freeze_min = freeze_min
        self.freeze_max = freeze_max
        self.patience_min = patience_min
        self.patience_max = patience_max

        # Stagnation thresholds
        self.fitness_plateau_cycles = fitness_plateau_cycles
        self.level_stall_cycles = level_stall_cycles
        self.diversity_collapse_threshold = diversity_collapse_threshold
        self.val_loss_regression_ratio = val_loss_regression_ratio
        self.checkpoint_acceptance_tolerance = checkpoint_acceptance_tolerance

        # Telemetry history
        # R34: Cap telemetry to last 100 entries. detect_stagnation only uses
        # the last 5, save_state/get_state_summary only use the last 5/20.
        # Unbounded growth on long multi-cycle runs wastes memory.
        self.telemetry: list[TelemetrySnapshot] = []
        self._max_telemetry_len: int = 100

        # Best known values
        self.best_fitness: float = -float("inf")
        self.best_level: int = 0
        self.best_champion_words: int = 0

        # Stagnation tracking
        self.cycles_since_fitness_improvement: int = 0
        self.cycles_since_level_up: int = 0
        self.cycles_since_diversity_improvement: int = 0
        self.last_level: int = -1

        # State
        self.recovery_mode: bool = False
        self.recovery_cycles_remaining: int = 0
        self._last_stagnation: StagnationResult = StagnationResult.NORMAL
        self._tuning_log: list[TuningAction] = []
        # R34: Cap tuning log to prevent unbounded memory growth.
        # get_state_summary only shows last 20, consecutive stagnation
        # counter only uses last 16 ring entries. 200 is sufficient.
        self._max_tuning_log_len: int = 200
        self._gated_count: int = 0
        self._rollback_count: int = 0
        self._last_snapshot_path: str = ""

        # Consecutive cycles with None val_loss (for degenerate-loop detection)
        self._consecutive_none_val_loss: int = 0

        # Compound stall counter (for meta-evolution trigger)
        self._compound_stall_counter: int = 0

        # Force-retrain flag: set when compound_stall persists with val_loss=None
        # for 5+ cycles. The main loop should check this and override skip logic.
        self.force_retrain: bool = False

        # R28: Fitness momentum tracking -- exponential moving average of
        # fitness improvement rate. Used to detect acceleration/deceleration
        # in improvement and make smarter tuning decisions.
        self._fitness_momentum: float = 0.0
        self._fitness_momentum_alpha: float = 0.3  # EMA smoothing factor
        self._best_fitness_at_last_tune: float = -float("inf")

        # O(1) consecutive stagnation counter -- tracks how many consecutive
        # tuning log entries match a given state value, avoiding the O(N)
        # backward scan of _count_consecutive_stagnation.
        # R19: Also maintain a deque of the last 16 state values
        # so we can provide exact counts for any stagnation state in O(1)
        # without the O(N) fallback scan.
        self._consecutive_stagnation_counter: int = 0
        self._last_stagnation_state: str = ""
        self._state_ring: collections.deque = collections.deque(maxlen=16)

        # Total consecutive stagnation events across the entire run.
        # Unlike _consecutive_stagnation_counter (which resets on state change),
        # this accumulates across all stagnation types for effectiveness scoring.
        self._total_consecutive_stagnation: int = 0

        # R16: Cache stagnation detection result per cycle (both tune_retrain_params
        # and tune_evolution_params call detect_stagnation; cache avoids redundant scan).
        self._cached_stagnation: StagnationResult | None = None
        self._cached_stagnation_len: int = -1  # len(self.telemetry) when cached

        # Escalation event history — tracks stagnation detections that triggered
        # tuning actions, with effectiveness score at time of detection.
        self._escalation_history: list[dict[str, Any]] = []

        # Lazy log file handle (avoids fd exhaustion on long runs)
        self._log_file = None

        # Logging
        self.log_file = log_file
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file) or ".", exist_ok=True)

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def update_cycle(self, snapshot: TelemetrySnapshot) -> None:
        """Ingest metrics from a completed cycle.

        Updates all internal state and stagnation counters.
        Should be called once per cycle, AFTER retrain decision.
        """
        # R28: Update fitness momentum (EMA of fitness delta)
        delta = snapshot.fitness - self.best_fitness if self.best_fitness > -float("inf") else 0.0
        alpha = self._fitness_momentum_alpha
        self._fitness_momentum = alpha * delta + (1.0 - alpha) * self._fitness_momentum

        # Update best values
        if snapshot.fitness > self.best_fitness:
            self.best_fitness = snapshot.fitness
            self.cycles_since_fitness_improvement = 0
        else:
            self.cycles_since_fitness_improvement += 1

        if snapshot.level > self.best_level:
            self.best_level = snapshot.level
            self.cycles_since_level_up = 0
        elif snapshot.level == self.last_level:
            self.cycles_since_level_up += 1
        else:
            # Level dropped — still count as stall if below best
            if snapshot.level < self.best_level:
                self.cycles_since_level_up += 1
            else:
                self.cycles_since_level_up = 0

        self.last_level = snapshot.level

        self.best_champion_words = max(self.best_champion_words, snapshot.champion_words)

        # Update best val loss if we got a valid measurement
        if snapshot.val_loss is not None and snapshot.val_loss > 0:
            self._best_val_loss = min(self._best_val_loss, snapshot.val_loss)
            self._consecutive_none_val_loss = 0
        else:
            self._consecutive_none_val_loss += 1

        # Append to history
        self.telemetry.append(snapshot)
        # R34: Cap telemetry to prevent unbounded memory growth on long runs.
        # Only the last _max_telemetry_len entries are needed for diagnostics
        # and stagnation detection (which uses a window of 5).
        if len(self.telemetry) > self._max_telemetry_len:
            self.telemetry = self.telemetry[-self._max_telemetry_len :]
        # R25: Invalidate stagnation cache when telemetry changes.
        # Without this, detect_stagnation() may return a stale cached result
        # from a previous call within the same cycle.
        self._cached_stagnation_len = -1
        self._log(
            f"Cycle {snapshot.cycle}: F={snapshot.fitness:.3f} L={snapshot.level} "
            f"val={snapshot.val_loss} div={snapshot.diversity:.3f} "
            f"retrain={snapshot.retrained}"
        )

    def accept_checkpoint(self, val_loss: float, new_checkpoint_path: str) -> bool:
        """Gate: should we accept this checkpoint?

        Returns True if val_loss is within tolerance of best known loss.
        If rejected, the caller should NOT overwrite the best checkpoint
        and should continue with the previous best model.

        Args:
            val_loss: Validation loss from the retrained model.
            new_checkpoint_path: Path to the new checkpoint file.

        Returns:
            True if checkpoint passes gating, False if it should be rejected.
        """
        if not os.path.isfile(new_checkpoint_path):
            self._log(f"Gate: checkpoint {new_checkpoint_path} not found — rejecting")
            return False

        if val_loss <= 0:
            # Invalid val loss — can't gate meaningfully, accept cautiously
            self._log(f"Gate: val_loss={val_loss} invalid — accepting cautiously")
            return True

        threshold = self._best_val_loss * self.checkpoint_acceptance_tolerance

        if val_loss <= threshold:
            # Accept — update best val loss
            old_best = self._best_val_loss
            self._best_val_loss = min(self._best_val_loss, val_loss)
            self._log(f"Gate: ACCEPT val_loss={val_loss:.4f} " f"(best={old_best:.4f}, threshold={threshold:.4f})")
            self._gated_count += 1
            return True
        else:
            # Reject — model is worse
            self._log(
                f"Gate: REJECT val_loss={val_loss:.4f} > threshold={threshold:.4f} " f"(best={self._best_val_loss:.4f})"
            )
            self._gated_count += 1
            return False

    def detect_stagnation(self) -> StagnationResult:
        """Classify current system state based on recent telemetry.

        R24: Single-pass over the window for ALL four indicators.
        Previous version scanned the window 3-4 separate times (fitness
        min/max, val_loss first/last, diversity min). This version
        extracts all signals in one iteration, reducing cache misses
        and Python loop overhead.

        R16: Caches result within a cycle -- telemetry length doesn't change
        between tune_retrain_params and tune_evolution_params calls.

        R35: Fast-path early return when all indicators are clearly
        below their thresholds (no telemetry window scan needed).
        """
        tel_len = len(self.telemetry)
        if tel_len < 2:
            return StagnationResult.NORMAL
        # R16: Return cached result if telemetry hasn't changed since last call
        if self._cached_stagnation is not None and self._cached_stagnation_len == tel_len:
            return self._cached_stagnation

        # R35: Fast-path -- if fitness is improving AND level just changed
        # AND diversity is above threshold, we can skip the full window scan
        # since no indicator is active. Only applies when we have at least
        # 3 data points for a reliable signal.
        if tel_len >= 3:
            last = self.telemetry[-1]
            prev = self.telemetry[-2]
            # Fitness improving: last > prev, and no cycle gap in improvement
            fitness_improving = last.fitness > prev.fitness and self.cycles_since_fitness_improvement <= 1
            # Level changed: not stalled
            level_fresh = self.cycles_since_level_up <= 1
            # Diversity healthy
            div_healthy = last.diversity >= self.diversity_collapse_threshold
            if fitness_improving and level_fresh and div_healthy:
                result = StagnationResult.NORMAL
                self._last_stagnation = result
                self._cached_stagnation = result
                self._cached_stagnation_len = tel_len
                return result

        tel = self.telemetry
        win_len = min(5, len(tel))
        start_idx = len(tel) - win_len

        # Precompute constants once
        fp_cycles = self.fitness_plateau_cycles
        vlr_ratio = self.val_loss_regression_ratio
        dc_thresh = self.diversity_collapse_threshold

        # R24: Single pass over the window -- extract fitness min/max,
        # val_loss first/last, and diversity min in one iteration.
        first = tel[start_idx]
        f_min = f_max = first.fitness
        div_min = first.diversity
        vl_first = vl_last = None
        vl0 = first.val_loss
        if vl0 is not None:
            vl_first = vl0
            vl_last = vl0

        for j in range(start_idx + 1, tel_len):
            snap = tel[j]
            fj = snap.fitness
            if fj < f_min:
                f_min = fj
            elif fj > f_max:
                f_max = fj
            dj = snap.diversity
            div_min = min(div_min, dj)
            vj = snap.val_loss
            if vj is not None:
                vl_last = vj
                if vl_first is None:
                    vl_first = vj

        # Classify indicators
        indications_count = 0
        ind_fitness = ind_level = ind_valloss = ind_diversity = False

        # 1. Fitness plateau (fast path: skip if counter too low)
        if self.cycles_since_fitness_improvement >= fp_cycles and f_max - f_min < 0.1:
            ind_fitness = True
            indications_count += 1

        # 2. Level stall (fast path: skip if counter too low)
        if self.cycles_since_level_up >= self.level_stall_cycles:
            ind_level = True
            indications_count += 1

        # 3. Val loss regression
        if vl_first is not None and vl_last is not None and vl_first != vl_last and vl_last > vl_first * vlr_ratio:
            ind_valloss = True
            indications_count += 1

        # 4. Diversity collapse — absolute threshold OR downward trend
        # Trend detection catches early diversity erosion before the absolute
        # threshold is reached: if diversity drops >40% across the window AND
        # current diversity is below a relaxed threshold (1.5x the collapse
        # threshold), flag it as an early warning.  This enables earlier
        # intervention (diversity injection, population expansion) to prevent
        # full collapse.
        div_first = first.diversity
        div_last = tel[tel_len - 1].diversity
        diversity_trend_detected = False
        if div_first > 0 and div_last < div_first * 0.6 and div_last < dc_thresh * 1.5:
            diversity_trend_detected = True
        if div_min < dc_thresh or diversity_trend_detected:
            ind_diversity = True
            indications_count += 1

        # Classify from counts (avoids list allocation entirely)
        if indications_count >= 2:
            result = StagnationResult.COMPOUND_STALL
        elif indications_count == 1:
            if ind_fitness:
                result = StagnationResult.FITNESS_PLATEAU
            elif ind_level:
                result = StagnationResult.LEVEL_STALLED
            elif ind_valloss:
                result = StagnationResult.VAL_LOSS_REGRESSION
            else:
                result = StagnationResult.DIVERSITY_COLLAPSE
        else:
            result = StagnationResult.NORMAL

        self._last_stagnation = result
        # R16: Cache result (avoids redundant scan when both tune_* methods call detect)
        self._cached_stagnation = result
        self._cached_stagnation_len = tel_len
        return result

    def tune_retrain_params(self, cycle: int) -> dict[str, Any]:
        """Return dynamically tuned retrain hyperparameters.

        This is the central tuning function for the retrain pipeline.
        Returns a dict of parameter names to values that should be passed
        to run_multi_cycle's retrain subprocess.

        Args:
            cycle: Current cycle number (1-indexed).

        Returns:
            Dict of retrain parameters (empty dict = use defaults).
        """
        params: dict[str, Any] = {}

        # ── Baseline tuning by cycle number ──────────────────────────
        if cycle == 1:
            base_lr = 3e-4
            base_freeze = 0
            base_patience = 5
            base_epochs = 50
        elif cycle <= 3:
            base_lr = 3e-5
            base_freeze = 2
            base_patience = 5
            base_epochs = 50
        elif cycle <= 8:
            base_lr = 1e-5
            base_freeze = 2
            base_patience = 5
            base_epochs = 40
        elif cycle <= 15:
            base_lr = 1e-5
            base_freeze = 4
            base_patience = 3
            base_epochs = 30
        else:
            base_lr = 5e-6
            base_freeze = 6
            base_patience = 3
            base_epochs = 25

        # ── Override based on stagnation state ───────────────────────
        state = self.detect_stagnation()
        consecutive = self._count_consecutive_stagnation(state)

        if state == StagnationResult.FITNESS_PLATEAU:
            # Break out: increase LR slightly, reduce freezing
            base_lr = min(self.lr_max, base_lr * 2.0)
            base_freeze = max(self.freeze_min, base_freeze - 1)
            self._record_tuning("retrain", "retrain_lr", None, base_lr, "stagnation: fitness_plateau")

        elif state == StagnationResult.LEVEL_STALLED:
            # Level stall: moderate LR, reduced freezing for plasticity
            base_lr = min(self.lr_max, base_lr * 1.5)
            base_freeze = max(self.freeze_min, base_freeze - 1)
            self._record_tuning("retrain", "retrain_lr", None, base_lr, "stagnation: level_stalled")

        elif state == StagnationResult.VAL_LOSS_REGRESSION:
            # Regression: protect existing knowledge, slow learning
            base_lr = max(self.lr_min, base_lr * 0.3)
            base_freeze = min(self.freeze_max, base_freeze + 2)
            self._record_tuning("retrain", "retrain_lr", None, base_lr, "regression: val_loss_regression")

        elif state == StagnationResult.DIVERSITY_COLLAPSE:
            # Diversity collapse: escalate retrain params
            if consecutive <= 2:
                # Initial: moderate LR, minimal freezing
                base_freeze = max(self.freeze_min, base_freeze - 1)
            elif consecutive <= 4:
                # Escalation 1: high LR, no freezing, more augment
                base_freeze = 0
                base_lr = min(self.lr_max, base_lr * 2.0)
            else:
                # Escalation 2: max LR, no freezing, max augment
                base_freeze = 0
                base_lr = self.lr_max
            self._record_tuning(
                "retrain",
                "retrain_freeze_layers",
                None,
                base_freeze,
                f"stagnation: diversity_collapse (tier {min(consecutive, 3)})",
            )

        elif state == StagnationResult.COMPOUND_STALL:
            # Multiple indicators: conservative approach
            if self._best_val_loss < float("inf"):
                # We have a good model — protect it aggressively
                base_freeze = min(self.freeze_max, base_freeze + 1)
            else:
                # No good model yet — explore
                base_lr = min(self.lr_max, base_lr * 1.5)
                base_freeze = max(self.freeze_min, base_freeze - 1)
            self._record_tuning("retrain", "retrain_freeze_layers", None, base_freeze, "stagnation: compound_stall")

        # ── Fitness-based fallback when val_loss is unavailable ──────
        # When val_loss has been None for 3+ consecutive cycles (e.g. retrain
        # skipped or val loss not captured), the controller can oscillate
        # between two LR values because it has no signal to guide tuning.
        # Use fitness improvement rate as a proxy: if fitness is improving,
        # keep the current LR; if stagnant, halve it to encourage exploration.
        if self._consecutive_none_val_loss >= 3:
            if self.cycles_since_fitness_improvement <= 1:
                # Fitness is improving — keep current LR (don't change base_lr)
                self._log(
                    f"  [Fallback] val_loss=None x{self._consecutive_none_val_loss}: "
                    f"fitness improving, keeping LR={base_lr:.2e}"
                )
            elif self.cycles_since_fitness_improvement >= 3:
                # Fitness is stagnant — halve LR to escape local minimum
                base_lr = max(self.lr_min, base_lr * 0.5)
                self._log(
                    f"  [Fallback] val_loss=None x{self._consecutive_none_val_loss}, "
                    f"fitness stagnant ({self.cycles_since_fitness_improvement} cycles): "
                    f"halving LR to {base_lr:.2e}"
                )
                self._record_tuning("retrain", "retrain_lr", None, base_lr, "fallback: val_loss_none_fitness_stagnant")
            else:
                self._log(
                    f"  [Fallback] val_loss=None x{self._consecutive_none_val_loss}: "
                    f"fitness slightly stale ({self.cycles_since_fitness_improvement}), "
                    f"keeping LR={base_lr:.2e}"
                )

        # ── Per-level data diversity ─────────────────────────────────
        if cycle >= 3:
            per_level_cap = min(200 + 50 * (cycle - 3), 500)
            params["retrain_max_per_level"] = per_level_cap

        # ── Recovery mode override ───────────────────────────────────
        if self.recovery_mode:
            base_lr = max(self.lr_min, base_lr * 0.5)
            base_patience = min(self.patience_max, base_patience + 2)
            base_freeze = min(self.freeze_max, base_freeze + 1)
            self._record_tuning("retrain", "retrain_lr", None, base_lr, "recovery_mode")
            self.recovery_cycles_remaining -= 1
            if self.recovery_cycles_remaining <= 0:
                self.recovery_mode = False

        # ── Assemble result ──────────────────────────────────────────
        params["retrain_lr"] = base_lr
        params["retrain_freeze_layers"] = base_freeze
        params["retrain_patience"] = min(self.patience_max, base_patience)
        params["retrain_max_epochs"] = base_epochs

        self._log(
            f"Retrain params for C{cycle}: LR={base_lr:.2e} "
            f"freeze={base_freeze} patience={base_patience} "
            f"epochs={base_epochs} "
            f"state={state.value}"
        )

        # v11 P3.2: Request meta-evolution if compound_stall persists
        if state == StagnationResult.COMPOUND_STALL:
            self._compound_stall_counter += 1
            if self._compound_stall_counter >= 3 and cycle >= 10:
                params["meta_evolution_requested"] = True
                self._log(f"  *** META-EVOLUTION REQUESTED (compound_stall x{self._compound_stall_counter})")
        else:
            self._compound_stall_counter = 0

        # Force retrain when stuck with no val_loss signal for 5+ consecutive cycles.
        # Uses _consecutive_none_val_loss as primary gate (counts from cycle 1,
        # unlike compound_stall_counter which can't fire until cycle 6+).
        # Requires either compound_stall detected OR fitness stagnant for robustness.
        if self._consecutive_none_val_loss >= 5 and (
            state == StagnationResult.COMPOUND_STALL or self.cycles_since_fitness_improvement >= 3
        ):
            if not self.force_retrain:
                self._log(
                    f"  *** FORCE RETRAIN REQUESTED "
                    f"(no val_loss x{self._consecutive_none_val_loss}, "
                    f"state={state.value}, fitness_stale={self.cycles_since_fitness_improvement})"
                )
            self.force_retrain = True

        return params

    def tune_evolution_params(self, cycle: int) -> dict[str, Any]:
        """Return dynamically tuned evolution hyperparameters.

        These are applied as overrides to EvolutionConfig before
        running evolution each cycle.

        Returns:
            Dict of evolution parameters (empty dict = use defaults).
        """
        params: dict[str, Any] = {}
        state = self.detect_stagnation()

        # Count consecutive stagnation events for escalation
        consecutive = self._count_consecutive_stagnation(state)

        if state == StagnationResult.FITNESS_PLATEAU:
            params["mutation_rate"] = 0.70
            params["diversity_threshold"] = 0.20
            params["diversity_inject_ratio"] = 0.20
            params["novelty_weight"] = 0.30
            self._record_tuning("evolution", "mutation_rate", None, 0.70, "stagnation: fitness_plateau")

        elif state == StagnationResult.LEVEL_STALLED:
            params["mutation_rate"] = 0.60
            params["diversity_threshold"] = 0.25
            params["diversity_inject_ratio"] = 0.25
            params["novelty_weight"] = 0.35
            self._record_tuning("evolution", "novelty_weight", None, 0.35, "stagnation: level_stalled")

        elif state == StagnationResult.DIVERSITY_COLLAPSE:
            # Escalate: progressively stronger fixes for persistent collapse
            if consecutive <= 2:
                # Initial fix: increase injection ratio
                params["diversity_inject_ratio"] = 0.30
                params["mutation_rate"] = 0.65
                params["diversity_threshold"] = 0.15
                self._record_tuning(
                    "evolution", "diversity_inject_ratio", None, 0.30, "stagnation: diversity_collapse (tier 1)"
                )
            elif consecutive <= 4:
                # Escalation 1: champion-level crossover, higher novelty
                params["diversity_inject_ratio"] = 0.50
                params["mutation_rate"] = 0.80
                params["novelty_weight"] = 0.50
                params["champion_crossover_prob"] = 0.85
                params["diversity_threshold"] = 0.10
                self._record_tuning(
                    "evolution", "champion_crossover_prob", None, 0.85, "stagnation: diversity_collapse (tier 2)"
                )
            else:
                # Escalation 2: drastic — large pop, high mutation, heavy injection
                params["diversity_inject_ratio"] = 0.75
                params["mutation_rate"] = 0.90
                params["population_size"] = 120
                params["novelty_weight"] = 0.60
                params["champion_crossover_prob"] = 0.90
                params["diversity_threshold"] = 0.08
                params["elite_ratio"] = 0.05
                self._record_tuning(
                    "evolution", "population_size", None, 120, "stagnation: diversity_collapse (tier 3)"
                )

        elif state == StagnationResult.VAL_LOSS_REGRESSION:
            # Regression: explore more cautiously — higher elite ratio
            if consecutive <= 2:
                params["elite_ratio"] = 0.15
                params["mutation_rate"] = 0.50
                params["novelty_weight"] = 0.20
            else:
                params["elite_ratio"] = 0.20
                params["mutation_rate"] = 0.40
                params["novelty_weight"] = 0.10
            self._record_tuning(
                "evolution",
                "elite_ratio",
                None,
                params.get("elite_ratio", 0.15),
                f"regression: val_loss_regression (tier {min(consecutive, 2)})",
            )

        elif state == StagnationResult.COMPOUND_STALL:
            # Multiple issues: aggressive exploration
            if consecutive <= 2:
                params["mutation_rate"] = 0.75
                params["diversity_inject_ratio"] = 0.30
                params["novelty_weight"] = 0.40
            else:
                params["mutation_rate"] = 0.85
                params["diversity_inject_ratio"] = 0.50
                params["novelty_weight"] = 0.50
                params["population_size"] = 100
            self._record_tuning(
                "evolution",
                "mutation_rate",
                None,
                params.get("mutation_rate", 0.75),
                f"stagnation: compound_stall (tier {min(consecutive, 2)})",
            )

        elif state == StagnationResult.NORMAL and self.recovery_mode:
            # In recovery from previous stagnation: moderate params
            params["mutation_rate"] = 0.55
            params["novelty_weight"] = 0.25
            self._record_tuning("evolution", "mutation_rate", None, 0.55, "recovery_mode")

        return params

    def should_rollback(self) -> bool:
        """Determine if we should roll back to best known checkpoint.

        Rollback triggers when:
        - Val loss regression persists for 2+ cycles AND
        - Fitness hasn't improved for 2+ cycles
        OR
        - Effectiveness score is critically low (<0.2) AND
        - Fitness hasn't improved for 3+ cycles (sustained stagnation)

        The effectiveness check catches cases where tuning actions are
        consistently ineffective and the system is stuck, even without
        an explicit val_loss regression signal.
        """
        if len(self.telemetry) < 2:
            return False

        state = self.detect_stagnation()
        if state in (StagnationResult.VAL_LOSS_REGRESSION, StagnationResult.COMPOUND_STALL):
            if self.cycles_since_fitness_improvement >= 2:
                self._log(
                    f"ROLLBACK: state={state.value}, "
                    f"cycles_since_fitness_improvement="
                    f"{self.cycles_since_fitness_improvement}"
                )
                self._rollback_count += 1
                return True

        # Trigger on critically low effectiveness with sustained stagnation.
        # Catches systems stuck in a local minimum where tuning actions
        # are consistently ineffective, even without explicit regression.
        effectiveness = self.get_effectiveness_score()
        if effectiveness < 0.2 and self.cycles_since_fitness_improvement >= 3:
            self._log(
                f"ROLLBACK: effectiveness_score={effectiveness:.3f} < 0.2, "
                f"cycles_since_fitness_improvement="
                f"{self.cycles_since_fitness_improvement}"
            )
            self._rollback_count += 1
            return True

        return False

    def get_best_checkpoint(self) -> str | None:
        """Return path to best known checkpoint, if it exists."""
        if os.path.isfile(self.best_checkpoint_path):
            return self.best_checkpoint_path
        return None

    def snapshot_before_cycle(self, cycle: int, checkpoint_path: str) -> None:
        """Save a pre-cycle checkpoint snapshot for potential rollback.

        Called before evolution+retrain to preserve the current checkpoint.
        If retrain produces a worse model, we can restore from this snapshot.
        """
        if not os.path.isfile(checkpoint_path):
            return
        snapshot_dir = os.path.join(os.path.dirname(self.best_checkpoint_path), "pre_cycle_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"snapshot_cycle_{cycle}.pt")
        try:
            shutil.copy2(checkpoint_path, snapshot_path)
            self._log(f"Snapshot: saved pre-cycle {cycle} checkpoint to {snapshot_path}")
            self._last_snapshot_path = snapshot_path
        except (OSError, shutil.Error) as e:
            self._log(f"Snapshot: failed to save pre-cycle checkpoint: {e}")

    def execute_rollback(self, current_checkpoint: str, use_snapshot: bool = False) -> str | None:
        """Perform rollback: copy best checkpoint (or pre-cycle snapshot) over current.

        Args:
            current_checkpoint: Path to the current (regressed) checkpoint.
            use_snapshot: If True, use the most recent pre-cycle snapshot
                instead of best_checkpoint.  Useful when checkpoint_best.pt
                was overwritten by a rejected retrain.

        Returns:
            Path to the rolled-back checkpoint, or None if rollback failed.
        """
        # Determine the source for rollback
        if use_snapshot and self._last_snapshot_path:
            source = self._last_snapshot_path
            if not os.path.isfile(source):
                self._log(f"Rollback snapshot not found at {source}, falling back to best")
                source = None
        else:
            source = None

        if source is None:
            source = self.get_best_checkpoint()
            if source is None or not os.path.isfile(source):
                self._log(f"Rollback failed: best checkpoint not found at " f"{self.best_checkpoint_path}")
                return None

        # Guard against same-file copy (source == current_checkpoint)
        abs_source = os.path.abspath(source)
        abs_current = os.path.abspath(current_checkpoint)
        if abs_source == abs_current:
            self._log(
                f"Rollback skipped: source and current are the same file "
                f"({source}). Retrain output path must differ from best."
            )
            # Don't treat this as a failure — the file itself is fine
            self.recovery_mode = True
            self.recovery_cycles_remaining = 2
            self.cycles_since_fitness_improvement = 0
            return current_checkpoint

        try:
            # Backup current checkpoint before overwriting
            backup_path = current_checkpoint + ".regressed"
            if os.path.isfile(current_checkpoint):
                shutil.copy2(current_checkpoint, backup_path)
                self._log(f"Backed up regressed checkpoint to {backup_path}")

            # Copy source checkpoint to current location
            shutil.copy2(source, current_checkpoint)
            label = "snapshot" if use_snapshot else "best"
            self._log(f"Rollback ({label}): copied {source} -> {current_checkpoint}")

            # Reset stagnation counters after rollback
            self.recovery_mode = True
            self.recovery_cycles_remaining = 2
            self.cycles_since_fitness_improvement = 0

            return current_checkpoint
        except (OSError, shutil.Error) as e:
            self._log(f"Rollback failed: {e}")
            return None

    def get_state_summary(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of current state."""
        return {
            "best_fitness": self.best_fitness,
            "best_level": self.best_level,
            "best_val_loss": None if self._best_val_loss == float("inf") else self._best_val_loss,
            "cycles_since_fitness_improvement": self.cycles_since_fitness_improvement,
            "cycles_since_level_up": self.cycles_since_level_up,
            "recovery_mode": self.recovery_mode,
            "recovery_cycles_remaining": self.recovery_cycles_remaining,
            "last_stagnation": self._last_stagnation.value,
            "total_gated": self._gated_count,
            "total_rollbacks": self._rollback_count,
            "total_cycles_observed": len(self.telemetry),
            "_consecutive_none_val_loss": self._consecutive_none_val_loss,
            "last_5_cycles": [
                {
                    "cycle": t.cycle,
                    "fitness": t.fitness,
                    "level": t.level,
                    "val_loss": t.val_loss,
                    "diversity": t.diversity,
                }
                for t in self.telemetry[-5:]
            ],
            "tuning_history": [a.to_dict() for a in self._tuning_log[-20:]],  # Last 20 actions
        }

    def get_tuning_history_summary(self) -> dict[str, Any]:
        """Return a summary of the tuning history.

        Provides aggregate stats and recent actions for dashboards
        and debugging. Computed in O(N) over the tuning log.
        """
        log = self._tuning_log
        total = len(log)

        # Build per-domain counts
        actions_by_domain: dict[str, int] = {}
        for action in log:
            actions_by_domain[action.domain] = actions_by_domain.get(action.domain, 0) + 1

        # Build per-stagnation-state counts from reason strings.
        # Reasons follow patterns: "stagnation: <value>", "regression: <value>",
        # "recovery", "fallback: <desc>". Extract the state label after the prefix.
        actions_by_state: dict[str, int] = {}
        for action in log:
            state_label: str
            if action.reason.startswith("stagnation: "):
                state_label = action.reason.split("(")[0].split(":")[1].strip()
            elif action.reason.startswith("regression: "):
                state_label = action.reason.split(":")[-1].strip()
            elif action.reason.startswith("recovery"):
                state_label = "recovery"
            elif action.reason.startswith("fallback:"):
                state_label = action.reason.split(":")[1].strip()
            else:
                state_label = "other"
            actions_by_state[state_label] = actions_by_state.get(state_label, 0) + 1

        # Recent actions: last 5 as dicts
        recent_actions = [a.to_dict() for a in log[-5:]]

        # Cycles since last tuning action
        if total > 0 and self.telemetry:
            current_cycle = self.telemetry[-1].cycle
            last_action_cycle = log[-1].cycle
            cycles_since = max(0, current_cycle - last_action_cycle)
        elif self.telemetry:
            cycles_since = self.telemetry[-1].cycle
        else:
            cycles_since = 0

        return {
            "total_actions": total,
            "actions_by_domain": actions_by_domain,
            "actions_by_state": actions_by_state,
            "recent_actions": recent_actions,
            "cycles_since_last_action": cycles_since,
        }

    def save_state(self, path: str) -> None:
        """Save controller state to a JSON file (for resumption).

        Persists all tuning-dependent state including stagnation tracking,
        fitness momentum, compound stall counter, and the state ring buffer
        so that load_state can fully restore the controller's behavior.

        Uses atomic write (temp file + rename) to prevent corruption
        if the process crashes mid-write.
        """
        state = self.get_state_summary()
        state["_best_val_loss"] = self._best_val_loss
        state["_gated_count"] = self._gated_count
        state["_rollback_count"] = self._rollback_count

        # R34: Save tuning-dependent state for accurate post-restart behavior.
        state["_compound_stall_counter"] = self._compound_stall_counter
        state["_fitness_momentum"] = self._fitness_momentum
        state["_best_fitness_at_last_tune"] = self._best_fitness_at_last_tune
        state["_last_stagnation_state"] = self._last_stagnation_state
        state["_consecutive_stagnation_counter"] = self._consecutive_stagnation_counter
        state["_state_ring"] = list(self._state_ring)
        state["cycles_since_fitness_improvement"] = self.cycles_since_fitness_improvement
        state["cycles_since_level_up"] = self.cycles_since_level_up
        state["last_level"] = self.last_level
        state["best_champion_words"] = self.best_champion_words
        state["_total_consecutive_stagnation"] = self._total_consecutive_stagnation

        try:
            dir_name = os.path.dirname(path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(state, f, indent=2)
                os.replace(tmp_path, path)
            except (OSError, PermissionError):
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, PermissionError) as e:
            self._log(f"Failed to save state: {e}")

    def load_state(self, path: str) -> bool:
        """Load controller state from a JSON file.

        Restores all tuning-dependent state including stagnation tracking,
        fitness momentum, compound stall counter, and the state ring buffer
        so that the controller can resume accurate detection after restart.

        Returns True on success, False if file doesn't exist.
        """
        if not os.path.isfile(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._best_val_loss = data.get("_best_val_loss", float("inf"))
            self.best_fitness = data.get("best_fitness", -float("inf"))
            self.best_level = data.get("best_level", 0)
            self._gated_count = data.get("_gated_count", 0)
            self._rollback_count = data.get("_rollback_count", 0)
            self.recovery_mode = data.get("recovery_mode", False)
            self.recovery_cycles_remaining = data.get("recovery_cycles_remaining", 0)
            self._consecutive_none_val_loss = data.get("_consecutive_none_val_loss", 0)

            # R34: Restore tuning-dependent state for accurate post-restart behavior.
            # Without these, the controller would lose its stagnation tracking
            # and fitness momentum, causing duplicate tuning decisions.
            self._compound_stall_counter = data.get("_compound_stall_counter", 0)
            self._fitness_momentum = data.get("_fitness_momentum", 0.0)
            self._best_fitness_at_last_tune = data.get("_best_fitness_at_last_tune", -float("inf"))
            self._last_stagnation_state = data.get("_last_stagnation_state", "")
            self._consecutive_stagnation_counter = data.get("_consecutive_stagnation_counter", 0)

            # Restore the state ring buffer (last 16 entries)
            saved_ring = data.get("_state_ring", [])
            self._state_ring.clear()
            for entry in saved_ring[-16:]:  # cap at maxlen
                self._state_ring.append(entry)

            # Restore cycles_since tracking
            self.cycles_since_fitness_improvement = data.get("cycles_since_fitness_improvement", 0)
            self.cycles_since_level_up = data.get("cycles_since_level_up", 0)
            self.last_level = data.get("last_level", -1)
            self.best_champion_words = data.get("best_champion_words", 0)
            self._total_consecutive_stagnation = data.get("_total_consecutive_stagnation", 0)

            return True
        except (json.JSONDecodeError, KeyError, OSError) as e:
            self._log(f"Failed to load state: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _count_consecutive_stagnation(self, state: StagnationResult) -> int:
        """Count how many consecutive tuning events match the given stagnation state.

        R19: Always O(1) -- uses the cached counter when state matches
        the tracked value.  For non-tracked states, scans the small
        ring buffer (max 16 entries) instead of the full tuning log.
        R23: When state matches but counter is 0, fall through to ring
        scan instead of returning 0 (which would be wrong if the ring
        has entries but no tuning log entry was recorded yet).
        R33: When state matches AND counter > 0, also cache the ring
        scan result so subsequent calls within the same cycle avoid
        redundant scan.
        """
        state_str = state.value
        if state_str == self._last_stagnation_state and self._consecutive_stagnation_counter > 0:
            return self._consecutive_stagnation_counter
        # O(K) scan of state buffer (K <= 16) instead of O(N) tuning log
        count = 0
        ring = self._state_ring
        ring_len = len(ring)
        for i in range(ring_len - 1, -1, -1):
            if ring[i] == state_str:
                count += 1
            else:
                break
        # Cache result so next call for same state is O(1)
        # Note: only cache non-zero counts. When count=0, we cannot cache
        # because _record_tuning may have added matching entries to the ring
        # between calls (within the same cycle), and a stale cached 0 would
        # mask the correct non-zero count.
        if count > 0:
            self._last_stagnation_state = state_str
            self._consecutive_stagnation_counter = count
        return count

    def _record_tuning(self, domain: str, param: str, old: Any, new: Any, reason: str) -> None:
        self._tuning_log.append(TuningAction(domain, param, old, new, reason))
        # R34: Cap tuning log to prevent unbounded memory growth on long runs.
        # The ring buffer and consecutive counter are O(1) — only the log list
        # grows. Truncate to keep last _max_tuning_log_len entries.
        if len(self._tuning_log) > self._max_tuning_log_len:
            self._tuning_log = self._tuning_log[-self._max_tuning_log_len :]
        # Maintain O(1) consecutive stagnation counter.
        # R24: Use str.partition once instead of str.find() + inner loop.
        # Reason strings follow patterns: "stagnation: <value> (tier N)",
        # "regression: <value>", "recovery", "fallback: <desc>".
        # str.partition("stagnation: ") is a single C call that extracts
        # everything after the prefix, then we partition again on " (" to
        # strip the tier suffix. Two C-level calls total, no Python loops.
        _state = None
        if reason.startswith("recovery"):
            _state = "recovery"
        else:
            # Single partition for "stagnation: " or "regression: "
            for prefix in ("stagnation: ", "regression: "):
                before, _, after = reason.partition(prefix)
                if before != reason:  # partition found the separator
                    # Strip tier suffix: "fitness_plateau (tier 2)" -> "fitness_plateau"
                    _state, _, _ = after.partition(" (")
                    if not _state:
                        _state = after
                    break
        if _state is not None and _state == self._last_stagnation_state:
            self._consecutive_stagnation_counter += 1
            self._total_consecutive_stagnation += 1
        elif _state is not None and _state in _STAGNATION_STATE_VALUES:
            self._last_stagnation_state = _state
            self._consecutive_stagnation_counter = 1
            self._total_consecutive_stagnation += 1
        else:
            # Non-stagnation reason resets the counter
            self._last_stagnation_state = ""
            self._consecutive_stagnation_counter = 0

        # Maintain capped state buffer for O(K) consecutive count on
        # non-tracked states.  K <= 16, effectively O(1).
        self._state_ring.append(_state or "")

        # Record escalation events: stagnation/regression detections that
        # triggered tuning actions.  Captures effectiveness_before so callers
        # can assess whether interventions are improving over time.
        if _state is not None and _state != "" and _state != "recovery":
            self._escalation_history.append(
                {
                    "cycle": len(self.telemetry),
                    "stagnation_type": _state,
                    "action_taken": f"{domain}.{param}={new}",
                    "effectiveness_before": self.get_effectiveness_score(),
                }
            )

    # ──────────────────────────────────────────────────────────────────
    # Effectiveness / tuning introspection
    # ──────────────────────────────────────────────────────────────────

    def get_effectiveness_report(self) -> dict[str, Any]:
        """Return a comprehensive effectiveness report.

        Aggregates stagnation events, tuning actions, and outcomes
        to assess how well the meta-controller is performing.
        """
        total_tuning = len(self._tuning_log)
        stagnation_actions = [a for a in self._tuning_log if "stagnation" in a.reason or "regression" in a.reason]
        recovery_actions = [a for a in self._tuning_log if "recovery" in a.reason]
        fallback_actions = [a for a in self._tuning_log if "fallback" in a.reason]

        # Count actions by domain
        domain_counts: dict[str, int] = {}
        for a in self._tuning_log:
            domain_counts[a.domain] = domain_counts.get(a.domain, 0) + 1

        return {
            "total_tuning_actions": total_tuning,
            "stagnation_actions": len(stagnation_actions),
            "recovery_actions": len(recovery_actions),
            "fallback_actions": len(fallback_actions),
            "domain_counts": domain_counts,
            "total_rollbacks": self._rollback_count,
            "total_gated": self._gated_count,
            "total_consecutive_stagnation": self._total_consecutive_stagnation,
            "current_stagnation": self._last_stagnation.value,
            "recovery_mode": self.recovery_mode,
            "cycles_observed": len(self.telemetry),
        }

    def suggest_parameter_reset(self) -> dict[str, Any]:
        """Suggest whether parameters should be reset and to what values.

        Returns a dict with:
          - should_reset: bool
          - reason: str
          - suggested_params: dict of suggested parameter overrides
        """
        total_stag = self._total_consecutive_stagnation
        cycles = len(self.telemetry)

        if cycles == 0:
            return {"should_reset": False, "reason": "no telemetry", "suggested_params": {}}

        # Reset if stagnation dominates (>60% of cycles observed)
        stag_ratio = total_stag / max(cycles, 1)
        if stag_ratio > 0.6 and cycles >= 5:
            return {
                "should_reset": True,
                "reason": f"stagnation dominates ({total_stag}/{cycles} cycles = {stag_ratio:.0%})",
                "suggested_params": {
                    "retrain_lr": 3e-4,
                    "retrain_freeze_layers": 0,
                    "retrain_patience": 5,
                    "mutation_rate": 0.50,
                },
            }

        # Reset if compound stall persists with many rollbacks
        if self._compound_stall_counter >= 5 and self._rollback_count >= 3:
            return {
                "should_reset": True,
                "reason": f"compound_stall x{self._compound_stall_counter}, rollbacks x{self._rollback_count}",
                "suggested_params": {
                    "retrain_lr": 1e-4,
                    "retrain_freeze_layers": 2,
                    "retrain_patience": 3,
                    "mutation_rate": 0.60,
                },
            }

        # Reset if cross-cycle fitness trend is declining with high confidence
        if cycles >= 5:
            trends = self.get_cross_cycle_trends()
            fitness_trend = trends.get("fitness", {})
            if fitness_trend.get("direction") == "declining" and fitness_trend.get("confidence", 0) > 0.5:
                return {
                    "should_reset": True,
                    "reason": (
                        f"cross-cycle fitness declining "
                        f"(confidence={fitness_trend['confidence']:.2f}, "
                        f"slope={fitness_trend['slope']:.4f})"
                    ),
                    "suggested_params": {
                        "retrain_lr": 5e-5,
                        "retrain_freeze_layers": 1,
                        "retrain_patience": 4,
                        "mutation_rate": 0.55,
                    },
                }

        return {"should_reset": False, "reason": "no reset needed", "suggested_params": {}}

    def get_effectiveness_score(self) -> float:
        """Return a score in [0.0, 1.0] indicating tuning effectiveness.

        1.0 = perfectly effective (no stagnation, frequent improvements).
        0.0 = completely ineffective (all stagnation, no improvements).
        """
        cycles = len(self.telemetry)
        if cycles == 0:
            return 0.5  # neutral

        improvement_cycles = cycles - self._total_consecutive_stagnation
        base = max(0.0, improvement_cycles / max(cycles, 1))

        # Bonus for avoiding rollbacks
        rollback_penalty = min(0.3, self._rollback_count * 0.05)
        # Bonus for recovery mode working
        recovery_bonus = 0.1 if self._last_stagnation == StagnationResult.NORMAL else 0.0

        score = max(0.0, min(1.0, base - rollback_penalty + recovery_bonus))
        return score

    def get_recovery_confidence(self) -> dict[str, Any]:
        """Assess whether recovery mode is working.

        Returns a dict with:
          - is_recovering (bool): True if recovery_mode is currently active
          - cycles_in_recovery (int): count of consecutive cycles with recovery_mode=True
          - fitness_trend_during_recovery (str): 'improving'/'stable'/'declining'/'unknown',
            computed from telemetry during recovery cycles
          - confidence (float 0-1): based on fitness trend
          - should_exit_recovery (bool): True if cycles_in_recovery >= 5 AND
            fitness_trend is 'stable' or 'declining'
        """
        is_recovering = self.recovery_mode

        # Count consecutive recovery cycles from tuning log entries
        cycles_in_recovery = 0
        for action in reversed(self._tuning_log):
            if "recovery" in action.reason:
                cycles_in_recovery += 1
            else:
                break

        # Collect recovery cycle numbers from tuning log
        recovery_cycles: set = set()
        for action in self._tuning_log:
            if "recovery" in action.reason:
                recovery_cycles.add(action.cycle)

        # Determine fitness trend during recovery from telemetry
        fitness_trend = "unknown"
        confidence = 0.3

        if recovery_cycles:
            recovery_fitness = [t.fitness for t in self.telemetry if t.cycle in recovery_cycles]
            if len(recovery_fitness) >= 3:
                improving_count = sum(
                    1 for i in range(1, len(recovery_fitness)) if recovery_fitness[i] > recovery_fitness[i - 1]
                )
                declining_count = sum(
                    1 for i in range(1, len(recovery_fitness)) if recovery_fitness[i] < recovery_fitness[i - 1]
                )

                if improving_count > declining_count:
                    fitness_trend = "improving"
                    confidence = 0.8
                elif declining_count > improving_count:
                    fitness_trend = "declining"
                    confidence = 0.1
                else:
                    fitness_trend = "stable"
                    confidence = 0.4

        should_exit_recovery = cycles_in_recovery >= 5 and fitness_trend in ("stable", "declining")

        return {
            "is_recovering": is_recovering,
            "cycles_in_recovery": cycles_in_recovery,
            "fitness_trend_during_recovery": fitness_trend,
            "confidence": confidence,
            "should_exit_recovery": should_exit_recovery,
        }

    def get_tuning_escalation_history(self) -> list[dict[str, Any]]:
        """Return a chronological list of all escalation events.

        Each event is derived from the tuning log, representing a tuning action
        that was taken in response to stagnation detection.
        Returns a list of dicts with:
          - cycle: the cycle number when the escalation occurred
          - stagnation_type: the stagnation state value (e.g. 'fitness_plateau')
          - action_taken: description of the tuning action applied
          - effectiveness_before: effectiveness score at time of detection
        """
        history: list[dict[str, Any]] = []
        for a in self._tuning_log:
            stagnation_type = a.reason.split(":")[0] if ":" in a.reason else a.reason
            history.append(
                {
                    "cycle": a.cycle,
                    "stagnation_type": stagnation_type,
                    "action_taken": f"{a.domain}.{a.param} = {a.new_value}",
                    "effectiveness_before": None,
                }
            )
        return history

    def get_cross_cycle_trends(self, window: int = 5) -> dict[str, Any]:
        """Analyze cross-cycle trends for fitness, val_loss, and diversity.

        Uses linear regression across the telemetry history to compute trend
        direction, slope, and confidence for each metric. Trends help identify
        whether the system is improving, stable, or declining over time.

        Args:
            window: Minimum number of cycles for trend analysis. Default is 5.
                With fewer cycles than this, confidence is reduced.

        Returns:
            Dict with keys 'fitness', 'val_loss', 'diversity', each containing:
            - 'direction': 'improving', 'stable', or 'declining'
            - 'slope': float (per-cycle change rate)
            - 'confidence': float in [0, 1] (higher = more consistent trend)

        Note:
            Direction is interpreted relative to metric semantics:
            - fitness: increasing slope -> improving
            - diversity: increasing slope -> improving
            - val_loss: decreasing slope -> improving (loss is getting better)
        """

        def _linear_regression(values):
            """Compute slope and R-squared for a list of values."""
            n = len(values)
            if n < 2:
                return (0.0, 0.0)

            x_mean = (n - 1) / 2.0
            y_mean = sum(values) / n

            numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                return (0.0, 0.0)

            slope = numerator / denominator

            # Compute R-squared
            y_pred = [slope * (i - x_mean) + y_mean for i in range(n)]
            ss_res = sum((v - p) ** 2 for v, p in zip(values, y_pred))
            ss_tot = sum((v - y_mean) ** 2 for v in values)

            if ss_tot == 0:
                r_squared = 1.0 if ss_res == 0 else 0.0
            else:
                r_squared = max(0.0, 1.0 - (ss_res / ss_tot))

            return (slope, r_squared)

        def _compute_trend(values):
            """Compute trend direction, slope, and confidence for a metric."""
            if len(values) < 2:
                return {"direction": "stable", "slope": 0.0, "confidence": 0.0}

            slope, r_squared = _linear_regression(values)

            # Adaptive threshold: 1% of the metric's range
            value_range = max(values) - min(values)
            slope_threshold = 0.01 * value_range if value_range > 0 else 1e-10

            if slope > slope_threshold:
                direction = "improving"
            elif slope < -slope_threshold:
                direction = "declining"
            else:
                direction = "stable"

            # Confidence based on R-squared and sample size
            sample_factor = min(1.0, len(values) / max(window, 1))
            confidence = min(1.0, max(0.0, r_squared * sample_factor))

            return {"direction": direction, "slope": slope, "confidence": confidence}

        # Extract metric series from telemetry
        fitness_values = [t.fitness for t in self.telemetry]
        val_loss_values = [t.val_loss for t in self.telemetry if t.val_loss is not None]
        diversity_values = [t.diversity for t in self.telemetry]

        # Compute fitness and diversity trends (higher = better)
        fitness_trend = _compute_trend(fitness_values)
        diversity_trend = _compute_trend(diversity_values)

        # Compute val_loss trend (lower = better, so invert direction)
        if val_loss_values:
            val_loss_trend = _compute_trend(val_loss_values)
            # Invert direction for val_loss: decreasing slope = improving
            if val_loss_trend["direction"] == "improving":
                val_loss_trend["direction"] = "declining"
            elif val_loss_trend["direction"] == "declining":
                val_loss_trend["direction"] = "improving"
        else:
            val_loss_trend = {"direction": "stable", "slope": 0.0, "confidence": 0.0}

        self._log(
            f"Trends: fitness={fitness_trend['direction']} "
            f"(slope={fitness_trend['slope']:.4f}, conf={fitness_trend['confidence']:.2f}), "
            f"val_loss={val_loss_trend['direction']} "
            f"(slope={val_loss_trend['slope']:.4f}, conf={val_loss_trend['confidence']:.2f}), "
            f"diversity={diversity_trend['direction']} "
            f"(slope={diversity_trend['slope']:.4f}, conf={diversity_trend['confidence']:.2f})"
        )

        return {
            "fitness": fitness_trend,
            "val_loss": val_loss_trend,
            "diversity": diversity_trend,
        }

    def _log(self, msg: str) -> None:
        """Log message via structured logger and optional log file.

        Uses lazy file handle caching (same pattern as EvolutionEngine._log)
        to avoid fd exhaustion on long multi-cycle runs where _log is called
        hundreds of times per cycle.
        """
        _logger.info(msg)
        if self.log_file:
            try:
                if self._log_file is None:
                    self._log_file = open(self.log_file, "a", encoding="utf-8")
                self._log_file.write(f"[MetaController] {msg}\n")
                self._log_file.flush()
            except (OSError, ValueError):
                # File handle stale or directory deleted -- reset and retry once
                self._log_file = None
                try:
                    self._log_file = open(self.log_file, "a", encoding="utf-8")
                    self._log_file.write(f"[MetaController] {msg}\n")
                    self._log_file.flush()
                except (OSError, ValueError):
                    pass


# ══════════════════════════════════════════════════════════════════════
# Convenience: parse val_loss from retrain output
# ══════════════════════════════════════════════════════════════════════


# Pre-compiled regex patterns for parse_val_loss_from_retrain_log
# (avoids re-compilation on every call — called once per cycle per retrain)
_VAL_PATTERNS = (
    _re.compile(r"val=([0-9]+\.[0-9]+)"),
    _re.compile(r"val_loss=([0-9]+\.[0-9]+)"),
    _re.compile(r"Val Loss:\s*([0-9]+\.[0-9]+)"),
)
_VAL_LOSS_RE = _re.compile(r"val[_=]?(?:loss)?:\s*([0-9]+\.[0-9]+)")
_GRAD_NORM_RE = _re.compile(r"grad_norm[/\s](\S+?)\s*[=:]\s*([0-9]+\.?[0-9]*)")


def parse_val_loss_from_retrain_log(log_path: str) -> float | None:
    """Extract final validation loss from a retrain log file.

    Matches multiple output formats from train_model.py:
      - "val=X.XXXX" (epoch line)
      - "val_loss=X.XXXX" (saved checkpoint line)
      - "Val Loss: X.XXXX" (legacy format)
    Returns the minimum (best) val_loss found, or None if not found.
    """
    if not os.path.isfile(log_path):
        return None
    try:
        best_val = None
        with open(log_path) as f:
            for line in f:
                for pat in _VAL_PATTERNS:
                    m = pat.search(line)
                    if m:
                        val = float(m.group(1))
                        if best_val is None or val < best_val:
                            best_val = val
                        break  # first matching pattern per line
        return best_val
    except (OSError, PermissionError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════
# Convenience: parse gradient norms from retrain output
# ══════════════════════════════════════════════════════════════════════


def parse_grad_norms_from_retrain_log(log_path: str) -> dict[str, float] | None:
    """Extract per-layer gradient norms from a retrain log file.

    Matches output formats like:
      - "grad_norm/layer.0=0.0234"
      - "grad_norm blocks.0.attn=0.0156"

    Returns dict mapping parameter names to gradient norm values,
    or None if no gradient norms found.
    """
    if not os.path.isfile(log_path):
        return None
    try:
        norms: dict[str, float] = {}
        with open(log_path) as f:
            for line in f:
                m = _GRAD_NORM_RE.search(line)
                if m:
                    key = m.group(1).strip()
                    val = float(m.group(2))
                    norms[key] = val
        return norms if norms else None
    except (OSError, PermissionError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════
# Simple test / demo
# ══════════════════════════════════════════════════════════════════════


def _demo() -> None:
    """Quick demo of MetaController tuning logic."""
    mc = MetaController(checkpoint_dir="/tmp/mc_demo")

    # Simulate first 3 cycles: improving
    for c, f, l, v in [(1, -3.0, 9, 3.0), (2, -2.5, 10, 2.5), (3, -2.0, 11, 2.0)]:
        snap = TelemetrySnapshot(
            cycle=c,
            fitness=f,
            level=l,
            champion_words=50,
            val_loss=v,
            diversity=0.3,
            novelty_size=10,
            qd_occupied=5,
            qd_total=100,
            traces_count=20,
            retrained=True,
            generations_completed=50,
        )
        mc.update_cycle(snap)
        _logger.info(f"C{c}: state={mc.detect_stagnation().value}, " f"retrain_params={mc.tune_retrain_params(c)}")

    # Simulate plateau: fitness flat for 3 cycles
    for c, f, l in [(4, -1.8, 11), (5, -1.8, 11), (6, -1.8, 11)]:
        snap = TelemetrySnapshot(
            cycle=c,
            fitness=f,
            level=l,
            champion_words=50,
            val_loss=1.8,
            diversity=0.25,
            novelty_size=10,
            qd_occupied=5,
            qd_total=100,
            traces_count=25,
            retrained=True,
            generations_completed=50,
        )
        mc.update_cycle(snap)
        _logger.info(f"C{c}: state={mc.detect_stagnation().value}, " f"retrain_params={mc.tune_retrain_params(c)}")

    # Simulate regression
    snap = TelemetrySnapshot(
        cycle=7,
        fitness=-1.9,
        level=11,
        champion_words=50,
        val_loss=7.7,
        diversity=0.2,
        novelty_size=10,
        qd_occupied=5,
        qd_total=100,
        traces_count=30,
        retrained=True,
        generations_completed=50,
    )
    mc.update_cycle(snap)
    _logger.info(
        f"C7: state={mc.detect_stagnation().value}, "
        f"rollback={mc.should_rollback()}, "
        f"retrain_params={mc.tune_retrain_params(7)}"
    )

    _logger.info("\nState summary:")
    _logger.info(json.dumps(mc.get_state_summary(), indent=2))
    _logger.info("\nDemo complete.")


if __name__ == "__main__":
    _demo()
