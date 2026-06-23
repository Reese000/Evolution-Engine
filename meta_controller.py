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

import json
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════════════════════════════


class StagnationResult(Enum):
    """Classifications of system state based on telemetry."""
    NORMAL = "normal"
    FITNESS_PLATEAU = "fitness_plateau"       # fitness flat for N cycles
    LEVEL_STALLED = "level_stalled"            # same level for N cycles
    VAL_LOSS_REGRESSION = "val_loss_regression"  # val loss spiked
    DIVERSITY_COLLAPSE = "diversity_collapse"  # diversity below threshold
    COMPOUND_STALL = "compound_stall"          # multiple indicators


# ══════════════════════════════════════════════════════════════════════
# Data containers
# ══════════════════════════════════════════════════════════════════════


@dataclass
class TelemetrySnapshot:
    """Immutable snapshot of system state at the end of a cycle."""
    cycle: int
    fitness: float
    level: int
    champion_words: int
    val_loss: Optional[float]          # None = retrain didn't run or wasn't captured
    diversity: float                   # population diversity at end of cycle
    novelty_size: int
    qd_occupied: int
    qd_total: int
    traces_count: int
    retrained: bool                    # whether retrain ran this cycle
    generations_completed: int

    def to_dict(self) -> Dict[str, Any]:
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
    def __init__(self, domain: str, param: str, old_value: Any, new_value: Any,
                 reason: str):
        self.domain = domain          # "retrain", "evolution", "recovery"
        self.param = param            # e.g. "retrain_lr"
        self.old_value = old_value
        self.new_value = new_value
        self.reason = reason          # e.g. "stagnation: fitness_plateau"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "param": self.param,
            "old": self.old_value,
            "new": self.new_value,
            "reason": self.reason,
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
        initial_val_loss: float = float('inf'),
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
        checkpoint_acceptance_tolerance: float = 1.05,
        # ── Logging ────────────────────────────────────────────────────
        log_file: Optional[str] = None,
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
        self.telemetry: List[TelemetrySnapshot] = []

        # Best known values
        self.best_fitness: float = -float('inf')
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
        self._tuning_log: List[TuningAction] = []
        self._gated_count: int = 0
        self._rollback_count: int = 0
        self._last_snapshot_path: str = ""

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

        if snapshot.champion_words > self.best_champion_words:
            self.best_champion_words = snapshot.champion_words

        # Update best val loss if we got a valid measurement
        if snapshot.val_loss is not None and snapshot.val_loss > 0:
            if snapshot.val_loss < self._best_val_loss:
                self._best_val_loss = snapshot.val_loss

        # Append to history
        self.telemetry.append(snapshot)
        self._log(f"Cycle {snapshot.cycle}: F={snapshot.fitness:.3f} L={snapshot.level} "
                  f"val={snapshot.val_loss} div={snapshot.diversity:.3f} "
                  f"retrain={snapshot.retrained}")

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
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
            self._log(f"Gate: ACCEPT val_loss={val_loss:.4f} "
                      f"(best={old_best:.4f}, threshold={threshold:.4f})")
            self._gated_count += 1
            return True
        else:
            # Reject — model is worse
            self._log(f"Gate: REJECT val_loss={val_loss:.4f} > threshold={threshold:.4f} "
                      f"(best={self._best_val_loss:.4f})")
            self._gated_count += 1
            return False

    def detect_stagnation(self) -> StagnationResult:
        """Classify current system state based on recent telemetry."""
        if len(self.telemetry) < 2:
            return StagnationResult.NORMAL

        # Collect recent window
        window = self.telemetry[-min(5, len(self.telemetry)):]
        fitnesses = [t.fitness for t in window]
        levels = [t.level for t in window]
        val_losses = [t.val_loss for t in window if t.val_loss is not None]
        diversities = [t.diversity for t in window]

        indications: List[StagnationResult] = []

        # 1. Fitness plateau
        if self.cycles_since_fitness_improvement >= self.fitness_plateau_cycles:
            if len(fitnesses) >= 2:
                f_range = max(fitnesses) - min(fitnesses)
                if f_range < 0.1:  # Flat plateau
                    indications.append(StagnationResult.FITNESS_PLATEAU)

        # 2. Level stall
        if self.cycles_since_level_up >= self.level_stall_cycles:
            indications.append(StagnationResult.LEVEL_STALLED)

        # 3. Val loss regression
        if len(val_losses) >= 2:
            if val_losses[-1] > val_losses[0] * self.val_loss_regression_ratio:
                indications.append(StagnationResult.VAL_LOSS_REGRESSION)

        # 4. Diversity collapse
        if diversities and min(diversities) < self.diversity_collapse_threshold:
            indications.append(StagnationResult.DIVERSITY_COLLAPSE)

        # Classify
        if len(indications) >= 2:
            result = StagnationResult.COMPOUND_STALL
        elif len(indications) == 1:
            result = indications[0]
        else:
            result = StagnationResult.NORMAL

        self._last_stagnation = result
        return result

    def tune_retrain_params(self, cycle: int) -> Dict[str, Any]:
        """Return dynamically tuned retrain hyperparameters.

        This is the central tuning function for the retrain pipeline.
        Returns a dict of parameter names to values that should be passed
        to run_multi_cycle's retrain subprocess.

        Args:
            cycle: Current cycle number (1-indexed).

        Returns:
            Dict of retrain parameters (empty dict = use defaults).
        """
        params: Dict[str, Any] = {}

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
            self._record_tuning("retrain", "retrain_lr", None, base_lr,
                                "stagnation: fitness_plateau")

        elif state == StagnationResult.LEVEL_STALLED:
            # Level stall: moderate LR, reduced freezing for plasticity
            base_lr = min(self.lr_max, base_lr * 1.5)
            base_freeze = max(self.freeze_min, base_freeze - 1)
            self._record_tuning("retrain", "retrain_lr", None, base_lr,
                                "stagnation: level_stalled")

        elif state == StagnationResult.VAL_LOSS_REGRESSION:
            # Regression: protect existing knowledge, slow learning
            base_lr = max(self.lr_min, base_lr * 0.3)
            base_freeze = min(self.freeze_max, base_freeze + 2)
            self._record_tuning("retrain", "retrain_lr", None, base_lr,
                                "regression: val_loss_regression")

        elif state == StagnationResult.DIVERSITY_COLLAPSE:
            # Diversity collapse: escalate retrain params
            if consecutive <= 2:
                # Initial: moderate LR, minimal freezing
                base_freeze = max(self.freeze_min, base_freeze - 1)
                params["ref_high_augment"] = 100
            elif consecutive <= 4:
                # Escalation 1: high LR, no freezing, more augment
                base_freeze = 0
                base_lr = min(self.lr_max, base_lr * 2.0)
                params["ref_high_augment"] = 200
            else:
                # Escalation 2: max LR, no freezing, max augment
                base_freeze = 0
                base_lr = self.lr_max
                params["ref_high_augment"] = 500
            self._record_tuning("retrain", "retrain_freeze_layers",
                                None, base_freeze,
                                f"stagnation: diversity_collapse (tier {min(consecutive, 3)})")

        elif state == StagnationResult.COMPOUND_STALL:
            # Multiple indicators: conservative approach
            if self._best_val_loss < float('inf'):
                # We have a good model — protect it aggressively
                base_freeze = min(self.freeze_max, base_freeze + 1)
            else:
                # No good model yet — explore
                base_lr = min(self.lr_max, base_lr * 1.5)
                base_freeze = max(self.freeze_min, base_freeze - 1)
            self._record_tuning("retrain", "retrain_freeze_layers",
                                None, base_freeze,
                                "stagnation: compound_stall")

        # ── Per-level data diversity ─────────────────────────────────
        if cycle >= 3:
            per_level_cap = min(200 + 50 * (cycle - 3), 500)
            params["retrain_max_per_level"] = per_level_cap

        # ── Recovery mode override ───────────────────────────────────
        if self.recovery_mode:
            base_lr = max(self.lr_min, base_lr * 0.5)
            base_patience = min(self.patience_max, base_patience + 2)
            base_freeze = min(self.freeze_max, base_freeze + 1)
            self._record_tuning("retrain", "retrain_lr", None, base_lr,
                                "recovery_mode")
            self.recovery_cycles_remaining -= 1
            if self.recovery_cycles_remaining <= 0:
                self.recovery_mode = False

        # ── Assemble result ──────────────────────────────────────────
        params["retrain_lr"] = base_lr
        params["retrain_freeze_layers"] = base_freeze
        params["retrain_patience"] = min(self.patience_max, base_patience)
        params["retrain_max_epochs"] = base_epochs

        # ── High-level reference augmentation ──────────────────────
        # When diversity is collapsed or multiple indicators, inject extra
        # L11-L13 reference examples to break out of local minima.
        if state in (StagnationResult.DIVERSITY_COLLAPSE,
                     StagnationResult.COMPOUND_STALL):
            params["ref_high_augment"] = 200
        elif state == StagnationResult.LEVEL_STALLED:
            params["ref_high_augment"] = 100
        else:
            params["ref_high_augment"] = 50

        self._log(f"Retrain params for C{cycle}: LR={base_lr:.2e} "
                  f"freeze={base_freeze} patience={base_patience} "
                  f"epochs={base_epochs} hl_ref={params['ref_high_augment']} "
                  f"state={state.value}")

        # v11 P3.2: Request meta-evolution if compound_stall persists
        if state == StagnationResult.COMPOUND_STALL:
            # Count consecutive compound_stalls
            if not hasattr(self, '_compound_stall_counter'):
                self._compound_stall_counter = 0
            self._compound_stall_counter += 1
            if self._compound_stall_counter >= 3 and cycle >= 10:
                params["meta_evolution_requested"] = True
                self._log(f"  *** META-EVOLUTION REQUESTED (compound_stall x{self._compound_stall_counter})")
        else:
            if hasattr(self, '_compound_stall_counter'):
                self._compound_stall_counter = 0

        return params

    def tune_evolution_params(self, cycle: int) -> Dict[str, Any]:
        """Return dynamically tuned evolution hyperparameters.

        These are applied as overrides to EvolutionConfig before
        running evolution each cycle.

        Returns:
            Dict of evolution parameters (empty dict = use defaults).
        """
        params: Dict[str, Any] = {}
        state = self.detect_stagnation()

        # Count consecutive stagnation events for escalation
        consecutive = self._count_consecutive_stagnation(state)

        if state == StagnationResult.FITNESS_PLATEAU:
            params["mutation_rate"] = 0.70
            params["diversity_threshold"] = 0.20
            params["diversity_inject_ratio"] = 0.20
            params["novelty_weight"] = 0.30
            self._record_tuning("evolution", "mutation_rate", None, 0.70,
                                "stagnation: fitness_plateau")

        elif state == StagnationResult.LEVEL_STALLED:
            params["mutation_rate"] = 0.60
            params["diversity_threshold"] = 0.25
            params["diversity_inject_ratio"] = 0.25
            params["novelty_weight"] = 0.35
            self._record_tuning("evolution", "novelty_weight", None, 0.35,
                                "stagnation: level_stalled")

        elif state == StagnationResult.DIVERSITY_COLLAPSE:
            # Escalate: progressively stronger fixes for persistent collapse
            if consecutive <= 2:
                # Initial fix: increase injection ratio
                params["diversity_inject_ratio"] = 0.30
                params["mutation_rate"] = 0.65
                params["diversity_threshold"] = 0.15
                self._record_tuning("evolution", "diversity_inject_ratio", None, 0.30,
                                    "stagnation: diversity_collapse (tier 1)")
            elif consecutive <= 4:
                # Escalation 1: champion-level crossover, higher novelty
                params["diversity_inject_ratio"] = 0.50
                params["mutation_rate"] = 0.80
                params["novelty_weight"] = 0.50
                params["champion_crossover_prob"] = 0.85
                params["diversity_threshold"] = 0.10
                self._record_tuning("evolution", "champion_crossover_prob", None, 0.85,
                                    "stagnation: diversity_collapse (tier 2)")
            else:
                # Escalation 2: drastic — large pop, high mutation, heavy injection
                params["diversity_inject_ratio"] = 0.75
                params["mutation_rate"] = 0.90
                params["population_size"] = 120
                params["novelty_weight"] = 0.60
                params["champion_crossover_prob"] = 0.90
                params["diversity_threshold"] = 0.08
                params["elite_ratio"] = 0.05
                self._record_tuning("evolution", "population_size", None, 120,
                                    "stagnation: diversity_collapse (tier 3)")

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
            self._record_tuning("evolution", "elite_ratio", None, params.get("elite_ratio", 0.15),
                                f"regression: val_loss_regression (tier {min(consecutive, 2)})")

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
            self._record_tuning("evolution", "mutation_rate", None, params.get("mutation_rate", 0.75),
                                f"stagnation: compound_stall (tier {min(consecutive, 2)})")

        elif state == StagnationResult.NORMAL and self.recovery_mode:
            # In recovery from previous stagnation: moderate params
            params["mutation_rate"] = 0.55
            params["novelty_weight"] = 0.25
            self._record_tuning("evolution", "mutation_rate", None, 0.55,
                                "recovery_mode")

        return params

    def should_rollback(self) -> bool:
        """Determine if we should roll back to best known checkpoint.

        Rollback triggers when:
        - Val loss regression persists for 2+ cycles AND
        - Fitness hasn't improved for 2+ cycles
        """
        if len(self.telemetry) < 2:
            return False

        state = self.detect_stagnation()
        if state in (StagnationResult.VAL_LOSS_REGRESSION,
                     StagnationResult.COMPOUND_STALL):
            if self.cycles_since_fitness_improvement >= 2:
                self._log(f"ROLLBACK: state={state.value}, "
                          f"cycles_since_fitness_improvement="
                          f"{self.cycles_since_fitness_improvement}")
                self._rollback_count += 1
                return True
        return False

    def get_best_checkpoint(self) -> Optional[str]:
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
        snapshot_dir = os.path.join(os.path.dirname(self.best_checkpoint_path),
                                    "pre_cycle_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"snapshot_cycle_{cycle}.pt")
        try:
            shutil.copy2(checkpoint_path, snapshot_path)
            self._log(f"Snapshot: saved pre-cycle {cycle} checkpoint to {snapshot_path}")
            self._last_snapshot_path = snapshot_path
        except (OSError, shutil.Error) as e:
            self._log(f"Snapshot: failed to save pre-cycle checkpoint: {e}")

    def execute_rollback(self, current_checkpoint: str,
                         use_snapshot: bool = False) -> Optional[str]:
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
        if use_snapshot and hasattr(self, '_last_snapshot_path'):
            source = self._last_snapshot_path
            if not os.path.isfile(source):
                self._log(f"Rollback snapshot not found at {source}, falling back to best")
                source = None
        else:
            source = None

        if source is None:
            source = self.get_best_checkpoint()
            if source is None or not os.path.isfile(source):
                self._log(f"Rollback failed: best checkpoint not found at "
                          f"{self.best_checkpoint_path}")
                return None

        # Guard against same-file copy (source == current_checkpoint)
        abs_source = os.path.abspath(source)
        abs_current = os.path.abspath(current_checkpoint)
        if abs_source == abs_current:
            self._log(f"Rollback skipped: source and current are the same file "
                      f"({source}). Retrain output path must differ from best.")
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

    def get_state_summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary of current state."""
        return {
            "best_fitness": self.best_fitness,
            "best_level": self.best_level,
            "best_val_loss": None if self._best_val_loss == float('inf') else self._best_val_loss,
            "cycles_since_fitness_improvement": self.cycles_since_fitness_improvement,
            "cycles_since_level_up": self.cycles_since_level_up,
            "recovery_mode": self.recovery_mode,
            "recovery_cycles_remaining": self.recovery_cycles_remaining,
            "last_stagnation": self._last_stagnation.value,
            "total_gated": self._gated_count,
            "total_rollbacks": self._rollback_count,
            "total_cycles_observed": len(self.telemetry),
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
            "tuning_history": [
                a.to_dict() for a in self._tuning_log[-20:]  # Last 20 actions
            ],
        }

    def save_state(self, path: str) -> None:
        """Save controller state to a JSON file (for resumption)."""
        state = self.get_state_summary()
        state["_best_val_loss"] = self._best_val_loss
        state["_gated_count"] = self._gated_count
        state["_rollback_count"] = self._rollback_count
        try:
            with open(path, "w") as f:
                json.dump(state, f, indent=2)
        except (OSError, PermissionError) as e:
            self._log(f"Failed to save state: {e}")

    def load_state(self, path: str) -> bool:
        """Load controller state from a JSON file.

        Returns True on success, False if file doesn't exist.
        """
        if not os.path.isfile(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._best_val_loss = data.get("_best_val_loss", float('inf'))
            self.best_fitness = data.get("best_fitness", -float('inf'))
            self.best_level = data.get("best_level", 0)
            self._gated_count = data.get("_gated_count", 0)
            self._rollback_count = data.get("_rollback_count", 0)
            self.recovery_mode = data.get("recovery_mode", False)
            self.recovery_cycles_remaining = data.get("recovery_cycles_remaining", 0)
            return True
        except (json.JSONDecodeError, KeyError, OSError) as e:
            self._log(f"Failed to load state: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _count_consecutive_stagnation(self, state: StagnationResult) -> int:
        """Count how many consecutive tuning events match the given stagnation state.

        Scans the tuning log backwards and counts consecutive entries
        whose reason string contains the state value.
        """
        count = 0
        state_str = state.value
        for t in reversed(self._tuning_log):
            if state_str in t.reason:
                count += 1
            else:
                break
        return count

    def _record_tuning(self, domain: str, param: str,
                       old: Any, new: Any, reason: str) -> None:
        self._tuning_log.append(TuningAction(domain, param, old, new, reason))

    def _log(self, msg: str) -> None:
        """Log message to stdout and optional log file."""
        print(f"[MetaController] {msg}", flush=True)
        if self.log_file:
            try:
                with open(self.log_file, "a") as f:
                    f.write(f"[MetaController] {msg}\n")
            except (OSError, PermissionError):
                pass


# ══════════════════════════════════════════════════════════════════════
# Convenience: parse val_loss from retrain output
# ══════════════════════════════════════════════════════════════════════


def parse_val_loss_from_retrain_log(log_path: str) -> Optional[float]:
    """Extract final validation loss from a retrain log file.

    Looks for the last occurrence of "Val Loss: X.XXXX" in the log.
    Returns None if not found or file doesn't exist.
    """
    if not os.path.isfile(log_path):
        return None
    try:
        import re
        best_val = None
        with open(log_path) as f:
            for line in f:
                m = re.search(r"Val Loss:\s*([0-9]+\.[0-9]+)", line)
                if m:
                    val = float(m.group(1))
                    if best_val is None or val < best_val:
                        best_val = val
        return best_val
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
            cycle=c, fitness=f, level=l, champion_words=50,
            val_loss=v, diversity=0.3, novelty_size=10, qd_occupied=5,
            qd_total=100, traces_count=20, retrained=True,
            generations_completed=50,
        )
        mc.update_cycle(snap)
        print(f"C{c}: state={mc.detect_stagnation().value}, "
              f"retrain_params={mc.tune_retrain_params(c)}")

    # Simulate plateau: fitness flat for 3 cycles
    for c, f, l in [(4, -1.8, 11), (5, -1.8, 11), (6, -1.8, 11)]:
        snap = TelemetrySnapshot(
            cycle=c, fitness=f, level=l, champion_words=50,
            val_loss=1.8, diversity=0.25, novelty_size=10, qd_occupied=5,
            qd_total=100, traces_count=25, retrained=True,
            generations_completed=50,
        )
        mc.update_cycle(snap)
        print(f"C{c}: state={mc.detect_stagnation().value}, "
              f"retrain_params={mc.tune_retrain_params(c)}")

    # Simulate regression
    snap = TelemetrySnapshot(
        cycle=7, fitness=-1.9, level=11, champion_words=50,
        val_loss=7.7, diversity=0.2, novelty_size=10, qd_occupied=5,
        qd_total=100, traces_count=30, retrained=True,
        generations_completed=50,
    )
    mc.update_cycle(snap)
    print(f"C7: state={mc.detect_stagnation().value}, "
          f"rollback={mc.should_rollback()}, "
          f"retrain_params={mc.tune_retrain_params(7)}")

    print("\nState summary:")
    print(json.dumps(mc.get_state_summary(), indent=2))
    print("\nDemo complete.")


if __name__ == "__main__":
    _demo()
