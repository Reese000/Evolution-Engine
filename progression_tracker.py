"""ProgressionMetrics — track improvement across evolution cycles.

Tracks not just "level reached" but quality metrics:
- Champion word count (efficiency)
- Instruction diversity
- Model val_loss trend
- Archive growth
- MG success rate

Used by MetaController to auto-tune training and evolution parameters.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from milestone1.training.scaling.types import TrendSnapshot


def _load_scaling_module(name: str):
    """Load a module from milestone1/training/scaling/ without triggering
    milestone1/training/__init__.py (which imports torch).

    Uses importlib to load the .py file directly and registers it
    in sys.modules so relative imports within the scaling package work.

    Crucially: restores any stubbed parent packages after loading
    to avoid polluting sys.modules for other test files.
    """
    import pathlib as _pl
    import types as _types_mod

    pkg_name = "milestone1.training.scaling"
    full_name = f"{pkg_name}.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    # Track which parents we stub (to restore after loading)
    stubbed_parents = []
    for parent in ("milestone1", "milestone1.training", pkg_name):
        if parent not in sys.modules:
            sys.modules[parent] = _types_mod.ModuleType(parent)
            sys.modules[parent].__path__ = []
            stubbed_parents.append(parent)

    scaling_dir = _pl.Path(__file__).resolve().parent / "milestone1" / "training" / "scaling"
    file_path = scaling_dir / f"{name}.py"
    spec = _ilu.spec_from_file_location(full_name, str(file_path), submodule_search_locations=[])
    mod = _ilu.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)

    # Restore: remove stubs we created so they don't interfere
    # with other modules' imports in pytest's full suite
    for parent in stubbed_parents:
        if parent in sys.modules and hasattr(sys.modules[parent], "__path__") and not sys.modules[parent].__path__:
            del sys.modules[parent]

    return mod


@dataclass
class CycleMetrics:
    """Metrics for a single evolution cycle."""

    cycle: int
    level: int
    fitness: float
    words: int
    archive_size: int
    val_loss: Optional[float]
    time_s: float
    mg_count: int
    total_challengers: int
    champion_improvements: int

    @property
    def mg_ratio(self) -> float:
        return self.mg_count / max(self.total_challengers, 1)


class ProgressionTracker:
    """Track progression across cycles for MetaController decisions."""

    def __init__(self, path: str = "champions/progression.jsonl"):
        self.path = path
        self.history: List[CycleMetrics] = []
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    self.history.append(CycleMetrics(**d))
        except Exception:
            pass

    def record(self, metrics: CycleMetrics) -> None:
        """Record a cycle's metrics."""
        self.history.append(metrics)
        self.save()

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            for m in self.history:
                f.write(json.dumps(asdict(m)) + "\n")

    @property
    def last_cycle(self) -> Optional[CycleMetrics]:
        return self.history[-1] if self.history else None

    def fitness_trend(self, window: int = 5) -> float:
        """Positive = improving, negative = regressing."""
        recent = self.history[-window:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1].fitness - recent[0].fitness) / max(len(recent) - 1, 1)

    def val_loss_trend(self, window: int = 5) -> float:
        """Negative = improving (lower loss is better)."""
        recent = [m for m in self.history[-window:] if m.val_loss is not None]
        if len(recent) < 2:
            return 0.0
        return (recent[-1].val_loss - recent[0].val_loss) / max(len(recent) - 1, 1)

    def words_trend(self, window: int = 5) -> float:
        """Positive = champion growing (more complex programs)."""
        recent = self.history[-window:]
        if len(recent) < 2:
            return 0.0
        return (recent[-1].words - recent[0].words) / max(len(recent) - 1, 1)

    def mg_ratio_avg(self, window: int = 5) -> float:
        recent = self.history[-window:]
        if not recent:
            return 0.0
        return sum(m.mg_ratio for m in recent) / len(recent)

    def is_stagnant(self, window: int = 5) -> bool:
        """True if no improvement in fitness or words over window."""
        if len(self.history) < window:
            return False
        recent = self.history[-window:]
        fitness_stable = max(m.fitness for m in recent) - min(m.fitness for m in recent) < 5.0
        words_stable = max(m.words for m in recent) - min(m.words for m in recent) < 3
        return fitness_stable and words_stable

    def suggest_adjustments(self) -> Dict[str, Any]:
        """Suggest hyperparameter adjustments based on progression."""
        suggestions = {}
        ft = self.fitness_trend()
        vt = self.val_loss_trend()
        wt = self.words_trend()
        mg_avg = self.mg_ratio_avg()

        if ft < 0:
            suggestions["action"] = "increase_exploration"
            suggestions["reason"] = f"fitness declining ({ft:.1f}/cycle)"
        elif wt < 0:
            suggestions["action"] = "increase_mg_ratio"
            suggestions["reason"] = f"champion shrinking ({wt:.1f} words/cycle)"
        elif self.is_stagnant():
            suggestions["action"] = "increase_training_budget"
            suggestions["reason"] = "stagnant for 5+ cycles"
        elif vt > 0.001:
            suggestions["action"] = "reduce_training_epochs"
            suggestions["reason"] = f"val_loss increasing ({vt:.4f}/cycle)"

        return suggestions

    def diversity_trend(self, window: int = 5) -> float:
        """Return average archive diversity over recent cycles.

        Uses the mg_ratio as a proxy for diversity (higher MG ratio
        means more model-guided exploration which correlates with
        archive diversity).  Returns a value in [0, 1].
        """
        recent = self.history[-window:]
        if not recent:
            return 0.5  # neutral default
        return self.mg_ratio_avg(window)

    def plateau_cycles(self, threshold: float = 1.0, window: int = 5) -> int:
        """Count consecutive cycles where fitness slope < threshold."""
        count = 0
        for i in range(len(self.history) - 1, 0, -1):
            slope = self.history[i].fitness - self.history[i - 1].fitness
            if abs(slope) < threshold:
                count += 1
            else:
                break
        return count

    def get_plateau_analysis(self, window: int = 10) -> Dict[str, Any]:
        """Detailed plateau analysis over the last *window* cycles.

        Returns
        -------
        dict with keys:
            is_plateau          – True when max-min < 5 % of mean fitness
            plateau_start_cycle – cycle number where fitness first stopped improving
            plateau_duration    – number of cycles since the plateau started
            fitness_variance    – variance of fitness values in the window
            trend_slope         – linear-regression slope (fitness per cycle)
            status              – 'plateau' | 'improving' | 'declining'
        When fewer than *window* entries exist:
            status              – 'insufficient_data'
            cycles_available    – how many entries exist
        """
        if len(self.history) < window:
            return {
                "status": "insufficient_data",
                "cycles_available": len(self.history),
            }

        recent = self.history[-window:]
        fitness_values = [m.fitness for m in recent]
        cycle_numbers = [m.cycle for m in recent]

        n = len(fitness_values)
        mean_val = sum(fitness_values) / n
        min_val = min(fitness_values)
        max_val = max(fitness_values)

        # --- is_plateau: range < threshold * mean (5 % relative variation) ---
        threshold = 0.05
        is_plateau = (max_val - min_val) < (threshold * mean_val) if mean_val != 0 else False

        # --- fitness_variance ---
        fitness_variance = sum((v - mean_val) ** 2 for v in fitness_values) / n

        # --- trend_slope via ordinary least-squares linear regression ---
        #   y = slope * x + intercept
        x = list(range(n))
        x_mean = sum(x) / n
        slope_num = sum((x[i] - x_mean) * (fitness_values[i] - mean_val) for i in range(n))
        slope_den = sum((x[i] - x_mean) ** 2 for i in range(n))
        trend_slope = slope_num / slope_den if slope_den != 0 else 0.0

        # --- plateau_start_cycle ---
        #   First cycle in the window after which fitness never exceeded
        #   the current running maximum.
        running_max = float("-inf")
        plateau_start_cycle = cycle_numbers[-1]  # default: no plateau
        for i, fv in enumerate(fitness_values):
            if fv > running_max:
                running_max = fv
                plateau_start_cycle = cycle_numbers[i]

        # If the peak is the very first entry, the plateau spans the
        # entire window – report the first cycle.
        plateau_duration = cycle_numbers[-1] - plateau_start_cycle

        # --- status classification ---
        if is_plateau:
            status = "plateau"
        elif trend_slope > 0:
            status = "improving"
        else:
            status = "declining"

        return {
            "is_plateau": is_plateau,
            "plateau_start_cycle": plateau_start_cycle,
            "plateau_duration": plateau_duration,
            "fitness_variance": fitness_variance,
            "trend_slope": trend_slope,
            "status": status,
        }

    def compare_windows(
        self,
        metric: str,
        window_a: int = 5,
        window_b: int = 5,
    ) -> Dict[str, Any]:
        """Compare a metric across two consecutive time windows.

        *window_b* is the older slice, *window_a* the newer.

        Parameters
        ----------
        metric : str
            CycleMetrics field name (e.g. 'fitness', 'val_loss', 'words').
        window_a, window_b : int
            Number of cycles in each window.

        Returns
        -------
        dict with keys:
            metric          – the metric name
            window_a_mean   – mean of the newer window
            window_b_mean   – mean of the older window
            improvement     – mean_a − mean_b
            percent_change  – (improvement / mean_b) × 100
            direction       – 'improving' | 'stable' | 'declining'
            window_a_stddev – standard deviation of the newer window
            window_b_stddev – standard deviation of the older window
        """
        if not self.history:
            return {
                "metric": metric,
                "window_a_mean": 0.0,
                "window_b_mean": 0.0,
                "improvement": 0.0,
                "percent_change": 0.0,
                "direction": "stable",
                "window_a_stddev": 0.0,
                "window_b_stddev": 0.0,
            }

        needed = window_a + window_b
        if len(self.history) < needed:
            # Fall back: use whatever is available, split as evenly as possible
            total = len(self.history)
            window_b_actual = min(window_b, total)
            window_a_actual = min(window_a, total - window_b_actual)
        else:
            window_a_actual = window_a
            window_b_actual = window_b

        # Slice: older (window_b) then newer (window_a)
        start_b = len(self.history) - window_a_actual - window_b_actual
        end_b = start_b + window_b_actual
        slice_b = self.history[start_b:end_b]  # older window
        slice_a = self.history[end_b:]  # newer window

        def _extract(entries: list, attr_name: str) -> List[float]:
            values = []
            for m in entries:
                v = getattr(m, attr_name, None)
                if v is not None:
                    values.append(float(v))
            return values

        vals_b = _extract(slice_b, metric)
        vals_a = _extract(slice_a, metric)

        def _stats(values: List[float]):
            if not values:
                return 0.0, 0.0
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            return mean, variance**0.5

        mean_b, std_b = _stats(vals_b)
        mean_a, std_a = _stats(vals_a)

        improvement = mean_a - mean_b
        percent_change = (improvement / mean_b * 100) if mean_b != 0 else 0.0

        if improvement > 0:
            direction = "improving"
        elif improvement < 0:
            direction = "declining"
        else:
            direction = "stable"

        return {
            "metric": metric,
            "window_a_mean": mean_a,
            "window_b_mean": mean_b,
            "improvement": improvement,
            "percent_change": percent_change,
            "direction": direction,
            "window_a_stddev": std_a,
            "window_b_stddev": std_b,
        }

    def emit_snapshot(self, window: int = 20, current_archive_size: int = None) -> "TrendSnapshot":
        """Create a TrendSnapshot from the tracker's history for SaturationDetector.

        Parameters
        ----------
        window : int
            Number of most-recent cycles to include (default 20).
        current_archive_size : int, optional
            Current archive size (overrides stale recorded value).
        """
        _types = _load_scaling_module("types")
        TrendSnapshot = _types.TrendSnapshot
        recent = self.history[-window:] if self.history else []
        archive_size = (
            current_archive_size if current_archive_size is not None else (recent[-1].archive_size if recent else 0)
        )
        return TrendSnapshot(
            fitness_history=[m.fitness for m in recent],
            val_loss_history=[m.val_loss for m in recent if m.val_loss is not None],
            max_level_history=[m.level for m in recent],
            archive_size=archive_size,
            cycles_observed=len(self.history),
        )

    def is_saturated(self, window: int = 20) -> bool:
        """Check if training has saturated by delegating to SaturationDetector.

        Returns True only if ALL four saturation criteria are met
        (fitness plateau, val_loss plateau, curriculum saturated, archive sufficient).
        """
        _sd = _load_scaling_module("saturation_detector")
        SaturationDetector = _sd.SaturationDetector
        snapshot = self.emit_snapshot(window)
        detector = SaturationDetector()
        report = detector.evaluate(snapshot)
        return report.should_scale

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"cycles": 0}
        return {
            "cycles": len(self.history),
            "best_fitness": max(m.fitness for m in self.history),
            "best_words": max(m.words for m in self.history),
            "fitness_trend": self.fitness_trend(),
            "val_loss_trend": self.val_loss_trend(),
            "words_trend": self.words_trend(),
            "mg_ratio_avg": self.mg_ratio_avg(),
            "is_stagnant": self.is_stagnant(),
            "suggestions": self.suggest_adjustments(),
        }


# ═══════════════════════════════════════════════════════════════════════
# Auto-tuning: TuningPolicy + ParameterActuator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TuningAdjustment:
    """A single hyperparameter change to apply."""

    parameter: str  # name on EvolutionConfig
    value: Any  # new value to set
    reason: str  # human-readable justification
    magnitude: float = 0.0  # absolute change from current (for logging)


class TuningPolicy:
    """Convert ProgressionTracker trends into TuningAdjustments.

    Rules (evaluated in priority order, one applied per call):
      1. Fitness plateau (slope < 1.0 for 5+ cycles)
         → increase mutation_rate, increase MG explorer ratio
      2. val_loss increasing
         → increase training_epochs (retrain_max_epochs)
      3. val_loss decreasing well
         → reduce training_epochs (prevent overfitting)
      4. Archive diversity dropping
         → boost explorer ratio, increase mutation_rate
      5. Word count stagnation
         → increase curriculum focus, boost SMC bonus

    Rate-limiting:
      - Never adjusts the same parameter two cycles in a row
      - Cooldown: each parameter must wait at least `cooldown` cycles
        after its last adjustment before being eligible again.
    """

    def __init__(self, cooldown: int = 2):
        self.cooldown = cooldown
        # param_name -> cycle number when last adjusted
        self._last_adjusted: Dict[str, int] = {}
        # the parameter adjusted in the most recent call (prevents
        # adjusting the same param twice in a row)
        self._last_param: Optional[str] = None
        # cumulative log of all adjustments applied
        self.adjustment_history: List[TuningAdjustment] = []

    def _eligible(self, param: str, current_cycle: int) -> bool:
        """Check if a parameter is eligible for adjustment."""
        # Never adjust the same param twice in a row
        if param == self._last_param:
            return False
        # Cooldown check
        last = self._last_adjusted.get(param, -999)
        if current_cycle - last < self.cooldown:
            return False
        return True

    def compute_adjustments(
        self,
        tracker: ProgressionTracker,
        current_cycle: int,
        current_config: Optional[Dict[str, Any]] = None,
    ) -> List[TuningAdjustment]:
        """Compute a prioritized list of adjustments.

        Returns the top-eligible adjustment (one per cycle, per the
        single-param constraint).  The caller should apply only the
        first returned adjustment.

        Parameters
        ----------
        tracker : ProgressionTracker
            The tracker containing cycle history.
        current_cycle : int
            The 1-based cycle number about to execute.
        current_config : dict, optional
            Current hyperparameter values (needed for magnitude
            calculations).  Keys should match EvolutionConfig fields.
        """
        if current_config is None:
            current_config = {}

        ft = tracker.fitness_trend()
        vt = tracker.val_loss_trend()
        wt = tracker.words_trend()
        pc = tracker.plateau_cycles()
        mg_avg = tracker.mg_ratio_avg()
        stagnant = tracker.is_stagnant()

        candidates: List[TuningAdjustment] = []

        # -- Rule 1: Fitness plateau ---------------------------------
        if pc >= 5 or (ft < 0 and stagnant):
            cur_mu = current_config.get("mutation_rate", 0.3)
            new_mu = min(cur_mu + 0.1, 0.8)
            candidates.append(
                TuningAdjustment(
                    parameter="mutation_rate",
                    value=new_mu,
                    reason=f"fitness plateau ({pc} cycles, slope={ft:.2f})",
                    magnitude=abs(new_mu - cur_mu),
                )
            )

        # -- Rule 2: val_loss increasing -----------------------------
        if vt > 0.001:
            cur_epochs = current_config.get("retrain_max_epochs", 50)
            new_epochs = min(cur_epochs + 10, 100)
            candidates.append(
                TuningAdjustment(
                    parameter="retrain_max_epochs",
                    value=new_epochs,
                    reason=f"val_loss increasing ({vt:.4f}/cycle)",
                    magnitude=abs(new_epochs - cur_epochs),
                )
            )

        # -- Rule 3: val_loss decreasing well ------------------------
        if vt < -0.01:
            cur_epochs = current_config.get("retrain_max_epochs", 50)
            new_epochs = max(cur_epochs - 5, 10)
            candidates.append(
                TuningAdjustment(
                    parameter="retrain_max_epochs",
                    value=new_epochs,
                    reason=f"val_loss decreasing well ({vt:.4f}/cycle), prevent overfitting",
                    magnitude=abs(new_epochs - cur_epochs),
                )
            )

        # -- Rule 4: Archive diversity dropping ----------------------
        if mg_avg < 0.15 or (ft < 0 and mg_avg < 0.3):
            cur_mu = current_config.get("mutation_rate", 0.3)
            new_mu = min(cur_mu + 0.15, 0.8)
            candidates.append(
                TuningAdjustment(
                    parameter="mutation_rate",
                    value=new_mu,
                    reason=f"diversity dropping (mg_ratio={mg_avg:.3f})",
                    magnitude=abs(new_mu - cur_mu),
                )
            )

        # -- Rule 5: Word count stagnation ---------------------------
        if wt < 0.5 and stagnant:
            cur_curriculum = current_config.get("curriculum_stall_threshold", 3)
            new_curriculum = max(cur_curriculum - 1, 1)
            candidates.append(
                TuningAdjustment(
                    parameter="curriculum_stall_threshold",
                    value=new_curriculum,
                    reason=f"word stagnation (wt={wt:.2f}, stagnant)",
                    magnitude=abs(new_curriculum - cur_curriculum),
                )
            )

        # Filter by eligibility
        eligible = [c for c in candidates if self._eligible(c.parameter, current_cycle)]

        # Return at most one (highest priority = first in list)
        return eligible[:1]

    def record_applied(self, adjustment: TuningAdjustment, cycle: int) -> None:
        """Record that an adjustment was applied (updates cooldown state)."""
        self._last_adjusted[adjustment.parameter] = cycle
        self._last_param = adjustment.parameter
        self.adjustment_history.append(adjustment)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of tuning policy state."""
        return {
            "cooldown": self.cooldown,
            "last_param": self._last_param,
            "last_adjusted": dict(self._last_adjusted),
            "total_adjustments": len(self.adjustment_history),
            "recent": [{"param": a.parameter, "reason": a.reason} for a in self.adjustment_history[-5:]],
        }


class ParameterActuator:
    """Apply TuningAdjustments to an EvolutionConfig-like object.

    Enforces constraints:
      - Only ONE parameter per cycle
      - Never the same param twice in a row (delegated to TuningPolicy)
      - Bounds checking: mutation_rate ∈ [0, 1], elite_ratio ∈ [0, 0.5], etc.

    Usage::
        actuator = ParameterActuator()
        actuator.apply(adjustments[0], engine.config)
    """

    # Parameter bounds: (min, max)
    BOUNDS: Dict[str, tuple[float, float]] = {
        "mutation_rate": (0.05, 0.8),
        "crossover_rate": (0.0, 0.5),
        "elite_ratio": (0.01, 0.3),
        "tournament_size": (2, 10),
        "novelty_weight": (0.0, 1.0),
        "qd_bonus_weight": (0.0, 1.0),
        "diversity_threshold": (0.0, 1.0),
        "diversity_inject_ratio": (0.0, 0.5),
        "retrain_max_epochs": (5, 100),
        "retrain_lr": (1e-6, 1e-3),
        "curriculum_stall_threshold": (1, 10),
    }

    def __init__(self):
        self._applied_count: int = 0
        self._last_applied: Optional[TuningAdjustment] = None

    def _clamp(self, param: str, value: Any) -> Any:
        """Clamp numeric values to valid bounds."""
        if param in self.BOUNDS and isinstance(value, (int, float)):
            lo, hi = self.BOUNDS[param]
            return type(value)(max(lo, min(hi, value)))
        return value

    def apply(self, adjustment: TuningAdjustment, config: Any) -> bool:
        """Apply a single adjustment to an EvolutionConfig.

        Parameters
        ----------
        adjustment : TuningAdjustment
            The adjustment to apply.
        config : EvolutionConfig
            The config object to modify (mutates in-place).

        Returns
        -------
        bool
            True if applied successfully, False if the parameter
            doesn't exist on the config.
        """
        param = adjustment.parameter
        if not hasattr(config, param):
            return False

        value = self._clamp(param, adjustment.value)
        setattr(config, param, value)
        self._applied_count += 1
        self._last_applied = adjustment
        return True

    def summary(self) -> Dict[str, Any]:
        """Return a summary of actuator state."""
        return {
            "applied_count": self._applied_count,
            "last_applied": (
                {
                    "param": self._last_applied.parameter,
                    "value": self._last_applied.value,
                    "reason": self._last_applied.reason,
                }
                if self._last_applied
                else None
            ),
        }
