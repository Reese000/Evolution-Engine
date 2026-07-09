"""Saturation detection for self-scaling architecture.

Phase 1: Analyzes training trends to determine if the model has
reached a plateau and should be scaled to a larger architecture.

Uses scipy.stats.linregress for slope-based trend analysis.
"""

from typing import Optional

import numpy as np

from .types import SaturationReport, TrendSnapshot


class SaturationDetector:
    """Detects when training has saturated across multiple metrics.

    Evaluates 4 criteria:
    1. fitness_plateau: Fitness slope < 0.005 relative to mean fitness
    2. val_loss_plateau: Val loss slope > -1e-4 (not decreasing meaningfully)
    3. curriculum_saturated: Max level L17 reached in 18/20 recent cycles
    4. archive_sufficient: Archive has >= 200 entries

    All 4 must hold for should_scale=True in the SaturationReport.
    """

    # Thresholds
    FITNESS_SLOPE_THRESHOLD = 0.005  # Relative to mean fitness
    VAL_LOSS_SLOPE_THRESHOLD = -1e-4  # Must be more negative to indicate improvement
    CURRICULUM_LEVEL_TARGET = 17
    CURRICULUM_WINDOW = 20
    CURRICULUM_THRESHOLD = 18  # Must appear in 18/20 cycles
    ARCHIVE_SIZE_THRESHOLD = 200

    def evaluate(self, snapshot: TrendSnapshot) -> SaturationReport:
        """Evaluate whether training has saturated.

        Args:
            snapshot: Current training trend data.

        Returns:
            SaturationReport with all criteria evaluated.
        """
        report = SaturationReport()

        # Criterion 1: Fitness plateau
        report.fitness_plateau = self._check_fitness_plateau(snapshot.fitness_history)

        # Criterion 2: Val loss plateau
        report.val_loss_plateau = self._check_val_loss_plateau(snapshot.val_loss_history)

        # Criterion 3: Curriculum saturated
        report.curriculum_saturated = self._check_curriculum_saturated(snapshot.max_level_history)

        # Criterion 4: Archive sufficient
        report.archive_sufficient = snapshot.archive_size >= self.ARCHIVE_SIZE_THRESHOLD

        # Calculate confidence as fraction of criteria met
        report.confidence = report.criteria_met() / 4.0

        return report

    def _check_fitness_plateau(self, fitness_history: list) -> bool:
        """Check if fitness has plateaued.

        Uses linear regression to compute slope. If slope relative to
        mean fitness is below threshold, fitness is considered plateaued.

        Args:
            fitness_history: List of fitness values over time.

        Returns:
            True if fitness has plateaued.
        """
        if len(fitness_history) < 3:
            return False

        try:
            x = np.arange(len(fitness_history), dtype=float)
            y = np.array(fitness_history, dtype=float)

            slope, intercept, r_value, p_value, std_err = _linregress(x, y)

            # Relative slope: slope / mean(abs(y))
            mean_abs_fitness = np.mean(np.abs(y))
            if mean_abs_fitness == 0:
                return True  # Zero fitness is definitely plateaued

            relative_slope = abs(slope) / mean_abs_fitness
            return relative_slope < self.FITNESS_SLOPE_THRESHOLD

        except Exception:
            return False

    def _check_val_loss_plateau(self, val_loss_history: list) -> bool:
        """Check if validation loss has plateaued.

        Uses linear regression to compute slope. If slope is above
        threshold (not decreasing), val loss is considered plateaued.

        Args:
            val_loss_history: List of validation loss values over time.

        Returns:
            True if validation loss has plateaued.
        """
        if len(val_loss_history) < 3:
            return False

        try:
            x = np.arange(len(val_loss_history), dtype=float)
            y = np.array(val_loss_history, dtype=float)

            slope, intercept, r_value, p_value, std_err = _linregress(x, y)

            # If slope is above threshold, loss is not decreasing
            return slope > self.VAL_LOSS_SLOPE_THRESHOLD

        except Exception:
            return False

    def _check_curriculum_saturated(self, max_level_history: list) -> bool:
        """Check if curriculum has saturated at max level.

        Returns True if the target level (L17) appears in at least
        CURRICULUM_THRESHOLD of the most recent CURRICULUM_WINDOW cycles.

        Args:
            max_level_history: List of max levels reached per cycle.

        Returns:
            True if curriculum is saturated.
        """
        if len(max_level_history) < self.CURRICULUM_WINDOW:
            return False

        # Check last CURRICULUM_WINDOW entries
        recent = max_level_history[-self.CURRICULUM_WINDOW :]
        count_at_target = sum(1 for level in recent if level >= self.CURRICULUM_LEVEL_TARGET)

        return count_at_target >= self.CURRICULUM_THRESHOLD


def _linregress(x, y):
    """Wrapper around scipy.stats.linregress with numpy fallback.

    Returns:
        slope, intercept, r_value, p_value, std_err
    """
    try:
        from scipy.stats import linregress

        return linregress(x, y)
    except ImportError:
        # Fallback: compute slope via least-squares with numpy
        from collections import namedtuple

        import numpy as np

        LinregressResult = namedtuple("LinregressResult", ["slope", "intercept", "rvalue", "pvalue", "stderr"])
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        if n < 2:
            return LinregressResult(slope=0, intercept=0, rvalue=0, pvalue=1, stderr=0)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)
        denom = n * sum_x2 - sum_x**2
        if abs(denom) < 1e-15:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        # Compute r_value
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_value = float(np.sqrt(1 - ss_res / ss_tot)) if ss_tot > 0 else 0.0
        return LinregressResult(slope=float(slope), intercept=float(intercept), rvalue=r_value, pvalue=0.0, stderr=0.0)
