"""Type definitions for the self-scaling architecture.

Phase 1: Trend analysis and saturation detection types.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TrendSnapshot:
    """Captures training trends over a window of cycles.

    Used by SaturationDetector to analyze whether the model has
    plateaued and should be scaled to a larger architecture.
    """

    fitness_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    max_level_history: list[int] = field(default_factory=list)
    archive_size: int = 0
    cycles_observed: int = 0


@dataclass
class SaturationReport:
    """Result of saturation detection analysis.

    Attributes:
        fitness_plateau: True if fitness has plateaued (slope < 0.005 relative).
        val_loss_plateau: True if validation loss has plateaued (slope > -1e-4).
        curriculum_saturated: True if max level L17 reached in 18/20 cycles.
        archive_sufficient: True if archive has >= 200 entries.
        confidence: Fraction of criteria met (0.0 to 1.0).
        should_scale: True only if ALL 4 criteria are met.
    """

    fitness_plateau: bool = False
    val_loss_plateau: bool = False
    curriculum_saturated: bool = False
    archive_sufficient: bool = False
    confidence: float = 0.0

    @property
    def should_scale(self) -> bool:
        """Return True only if all 4 criteria are met."""
        return self.fitness_plateau and self.val_loss_plateau and self.curriculum_saturated and self.archive_sufficient

    def criteria_met(self) -> int:
        """Return the number of criteria currently met."""
        return sum(
            [
                self.fitness_plateau,
                self.val_loss_plateau,
                self.curriculum_saturated,
                self.archive_sufficient,
            ]
        )


@dataclass
class ScalingDecision:
    """Describes a planned architecture scaling action.

    Attributes:
        current_stage: Index into the scaling ladder of current config.
        target_stage: Index into the scaling ladder of the target config.
        current_config: Dict of current architecture hyperparameters.
        target_config: Dict of target architecture hyperparameters.
        reason: Human-readable explanation of why scaling was triggered.
    """

    current_stage: int = 0
    target_stage: int = 0
    current_config: dict = field(default_factory=dict)
    target_config: dict = field(default_factory=dict)
    reason: str = ""


@dataclass
class MigrationResult:
    """Result of migrating weights from old to new architecture.

    Attributes:
        success: Whether migration completed without errors.
        old_val_loss: Validation loss before migration.
        new_val_loss: Validation loss after migration.
        message: Human-readable status or error message.
    """

    success: bool = False
    old_val_loss: float = 0.0
    new_val_loss: float = 0.0
    message: str = ""


@dataclass
class ValidationResult:
    """Result of validating a scaled architecture post-migration.

    Attributes:
        passed: Whether the validation passed acceptance criteria.
        fitness_improvement: Change in fitness (new - old).
        loss_improvement: Change in val_loss (old - new, positive = better).
        level_maintained: Whether curriculum level was maintained.
        recommendation: Human-readable recommendation (e.g., 'proceed', 'rollback').
    """

    passed: bool = False
    fitness_improvement: float = 0.0
    loss_improvement: float = 0.0
    level_maintained: bool = False
    recommendation: str = ""
