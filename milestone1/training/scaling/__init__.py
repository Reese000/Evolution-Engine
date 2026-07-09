"""Self-scaling architecture for the Evolution Engine.

Phase 1: Saturation detection and scaling planning.

This package provides tools for automatically detecting when a model
has reached a training plateau and should be scaled to a larger architecture.

Usage:
    from milestone1.training.scaling import (
        SaturationDetector,
        ScalingPlanner,
        TrendSnapshot,
        SaturationReport,
        ScalingDecision,
    )

    # Create a trend snapshot from training history
    snapshot = TrendSnapshot(
        fitness_history=[0.1, 0.15, 0.16, 0.16, 0.16],
        val_loss_history=[2.0, 1.5, 1.2, 1.19, 1.18],
        max_level_history=[17] * 20,
        archive_size=250,
        cycles_observed=20,
    )

    # Detect saturation
    detector = SaturationDetector()
    report = detector.evaluate(snapshot)

    if report.should_scale:
        # Plan scaling
        planner = ScalingPlanner()
        decision = planner.plan(current_stage=0, report=report)
        if decision:
            print(f"Scale to stage {decision.target_stage}: {decision.reason}")
"""

from .saturation_detector import SaturationDetector
from .scaling_planner import ScalingPlanner
from .types import MigrationResult, SaturationReport, ScalingDecision, TrendSnapshot, ValidationResult

# Torch-dependent imports are lazy so the types/subpackages can be
# imported without requiring torch (e.g. in tests).
try:
    from .model_migrator import ModelMigrator
    from .warm_start_trainer import WarmStartConfig, WarmStartTrainer
except ImportError:
    ModelMigrator = None  # type: ignore[assignment,misc]
    WarmStartTrainer = None  # type: ignore[assignment,misc]
    WarmStartConfig = None  # type: ignore[assignment,misc]

__all__ = [
    "SaturationDetector",
    "ScalingPlanner",
    "ModelMigrator",
    "WarmStartTrainer",
    "WarmStartConfig",
    "TrendSnapshot",
    "SaturationReport",
    "ScalingDecision",
    "MigrationResult",
    "ValidationResult",
]
