"""Unit tests for scaling package types and properties.

Tests all dataclasses and their properties in the self-scaling architecture.
"""

import pytest

from milestone1.training.scaling.types import (
    MigrationResult,
    SaturationReport,
    ScalingDecision,
    TrendSnapshot,
    ValidationResult,
)


class TestTrendSnapshot:
    """Tests for TrendSnapshot dataclass."""

    def test_default_initialization(self):
        """TrendSnapshot defaults to empty lists and zero values."""
        snapshot = TrendSnapshot()
        assert snapshot.fitness_history == []
        assert snapshot.val_loss_history == []
        assert snapshot.max_level_history == []
        assert snapshot.archive_size == 0
        assert snapshot.cycles_observed == 0

    def test_custom_initialization(self):
        """TrendSnapshot accepts custom values."""
        snapshot = TrendSnapshot(
            fitness_history=[0.1, 0.2, 0.3],
            val_loss_history=[2.0, 1.5, 1.0],
            max_level_history=[5, 10, 15],
            archive_size=150,
            cycles_observed=3,
        )
        assert len(snapshot.fitness_history) == 3
        assert snapshot.archive_size == 150
        assert snapshot.cycles_observed == 3

    def test_mutable_fields(self):
        """TrendSnapshot fields can be modified after creation."""
        snapshot = TrendSnapshot()
        snapshot.fitness_history.append(0.5)
        snapshot.archive_size = 100
        assert snapshot.fitness_history == [0.5]
        assert snapshot.archive_size == 100


class TestSaturationReport:
    """Tests for SaturationReport dataclass and properties."""

    def test_default_initialization(self):
        """SaturationReport defaults to False for all criteria."""
        report = SaturationReport()
        assert report.fitness_plateau is False
        assert report.val_loss_plateau is False
        assert report.curriculum_saturated is False
        assert report.archive_sufficient is False
        assert report.confidence == 0.0

    def test_should_scale_all_false(self):
        """should_scale returns False when no criteria are met."""
        report = SaturationReport()
        assert report.should_scale is False

    def test_should_scale_partial(self):
        """should_scale returns False when only some criteria are met."""
        report = SaturationReport(
            fitness_plateau=True,
            val_loss_plateau=True,
            curriculum_saturated=False,
            archive_sufficient=False,
        )
        assert report.should_scale is False

    def test_should_scale_three_of_four(self):
        """should_scale returns False with 3 of 4 criteria."""
        report = SaturationReport(
            fitness_plateau=True,
            val_loss_plateau=True,
            curriculum_saturated=True,
            archive_sufficient=False,
        )
        assert report.should_scale is False

    def test_should_scale_all_true(self):
        """should_scale returns True only when all 4 criteria are met."""
        report = SaturationReport(
            fitness_plateau=True,
            val_loss_plateau=True,
            curriculum_saturated=True,
            archive_sufficient=True,
        )
        assert report.should_scale is True

    def test_criteria_met_count_zero(self):
        """criteria_met returns 0 when no criteria are met."""
        report = SaturationReport()
        assert report.criteria_met() == 0

    def test_criteria_met_count_partial(self):
        """criteria_met returns correct count for partial criteria."""
        report = SaturationReport(
            fitness_plateau=True,
            val_loss_plateau=False,
            curriculum_saturated=True,
            archive_sufficient=False,
        )
        assert report.criteria_met() == 2

    def test_criteria_met_count_all(self):
        """criteria_met returns 4 when all criteria are met."""
        report = SaturationReport(
            fitness_plateau=True,
            val_loss_plateau=True,
            curriculum_saturated=True,
            archive_sufficient=True,
        )
        assert report.criteria_met() == 4

    def test_confidence_values(self):
        """Confidence correctly reflects fraction of criteria met."""
        # 0/4 = 0.0
        report = SaturationReport(confidence=0.0)
        assert report.confidence == 0.0

        # 2/4 = 0.5
        report = SaturationReport(confidence=0.5)
        assert report.confidence == 0.5

        # 4/4 = 1.0
        report = SaturationReport(confidence=1.0)
        assert report.confidence == 1.0


class TestScalingDecision:
    """Tests for ScalingDecision dataclass."""

    def test_default_initialization(self):
        """ScalingDecision defaults to stage 0 with empty dicts."""
        decision = ScalingDecision()
        assert decision.current_stage == 0
        assert decision.target_stage == 0
        assert decision.current_config == {}
        assert decision.target_config == {}
        assert decision.reason == ""

    def test_custom_initialization(self):
        """ScalingDecision accepts custom values."""
        decision = ScalingDecision(
            current_stage=0,
            target_stage=1,
            current_config={"embed_dim": 8, "ff_dim": 256},
            target_config={"embed_dim": 14, "ff_dim": 512},
            reason="Training saturated at stage 0",
        )
        assert decision.current_stage == 0
        assert decision.target_stage == 1
        assert decision.current_config["embed_dim"] == 8
        assert decision.target_config["embed_dim"] == 14
        assert "saturated" in decision.reason

    def test_mutable_config_dicts(self):
        """Config dicts can be modified after creation."""
        decision = ScalingDecision()
        decision.current_config["embed_dim"] = 8
        decision.target_config["embed_dim"] = 14
        assert decision.current_config["embed_dim"] == 8
        assert decision.target_config["embed_dim"] == 14


class TestMigrationResult:
    """Tests for MigrationResult dataclass."""

    def test_default_initialization(self):
        """MigrationResult defaults to failed with zero losses."""
        result = MigrationResult()
        assert result.success is False
        assert result.old_val_loss == 0.0
        assert result.new_val_loss == 0.0
        assert result.message == ""

    def test_successful_migration(self):
        """MigrationResult for successful migration."""
        result = MigrationResult(
            success=True,
            old_val_loss=1.5,
            new_val_loss=1.2,
            message="Weights migrated successfully",
        )
        assert result.success is True
        assert result.old_val_loss == 1.5
        assert result.new_val_loss == 1.2
        assert "successfully" in result.message

    def test_failed_migration(self):
        """MigrationResult for failed migration."""
        result = MigrationResult(
            success=False,
            old_val_loss=1.5,
            new_val_loss=0.0,
            message="Dimension mismatch in layer 3",
        )
        assert result.success is False
        assert result.new_val_loss == 0.0
        assert "mismatch" in result.message


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_initialization(self):
        """ValidationResult defaults to failed with zero improvements."""
        result = ValidationResult()
        assert result.passed is False
        assert result.fitness_improvement == 0.0
        assert result.loss_improvement == 0.0
        assert result.level_maintained is False
        assert result.recommendation == ""

    def test_successful_validation(self):
        """ValidationResult for successful validation."""
        result = ValidationResult(
            passed=True,
            fitness_improvement=0.05,
            loss_improvement=0.3,
            level_maintained=True,
            recommendation="proceed",
        )
        assert result.passed is True
        assert result.fitness_improvement == 0.05
        assert result.loss_improvement == 0.3
        assert result.level_maintained is True
        assert result.recommendation == "proceed"

    def test_failed_validation(self):
        """ValidationResult for failed validation."""
        result = ValidationResult(
            passed=False,
            fitness_improvement=-0.02,
            loss_improvement=-0.1,
            level_maintained=False,
            recommendation="rollback",
        )
        assert result.passed is False
        assert result.fitness_improvement < 0
        assert result.loss_improvement < 0
        assert result.level_maintained is False
        assert result.recommendation == "rollback"

    def test_rollback_recommendation(self):
        """ValidationResult can have rollback recommendation."""
        result = ValidationResult(
            passed=False,
            recommendation="rollback to stage 0",
        )
        assert "rollback" in result.recommendation
