"""Tests for ScaleValidator — pass/fail gate with rollback trigger."""

import pytest

from milestone1.training.scaling.scale_validator import ScaleValidator

# ---------------------------------------------------------------------------
# Helpers — lightweight mock models for testing
# ---------------------------------------------------------------------------


class MockModel:
    """Minimal model stub with controllable fitness/loss/level."""

    def __init__(self, fitness: float = 0.8, loss: float = 0.2, level: int = 5):
        self._fitness = fitness
        self._loss = loss
        self.max_level = level


class MockArchive:
    """Archive stub that returns deterministic test entries."""

    def __init__(self, n: int = 100):
        self._entries = [{"input": i, "target": float(i % 2)} for i in range(n)]

    def get_all(self):
        return list(self._entries)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScaleValidatorPasses:
    """1. test_validator_passes_on_good_migration — new model slightly better → COMMIT"""

    def test_good_migration_commits(self):
        old = MockModel(fitness=0.80, loss=0.20, level=5)
        new = MockModel(fitness=0.85, loss=0.15, level=5)  # better
        archive = MockArchive(100)

        validator = ScaleValidator(old, new, archive)
        result = validator.validate()

        assert result.recommendation == "COMMIT"
        assert result.passed is True
        assert result.fitness_improvement > 0
        assert result.loss_improvement > 0
        assert result.level_maintained is True


class TestScaleValidatorRollback:
    """2. test_validator_rollback_on_bad_migration — new model worse → ROLLBACK"""

    def test_bad_migration_rollback(self):
        old = MockModel(fitness=0.80, loss=0.20, level=5)
        new = MockModel(fitness=0.70, loss=0.35, level=3)  # worse
        archive = MockArchive(100)

        validator = ScaleValidator(old, new, archive)
        result = validator.validate()

        assert result.recommendation == "ROLLBACK"
        assert result.passed is False
        assert result.fitness_improvement < 0
        assert result.level_maintained is False


class TestScaleValidatorExtendGate:
    """3. test_validator_extends_on_inconclusive — new model same → EXTEND_GATE"""

    def test_inconclusive_extends_gate(self):
        old = MockModel(fitness=0.80, loss=0.20, level=5)
        new = MockModel(fitness=0.80, loss=0.20, level=5)  # same
        archive = MockArchive(100)

        validator = ScaleValidator(old, new, archive)
        result = validator.validate()

        assert result.recommendation == "EXTEND_GATE"
        assert result.passed is False
        assert result.fitness_improvement == pytest.approx(0.0)


class TestScaleValidatorEdgeCases:
    """Additional edge-case coverage."""

    def test_empty_archive_returns_extend_gate(self):
        old = MockModel()
        new = MockModel()
        archive = MockArchive(0)

        validator = ScaleValidator(old, new, archive)
        result = validator.validate()

        assert result.recommendation == "EXTEND_GATE"
        assert result.passed is False

    def test_zero_fitness_absolute_threshold(self):
        old = MockModel(fitness=0.0, loss=0.5, level=5)
        new = MockModel(fitness=0.05, loss=0.45, level=5)
        archive = MockArchive(100)

        validator = ScaleValidator(old, new, archive)
        result = validator.validate()

        assert result.recommendation == "COMMIT"

    def test_custom_thresholds(self):
        old = MockModel(fitness=0.80, loss=0.20, level=5)
        new = MockModel(fitness=0.81, loss=0.19, level=5)  # +1.25% fitness, +5% loss
        archive = MockArchive(100)

        # With default threshold (2%), fitness alone is inconclusive (1.25% < 2%)
        # but the secondary gate commits on strong loss improvement (5% > 2%)
        validator = ScaleValidator(old, new, archive)
        result = validator.validate()
        assert result.recommendation == "COMMIT"  # secondary gate: loss improvement

        # With 1% threshold, this passes on primary gate
        validator2 = ScaleValidator(old, new, archive, config={"pass_threshold": 0.01})
        result2 = validator2.validate()
        assert result2.recommendation == "COMMIT"
