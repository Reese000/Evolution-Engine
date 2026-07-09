"""ScaleValidator — pass/fail gate for post-migration validation.

Compares old and new models on held-out archive data and produces a
ValidationResult with a recommendation: COMMIT, EXTEND_GATE, or ROLLBACK.

Used by ScaleCoordinator in the GATING state of the scaling lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from milestone1.training.scaling.types import ValidationResult

logger = logging.getLogger(__name__)


class ScaleValidator:
    """Validates that a migrated model performs at least as well as the old model.

    The validator evaluates both models on held-out archive data and computes
    relative fitness and loss improvements. Based on these metrics it issues
    a recommendation:
      - 'COMMIT': new model meets or exceeds old within threshold.
      - 'EXTEND_GATE': results are inconclusive — extend observation.
      - 'ROLLBACK': new model is significantly worse.
    """

    def __init__(
        self,
        old_model: Any,
        new_model: Any,
        archive: Any,
        config: Optional[dict] = None,
    ):
        self.old_model = old_model
        self.new_model = new_model
        self.archive = archive
        config = config or {}
        self.pass_threshold: float = config.get("pass_threshold", 0.02)
        self.rollback_threshold: float = config.get("rollback_threshold", -0.02)
        self.test_split: float = config.get("test_split", 0.2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> ValidationResult:
        """Compare old and new model on held-out archive data.

        Returns:
            ValidationResult with passed, fitness_improvement, loss_improvement,
            level_maintained, and recommendation fields populated.
        """
        test_data = self._get_test_data()
        if not test_data:
            logger.warning("No test data available; treating as inconclusive")
            return ValidationResult(
                passed=False,
                recommendation="EXTEND_GATE",
            )

        old_metrics = self._evaluate_model(self.old_model, test_data)
        new_metrics = self._evaluate_model(self.new_model, test_data)

        fitness_improvement = new_metrics["fitness"] - old_metrics["fitness"]
        loss_improvement = old_metrics["loss"] - new_metrics["loss"]  # positive = better
        level_maintained = new_metrics["level"] >= old_metrics["level"]

        recommendation = self._compare_results(old_metrics, new_metrics)
        passed = recommendation == "COMMIT"

        return ValidationResult(
            passed=passed,
            fitness_improvement=fitness_improvement,
            loss_improvement=loss_improvement,
            level_maintained=level_maintained,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_test_data(self) -> list:
        """Extract a held-out subset from the archive for validation."""
        if hasattr(self.archive, "get_all"):
            entries = self.archive.get_all()
        elif isinstance(self.archive, (list, tuple)):
            entries = list(self.archive)
        else:
            return []

        if not entries:
            return []

        # Deterministic split — take last N entries as held-out test set
        n_test = max(1, int(len(entries) * self.test_split))
        return entries[-n_test:]

    def _evaluate_model(self, model: Any, test_data: list) -> dict:
        """Evaluate a model on test data, returning avg loss and accuracy.

        Supports models with a ``.predict(data)`` method, or falls back to
        a simple mock evaluation (for testing / stubs).
        """
        if hasattr(model, "predict"):
            predictions = model.predict(test_data)
            loss = self._compute_loss(predictions, test_data)
            fitness = self._compute_fitness(predictions, test_data)
            level = getattr(model, "max_level", 0)
        elif hasattr(model, "evaluate"):
            result = model.evaluate(test_data)
            loss = result.get("loss", 0.0)
            fitness = result.get("fitness", 0.0)
            level = result.get("level", 0)
        else:
            # Stub / mock — extract metrics directly if available
            loss = getattr(model, "_loss", 0.0)
            fitness = getattr(model, "_fitness", 0.0)
            level = getattr(model, "max_level", 0)

        return {"loss": loss, "fitness": fitness, "level": level}

    def _compare_results(self, old_metrics: dict, new_metrics: dict) -> str:
        """Compare old vs new metrics and return a recommendation string.

        Decision logic:
          - If new fitness >= old fitness * (1 + pass_threshold): COMMIT
          - If new fitness < old fitness * (1 + rollback_threshold): ROLLBACK
          - Otherwise: EXTEND_GATE
        """
        old_fitness = old_metrics["fitness"]
        new_fitness = new_metrics["fitness"]
        old_loss = old_metrics["loss"]
        new_loss = new_metrics["loss"]

        if old_fitness == 0:
            # Avoid division-by-zero; use absolute thresholds
            if new_fitness >= self.pass_threshold:
                return "COMMIT"
            elif new_fitness <= self.rollback_threshold:
                return "ROLLBACK"
            return "EXTEND_GATE"

        relative_fitness = (new_fitness - old_fitness) / abs(old_fitness)

        # Also consider loss improvement
        if old_loss > 0:
            relative_loss = (old_loss - new_loss) / abs(old_loss)
        else:
            relative_loss = 0.0

        # Primary gate: fitness must meet threshold
        if relative_fitness >= self.pass_threshold:
            return "COMMIT"
        elif relative_fitness <= self.rollback_threshold:
            return "ROLLBACK"

        # Secondary gate: loss can salvage a borderline fitness result
        if relative_fitness >= 0 and relative_loss >= self.pass_threshold:
            return "COMMIT"
        elif relative_loss <= self.rollback_threshold:
            return "ROLLBACK"

        return "EXTEND_GATE"

    # ------------------------------------------------------------------
    # Prediction helpers (for models with .predict)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_loss(predictions: Any, test_data: list) -> float:
        """Compute average loss from predictions vs expected outputs."""
        if not hasattr(predictions, "__len__") or not predictions:
            return 0.0
        total = 0.0
        for pred, entry in zip(predictions, test_data):
            expected = entry.get("target", 0.0) if isinstance(entry, dict) else 0.0
            total += (pred - expected) ** 2 if isinstance(pred, (int, float)) else 0.0
        return total / len(predictions)

    @staticmethod
    def _compute_fitness(predictions: Any, test_data: list) -> float:
        """Compute accuracy/fitness from predictions vs expected outputs."""
        if not hasattr(predictions, "__len__") or not predictions:
            return 0.0
        correct = 0
        for pred, entry in zip(predictions, test_data):
            expected = entry.get("target", 0.0) if isinstance(entry, dict) else 0.0
            if isinstance(pred, (int, float)) and abs(pred - expected) < 0.5:
                correct += 1
        return correct / len(predictions)
