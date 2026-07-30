#!/usr/bin/env python3
"""Wave 23 final smoke tests — lightweight health checks for the core classes.

Tests verify that EvolutionEngine and EvolutionConfig can be instantiated
with defaults and expose the expected interface.  No evolution is run.
"""
from __future__ import annotations

import pytest

from evolution_engine import EvolutionConfig, EvolutionEngine


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineHealth:
    """Verify EvolutionEngine basic construction and interface via conftest fixture."""

    def test_engine_creation(self, engine: EvolutionEngine) -> None:
        """Engine can be created with default config via _make_engine fixture."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine) -> None:
        """Engine exposes all critical attributes expected by the system."""
        expected_attrs = [
            "config",
            "champion",
            "champion_trace",
            "champion_fitness",
            "champion_level",
            "current_generation",
            "generations_stalled",
            "task_instance",
            "novelty_archive",
            "qd_archive",
            "_hall_of_fame",
            "vm_farm",
            "curriculum",
            "strategy_archive",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"engine missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine) -> None:
        """engine.run is callable (we do NOT invoke it — too slow)."""
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_run_diagnostics returns a dict with expected top-level keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        # Core diagnostic keys that must always be present
        for key in ("champion_level", "current_generation", "champion_fitness",
                     "generations_stalled", "champion_words"):
            assert key in diag, f"diagnostics missing key: {key}"

    def test_engine_get_performance_metrics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_performance_metrics returns a dict with expected keys."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        expected_keys = [
            "current_generation",
            "population_size",
            "elite_ratio",
            "eval_budget",
            "mutation_rate",
            "champion_words",
            "novelty_weight",
            "qd_bonus_weight",
            "hof_size",
        ]
        for key in expected_keys:
            assert key in metrics, f"performance metrics missing key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigHealth:
    """Verify EvolutionConfig construction, attribute access, and validation."""

    def test_config_creation_defaults(self) -> None:
        """EvolutionConfig can be created with all defaults."""
        cfg = EvolutionConfig()
        assert cfg is not None
        assert isinstance(cfg, EvolutionConfig)

    def test_config_has_expected_fields(self) -> None:
        """Config exposes key numeric / structural fields with sane defaults."""
        cfg = EvolutionConfig()
        assert cfg.population_size >= 2
        assert cfg.max_generations >= 1
        assert 0.0 <= cfg.mutation_rate <= 1.0
        assert 0.0 <= cfg.crossover_rate <= 1.0
        assert 0.0 <= cfg.elite_ratio <= 0.5
        assert cfg.tournament_size >= 2
        assert cfg.instruction_budget >= 100
        assert cfg.eval_budget >= 100

    def test_config_summary_as_dict(self) -> None:
        """Config fields are accessible as a dict-like snapshot (via __dict__)."""
        cfg = EvolutionConfig()
        summary = vars(cfg)
        assert isinstance(summary, dict)
        # Must contain the most important config knobs
        for key in ("population_size", "max_generations", "mutation_rate",
                     "elite_ratio", "eval_budget", "novelty_weight",
                     "qd_bonus_weight"):
            assert key in summary, f"config summary missing key: {key}"

    def test_config_validate_returns_no_errors_on_defaults(self) -> None:
        """Default config passes __post_init__ validation without raising."""
        # If __post_init__ raised, the constructor itself would fail.
        # This test makes that explicit: no ValueError on defaults.
        cfg = EvolutionConfig()
        # Additionally verify the validated fields are in valid ranges
        assert cfg.retrain_val_split >= 0.0
        assert cfg.retrain_val_split <= 1.0
        assert cfg.retrain_max_epochs >= 1
        assert cfg.retrain_batch_size >= 1
        assert cfg.retrain_lr > 0

    def test_config_no_warnings_on_default(self) -> None:
        """Creating a default config does not emit log warnings."""
        import io
        import logging

        handler = logging.StreamHandler(io.StringIO())
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("evolution_engine")
        logger.addHandler(handler)
        try:
            EvolutionConfig()
        finally:
            logger.removeHandler(handler)
            output = handler.stream.getvalue()
            # Default elite_ratio is 0.1 so the elite_ratio==0.0 warning should not fire
            assert "elite_ratio=0.0" not in output, (
                f"Unexpected warning on default config: {output}"
            )
