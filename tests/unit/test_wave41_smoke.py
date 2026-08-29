#!/usr/bin/env python3
"""Wave 41 — Final smoke tests.

Verifies that EvolutionEngine and EvolutionConfig can be instantiated
with defaults, that core diagnostic/config APIs return the expected
shapes, and that the default configuration has no validation warnings.

All tests are fast and do NOT run actual evolution.
"""
from __future__ import annotations

import pytest

from config import Config
from evolution_engine import EvolutionConfig, EvolutionEngine

# ══════════════════════════════════════════════════════════════════════════════
# TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Smoke tests for EvolutionEngine instantiation and core APIs."""

    def test_engine_creation_default_config(self, engine: EvolutionEngine):
        """EvolutionEngine can be created (via fixture) with default config."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine):
        """Engine exposes all critical attributes after construction."""
        expected_attrs = [
            "config",
            "rng",
            "champion",
            "champion_trace",
            "champion_fitness",
            "champion_level",
            "current_generation",
            "generations_stalled",
            "task_instance",
            "novelty_archive",
            "qd_archive",
            "strategy_archive",
            "_hall_of_fame",
            "circuit_breaker",
            "vm_farm",
            "champion_anchor",
            "use_gpu",
            "curriculum",
            "best_fitness_history",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine):
        """engine.run is a callable method (we do NOT invoke it)."""
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine):
        """get_run_diagnostics() returns a dict with known keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        # Check a representative subset of guaranteed keys
        key_subset = [
            "champion_level",
            "champion_fitness",
            "champion_words",
            "generations_stalled",
            "current_generation",
            "novelty_archive_size",
            "qd_archive_size",
            "hall_of_fame_size",
            "fitness_velocity",
            "mutation_rate",
            "population_size",
        ]
        for key in key_subset:
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_engine_get_performance_metrics_returns_dict(self, engine: EvolutionEngine):
        """get_performance_metrics() returns a dict with known keys."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        key_subset = [
            "current_generation",
            "population_size",
            "elite_ratio",
            "eval_budget",
            "model_guided_ratio",
            "mutation_rate",
            "champion_words",
            "novelty_weight",
        ]
        for key in key_subset:
            assert key in metrics, f"Missing metric key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Smoke tests for EvolutionConfig and the unified Config system."""

    def test_config_creation_defaults(self):
        """EvolutionConfig can be created with default values."""
        cfg = EvolutionConfig()
        assert isinstance(cfg, EvolutionConfig)
        assert cfg.population_size == 50
        assert cfg.max_generations == 200
        assert cfg.mutation_rate == 0.3
        assert cfg.seed == 42

    def test_config_summary_returns_dict_with_expected_keys(self):
        """Config.evolution.get_config_summary() returns expected keys."""
        cfg = Config()
        summary = cfg.evolution.get_config_summary()
        assert isinstance(summary, dict)
        expected_keys = [
            "seed",
            "use_gpu",
            "auto_curriculum",
            "archives_enabled",
            "novelty_weight",
            "qd_bonus_weight",
            "mutation_rate",
            "population_size",
            "eval_budget",
            "max_consecutive_same_strategy",
            "total_config_fields",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing config summary key: {key}"

    def test_validate_config_returns_list(self):
        """Config.evolution.validate_config() returns a list."""
        cfg = Config()
        warnings = cfg.evolution.validate_config()
        assert isinstance(warnings, list)

    def test_no_warnings_on_default_config(self):
        """Default configuration produces no validation warnings."""
        cfg = Config()
        warnings = cfg.evolution.validate_config()
        assert len(warnings) == 0, f"Unexpected warnings on default config: {warnings}"
