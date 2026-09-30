#!/usr/bin/env python3
"""Wave 46 smoke tests — lightweight health checks for engine and config.

These tests verify that core objects can be instantiated, key methods
are callable and return the correct types, and default configs produce
no warnings.  No actual evolution is run.
"""
from __future__ import annotations

from config import EvolutionSection

# Ensure project root is on the path
from evolution_engine import EvolutionEngine

# ══════════════════════════════════════════════════════════════════════════════
# TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine basics via the conftest factory."""

    def test_engine_creation_with_default_config(self, engine):
        """Engine object is created successfully with default config."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self, engine):
        """Engine exposes all core attributes expected by downstream code."""
        expected_attrs = [
            "config",
            "champion",
            "champion_level",
            "champion_fitness",
            "current_generation",
            "generations_stalled",
            "champion_trace",
            "novelty_archive",
            "qd_archive",
            "strategy_archive",
            "_hall_of_fame",
            "best_fitness_history",
            "task_instance",
            "rng",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine):
        """engine.run is a callable method."""
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine):
        """get_run_diagnostics returns a dict with key diagnostic fields."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        # Must contain at least these keys
        for key in [
            "champion_level",
            "champion_fitness",
            "current_generation",
            "generations_stalled",
            "mutation_rate",
            "population_size",
        ]:
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_engine_get_performance_metrics_returns_dict(self, engine):
        """get_performance_metrics returns a dict with key metric fields."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        for key in [
            "current_generation",
            "population_size",
            "mutation_rate",
            "elite_ratio",
            "eval_budget",
        ]:
            assert key in metrics, f"Missing metric key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionSection config creation and validation."""

    def test_config_creation_with_defaults(self):
        """EvolutionSection can be created with all defaults."""
        cfg = EvolutionSection()
        assert cfg is not None
        assert isinstance(cfg, EvolutionSection)

    def test_config_summary_returns_dict_with_expected_keys(self):
        """get_config_summary returns a dict containing required keys."""
        cfg = EvolutionSection()
        summary = cfg.get_config_summary()
        assert isinstance(summary, dict)
        required_keys = [
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
        for key in required_keys:
            assert key in summary, f"Missing config summary key: {key}"

    def test_config_validate_returns_list(self):
        """validate_config returns a list (possibly empty)."""
        cfg = EvolutionSection()
        result = cfg.validate_config()
        assert isinstance(result, list)

    def test_config_no_warnings_on_defaults(self):
        """Default EvolutionSection produces no validation warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"Unexpected warnings on default config: {warnings}"
