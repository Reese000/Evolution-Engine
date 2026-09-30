#!/usr/bin/env python3
"""Wave 64 — Final smoke tests for Evolution Engine core health.

Verifies that EvolutionEngine and EvolutionSection (config) can be
instantiated, have the expected API surface, and return sensible
structures. Tests are fast — no actual evolution runs.
"""
from __future__ import annotations

from config import EvolutionSection
from evolution_engine import EvolutionEngine

# ═══════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ═══════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine can be created and exposes the expected API."""

    def test_engine_creation_with_default_config(self, engine):
        """EvolutionEngine can be created with default config via fixture."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self, engine):
        """Engine exposes all core attributes needed for operation."""
        expected_attrs = [
            "config",
            "champion",
            "champion_level",
            "champion_fitness",
            "current_generation",
            "generations_stalled",
            "task_instance",
            "novelty_archive",
            "qd_archive",
            "strategy_archive",
            "_hall_of_fame",
            "best_fitness_history",
            "rng",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine):
        """engine.run is a callable method."""
        assert callable(engine.run)

    def test_get_run_diagnostics_returns_dict(self, engine):
        """get_run_diagnostics returns a dict with expected keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        expected_keys = [
            "champion_level",
            "champion_fitness",
            "champion_words",
            "champion_verified",
            "generations_stalled",
            "current_generation",
            "novelty_archive_size",
            "qd_archive_size",
            "strategy_archive_records",
            "hall_of_fame_size",
            "fitness_velocity",
            "mutation_rate",
            "population_size",
            "elite_ratio",
        ]
        for key in expected_keys:
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_get_performance_metrics_returns_dict(self, engine):
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
            assert key in metrics, f"Missing metrics key: {key}"


# ═══════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ═══════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionSection (config) can be created and its API works."""

    def test_config_creation_with_defaults(self):
        """EvolutionSection can be created with default values."""
        cfg = EvolutionSection()
        assert cfg is not None
        assert isinstance(cfg, EvolutionSection)
        assert cfg.population_size == 50
        assert cfg.seed == 42

    def test_get_config_summary_returns_dict_with_expected_keys(self):
        """get_config_summary returns a dict with all expected keys."""
        cfg = EvolutionSection()
        summary = cfg.get_config_summary()
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
            assert key in summary, f"Missing summary key: {key}"

    def test_validate_config_returns_list(self):
        """validate_config returns a list."""
        cfg = EvolutionSection()
        result = cfg.validate_config()
        assert isinstance(result, list)

    def test_no_warnings_on_default_config(self):
        """Default EvolutionSection produces no validation warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"Unexpected warnings on default config: {warnings}"
