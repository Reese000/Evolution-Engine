#!/usr/bin/env python3
"""Wave-60 final smoke tests for Evolution Engine.

Two lightweight test classes that verify core component health without
running actual evolution — fast, deterministic, fixture-based.
"""
from __future__ import annotations

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine

# ──────────────────────────────────────────────────────────────────────────────
# 1. TestEngineHealth — EvolutionEngine basics
# ──────────────────────────────────────────────────────────────────────────────


class TestEngineHealth:
    """Verify EvolutionEngine can be instantiated and exposes expected API."""

    def test_engine_creation_with_defaults(self):
        """EvolutionEngine can be constructed from defaults (no VM needed)."""
        EvolutionConfig()
        engine = EvolutionEngine.__new__(EvolutionEngine)
        assert engine is not None

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine):
        """Engine created via fixture carries all critical internal attributes."""
        expected = [
            "config",
            "rng",
            "champion_level",
            "current_generation",
            "generations_stalled",
            "champion",
            "champion_trace",
            "champion_fitness",
            "task_instance",
            "novelty_archive",
            "qd_archive",
            "best_fitness_history",
            "circuit_breaker",
            "strategy_archive",
            "champion_anchor",
            "vm_farm",
            "curriculum",
            "use_gpu",
        ]
        for attr in expected:
            assert hasattr(engine, attr), f"engine missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine):
        """engine.run exists and is callable (we do NOT call it)."""
        assert hasattr(engine, "run")
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine):
        """get_run_diagnostics() returns a dict with expected keys."""
        diagnostics = engine.get_run_diagnostics()
        assert isinstance(diagnostics, dict)
        expected_keys = [
            "champion_level",
            "champion_fitness",
            "champion_words",
            "champion_verified",
            "generations_stalled",
            "current_generation",
            "novelty_archive_size",
            "qd_archive_size",
            "hall_of_fame_size",
            "fitness_velocity",
            "fitness_momentum",
            "mutation_rate",
            "population_size",
            "elite_ratio",
        ]
        for key in expected_keys:
            assert key in diagnostics, f"diagnostics missing key: {key}"

    def test_engine_get_performance_metrics_returns_dict(self, engine: EvolutionEngine):
        """get_performance_metrics() returns a dict with expected keys."""
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
            assert key in metrics, f"metrics missing key: {key}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. TestConfigHealth — EvolutionConfig / EvolutionSection config basics
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigHealth:
    """Verify EvolutionConfig and EvolutionSection config health."""

    def test_config_creation_with_defaults(self):
        """EvolutionConfig can be instantiated with all defaults."""
        cfg = EvolutionConfig()
        assert cfg is not None

    def test_get_config_summary_returns_dict(self):
        """get_config_summary() returns a dict with expected keys."""
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
            assert key in summary, f"config summary missing key: {key}"

    def test_validate_config_returns_list(self):
        """validate_config() returns a list (empty for defaults)."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert isinstance(warnings, list)

    def test_no_warnings_on_default_config(self):
        """Default config should produce zero validation warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"Default config produced warnings: {warnings}"
