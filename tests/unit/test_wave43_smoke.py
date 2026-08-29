#!/usr/bin/env python3
"""Wave 43 — Final smoke tests for EvolutionEngine and EvolutionConfig.

Fast health-check tests that verify core objects can be instantiated and
their key methods return the expected types. No actual evolution is run.
"""
from __future__ import annotations

import pytest

from config import EvolutionSection
from evolution_engine import EvolutionConfig

# ══════════════════════════════════════════════════════════════════════════════
# 1. Engine health
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineHealth:
    """Verify that EvolutionEngine can be created and its diagnostic
    surface works without running any evolution."""

    def test_engine_created_with_default_config(self, engine):
        """Engine fixture produces a valid EvolutionEngine."""
        assert engine is not None
        assert hasattr(engine, "config")
        assert isinstance(engine.config, EvolutionConfig)

    def test_engine_has_expected_attributes(self, engine):
        """Engine exposes all attributes that downstream code depends on."""
        expected_attrs = [
            "config",
            "rng",
            "champion",
            "champion_fitness",
            "champion_level",
            "champion_trace",
            "current_generation",
            "generations_stalled",
            "novelty_archive",
            "qd_archive",
            "strategy_archive",
            "_hall_of_fame",
            "task_instance",
            "recurrent_examples_count",
            "best_fitness_history",
            "circuit_breaker",
            "use_gpu",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine):
        """engine.run exists and is callable (we don't invoke it)."""
        assert callable(getattr(engine, "run", None))

    def test_get_run_diagnostics_returns_dict(self, engine):
        """get_run_diagnostics returns a dict with expected top-level keys."""
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
            "hall_of_fame_size",
            "mutation_rate",
            "population_size",
            "elite_ratio",
        ]
        for key in expected_keys:
            assert key in diag, f"Missing key in diagnostics: {key}"

    def test_get_performance_metrics_returns_dict(self, engine):
        """get_performance_metrics returns a dict with expected top-level keys."""
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
            "total_recurrent_examples",
            "hof_size",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key in performance metrics: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Config health
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigHealth:
    """Verify EvolutionConfig (dataclass) and EvolutionSection (config.py)
    can be created with defaults and report sane summaries."""

    def test_evolution_config_defaults(self):
        """EvolutionConfig can be instantiated with all defaults."""
        cfg = EvolutionConfig()
        assert cfg is not None
        assert cfg.population_size == 50
        assert cfg.mutation_rate == 0.3
        assert cfg.eval_budget == 5000

    def test_evolution_section_defaults(self):
        """EvolutionSection can be instantiated with all defaults."""
        section = EvolutionSection()
        assert section is not None
        assert section.population_size == 50

    def test_get_config_summary_returns_dict_with_expected_keys(self):
        """EvolutionSection.get_config_summary returns expected keys."""
        section = EvolutionSection()
        summary = section.get_config_summary()
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
            assert key in summary, f"Missing key in config summary: {key}"

    def test_validate_config_returns_list(self):
        """EvolutionSection.validate_config returns a list."""
        section = EvolutionSection()
        result = section.validate_config()
        assert isinstance(result, list)

    def test_no_warnings_on_default_config(self):
        """Default EvolutionSection config should produce no warnings."""
        section = EvolutionSection()
        warnings = section.validate_config()
        assert len(warnings) == 0, f"Unexpected warnings on defaults: {warnings}"
