#!/usr/bin/env python3
"""Wave 71 smoke tests — engine health and config health.

Fast smoke tests that verify core components can be created and their
APIs return expected shapes.  No actual evolution is run.
"""
from __future__ import annotations

import os
import sys

import pytest

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine

# Re-use the conftest factory for engine creation
from tests.conftest import make_engine

# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineHealth:
    """Verify EvolutionEngine can be instantiated and exposes expected API."""

    def test_engine_creation_with_default_config(self):
        """EvolutionEngine can be created with a default EvolutionConfig."""
        engine = make_engine()
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_all_expected_attributes(self, engine):
        """Engine carries every attribute the evolution loop expects."""
        expected_attrs = [
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
            "strategy_archive",
            "_hall_of_fame",
            "best_fitness_history",
            "circuit_breaker",
            "use_gpu",
            "vm_farm",
            "curriculum",
            "champion_anchor",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine):
        """engine.run is a callable method (we don't invoke it here)."""
        assert callable(getattr(engine, "run", None))

    def test_engine_get_run_diagnostics_returns_dict(self, engine):
        """get_run_diagnostics() returns a dict with expected keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        # Must contain at least these diagnostic keys
        for key in (
            "champion_level",
            "champion_fitness",
            "champion_words",
            "generations_stalled",
            "current_generation",
            "hall_of_fame_size",
            "mutation_rate",
            "population_size",
            "elite_ratio",
        ):
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_engine_get_performance_metrics_returns_dict(self, engine):
        """get_performance_metrics() returns a dict with expected keys."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        for key in (
            "current_generation",
            "population_size",
            "elite_ratio",
            "eval_budget",
            "mutation_rate",
            "champion_words",
            "hof_size",
        ):
            assert key in metrics, f"Missing metrics key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigHealth:
    """Verify EvolutionConfig / EvolutionSection default construction."""

    def test_evolution_config_creation_with_defaults(self):
        """EvolutionConfig can be instantiated with no arguments."""
        cfg = EvolutionConfig()
        assert cfg is not None
        assert cfg.population_size == 50
        assert cfg.mutation_rate == 0.3

    def test_evolution_section_get_config_summary_returns_dict(self):
        """EvolutionSection.get_config_summary() returns a dict with expected keys."""
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
            assert key in summary, f"Missing config summary key: {key}"

    def test_evolution_section_validate_config_returns_list(self):
        """EvolutionSection.validate_config() returns a list (empty on defaults)."""
        section = EvolutionSection()
        result = section.validate_config()
        assert isinstance(result, list)

    def test_evolution_section_no_warnings_on_defaults(self):
        """Default EvolutionSection produces zero validation warnings."""
        section = EvolutionSection()
        warnings = section.validate_config()
        assert warnings == [], f"Unexpected warnings on default config: {warnings}"
