#!/usr/bin/env python3
"""Wave 31 — Final smoke tests.

Fast health-checks verifying that the core engine and config objects
can be constructed and their key read-only methods return the expected
types.  No actual evolution runs.
"""

from __future__ import annotations

import pytest

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine

# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine construction and core read-only API."""

    def test_engine_creates_with_default_config(self, engine):
        """Engine fixture produces a usable instance."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self, engine):
        """Engine exposes all key runtime attributes."""
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
            "vm_farm",
            "curriculum",
            "circuit_breaker",
            "_fitness_velocity",
            "_current_diversity",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine):
        """engine.run exists and is callable."""
        assert hasattr(engine, "run")
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine):
        """get_run_diagnostics returns a dict with champion info."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        assert "champion_level" in diag
        assert "champion_fitness" in diag
        assert "current_generation" in diag
        assert "hall_of_fame_size" in diag

    def test_engine_get_performance_metrics_returns_dict(self, engine):
        """get_performance_metrics returns a dict with population info."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "current_generation" in metrics
        assert "population_size" in metrics
        assert "mutation_rate" in metrics
        assert "eval_budget" in metrics


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify config objects construct cleanly and expose diagnostic APIs."""

    def test_engine_config_creates_with_defaults(self):
        """EvolutionConfig() succeeds without arguments."""
        cfg = EvolutionConfig()
        assert cfg is not None
        assert isinstance(cfg, EvolutionConfig)
        assert cfg.mutation_rate == 0.3
        assert cfg.population_size == 50
        assert cfg.seed == 42

    def test_config_section_creates_with_defaults(self):
        """EvolutionSection() succeeds without arguments."""
        evo = EvolutionSection()
        assert evo is not None
        assert isinstance(evo, EvolutionSection)
        assert evo.population_size == 50
        assert evo.seed == 42

    def test_config_summary_returns_dict_with_expected_keys(self):
        """get_config_summary returns a dict with standard keys."""
        evo = EvolutionSection()
        summary = evo.get_config_summary()
        assert isinstance(summary, dict)
        expected_keys = {
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
        }
        assert expected_keys.issubset(summary.keys()), (
            f"Missing keys: {expected_keys - summary.keys()}"
        )

    def test_validate_config_returns_list(self):
        """validate_config returns a list."""
        evo = EvolutionSection()
        warnings = evo.validate_config()
        assert isinstance(warnings, list)

    def test_no_warnings_on_default_config(self):
        """Default EvolutionSection produces zero validation warnings."""
        evo = EvolutionSection()
        warnings = evo.validate_config()
        assert warnings == [], f"Unexpected warnings on default config: {warnings}"
