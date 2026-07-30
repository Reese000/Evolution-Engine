#!/usr/bin/env python3
"""Wave 26 — Final smoke tests.

Two fast, read-only test classes:

1. TestEngineHealth — verify EvolutionEngine can be created and has all
   expected attributes and callable methods.

2. TestConfigHealth — verify EvolutionSection (from config.py) can be
   created with defaults, returns valid summaries, validates cleanly,
   and produces no warnings on default config.

Uses conftest fixtures (engine). All tests are fast — no actual evolution
is run. No existing files are modified.
"""

from __future__ import annotations

import os
import sys

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine

# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth — engine creation, attributes, callable methods
# ══════════════════════════════════════════════════════════════════════════════

# Attributes every healthy engine must have (set by conftest's make_engine)
_EXPECTED_ENGINE_ATTRS: list[str] = [
    "config",
    "rng",
    "champion_level",
    "current_generation",
    "generations_stalled",
    "champion",
    "champion_trace",
    "champion_fitness",
    "novelty_archive",
    "qd_archive",
    "strategy_archive",
    "_hall_of_fame",
    "_fitness_velocity",
    "_champion_replacement_count",
    "best_fitness_history",
    "circuit_breaker",
]


class TestEngineHealth:
    """Verify EvolutionEngine can be created and has all expected attributes."""

    def test_engine_creation_with_default_config(self, engine):
        """Engine created via conftest fixture should be a valid instance."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    @pytest.mark.parametrize("attr", _EXPECTED_ENGINE_ATTRS)
    def test_engine_has_attribute(self, engine, attr):
        """Engine must have every expected attribute."""
        assert hasattr(engine, attr), (
            f"Engine missing attribute '{attr}'"
        )

    def test_engine_run_is_callable(self, engine):
        """engine.run must be callable."""
        assert hasattr(engine, "run")
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine):
        """engine.get_run_diagnostics() must return a dict."""
        diagnostics = engine.get_run_diagnostics()
        assert isinstance(diagnostics, dict)
        # Must contain key diagnostic fields
        expected_keys = {
            "champion_level",
            "champion_fitness",
            "current_generation",
            "generations_stalled",
            "hall_of_fame_size",
        }
        actual_keys = set(diagnostics.keys())
        missing = expected_keys - actual_keys
        assert not missing, (
            f"get_run_diagnostics missing keys: {missing}"
        )

    def test_engine_get_performance_metrics_returns_dict(self, engine):
        """engine.get_performance_metrics() must return a dict."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        # Must contain key performance fields
        expected_keys = {
            "current_generation",
            "population_size",
            "elite_ratio",
            "eval_budget",
            "mutation_rate",
        }
        actual_keys = set(metrics.keys())
        missing = expected_keys - actual_keys
        assert not missing, (
            f"get_performance_metrics missing keys: {missing}"
        )

    def test_expected_attribute_count(self, engine):
        """Engine should have at least as many attrs as expected list."""
        for attr in _EXPECTED_ENGINE_ATTRS:
            assert hasattr(engine, attr)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth — EvolutionSection creation, summary, validation
# ══════════════════════════════════════════════════════════════════════════════

# Keys expected in get_config_summary output
_EXPECTED_SUMMARY_KEYS: list[str] = [
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


class TestConfigHealth:
    """Verify EvolutionSection can be created, summarized, and validated."""

    @pytest.fixture
    def default_section(self):
        """Create an EvolutionSection with all defaults."""
        return EvolutionSection()

    def test_config_creation_with_defaults(self, default_section):
        """EvolutionSection() must succeed with default arguments."""
        assert default_section is not None
        assert isinstance(default_section, EvolutionSection)

    def test_get_config_summary_returns_dict(self, default_section):
        """get_config_summary must return a dict."""
        summary = default_section.get_config_summary()
        assert isinstance(summary, dict)

    def test_get_config_summary_has_expected_keys(self, default_section):
        """get_config_summary must include all expected keys."""
        summary = default_section.get_config_summary()
        actual_keys = set(summary.keys())
        missing = set(_EXPECTED_SUMMARY_KEYS) - actual_keys
        assert not missing, (
            f"get_config_summary missing keys: {missing}"
        )

    def test_validate_config_returns_list(self, default_section):
        """validate_config must return a list."""
        warnings = default_section.validate_config()
        assert isinstance(warnings, list)

    def test_no_warnings_on_default_config(self, default_section):
        """Default EvolutionSection config should produce no warnings."""
        warnings = default_section.validate_config()
        assert warnings == [], (
            f"Default config produced unexpected warnings: {warnings}"
        )

    def test_config_field_defaults(self, default_section):
        """Verify key default values are as expected."""
        assert default_section.population_size == 50
        assert default_section.max_generations == 200
        assert default_section.mutation_rate == 0.3
        assert default_section.seed == 42
        assert default_section.use_gpu is False
