#!/usr/bin/env python3
"""test_wave76_smoke.py — Final smoke tests for Evolution Engine.

Two test classes:
  1. TestEngineHealth  — engine creation, attributes, key methods
  2. TestConfigHealth  — config creation, summary, validation

Uses pytest and conftest fixtures (_make_engine). Fast — no VM, no evolution.
Does NOT modify any existing files.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EvolutionSection
from evolution_engine import EvolutionEngine
from tests.conftest import make_engine as _make_engine

# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════

# Attributes every engine instance must have after construction
EXPECTED_ENGINE_ATTRS = [
    "config",
    "champion",
    "champion_fitness",
    "champion_trace",
    "champion_level",
    "current_generation",
    "generations_stalled",
    "best_fitness_history",
    "circuit_breaker",
    "novelty_archive",
    "qd_archive",
    "strategy_archive",
    "_hall_of_fame",
    "vm_farm",
    "curriculum",
    "use_gpu",
]


class TestEngineHealth:
    """Verify EvolutionEngine basic construction and interface."""

    def test_engine_creation_with_defaults(self):
        """Engine can be created with default config."""
        engine = _make_engine()
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_all_expected_attributes(self):
        """Engine has all required attributes after construction."""
        engine = _make_engine()
        for attr in EXPECTED_ENGINE_ATTRS:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self):
        """engine.run is a callable method."""
        engine = _make_engine()
        assert callable(getattr(engine, "run", None)), "engine.run is not callable"

    def test_engine_get_run_diagnostics_returns_dict(self):
        """engine.get_run_diagnostics() returns a dict."""
        engine = _make_engine()
        diagnostics = engine.get_run_diagnostics()
        assert isinstance(diagnostics, dict), f"Expected dict, got {type(diagnostics)}"
        # Should contain known keys
        assert "champion_level" in diagnostics
        assert "champion_fitness" in diagnostics
        assert "current_generation" in diagnostics

    def test_engine_get_performance_metrics_returns_dict(self):
        """engine.get_performance_metrics() returns a dict."""
        engine = _make_engine()
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        # Should contain known keys
        assert "current_generation" in metrics
        assert "population_size" in metrics
        assert "eval_budget" in metrics


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════

# Keys expected from get_config_summary()
EXPECTED_CONFIG_SUMMARY_KEYS = {
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


class TestConfigHealth:
    """Verify EvolutionSection config construction, summary, and validation."""

    def test_config_creation_with_defaults(self):
        """EvolutionSection can be created with default values."""
        config = EvolutionSection()
        assert config is not None
        assert isinstance(config, EvolutionSection)

    def test_config_summary_returns_dict_with_expected_keys(self):
        """get_config_summary() returns a dict with expected keys."""
        config = EvolutionSection()
        summary = config.get_config_summary()
        assert isinstance(summary, dict), f"Expected dict, got {type(summary)}"
        missing = EXPECTED_CONFIG_SUMMARY_KEYS - set(summary.keys())
        assert not missing, f"Missing keys in config summary: {missing}"

    def test_validate_config_returns_list(self):
        """validate_config() returns a list."""
        config = EvolutionSection()
        warnings = config.validate_config()
        assert isinstance(warnings, list), f"Expected list, got {type(warnings)}"

    def test_no_warnings_on_default_config(self):
        """Default config should produce no validation warnings."""
        config = EvolutionSection()
        warnings = config.validate_config()
        assert len(warnings) == 0, f"Default config produced unexpected warnings: {warnings}"
