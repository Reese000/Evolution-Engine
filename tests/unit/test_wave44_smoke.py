"""Wave 44 — final smoke tests for engine and config health.

Fast, no-evolution tests that verify core object creation and API
surface without running any actual evolution cycles.
"""
from __future__ import annotations

# Import shared factory from conftest
import os
import sys

import pytest

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine
from tests.conftest import make_engine

# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine creation and API surface."""

    def test_engine_creation_with_default_config(self):
        """Engine can be created with default config."""
        engine = make_engine()
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self):
        """Engine has all expected core attributes."""
        engine = make_engine()
        expected_attrs = [
            "config",
            "rng",
            "champion_level",
            "current_generation",
            "champion_fitness",
            "champion",
            "champion_trace",
            "circuit_breaker",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self):
        """Engine.run is a callable method."""
        engine = make_engine()
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self):
        """Engine.get_run_diagnostics returns a dict."""
        engine = make_engine()
        result = engine.get_run_diagnostics()
        assert isinstance(result, dict)

    def test_engine_get_performance_metrics_returns_dict(self):
        """Engine.get_performance_metrics returns a dict."""
        engine = make_engine()
        result = engine.get_performance_metrics()
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionConfig and EvolutionSection config creation and API surface."""

    def test_config_creation_with_defaults(self):
        """EvolutionConfig can be created with defaults."""
        cfg = EvolutionConfig()
        assert isinstance(cfg, EvolutionConfig)

    def test_get_config_summary_returns_dict(self):
        """get_config_summary returns a dict with expected keys."""
        cfg = EvolutionSection()
        summary = cfg.get_config_summary()
        assert isinstance(summary, dict)
        expected_keys = [
            "total_config_fields",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_validate_config_returns_list(self):
        """validate_config returns a list."""
        cfg = EvolutionSection()
        result = cfg.validate_config()
        assert isinstance(result, list)

    def test_no_warnings_on_default_config(self):
        """Default config produces no validation warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"Unexpected warnings: {warnings}"
