#!/usr/bin/env python3
"""Wave 29 — Final smoke tests for EvolutionEngine and EvolutionConfig.

Fast, non-running tests that verify core object creation, attribute presence,
method signatures, and return types. No actual evolution is executed.
"""

from __future__ import annotations

import types

import pytest

from config import EvolutionSection
from evolution_engine import EvolutionConfig

# ══════════════════════════════════════════════════════════════════════════════
# TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify the core EvolutionEngine is wired up correctly."""

    # -- attributes present on a default-config engine -----------------------

    EXPECTED_ATTRS: list[str] = [
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
        "use_gpu",
        "curriculum",
        "recurrent_examples_count",
    ]

    def test_engine_creation(self, engine):
        """EvolutionEngine can be created via conftest fixture."""
        assert engine is not None

    def test_engine_has_expected_attributes(self, engine):
        """Engine exposes all core attributes after construction."""
        for attr in self.EXPECTED_ATTRS:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine):
        """engine.run is a callable (method/function)."""
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine):
        """get_run_diagnostics() returns a dict with known keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict), f"Expected dict, got {type(diag)}"
        # Key fields that must be present
        for key in ("champion_level", "champion_fitness", "generations_stalled"):
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_engine_get_performance_metrics_returns_dict(self, engine):
        """get_performance_metrics() returns a dict with known keys."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        for key in ("current_generation", "population_size", "eval_budget"):
            assert key in metrics, f"Missing metrics key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionSection (config.py) default construction and helpers."""

    def test_config_creation_with_defaults(self):
        """EvolutionSection() instantiates without error on defaults."""
        cfg = EvolutionSection()
        assert cfg is not None

    def test_get_config_summary_returns_dict_with_expected_keys(self):
        """get_config_summary() returns a dict containing core keys."""
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

    def test_validate_config_returns_list(self):
        """validate_config() returns a list (possibly empty)."""
        cfg = EvolutionSection()
        result = cfg.validate_config()
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_no_warnings_on_default_config(self):
        """Default EvolutionSection passes validation with zero warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"Unexpected warnings on defaults: {warnings}"
