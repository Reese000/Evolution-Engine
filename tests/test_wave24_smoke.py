#!/usr/bin/env python3
"""tests/test_wave24_smoke.py — Final smoke tests for Evolution Engine wave 24.

Two lightweight test classes that verify core object construction and API
surfaces without running any actual evolution.  All tests are <100ms.
"""
from __future__ import annotations

import warnings

import pytest

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine construction, attributes, and key API methods."""

    def test_engine_creation_with_default_config(self, engine: EvolutionEngine) -> None:
        """Engine created via conftest has expected type."""
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine) -> None:
        """Engine exposes all core attributes needed by downstream code."""
        expected_attrs = [
            "config",
            "champion",
            "champion_level",
            "champion_fitness",
            "current_generation",
            "generations_stalled",
            "rng",
            "novelty_archive",
            "qd_archive",
            "strategy_archive",
            "_hall_of_fame",
            "best_fitness_history",
            "circuit_breaker",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine) -> None:
        """engine.run exists and is callable (we don't invoke it)."""
        assert callable(engine.run)

    def test_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_run_diagnostics returns a dict with expected keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        expected_keys = [
            "champion_level",
            "champion_fitness",
            "champion_words",
            "generations_stalled",
            "current_generation",
            "novelty_archive_size",
            "qd_archive_size",
            "hall_of_fame_size",
            "mutation_rate",
            "population_size",
        ]
        for key in expected_keys:
            assert key in diag, f"Missing diagnostics key: {key}"

    def test_get_performance_metrics_returns_dict(self, engine: EvolutionEngine) -> None:
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
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metrics key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionConfig creation and EvolutionSection API surfaces."""

    def test_evolution_config_defaults(self) -> None:
        """EvolutionConfig can be instantiated with default values."""
        cfg = EvolutionConfig()
        assert cfg.population_size == 50
        assert cfg.mutation_rate == 0.3
        assert cfg.max_generations == 200

    def test_evolution_section_defaults(self) -> None:
        """EvolutionSection (config.py) can be instantiated with defaults."""
        section = EvolutionSection()
        assert section.population_size == 50
        assert section.mutation_rate == 0.3
        assert section.seed == 42

    def test_get_config_summary_returns_dict_with_expected_keys(self) -> None:
        """EvolutionSection.get_config_summary returns a dict with expected keys."""
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
            assert key in summary, f"Missing summary key: {key}"

    def test_validate_config_returns_list(self) -> None:
        """EvolutionSection.validate_config returns a list."""
        section = EvolutionSection()
        result = section.validate_config()
        assert isinstance(result, list)

    def test_no_warnings_on_default_config(self) -> None:
        """Default EvolutionSection should produce no validation warnings."""
        section = EvolutionSection()
        warnings_list = section.validate_config()
        assert warnings_list == [], f"Unexpected warnings on defaults: {warnings_list}"
