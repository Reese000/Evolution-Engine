#!/usr/bin/env python3
"""test_wave77_smoke.py — Final smoke tests for Evolution Engine.

Two test classes:
  1. TestEngineHealth  — verify EvolutionEngine creation, attributes, and
     diagnostic/metric methods return correct types without running evolution.
  2. TestConfigHealth  — verify EvolutionSection config summary, validation,
     and clean defaults.

Uses conftest make_engine() helper and config.EvolutionSection.
Fast — no VM, no evolution. Does NOT modify any existing files.
"""
from __future__ import annotations

import sys
import os

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.conftest import make_engine
from config import EvolutionSection
from evolution_engine import EvolutionConfig


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine creation, attributes, and diagnostic methods."""

    @pytest.fixture()
    def engine(self):
        """Provide a minimal EvolutionEngine via conftest factory."""
        return make_engine()

    # ── creation ──────────────────────────────────────────────────────────

    def test_engine_created_with_default_config(self, engine):
        """Engine is created successfully with default EvolutionConfig."""
        assert engine is not None
        assert engine.config is not None
        assert isinstance(engine.config, EvolutionConfig)

    # ── attributes ────────────────────────────────────────────────────────

    _EXPECTED_ATTRS = [
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
        "_hall_of_fame",
        "_fitness_velocity",
        "_velocity_alpha",
        "_adaptive_mutation_floor",
        "_adaptive_mutation_ceil",
        "_original_mutation_rate",
        "_consecutive_same_strategy",
        "_last_strategy_used",
        "best_fitness_history",
        "_max_history_len",
        "_champion_replacement_count",
        "circuit_breaker",
        "strategy_archive",
        "vm_farm",
        "curriculum",
        "use_gpu",
        "recurrent_examples_count",
        "_secondary_task_results",
    ]

    @pytest.mark.parametrize("attr", _EXPECTED_ATTRS)
    def test_engine_has_expected_attribute(self, engine, attr):
        """Engine has all critical attributes set during construction."""
        assert hasattr(engine, attr), f"Missing attribute: {attr}"

    # ── run is callable ───────────────────────────────────────────────────

    def test_run_is_callable(self, engine):
        """engine.run is a callable method (we do NOT invoke it)."""
        assert callable(engine.run)

    # ── get_run_diagnostics ──────────────────────────────────────────────

    def test_get_run_diagnostics_returns_dict(self, engine):
        """get_run_diagnostics returns a dict with expected top-level keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        # Must contain at least these core keys
        expected_keys = {
            "champion_level",
            "champion_fitness",
            "champion_words",
            "generations_stalled",
        }
        assert expected_keys.issubset(diag.keys()), (
            f"Missing keys: {expected_keys - diag.keys()}"
        )

    # ── get_performance_metrics ──────────────────────────────────────────

    def test_get_performance_metrics_returns_dict(self, engine):
        """get_performance_metrics returns a dict with expected keys."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        expected_keys = {
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
        }
        assert expected_keys.issubset(metrics.keys()), (
            f"Missing keys: {expected_keys - metrics.keys()}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionSection config creation, summary, and validation."""

    @pytest.fixture()
    def config(self):
        """Provide default EvolutionSection."""
        return EvolutionSection()

    # ── creation with defaults ───────────────────────────────────────────

    def test_config_created_with_defaults(self, config):
        """EvolutionSection can be created with all defaults."""
        assert config is not None
        assert isinstance(config, EvolutionSection)
        assert config.population_size == 50
        assert config.mutation_rate == 0.3
        assert config.seed == 42

    # ── get_config_summary ───────────────────────────────────────────────

    _SUMMARY_KEYS = {
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

    def test_config_summary_returns_dict_with_expected_keys(self, config):
        """get_config_summary returns a dict containing all expected keys."""
        summary = config.get_config_summary()
        assert isinstance(summary, dict)
        assert self._SUMMARY_KEYS.issubset(summary.keys()), (
            f"Missing keys: {self._SUMMARY_KEYS - summary.keys()}"
        )

    def test_config_summary_values_match_defaults(self, config):
        """Summary values match the dataclass defaults."""
        summary = config.get_config_summary()
        assert summary["seed"] == 42
        assert summary["population_size"] == 50
        assert summary["eval_budget"] == 5000
        assert summary["mutation_rate"] == 0.3
        assert summary["novelty_weight"] == 0.3
        assert summary["qd_bonus_weight"] == 0.1
        assert summary["total_config_fields"] == len(
            EvolutionSection.__dataclass_fields__
        )

    # ── validate_config ──────────────────────────────────────────────────

    def test_validate_config_returns_list(self, config):
        """validate_config returns a list (empty on default config)."""
        result = config.validate_config()
        assert isinstance(result, list)

    def test_no_warnings_on_default_config(self, config):
        """Default config produces no validation warnings."""
        result = config.validate_config()
        assert result == [], f"Unexpected warnings on default config: {result}"
