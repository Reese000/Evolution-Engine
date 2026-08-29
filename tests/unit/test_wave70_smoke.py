#!/usr/bin/env python3
"""test_wave70_smoke.py — Final smoke tests for EvolutionEngine and config.

Two test classes:
  1. TestEngineHealth   — engine creation, attributes, run, diagnostics, metrics
  2. TestConfigHealth   — EvolutionConfig defaults, summary, validation

Fast tests — no VM, no training, no actual evolution runs.
Does NOT modify any existing files.
"""
from __future__ import annotations

import pytest

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine

# ── TestEngineHealth ────────────────────────────────────────────────────────


class TestEngineHealth:
    """Verify EvolutionEngine creation, attributes, and diagnostic surfaces."""

    def test_engine_creation_with_default_config(
        self, engine: EvolutionEngine
    ) -> None:
        """Engine can be created via the conftest engine fixture."""
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_all_expected_attributes(
        self, engine: EvolutionEngine
    ) -> None:
        """Engine exposes every attribute expected by the hot-path methods."""
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
            "_hall_of_fame",
            "circuit_breaker",
            "vm_farm",
            "curriculum",
            "best_fitness_history",
            "use_gpu",
            "strategy_archive",
            "champion_anchor",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"engine missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine) -> None:
        """engine.run is a callable method."""
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(
        self, engine: EvolutionEngine
    ) -> None:
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
            "strategy_archive_records",
            "hall_of_fame_size",
            "mutation_rate",
            "population_size",
        ]
        for key in expected_keys:
            assert key in diag, f"diagnostics missing key: {key}"

    def test_engine_get_performance_metrics_returns_dict(
        self, engine: EvolutionEngine
    ) -> None:
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
            "hof_size",
        ]
        for key in expected_keys:
            assert key in metrics, f"metrics missing key: {key}"


# ── TestConfigHealth ────────────────────────────────────────────────────────


class TestConfigHealth:
    """Verify EvolutionConfig defaults and EvolutionSection validation surfaces."""

    def test_evolution_config_creation_with_defaults(self) -> None:
        """EvolutionConfig can be instantiated with all defaults."""
        cfg = EvolutionConfig()
        assert isinstance(cfg, EvolutionConfig)
        assert cfg.population_size == 50
        assert cfg.mutation_rate == 0.3
        assert cfg.eval_budget == 5000

    def test_config_summary_returns_dict_with_expected_keys(self) -> None:
        """EvolutionSection.get_config_summary returns a dict with expected keys."""
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
            assert key in summary, f"summary missing key: {key}"

    def test_validate_config_returns_list(self) -> None:
        """EvolutionSection.validate_config returns a list (possibly empty)."""
        cfg = EvolutionSection()
        result = cfg.validate_config()
        assert isinstance(result, list)

    def test_no_warnings_on_default_config(self) -> None:
        """Default EvolutionSection configuration produces no validation warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"unexpected warnings on default config: {warnings}"
