#!/usr/bin/env python3
"""Wave 68 smoke tests — quick health checks for EvolutionEngine and config.

Fast tests that verify core objects can be instantiated and key methods
return the expected types.  No actual evolution runs are executed.
"""
from __future__ import annotations

import pytest

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine


# ── TestEngineHealth ────────────────────────────────────────────────────────


class TestEngineHealth:
    """Verify the EvolutionEngine and EvolutionConfig core surfaces."""

    def test_engine_creation(self, engine: EvolutionEngine) -> None:
        """Engine can be created via the conftest engine fixture."""
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_config(self, engine: EvolutionEngine) -> None:
        """Engine carries an EvolutionConfig instance."""
        assert isinstance(engine.config, EvolutionConfig)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine) -> None:
        """Engine exposes all attributes expected by the hot-path methods."""
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
    """Verify EvolutionSection (config.py) defaults and validation surfaces."""

    def test_config_creation_with_defaults(self) -> None:
        """EvolutionSection can be created with all defaults."""
        cfg = EvolutionSection()
        assert isinstance(cfg, EvolutionSection)
        assert cfg.population_size == 50
        assert cfg.mutation_rate == 0.3
        assert cfg.eval_budget == 5000

    def test_config_get_config_summary_returns_dict(self) -> None:
        """get_config_summary returns a dict with expected keys."""
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

    def test_config_validate_config_returns_list(self) -> None:
        """validate_config returns a list (possibly empty)."""
        cfg = EvolutionSection()
        result = cfg.validate_config()
        assert isinstance(result, list)

    def test_config_no_warnings_on_defaults(self) -> None:
        """Default configuration produces no validation warnings."""
        cfg = EvolutionSection()
        warnings = cfg.validate_config()
        assert warnings == [], f"unexpected warnings on default config: {warnings}"
