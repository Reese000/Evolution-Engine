#!/usr/bin/env python3
"""Wave 35 — Final smoke tests for Evolution Engine.

Fast health-checks verifying that the core engine and config objects
can be constructed and their key read-only methods return the expected
types and contents.  No actual evolution runs.  Uses conftest fixtures.
"""
from __future__ import annotations

import pytest
from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine construction, attributes, and core read-only API."""

    def test_engine_creates_with_default_config(self, engine: EvolutionEngine) -> None:
        """Engine fixture produces a usable instance."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine) -> None:
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
            "best_fitness_history",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine) -> None:
        """engine.run exists and is callable."""
        assert hasattr(engine, "run")
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_run_diagnostics returns a dict with champion info."""
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

    def test_engine_get_performance_metrics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_performance_metrics returns a dict with population info."""
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
    """Verify config objects construct cleanly and expose diagnostic APIs."""

    def test_evolution_config_creates_with_defaults(self) -> None:
        """EvolutionConfig() succeeds without arguments."""
        cfg = EvolutionConfig()
        assert cfg is not None
        assert isinstance(cfg, EvolutionConfig)
        assert cfg.mutation_rate == 0.3
        assert cfg.population_size == 50
        assert cfg.seed == 42

    def test_config_section_creates_with_defaults(self) -> None:
        """EvolutionSection() succeeds without arguments."""
        evo = EvolutionSection()
        assert evo is not None
        assert isinstance(evo, EvolutionSection)
        assert evo.population_size == 50
        assert evo.seed == 42

    def test_config_summary_returns_dict_with_expected_keys(self) -> None:
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

    def test_validate_config_returns_list(self) -> None:
        """validate_config returns a list."""
        evo = EvolutionSection()
        warnings = evo.validate_config()
        assert isinstance(warnings, list)

    def test_no_warnings_on_default_config(self) -> None:
        """Default EvolutionSection produces zero validation warnings."""
        evo = EvolutionSection()
        warnings = evo.validate_config()
        assert warnings == [], f"Unexpected warnings on default config: {warnings}"
