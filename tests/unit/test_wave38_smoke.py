#!/usr/bin/env python3
"""Wave 38 — Final smoke tests for EvolutionEngine and EvolutionConfig.

Fast health-check tests that verify core construction, attributes, and
API contracts without running any actual evolution.
"""
from __future__ import annotations

import dataclasses
import warnings

import pytest

from evolution_engine import EvolutionConfig, EvolutionEngine

# ══════════════════════════════════════════════════════════════════════════════
# TestEngineHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineHealth:
    """Verify EvolutionEngine construction, attributes, and API contracts."""

    def test_engine_creation(self, engine: EvolutionEngine) -> None:
        """EvolutionEngine can be created via conftest _make_engine fixture."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine) -> None:
        """Engine exposes all core attributes expected by the hot path."""
        expected_attrs = [
            "config",
            "champion",
            "champion_level",
            "champion_fitness",
            "champion_trace",
            "current_generation",
            "generations_stalled",
            "rng",
            "novelty_archive",
            "qd_archive",
            "_hall_of_fame",
            "best_fitness_history",
            "circuit_breaker",
        ]
        for attr in expected_attrs:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine) -> None:
        """engine.run is a callable method."""
        assert callable(engine.run)

    def test_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine) -> None:
        """get_run_diagnostics returns a dict with expected diagnostic keys."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        expected_keys = [
            "champion_level",
            "champion_fitness",
            "generations_stalled",
            "current_generation",
            "mutation_rate",
        ]
        for key in expected_keys:
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_get_performance_metrics_returns_dict(
        self, engine: EvolutionEngine
    ) -> None:
        """get_performance_metrics returns a dict with expected metric keys."""
        perf = engine.get_performance_metrics()
        assert isinstance(perf, dict)
        expected_keys = [
            "current_generation",
            "population_size",
            "elite_ratio",
            "eval_budget",
            "champion_words",
        ]
        for key in expected_keys:
            assert key in perf, f"Missing metric key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigHealth:
    """Verify EvolutionConfig construction, serialization, and validation."""

    def test_config_creation_with_defaults(self) -> None:
        """EvolutionConfig can be created with no arguments (all defaults)."""
        cfg = EvolutionConfig()
        assert cfg is not None
        assert isinstance(cfg, EvolutionConfig)

    def test_config_summary_returns_dict_with_expected_keys(self) -> None:
        """EvolutionConfig serialises to a dict containing all core keys."""
        cfg = EvolutionConfig()
        summary = dataclasses.asdict(cfg)
        assert isinstance(summary, dict)
        expected_keys = [
            "population_size",
            "max_generations",
            "mutation_rate",
            "crossover_rate",
            "elite_ratio",
            "eval_budget",
            "instruction_budget",
            "seed",
            "curriculum_level",
            "use_novelty_archive",
            "use_qd_archive",
            "use_gpu",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing summary key: {key}"
        assert summary["population_size"] == 50
        assert summary["max_generations"] == 200

    def test_validate_config_rejects_invalid_values(self) -> None:
        """Invalid config values raise ValueError in __post_init__."""
        with pytest.raises(ValueError, match="population_size"):
            EvolutionConfig(population_size=0)
        with pytest.raises(ValueError, match="mutation_rate"):
            EvolutionConfig(mutation_rate=1.5)
        with pytest.raises(ValueError, match="elite_ratio"):
            EvolutionConfig(elite_ratio=1.5)

    def test_no_warnings_on_default_config(self) -> None:
        """Creating a default EvolutionConfig emits no warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EvolutionConfig()
            # Filter out non-config warnings (e.g. third-party)
            config_warnings = [
                x for x in w if "evolution_engine" in str(x.filename)
            ]
            assert config_warnings == [], (
                f"Default config produced {len(config_warnings)} warning(s)"
            )
