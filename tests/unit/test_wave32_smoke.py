#!/usr/bin/env python3
"""Wave 32: Final smoke tests for Evolution Engine core classes.

Tests basic health of EvolutionEngine and EvolutionConfig — creation,
attribute presence, method callability, and return types. All tests are
instant (no actual evolution runs).
"""
from __future__ import annotations

import warnings

from evolution_engine import EvolutionConfig, EvolutionEngine

# Re-use the conftest factory (not the fixture, to keep tests standalone)
from tests.conftest import make_engine

# ─── TestEngineHealth ───────────────────────────────────────────────────────


class TestEngineHealth:
    """Verify EvolutionEngine basic health via the conftest factory."""

    def test_engine_creates_with_default_config(self):
        engine = make_engine()
        assert isinstance(engine, EvolutionEngine)
        assert isinstance(engine.config, EvolutionConfig)

    def test_engine_has_expected_attributes(self):
        engine = make_engine()
        expected = [
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
        ]
        for attr in expected:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self):
        engine = make_engine()
        assert callable(engine.run)

    def test_engine_get_run_diagnostics_returns_dict(self):
        engine = make_engine()
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        # Must contain core diagnostic keys
        for key in (
            "champion_level",
            "champion_fitness",
            "generations_stalled",
            "current_generation",
        ):
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_engine_get_performance_metrics_returns_dict(self):
        engine = make_engine()
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        for key in (
            "current_generation",
            "population_size",
            "elite_ratio",
            "mutation_rate",
        ):
            assert key in metrics, f"Missing metric key: {key}"


# ─── TestConfigHealth ───────────────────────────────────────────────────────


class TestConfigHealth:
    """Verify EvolutionConfig basic health with defaults."""

    def test_config_creates_with_defaults(self):
        cfg = EvolutionConfig()
        assert isinstance(cfg, EvolutionConfig)

    def test_config_has_expected_attributes(self):
        cfg = EvolutionConfig()
        key_attrs = [
            "population_size",
            "mutation_rate",
            "elite_ratio",
            "crossover_rate",
            "tournament_size",
            "max_generations",
            "stall_threshold",
            "eval_budget",
            "use_gpu",
            "seed",
        ]
        for attr in key_attrs:
            assert hasattr(cfg, attr), f"Missing config attribute: {attr}"

    def test_config_population_size_positive(self):
        cfg = EvolutionConfig()
        assert cfg.population_size > 0

    def test_config_mutation_rate_in_range(self):
        cfg = EvolutionConfig()
        assert 0.0 <= cfg.mutation_rate <= 1.0

    def test_config_elite_ratio_in_range(self):
        cfg = EvolutionConfig()
        assert 0.0 <= cfg.elite_ratio <= 1.0

    def test_no_warnings_on_default_config(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = EvolutionConfig()
            # Filter out irrelevant warnings (deprecation, import, etc.)
            relevant = [x for x in w if "config" in str(x.message).lower() or "evolution" in str(x.message).lower()]
            assert len(relevant) == 0, f"Unexpected config warnings: {relevant}"
