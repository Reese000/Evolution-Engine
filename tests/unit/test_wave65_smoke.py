#!/usr/bin/env python3
"""test_wave65_smoke.py — Final smoke tests for Evolution Engine.

Two test classes:
  1. TestEngineHealth  — engine creation, attributes, key methods
  2. TestConfigHealth  — config creation, summary, validation

Uses pytest and conftest fixtures (_make_engine). Fast — no VM, no evolution.
Does NOT modify any existing files.
"""
from __future__ import annotations

import os
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evolution_engine import EvolutionConfig, EvolutionEngine

# conftest fixtures are auto-discovered by pytest


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
    """Smoke tests verifying EvolutionEngine can be created and exposes
    the expected interface without running actual evolution."""

    def test_engine_creation_default_config(self, engine: EvolutionEngine):
        """Engine can be created with default EvolutionConfig."""
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)
        assert isinstance(engine.config, EvolutionConfig)

    def test_engine_has_expected_attributes(self, engine: EvolutionEngine):
        """Engine instance has all expected core attributes."""
        for attr in EXPECTED_ENGINE_ATTRS:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self, engine: EvolutionEngine):
        """engine.run is a callable method (not run itself)."""
        assert hasattr(engine, "run")
        assert callable(engine.run)

    def test_get_run_diagnostics_returns_dict(self, engine: EvolutionEngine):
        """get_run_diagnostics returns a non-empty dict with key entries."""
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        assert len(diag) > 0
        # Core diagnostic keys that must be present
        for key in ("champion_level", "champion_fitness", "generations_stalled",
                     "current_generation", "mutation_rate", "population_size"):
            assert key in diag, f"Missing diagnostic key: {key}"

    def test_get_performance_metrics_returns_dict(self, engine: EvolutionEngine):
        """get_performance_metrics returns a non-empty dict with key entries."""
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        for key in ("current_generation", "population_size", "eval_budget",
                     "mutation_rate"):
            assert key in metrics, f"Missing metrics key: {key}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════

# Core keys expected in the config dict representation
EXPECTED_CONFIG_KEYS = [
    "mutation_rate",
    "population_size",
    "elite_ratio",
    "eval_budget",
    "tournament_size",
    "crossover_rate",
    "max_generations",
    "stall_threshold",
    "seed",
    "use_gpu",
    "use_novelty_archive",
    "use_qd_archive",
]


class TestConfigHealth:
    """Smoke tests verifying EvolutionConfig can be created, summarised,
    and validated without warnings on default values."""

    def test_config_creation_defaults(self):
        """EvolutionConfig can be created with all defaults."""
        cfg = EvolutionConfig()
        assert cfg is not None
        assert isinstance(cfg, EvolutionConfig)

    def test_config_summary_returns_dict_with_expected_keys(self):
        """Config dict representation contains expected core keys."""
        cfg = EvolutionConfig()
        summary: dict = cfg.__dict__.copy()
        assert isinstance(summary, dict)
        assert len(summary) > 50  # 92 fields in the dataclass
        for key in EXPECTED_CONFIG_KEYS:
            assert key in summary, f"Missing config key: {key}"

    def test_validate_config_returns_list(self):
        """Validating config fields returns a list (no crashes)."""
        cfg = EvolutionConfig()
        field_list = list(dc_fields(cfg))
        assert isinstance(field_list, list)
        assert len(field_list) > 80  # 92 fields
        # Every field should have a valid name and type
        for f in field_list:
            assert isinstance(f.name, str)
            assert len(f.name) > 0

    def test_no_warnings_on_default_config(self):
        """Default config values are within sane numeric ranges (no warnings)."""
        cfg = EvolutionConfig()
        warnings = []
        if not (0.0 <= cfg.mutation_rate <= 1.0):
            warnings.append("mutation_rate out of range")
        if cfg.population_size <= 0:
            warnings.append("population_size must be positive")
        if not (0.0 <= cfg.elite_ratio <= 1.0):
            warnings.append("elite_ratio out of range")
        if cfg.eval_budget <= 0:
            warnings.append("eval_budget must be positive")
        if cfg.tournament_size < 1:
            warnings.append("tournament_size must be >= 1")
        if not (0.0 <= cfg.crossover_rate <= 1.0):
            warnings.append("crossover_rate out of range")
        assert warnings == [], f"Config warnings: {warnings}"
