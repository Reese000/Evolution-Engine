#!/usr/bin/env python3
"""Wave 12 pipeline smoke tests — verifies EvolutionEngine can be instantiated
and basic diagnostic/state/config methods work without running actual evolution.

4 test classes, ~13 tests total. All fast, no VM farm, no real mutation.
"""
from __future__ import annotations

import collections
import os
import random
import sys

import pytest

# Ensure project root is on sys.path
from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame
from mutation_proposer import EBREAK, make_rv32i


def _make_engine(**overrides):
    """Create a minimal EvolutionEngine via __new__ with sensible defaults.

    Mirrors the conftest ``make_engine`` pattern but lives in-file so
    this module is self-contained.  Any keyword overrides are applied
    after the base setup.
    """
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.config = EvolutionConfig()
    engine.rng = random.Random(42)
    engine.champion_level = 0
    engine.current_generation = 0
    engine.generations_stalled = 0
    engine.champion = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
    engine.champion_trace = {
        "status": "ok",
        "final_regs": {10: 42},
        "final_memory": {},
        "instructions_count": 2,
        "trap_cause": "",
    }
    engine.champion_fitness = 100.0
    engine._raw_task_score = 100.0
    engine._champion_raw_task_score = 100.0
    engine._bonus_threshold_cache = None
    engine._novelty_weight_cache = 0.3
    engine._cached_eval_budget = 5000
    engine._current_diversity = 0.5
    engine._qd_coverage = 0.0
    engine.task_instance = {
        "expected_output": {"registers": {10: 42, 11: 7}},
        "inputs": {"registers": {}, "memory": {}},
    }
    engine.novelty_archive = None
    engine.qd_archive = None
    engine._extract_behavioral_descriptor = None
    engine._compute_qd_axes_fn = None
    engine._cached_task_instance_id = None
    engine._cached_expected_regs = None
    engine._cached_expected_total = 0
    engine._hall_of_fame = HallOfFame(capacity=10)
    engine._unsolved_levels_set = frozenset()
    engine._last_challengers = []
    engine._last_results = []
    engine._last_challenger_strategies = []
    engine._last_challenger_programs = []
    engine._last_challenger_fitnesses = []
    engine._generation_state_consumed = True
    engine._champion_changed_since_last_qd = False
    engine._qd_champion_reset_tried = False
    engine._champion_qd_valid = False
    engine._cached_champion_qd_desc = None
    engine._fitness_velocity = 0.0
    engine._velocity_alpha = 0.3
    engine._adaptive_mutation_floor = 0.15
    engine._adaptive_mutation_ceil = 0.85
    engine._original_mutation_rate = engine.config.mutation_rate
    engine._consecutive_same_strategy = 0
    engine._last_strategy_used = ""
    engine.best_fitness_history = []
    engine._max_history_len = 500
    engine._champion_replacement_count = 0
    engine._oscillation_positive_count = 0
    engine._base_hysteresis_min_delta = 0.5
    engine._hysteresis_min_delta = 0.5
    engine._oscillation_threshold = 3
    engine._oscillation_window_size = 10
    engine._champion_oscillation_window = collections.deque(maxlen=10)
    engine.circuit_breaker = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
    engine._seed_model_path = ""
    engine._prefix_generator = None
    engine._external_evolution_params = {}
    engine.recurrent_examples_count = 0
    engine._log_file = None
    engine._log_path_cache = None
    engine._log_handle_closed = True
    engine._cached_checkpoint_dir = None
    engine.strategy_archive = None
    engine.champion_anchor = None
    engine._log = lambda msg: None
    engine.vm_farm = None
    engine.curriculum = None
    engine.use_gpu = False
    engine._secondary_task_results = None
    engine._instruction_program = None
    engine.champion_archive = None

    for k, v in overrides.items():
        setattr(engine, k, v)
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestEngineInstantiation
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineInstantiation:
    """Verify EvolutionEngine can be created and has expected attributes."""

    def test_engine_creates_with_default_config(self):
        engine = _make_engine()
        assert engine is not None
        assert engine.config is not None

    def test_engine_has_expected_attributes(self):
        engine = _make_engine()
        assert hasattr(engine, "strategy_archive")
        assert hasattr(engine, "champion_archive")
        assert hasattr(engine, "config")
        assert hasattr(engine, "rng")

    def test_engine_run_is_callable(self):
        engine = _make_engine()
        assert callable(getattr(engine, "run", None))


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestEngineDiagnostics
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineDiagnostics:
    """Verify diagnostic methods return dicts."""

    def test_get_run_diagnostics_returns_dict(self):
        engine = _make_engine()
        result = engine.get_run_diagnostics()
        assert isinstance(result, dict)

    def test_get_performance_metrics_returns_dict(self):
        engine = _make_engine()
        result = engine.get_performance_metrics()
        assert isinstance(result, dict)

    def test_get_archive_summary_returns_dict(self):
        engine = _make_engine()
        result = engine.get_archive_summary()
        assert isinstance(result, dict)

    def test_get_training_summary_returns_dict(self):
        engine = _make_engine()
        result = engine.get_training_summary()
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TestEngineState
# ══════════════════════════════════════════════════════════════════════════════


class TestEngineState:
    """Verify state introspection methods."""

    def test_get_evolution_state_returns_dict_with_expected_keys(self):
        engine = _make_engine()
        state = engine.get_evolution_state()
        assert isinstance(state, dict)
        # Top-level keys are structural; the actual data lives in sub-dicts
        expected_top = {"champion", "run_summary", "state", "config"}
        assert expected_top & set(state.keys()), (
            f"Expected at least one of {expected_top}, got {set(state.keys())}"
        )

    def test_get_champion_summary_returns_string(self):
        engine = _make_engine()
        summary = engine.get_champion_summary()
        assert isinstance(summary, str)

    def test_get_comprehensive_state_returns_nested_dict(self):
        engine = _make_engine()
        state = engine.get_comprehensive_state()
        assert isinstance(state, dict)
        # Should have at least one nested dict value (diagnostics, state, etc.)
        has_nested = any(isinstance(v, dict) for v in state.values())
        assert has_nested, "Expected at least one nested dict in comprehensive state"


# ══════════════════════════════════════════════════════════════════════════════
# 4. TestConfigIntegration
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigIntegration:
    """Verify EvolutionSection (config.py) has summary and validation methods."""

    def test_config_has_get_config_summary(self):
        from config import EvolutionSection
        cfg = EvolutionSection()
        assert hasattr(cfg, "get_config_summary")
        assert callable(cfg.get_config_summary)

    def test_config_has_validate_config(self):
        from config import EvolutionSection
        cfg = EvolutionSection()
        assert hasattr(cfg, "validate_config")
        assert callable(cfg.validate_config)

    def test_validate_config_returns_list(self):
        from config import EvolutionSection
        cfg = EvolutionSection()
        result = cfg.validate_config()
        assert isinstance(result, list)
