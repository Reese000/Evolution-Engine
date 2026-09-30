#!/usr/bin/env python3
"""test_wave72_smoke.py — Final smoke tests for Evolution Engine.

Two test classes:
  1. TestEngineHealth  — engine creation, attributes, key methods
  2. TestConfigHealth  — config creation, summary, validation

Uses pytest and conftest fixtures (_make_engine). Fast — no VM, no evolution.
Does NOT modify any existing files.
"""
from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import EvolutionSection
from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame

# conftest fixtures are auto-discovered by pytest


# ── Local helper (mirrors conftest.make_engine for direct use) ───────────────
def _make_engine(**overrides) -> EvolutionEngine:
    """Create a minimal EvolutionEngine for unit testing (conftest-compatible)."""
    from mutation_proposer import EBREAK, make_rv32i

    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.config = overrides.pop("config", EvolutionConfig())
    engine.rng = random.Random(overrides.pop("seed", 42))
    engine.champion_level = overrides.pop("champion_level", 0)
    engine.current_generation = overrides.pop("current_generation", 0)
    engine.generations_stalled = overrides.pop("generations_stalled", 0)
    engine.champion = overrides.pop(
        "champion",
        [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK],
    )
    engine.champion_trace = {
        "status": "ok",
        "final_regs": {10: 42},
        "final_memory": {},
        "instructions_count": 2,
        "trap_cause": "",
    }
    engine.champion_fitness = overrides.pop("champion_fitness", 100.0)
    engine._raw_task_score = engine.champion_fitness
    engine._champion_raw_task_score = engine.champion_fitness
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
    engine._champion_oscillation_window = deque(maxlen=10)
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
    engine.champion_archive = None
    engine.champion_anchor = None
    engine._log = lambda msg: None
    engine.vm_farm = None
    engine.curriculum = None
    engine.use_gpu = False
    engine._secondary_task_results = None
    engine._instruction_program = None
    engine._cycle_number = 1
    engine._champion_verified = False
    engine._fitness_momentum = 0.0
    engine._champion_replacement_history = []
    return engine


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
    """Verify EvolutionEngine basic construction and interface."""

    def test_engine_creation_with_defaults(self):
        """Engine can be created with default config."""
        engine = _make_engine()
        assert engine is not None
        assert isinstance(engine, EvolutionEngine)

    def test_engine_has_all_expected_attributes(self):
        """Engine has all required attributes after construction."""
        engine = _make_engine()
        for attr in EXPECTED_ENGINE_ATTRS:
            assert hasattr(engine, attr), f"Missing attribute: {attr}"

    def test_engine_run_is_callable(self):
        """engine.run is a callable method."""
        engine = _make_engine()
        assert callable(getattr(engine, "run", None)), "engine.run is not callable"

    def test_engine_get_run_diagnostics_returns_dict(self):
        """engine.get_run_diagnostics() returns a dict."""
        engine = _make_engine()
        diagnostics = engine.get_run_diagnostics()
        assert isinstance(diagnostics, dict), f"Expected dict, got {type(diagnostics)}"
        # Should contain known keys
        assert "champion_level" in diagnostics
        assert "champion_fitness" in diagnostics
        assert "current_generation" in diagnostics

    def test_engine_get_performance_metrics_returns_dict(self):
        """engine.get_performance_metrics() returns a dict."""
        engine = _make_engine()
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict), f"Expected dict, got {type(metrics)}"
        # Should contain known keys
        assert "current_generation" in metrics
        assert "population_size" in metrics
        assert "eval_budget" in metrics


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestConfigHealth
# ══════════════════════════════════════════════════════════════════════════════

# Keys expected from get_config_summary()
EXPECTED_CONFIG_SUMMARY_KEYS = {
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


class TestConfigHealth:
    """Verify EvolutionSection config construction, summary, and validation."""

    def test_config_creation_with_defaults(self):
        """EvolutionSection can be created with default values."""
        config = EvolutionSection()
        assert config is not None
        assert isinstance(config, EvolutionSection)

    def test_config_summary_returns_dict_with_expected_keys(self):
        """get_config_summary() returns a dict with expected keys."""
        config = EvolutionSection()
        summary = config.get_config_summary()
        assert isinstance(summary, dict), f"Expected dict, got {type(summary)}"
        missing = EXPECTED_CONFIG_SUMMARY_KEYS - set(summary.keys())
        assert not missing, f"Missing keys in config summary: {missing}"

    def test_validate_config_returns_list(self):
        """validate_config() returns a list."""
        config = EvolutionSection()
        warnings = config.validate_config()
        assert isinstance(warnings, list), f"Expected list, got {type(warnings)}"

    def test_no_warnings_on_default_config(self):
        """Default config should produce no validation warnings."""
        config = EvolutionSection()
        warnings = config.validate_config()
        assert len(warnings) == 0, f"Default config produced unexpected warnings: {warnings}"
