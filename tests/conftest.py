#!/usr/bin/env python3
"""Shared test fixtures for the Evolution Engine test suite.

R38: Consolidates repeated helper functions across test files into
shared fixtures. Each round file previously redefined _make_engine,
_make_replace_engine, _snap, _mc, etc. with identical implementations.
This conftest provides pytest fixtures and reusable factory functions.
"""
from __future__ import annotations

import collections
import os
import random
import sys
from typing import Optional

import pytest

# Path setup -- add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame, compute_qd_axes
from meta_controller import MetaController, TelemetrySnapshot
from mutation_proposer import EBREAK, NOP, generate_seed_program, make_rv32i

# ══════════════════════════════════════════════════════════════════════════════
# Engine factory
# ══════════════════════════════════════════════════════════════════════════════


def make_engine(
    expected_regs: Optional[dict[int, int]] = None,
    champion_fitness: float = 100.0,
    champion_level: int = 0,
    diversity: float = 0.5,
    generations_stalled: int = 0,
    with_archive: bool = False,
) -> EvolutionEngine:
    """Create a minimal EvolutionEngine for unit testing.

    Bypasses __init__ to avoid needing VM farm, champion anchor, etc.
    All attributes needed by hot-path methods are pre-initialized.
    """
    if expected_regs is None:
        expected_regs = {10: 42, 11: 7}

    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.config = EvolutionConfig()
    engine.rng = random.Random(42)
    engine.champion_level = champion_level
    engine.current_generation = 0
    engine.generations_stalled = generations_stalled
    engine.champion = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
    engine.champion_trace = {
        "status": "ok",
        "final_regs": {10: 42},
        "final_memory": {},
        "instructions_count": 2,
        "trap_cause": "",
    }
    engine.champion_fitness = champion_fitness
    engine._raw_task_score = champion_fitness
    engine._champion_raw_task_score = champion_fitness
    engine._bonus_threshold_cache = None
    engine._novelty_weight_cache = 0.3
    engine._cached_eval_budget = 5000
    engine._current_diversity = diversity
    engine._qd_coverage = 0.0
    engine.task_instance = {
        "expected_output": {"registers": expected_regs},
        "inputs": {"registers": {}, "memory": {}},
    }
    engine.novelty_archive = None
    engine.qd_archive = None
    engine._extract_behavioral_descriptor = None
    engine._compute_qd_axes_fn = compute_qd_axes
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

    if with_archive:
        from strategy_archive import StrategyArchive

        engine.strategy_archive = StrategyArchive()

    return engine


def make_engine_with_task(expected_regs: Optional[dict[int, int]] = None) -> EvolutionEngine:
    """Create engine with minimal task instance (for fitness computation)."""
    if expected_regs is None:
        expected_regs = {10: 42, 11: 7}
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.task_instance = {"expected_output": {"registers": expected_regs}}
    engine.champion_level = 0
    engine._bonus_threshold_cache = None
    engine._novelty_weight_cache = 0.0
    engine._cached_eval_budget = 5000
    engine._current_diversity = 0.5
    engine._qd_coverage = 0.0
    engine._raw_task_score = 0.0
    engine._champion_raw_task_score = 0.0
    engine.config = EvolutionConfig()
    engine.champion = None
    engine.champion_trace = None
    engine.novelty_archive = None
    engine.qd_archive = None
    engine._extract_behavioral_descriptor = None
    engine._compute_qd_axes_fn = compute_qd_axes
    engine.generations_stalled = 0
    engine.current_generation = 0
    engine._cached_task_instance_id = None
    engine._cached_expected_regs = None
    engine._cached_expected_total = 0
    engine._secondary_task_results = None
    engine._instruction_program = None
    return engine


# ══════════════════════════════════════════════════════════════════════════════
# MetaController factory
# ══════════════════════════════════════════════════════════════════════════════


def make_meta_controller(checkpoint_dir: str = "/tmp/mc_test") -> MetaController:
    """Create a MetaController with a temporary checkpoint directory."""
    return MetaController(checkpoint_dir=checkpoint_dir)


def make_snapshot(
    cycle: int = 1,
    fitness: float = 1.0,
    level: int = 0,
    champion_words: int = 10,
    val_loss: Optional[float] = None,
    diversity: float = 0.5,
    novelty_size: int = 0,
    qd_occupied: int = 0,
    qd_total: int = 100,
    traces_count: int = 0,
    retrained: bool = False,
    generations_completed: int = 50,
) -> TelemetrySnapshot:
    """Create a TelemetrySnapshot with sensible defaults."""
    return TelemetrySnapshot(
        cycle=cycle,
        fitness=fitness,
        level=level,
        champion_words=champion_words,
        val_loss=val_loss,
        diversity=diversity,
        novelty_size=novelty_size,
        qd_occupied=qd_occupied,
        qd_total=qd_total,
        traces_count=traces_count,
        retrained=retrained,
        generations_completed=generations_completed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Program generators
# ══════════════════════════════════════════════════════════════════════════════


def make_program(*instr_names: str, seed: int = 42) -> list[int]:
    """Create a program from instruction names. Appends EBREAK."""
    rng = random.Random(seed)
    prog = []
    for name in instr_names:
        if name == "NOP":
            prog.append(NOP)
        elif name == "EBREAK":
            prog.append(EBREAK)
        else:
            prog.append(
                make_rv32i(
                    name,
                    rd=rng.randint(1, 31),
                    rs1=rng.randint(0, 31),
                    rs2=rng.randint(0, 31),
                )
            )
    if not prog or prog[-1] != EBREAK:
        prog.append(EBREAK)
    return prog


def make_random_programs(n: int, length: int = 5, seed: int = 42) -> list[list[int]]:
    """Generate n random programs of the given length."""
    rng = random.Random(seed)
    return [generate_seed_program(length=length, rng=rng) for _ in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# Pytest fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine():
    """Minimal EvolutionEngine for unit testing."""
    return make_engine()


@pytest.fixture
def engine_with_archive():
    """EvolutionEngine with strategy archive enabled."""
    return make_engine(with_archive=True)


@pytest.fixture
def meta_controller():
    """MetaController with temporary checkpoint directory."""
    return make_meta_controller()


@pytest.fixture
def rng():
    """Seeded random.Random for deterministic tests."""
    return random.Random(42)
