#!/usr/bin/env python3
"""RV32I Evolution Engine -- Champion/Challenger with Directed Mutation.

Implements the core evolutionary loop: maintain a champion program,
generate challengers via 4 mutation strategies (RANDOM, TASK_AWARE,
DIRECTED, PERTURBATION_CAPPED), evaluate on the VM farm, and replace the
champion when a challenger achieves higher fitness.
"""

from __future__ import annotations

import heapq
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any

# Shared immutable empty tuples for trace field allocation optimization.
# Avoids creating new list objects on every _eval_result_to_trace call
# (~50 per generation, thousands over a full run).
_EMPTY_TUPLE: tuple = ()

# R36: Pre-computed EBREAK constant to avoid repeated hex literal lookups
# in hot mutation/evaluation loops (~100+ comparisons per generation).
_EBREAK_INST: int = 0x00100073

# R36: Frozen frozenset of crash/exception statuses for O(1) membership test
# in _select_elite and _compute_fitness hot paths (avoids repeated string
# comparisons or set creation per challenger).
_CRASH_STATUSES: frozenset[str] = frozenset(("crash", "trap", "exception"))
_TIMEOUT_STATUSES: frozenset[str] = frozenset(("timeout",))

# Phase 3: Training signal quality thresholds
# Only save traces with fitness above this threshold to the recurrent buffer.
# Prevents training on low-quality/crash traces that contaminate the model.
MIN_FITNESS_FOR_TRACE: float = 0.1
# Maximum entries in recurrent_data.jsonl before quality-based eviction kicks in.
# Evicts lowest-fitness traces when exceeded.
MAX_RECURRENT_ENTRIES: int = 500
# Recency decay rate for training sample weights (Phase 3.2).
# weight = exp(-decay_rate * age_in_cycles). At age=100, weight ~ 37%.
RECURRENT_DECAY_RATE: float = 0.01

# ----------------------------------------------------------------------
# Path setup -- import from sibling modules and milestone0
# ----------------------------------------------------------------------

from logging_config import get_logger

_logger = get_logger("EvolutionEngine")

from paths import MILESTONE0_DIR, MILESTONE1_DIR, ensure_project_on_path
from paths import PROJECT_ROOT as SCRIPT_DIR

ensure_project_on_path()

# ----------------------------------------------------------------------
# Imports from the real vm_farm module
# ----------------------------------------------------------------------

from vm_farm import EvalResult, _run_one_evaluation  # type: ignore


class VMFarm:
    """Wrapper around the safety-kernel RV32I VM farm.

    Provides evaluate/evaluate_batch interface expected by EvolutionEngine,
    delegating to _run_one_evaluation from the vm_farm module.
    When use_gpu=True and GpuVMFarm is available, uses GPU-accelerated
    batch evaluation via CuPy CUDA kernel (30-300x faster than CPU).
    """

    @staticmethod
    def _resolve_kernel_path(kernel_path: str) -> str:
        """Resolve the safety-kernel binary path.

        If kernel_path is the default string or doesn't exist as a file,
        try to locate the actual safety-kernel binary relative to this file.

        Returns the path if found, or an empty string when the VM is pure
        Python and no external kernel binary is needed.
        """
        if os.path.isfile(kernel_path):
            return kernel_path
        # Auto-detect relative to milestone0/
        candidates = [
            os.path.join(MILESTONE0_DIR, "safety-kernel", "target", "release", "safety-kernel"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                _logger.info(f"Kernel binary resolved: {c}")
                return c
        # No external kernel found -- pure-Python VM is used instead.
        return ""

    def __init__(self, num_workers: int = 1, kernel_path: str = "safety_kernel", use_gpu: bool = False, **kwargs):
        # Input validation
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError(f"num_workers must be a positive int, got {num_workers!r}")
        self.num_workers = num_workers
        self.kernel_path = self._resolve_kernel_path(kernel_path)
        self.use_gpu = bool(use_gpu)
        self._gpu_farm = None
        self._gpu_fallback_count: int = 0
        self._gpu_eval_count: int = 0
        self._pool_executor = None  # R17: cached ProcessPoolExecutor
        if self.use_gpu:
            try:
                import gpu_vm_farm as _gvm

                _gvm._ensure_cupy()  # triggers lazy CuPy import
                if not _gvm.HAS_CUDA:
                    self._log("CUDA not available -- GPU acceleration disabled")
                    self.use_gpu = False
                else:
                    self._gpu_farm = _gvm.GpuVMFarm(batch_size=128, budget=5000)
                    self._log("GPU VM farm initialized (batch_size=128, budget=5000)")
                    self._run_bit_exact_check()
            except ImportError:
                self._log("CuPy not installed -- GPU acceleration disabled")
                self._gpu_farm = None
                self.use_gpu = False
            except Exception as e:
                self._log(f"GPU init failed: {e} -- falling back to CPU")
                self._gpu_farm = None
                self.use_gpu = False

    def _log(self, msg: str) -> None:
        _logger.info(f"[VMFarm] {msg}")

    def _run_bit_exact_check(self, n_samples: int = 10) -> None:
        """Verify GPU and CPU produce identical results for random programs."""
        if self._gpu_farm is None:
            return
        import random as _rnd

        _rnd.seed(42)
        for s in range(n_samples):
            prog = [_rnd.randint(0, 0xFFFFFFFF) for _ in range(_rnd.randint(3, 12))]
            prog[-1] = 0x00100073
            inputs = {"registers": {10: _rnd.randint(0, 255)}, "memory": {}}
            try:
                cpu = _run_one_evaluation(
                    kernel_path=self.kernel_path,
                    program=prog,
                    budget=5000,
                    init_regs=inputs.get("registers", {}),
                    init_mem=inputs.get("memory", {}),
                )
                gpu_res = self._gpu_farm.evaluate_batch(prog, [inputs], budget=5000)
                if gpu_res:
                    cpu_ok = getattr(cpu, "success", True)
                    gpu_ok = gpu_res[0].get("success", True)
                    if cpu_ok != gpu_ok:
                        self._log(f"WARNING: GPU/CPU mismatch at sample {s} -- disabling GPU")
                        self._gpu_farm = None
                        self.use_gpu = False
                        return
            except Exception as e:
                self._log(f"Bit-exact check failed: {e} -- disabling GPU")
                self._gpu_farm = None
                self.use_gpu = False
                return
        self._log(f"GPU bit-exact: {n_samples}/{n_samples} samples match CPU")

    def evaluate(self, program: list[int], inputs: dict[str, Any], budget: int = 10000) -> EvalResult:
        init_regs = inputs.get("registers", {})
        init_mem = inputs.get("memory", {})
        return _run_one_evaluation(
            kernel_path=self.kernel_path,
            program=program,
            budget=budget,
            init_regs=init_regs,
            init_mem=init_mem,
        )

    def _eval_single_worker(self, program, task, budget):
        """Evaluate a single program on a single worker."""
        try:
            result = self.evaluate(program, task, budget)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e), "instructions_count": 0}

    def evaluate_batch(self, programs, task, budget=None):
        """Evaluate a batch of programs using the VM farm."""
        if not programs:
            return []
        budget = budget or self.config.eval_budget
        # R32: Try GPU farm first when available
        if self.use_gpu and self._gpu_farm is not None and len(programs) > 1:
            try:
                gpu_results = self._gpu_farm.evaluate_multi(programs, task, budget=budget)
                if gpu_results is not None:
                    self._gpu_eval_count += 1
                    return gpu_results
            except Exception:
                self._gpu_fallback_count += 1
                self._gpu_farm = None
                self.use_gpu = False
                self._log("GPU batch failed — falling back to CPU")
        if self.num_workers <= 1:
            return [self._eval_single_worker(p, task, budget) for p in programs]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._eval_single_worker, p, task, budget) for p in programs]
            return [f.result() for f in futures]

    @staticmethod
    def _dict_to_eval_result(d: dict) -> EvalResult:
        """Convert GPU result dict to EvalResult namedtuple."""
        return EvalResult(
            success=d.get("success", False),
            registers=d.get("registers", {}),
            memory=d.get("memory", {}),
            trace=[],
            cycles=d.get("cycles", 0),
            error=d.get("error"),
            blocked_writes=d.get("blocked_writes", 0),
        )


# Try importing from milestone2 modules
from champion_anchor import ChampionAnchor, IntegrityError, sha256_program
from mutation_proposer import (
    NUM_OPCODE_CATEGORIES,
    OPCODE_CATEGORY_MAP,
    MutationStrategy,
    TraceAnalysis,
    TraceAnalyzer,
    _crossover_mutation,
    _encode_inst,
    generate_seed_program,
    propose_mutation,
    trim_dead_code,
)

# Pre-built 128-element list for O(1) opcode-to-category lookup without
# hash computation. Index = low 7 bits of instruction word; value = category
# index (0-10) or -1 for unmapped. ~30% faster than dict.get() in hot loops.
_OPCODE_CAT_ARRAY: list[int] = [-1] * 128
for _opcode_val, _cat_val in OPCODE_CATEGORY_MAP.items():
    _OPCODE_CAT_ARRAY[_opcode_val & 0x7F] = _cat_val
del _opcode_val, _cat_val

# R33: Pre-built MutationStrategy enum lookup dict for O(1) string->enum
# conversion (avoids Python enum __new__ overhead on every call to
# _choose_mutation_strategy, which runs pop_size times per generation).
_STRATEGY_ENUM_CACHE: dict[str, MutationStrategy] = {s.value: s for s in MutationStrategy}
from evolution.diversity_helpers import check_level_circuit_breaker as _check_level_circuit_breaker_fn
from evolution.diversity_helpers import compute_population_diversity as _compute_population_diversity_fn
from evolution.diversity_helpers import count_backward_branches as _count_backward_branches_fn
from evolution.diversity_helpers import inject_diversity as _inject_diversity_fn
from evolution.diversity_helpers import is_level_disabled as _is_level_disabled_fn
from evolution.diversity_helpers import record_cycle_level_outcome as _record_cycle_level_outcome_fn
from evolution.diversity_helpers import record_level_circuit_outcome as _record_level_circuit_outcome_fn

# --- Extracted helper modules (P1-1 refactoring) ---
from evolution.fitness_helpers import apply_bonuses_and_normalize as _apply_bonuses_and_normalize_fn
from evolution.fitness_helpers import apply_novelty_qd_blending as _apply_novelty_qd_blending_fn
from evolution.fitness_helpers import batch_apply_bonuses as _batch_apply_bonuses_fn
from evolution.fitness_helpers import batch_apply_crash_penalties as _batch_apply_crash_penalties_fn
from evolution.fitness_helpers import batch_blend_novelty_qd as _batch_blend_novelty_qd_fn
from evolution.fitness_helpers import batch_classify_programs as _batch_classify_programs_fn
from evolution.fitness_helpers import batch_compute_base as _batch_compute_base_fn
from evolution.fitness_helpers import batch_compute_efficiency as _batch_compute_efficiency_fn
from evolution.fitness_helpers import batch_compute_fitness as batch_compute_fitness_fn
from evolution.fitness_helpers import batch_compute_structural_bonuses as _batch_compute_structural_bonuses_fn
from evolution.fitness_helpers import batch_get_expected as _batch_get_expected_fn
from evolution.fitness_helpers import batch_score_registers as _batch_score_registers_fn
from evolution.fitness_helpers import compute_all_bonuses as _compute_all_bonuses_fn
from evolution.fitness_helpers import compute_base_score as _compute_base_score_fn
from evolution.fitness_helpers import compute_efficiency_score as _compute_efficiency_score_fn
from evolution.fitness_helpers import compute_fitness as _compute_fitness_fn
from evolution.fitness_helpers import compute_multi_task_score as _compute_multi_task_score_fn
from evolution.fitness_helpers import compute_smc_bonus as _compute_smc_bonus_fn
from evolution.fitness_helpers import compute_structural_bonuses as _compute_structural_bonuses_fn
from evolution.fitness_helpers import compute_time_credit as _compute_time_credit_fn
from evolution.fitness_helpers import extract_trace_info as _extract_trace_info_fn
from evolution.fitness_helpers import finalize_with_blending as _finalize_with_blending_fn
from evolution.fitness_helpers import log_bonus_diagnostics as _log_bonus_diagnostics_fn
from evolution.fitness_helpers import score_register_match as _score_register_match_fn
from evolution.mutation_helpers import apply_affinity_bias as _apply_affinity_bias_fn
from evolution.mutation_helpers import apply_cycle_phase_bias as _apply_cycle_phase_bias_fn
from evolution.mutation_helpers import apply_level_aware_strategy_bias as _apply_level_aware_strategy_bias_fn
from evolution.mutation_helpers import check_priority_overrides as _check_priority_overrides_fn
from evolution.mutation_helpers import check_variance_override as _check_variance_override_fn
from evolution.mutation_helpers import compute_model_temperature as _compute_model_temperature_fn
from evolution.mutation_helpers import get_model_guided_prefix as _get_model_guided_prefix_fn
from evolution.mutation_helpers import get_model_guided_ratio as _get_model_guided_ratio_fn
from tasks_rv32i import LEVELS, get_task_level, reference_solution

# Optional: curriculum integration from milestone1
try:
    from curriculum_generator import CurriculumConfig, CurriculumGenerator  # type: ignore

    _HAS_CURRICULUM = True
except ImportError:
    _HAS_CURRICULUM = False

try:
    from difficulty_metrics import calibrate_task_instance  # type: ignore

    _HAS_DIFFICULTY = True
except ImportError:
    _HAS_DIFFICULTY = False

# Optional: StrategyArchive
try:
    from strategy_archive import StrategyArchive  # type: ignore

    _HAS_STRATEGY_ARCHIVE = True
except ImportError:
    _HAS_STRATEGY_ARCHIVE = False

# Optional: Novelty archive
try:
    from novelty import SparseArchive  # type: ignore

    _HAS_NOVELTY = True
except ImportError:
    _HAS_NOVELTY = False

# Optional: QD archive
try:
    from qd_archive import MAPElitesArchive  # type: ignore

    _HAS_QD = True
except ImportError:
    _HAS_QD = False


# ----------------------------------------------------------------------
# QD behavioural axes computation
# ----------------------------------------------------------------------


def compute_qd_axes(
    program: list[int] | None,
    trace: dict[str, Any] | None,
) -> list[float] | None:
    """Compute behavioural axes for MAP-Elites QD grid.

    Returns a 2-element vector [instruction_diversity, register_pressure]
    both in [-1, 1], or [-1.0, -1.0] when data is unavailable.

    * **instruction_diversity**: fraction of distinct RV32I opcode categories
      used (0 of 11 -> -1.0, 11 of 11 -> 1.0).
    * **register_pressure**: fraction of non-x0 registers with non-zero
      values in trace final_regs (0 of 31 -> -1.0, 31 of 31 -> 1.0).

    Uses module-level OPCODE_CATEGORY_MAP to avoid per-call dict creation.
    R15: Uses integer bitset for opcode tracking (~2x faster than set).
    """
    if program and isinstance(program, list) and len(program) > 0:
        # Fast opcode category scan using 128-element array for O(1) lookup
        # without hash computation per word (~30% faster than dict.get).
        seen_bits = 0
        cat_arr = _OPCODE_CAT_ARRAY
        for word in program:
            # Type guard: skip non-integer words (corrupt programs, dicts, etc.)
            if not isinstance(word, int):
                continue
            cat = cat_arr[word & 0x7F]
            if cat >= 0:
                seen_bits |= 1 << cat
        # Count set bits via C-implemented int.bit_count() (R16)
        count = seen_bits.bit_count()
        inst_div = (count / NUM_OPCODE_CATEGORIES) * 2.0 - 1.0
    else:
        inst_div = -1.0

    if trace and isinstance(trace, dict) and "final_regs" in trace:
        regs = trace.get("final_regs", {})
        if isinstance(regs, dict):
            active = sum(1 for idx, val in regs.items() if isinstance(idx, int) and 1 <= idx <= 31 and val != 0)
            reg_pressure = (active / 31.0) * 2.0 - 1.0
        else:
            reg_pressure = -1.0
    else:
        reg_pressure = -1.0

    return [inst_div, reg_pressure]


# ----------------------------------------------------------------------
# Hall of Fame — prevents losing good programs to aggressive mutation
# ----------------------------------------------------------------------


class HallOfFame:
    """Fixed-capacity hall of fame for the best programs encountered.

    Retains the top-K programs by fitness across the entire run, even
    when the champion is replaced or the curriculum advances. This
    prevents losing good programs to aggressive mutation or curriculum
    transitions. Programs are injected back as challengers during deep
    stall to re-inject proven quality.

    R28: Adds recency weighting to injection selection -- when multiple
    programs qualify, prefer more recent ones to avoid stale entries
    dominating injection. Uses generation timestamp in the heap tuple.

    Capacity defaults to 10. Uses a min-heap for O(log k) insert and
    O(1) access to the worst entry, avoiding the previous O(K log K)
    re-sort on every add.
    """

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        # Min-heap of (fitness, generation, program, level) tuples.
        # The smallest fitness is at index 0, enabling O(1) worst check
        # and O(log K) replacement via heapreplace.
        self._heap: list[tuple[float, int, list[int], int]] = []

    def add(self, program: list[int], fitness: float, level: int, generation: int = 0) -> bool:
        """Add a program if it qualifies for the hall of fame.

        R28: Now accepts generation parameter for recency tracking.

        Returns True if the program was added (and one was evicted or
        the hall was not yet full).
        """
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, (fitness, generation, list(program), level))
            return True
        if fitness > self._heap[0][0]:
            heapq.heapreplace(self._heap, (fitness, generation, list(program), level))
            return True
        return False

    def get_best(self, level: int | None = None) -> list[int] | None:
        """Get the highest-fitness program, optionally filtered by level.

        Uses min() on heap tuples directly (avoids building a second list
        for level filtering when possible).
        """
        if not self._heap:
            return None
        if level is not None:
            # Filter in-place on the min-heap (max fitness = max first elem)
            best_fit = float("-inf")
            best_prog = None
            for f, _g, p, l in self._heap:
                if l == level and f > best_fit:
                    best_fit = f
                    best_prog = p
            if best_prog is not None:
                return list(best_prog)
            # Fall through to unfiltered
        best = max(self._heap, key=lambda e: e[0])
        return list(best[2])

    def get_random(self, rng=None, level=None):
        """Return a random program from the Hall of Fame, with recency bias.

        Args:
            rng: Random number generator (default: stdlib random.Random()).
            level: If given, only return programs at this level.

        Returns:
            A defensive copy of the program list, or None if empty.
        """
        import random

        if rng is None:
            rng = random.Random()
        heap = getattr(self, "_heap", [])
        if not heap:
            return None
        # Heap tuple: (fitness, generation, program, level)
        if level is not None:
            candidates = [(f, g, p, l) for f, g, p, l in heap if l == level]
            if not candidates:
                candidates = heap
        else:
            candidates = heap
        # R28: Weight by generation (recency bias) — recent programs are
        # more likely to be selected, giving them higher novelty pressure.
        if len(candidates) > 1:
            max_gen = max(g for _, g, _, _ in candidates)
            weights = [1.0 + (g / max(max_gen, 1)) * 9.0 for _, g, _, _ in candidates]
            chosen = rng.choices(candidates, weights=weights, k=1)[0]
        else:
            chosen = candidates[0]
        return list(chosen[2])  # entry[2] is the program

    @property
    def size(self) -> int:
        return len(self._heap)

    @property
    def best_fitness(self) -> float:
        # For a min-heap, best fitness is the max element.
        # O(K) scan but K=10 so negligible.
        return max((e[0] for e in self._heap), default=0.0)

    @property
    def worst_fitness(self) -> float:
        """Return the worst (minimum) fitness in the hall of fame.

        For a min-heap, the worst fitness is at index 0 (O(1)).
        Returns -inf when empty so callers can always compare.
        """
        if self._heap:
            return self._heap[0][0]
        return float("-inf")


# ----------------------------------------------------------------------
# Evolution Engine
# ----------------------------------------------------------------------

# Module-level default values for EvolutionConfig._set_config_defaults()
_CONFIG_DEFAULTS: dict[str, object] = {
    "mutation_rate": 0.3,
    "crossover_rate": 0.7,
    "elite_ratio": 0.1,
    "tournament_size": 3,
    "population_size": 50,
    "max_generations": 100,
    "eval_budget": 500,
    "checkpoint_interval": 10,
    "curriculum_level": 0,
    "auto_curriculum": True,
    "curriculum_stall_threshold": 20,
    "use_gpu": False,
    "seed": 42,
    "log_dir": "logs",
    "champion_dir": "",
    "use_novelty_archive": False,
    "novelty_weight": 0.1,
    "novelty_threshold": 0.5,
    "use_qd_archive": False,
    "qd_bonus_weight": 0.1,
    "qd_bins": [10, 10],
    "max_bonus_fraction": 1.0,
    "bonus_log_interval": 50,
    "efficiency_score_weight": 1.0,
    "secondary_task_weight": 1.0,
    "use_secondary_tasks": False,
    "use_mutation_scheduler": False,
    "mutation_rate_start": 0.5,
    "mutation_rate_end": 0.1,
    "mutation_schedule_generations": 50,
    "use_temperature_annealing": False,
    "temperature_start": 2.0,
    "temperature_end": 0.5,
    "temperature_schedule_generations": 50,
    "model_guided_ratio": None,
    "model_guided_ratio_low": 0.1,
    "model_guided_ratio_mid": 0.2,
    "model_guided_ratio_high": 0.3,
    "use_strategy_archive": True,
    "strategy_archive_path": "",
    "recurrent_feedback_dir": "",
    "max_level": 30,
    "max_consecutive_same_strategy": 30,
    "perturbation_capped_probability_unsolved": 0.4,
    "perturbation_capped_unsolved_levels": [9, 10, 11],
    "champion_crossover_prob": 0.3,
}


@dataclass
class EvolutionConfig:
    """Configuration for the evolution engine."""

    # Circuit breaker sentinel (distinct singleton for barrier tuner integration)
    class _CircuitBreakerSentinel:
        """Null-object sentinel when no circuit breaker is configured.

        Used as the default value for EvolutionEngine.circuit_breaker
        when BarrierTuner integration is not active. The test suite
        verifies this sentinel's singleton identity to distinguish
        'no breaker configured' from 'breakers with same state'.
        Always reports not breached (open) to avoid blocking evolution.
        """

        def is_breached(self) -> bool:
            """Sentinel never reports breached (open circuit)."""
            return False

        def __repr__(self) -> str:
            return "<CircuitBreakerSentinel: not configured, always open>"

    CIRCUIT_BREAKER_SENTINEL = _CircuitBreakerSentinel()

    population_size: int = 50
    max_generations: int = 200
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    elite_ratio: float = 0.1
    tournament_size: int = 3
    instruction_budget: int = 10000
    eval_budget: int = 5000
    seed: int = 42
    curriculum_level: int = 0
    auto_curriculum: bool = True
    curriculum_stall_threshold: int = 3
    checkpoint_interval: int = 25
    champion_dir: str = ""
    log_dir: str = ""
    recurrent_feedback_dir: str = ""
    strategy_archive_path: str = ""

    # Dashboard / state persistence
    dashboard_state_dir: str = ""

    # Champion loading
    best_champion_path: str = ""

    # Novelty archive
    novelty_weight: float = 0.15
    use_novelty_archive: bool = True
    novelty_threshold: float = 0.15
    novelty_archive_path: str = ""

    # QD archive
    use_qd_archive: bool = True
    qd_bonus_weight: float = 0.1
    qd_archive_path: str = ""
    qd_bins: int = 20

    # GPU acceleration
    use_gpu: bool = False

    # Retrain hyperparameters (exposed for BarrierTuner tuning)
    retrain_lr: float = 3e-5
    retrain_batch_size: int = 8
    retrain_weight_decay: float = 0.01
    retrain_grad_clip: float = 1.0
    retrain_patience: int = 5
    retrain_freeze_layers: int = 0
    retrain_model_size: str = "medium"
    retrain_max_epochs: int = 50
    retrain_val_split: float = 0.05
    retrain_max_per_level: int = 0  # 0 = unlimited
    retrain_timeout: int = 3600
    retrain_min_examples: int = 5

    # Self-calibration / BarrierTuner params
    # NOTE: adaptive_hyperparams is DEPRECATED in v13 -- kept for backward
    # compatibility with saved configs. Its functionality (curriculum-adaptive
    # elite_ratio, etc.) is now handled by explicit fields like
    # model_guided_ratio_low/mid/high and adaptive_elite_low/high.
    adaptive_hyperparams: bool = True

    # MetaGenome / BarrierTuner evolution params
    meta_weight_improvement: float = 0.3
    meta_weight_depth: float = 0.3
    meta_weight_diversity: float = 0.2
    max_fitness: float = 10.0
    bias_register_prob: float = 0.3
    anchor_retention_floor: float = 0.3
    adaptive_elite_low: float = 0.05
    adaptive_elite_mid: float = 0.10
    adaptive_elite_high: float = 0.15
    adaptive_elite_max: float = 0.20

    # Adaptive crossover / tournament params
    adaptive_cx_mid_add: float = 0.2
    adaptive_cx_high_add: float = 0.3
    adaptive_tourney_mid_add: int = 3
    adaptive_tourney_high_add: int = 5

    # Archive system params
    archive_crossover_success_trigger: float = 0.80
    archive_all_low_trigger: float = 0.30
    archive_cx_adjust_factor: float = 1.1
    archive_pop_adjust_factor: float = 1.25
    archive_elite_reduce_factor: float = 0.80
    archive_consecutive_wins_trigger: int = 3

    # Misc
    variance_penalty_coeff: float = 0.1
    stall_threshold: int = 10
    max_level: int = 30
    max_champion_words: int = 500
    num_eval_seeds: int = 3
    self_calibrate_delta_threshold: float = 0.05
    use_circuit_breaker: bool = False
    circuit_breaker_stall_limit: int = 50

    # Level circuit breaker: auto-disable impossible levels
    # If a level has 0% success rate after N consecutive cycles,
    # it is auto-disabled and skipped in future cycles.
    level_circuit_breaker_enabled: bool = True
    level_circuit_breaker_threshold: int = 10

    # Multi-task fitness extension
    use_secondary_tasks: bool = True
    secondary_task_weight: float = 3.0
    efficiency_score_weight: float = 2.0

    # Bonus normalization -- cap each bonus at a fraction of the base score
    # to prevent any single fitness dimension from dominating the total.
    # 0.30 = no single bonus can contribute more than 30% of the base score.
    max_bonus_fraction: float = 0.30
    # How often (in generations) to log bonus distribution diagnostics.
    # Runs regardless of whether bonus normalization is active, helping
    # diagnose the fitness landscape at any setting. 0 = never log.
    bonus_log_interval: int = 50

    diversity_threshold: float = 0.3
    diversity_inject_ratio: float = 0.1
    champion_crossover_prob: float = 0.7
    crossover_dedicated_prob: float = 0.05
    max_consecutive_same_strategy: int = 30

    # Model-guided generation — boosted after torch.compile removal
    # (inference now 300ms per call, can afford higher MG ratios)
    model_guided_ratio: float | None = None

    # Mutation rate scheduling — decay from high to low over generations
    # to enable broad exploration early and fine-grained exploitation later.
    # Works alongside adaptive mutation: scheduler sets the base rate, adaptive
    # adjusts around it based on fitness velocity.
    use_mutation_scheduler: bool = True
    mutation_rate_start: float = 0.5
    mutation_rate_end: float = 0.1
    mutation_schedule_generations: int = 200

    # Temperature annealing for model-guided generation — smooth decay from
    # high temperature (exploration) to low temperature (exploitation) over
    # the evolution run. Replaces the stepped hard-coded gen ranges.
    use_temperature_annealing: bool = True
    temperature_start: float = 2.0
    temperature_end: float = 0.5
    temperature_schedule_generations: int = 200
    model_guided_ratio_low: float = 0.2
    model_guided_ratio_mid: float = 0.5
    model_guided_ratio_high: float = 0.6

    # PERTURBATION_CAPPED configuration
    perturbation_capped_unsolved_levels: tuple[int, ...] = (9, 10, 11)
    perturbation_capped_probability_unsolved: float = 0.20
    use_strategy_archive: bool = True

    def _set_config_defaults(self):
        """Set default values for configuration fields."""
        for key, val in _CONFIG_DEFAULTS.items():
            if not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, val)

    def __post_init__(self):
        """Apply defaults and validate configuration.

        Raises ValueError for truly invalid configs that cannot be
        sensibly clamped. Values within an acceptable range are clamped
        (the Evolution Engine's design contract).
        """
        self._set_config_defaults()
        self._validate_config()
        self._clamp_config()

    def _validate_config(self):
        """Raise ValueError for invalid config values."""
        self._validate_general()
        self._validate_retrain()
        self._validate_diversity()

    def _validate_general(self):
        """Validate general config parameters."""
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2 for tournament selection, got {self.population_size}")
        if self.max_generations < 1:
            raise ValueError(f"max_generations must be >= 1, got {self.max_generations}")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0,1], got {self.mutation_rate}")
        if self.crossover_rate < 0.0 or self.crossover_rate > 1.0:
            raise ValueError(f"crossover_rate must be in [0,1], got {self.crossover_rate}")
        if self.elite_ratio > 0.5:
            raise ValueError(f"elite_ratio must be <= 0.5, got {self.elite_ratio}")
        if self.tournament_size < 2:
            raise ValueError(f"tournament_size must be >= 2, got {self.tournament_size}")
        if self.tournament_size > self.population_size:
            raise ValueError(
                f"tournament_size ({self.tournament_size}) must be <= population_size ({self.population_size})"
            )
        if self.instruction_budget < 100:
            raise ValueError(f"instruction_budget must be >= 100, got {self.instruction_budget}")
        if self.eval_budget < 51:
            raise ValueError(f"eval_budget must be >= 51, got {self.eval_budget}")
        if self.stall_threshold < 1:
            raise ValueError(f"stall_threshold must be >= 1, got {self.stall_threshold}")
        if self.max_level < 1:
            raise ValueError(f"max_level must be >= 1, got {self.max_level}")
        if self.curriculum_stall_threshold < 1:
            raise ValueError(f"curriculum_stall_threshold must be >= 1, got {self.curriculum_stall_threshold}")
        if self.max_consecutive_same_strategy < 5:
            raise ValueError(f"max_consecutive_same_strategy must be >= 5, got {self.max_consecutive_same_strategy}")
        if self.circuit_breaker_stall_limit < 5:
            raise ValueError(f"circuit_breaker_stall_limit must be >= 5, got {self.circuit_breaker_stall_limit}")
        if self.novelty_weight < 0.0:
            raise ValueError(f"novelty_weight must be >= 0, got {self.novelty_weight}")
        if self.qd_bonus_weight < 0.0:
            raise ValueError(f"qd_bonus_weight must be >= 0, got {self.qd_bonus_weight}")
        if self.meta_weight_improvement < 0.0:
            raise ValueError(f"meta_weight_improvement must be >= 0, got {self.meta_weight_improvement}")

    def _validate_retrain(self):
        """Validate retrain hyperparameter config."""
        if self.retrain_lr <= 0:
            raise ValueError(f"retrain_lr must be > 0, got {self.retrain_lr}")
        if self.retrain_batch_size < 1:
            raise ValueError(f"retrain_batch_size must be >= 1, got {self.retrain_batch_size}")
        if self.retrain_max_epochs < 1:
            raise ValueError(f"retrain_max_epochs must be >= 1, got {self.retrain_max_epochs}")
        if self.retrain_freeze_layers < 0:
            raise ValueError(f"retrain_freeze_layers must be >= 0, got {self.retrain_freeze_layers}")
        if self.retrain_patience < 1:
            raise ValueError(f"retrain_patience must be >= 1, got {self.retrain_patience}")
        if self.retrain_timeout < 60:
            raise ValueError(f"retrain_timeout must be >= 60, got {self.retrain_timeout}")
        if self.retrain_weight_decay >= 1.0:
            raise ValueError(f"retrain_weight_decay must be < 1.0, got {self.retrain_weight_decay}")
        if not 0.0 <= self.retrain_val_split <= 1.0:
            raise ValueError(f"retrain_val_split must be in [0,1], got {self.retrain_val_split}")

    def _validate_diversity(self):
        """Validate diversity and crossover range checks."""
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError(f"diversity_threshold must be in [0,1], got {self.diversity_threshold}")
        if not 0.0 <= self.diversity_inject_ratio <= 1.0:
            raise ValueError(f"diversity_inject_ratio must be in [0,1], got {self.diversity_inject_ratio}")
        if not 0.0 <= self.champion_crossover_prob <= 1.0:
            raise ValueError(f"champion_crossover_prob must be in [0,1], got {self.champion_crossover_prob}")

    def _clamp_config(self):
        """Soft-clamp config to acceptable ranges."""
        self.curriculum_level = max(self.curriculum_level, 0)
        if self.curriculum_level >= self.max_level:
            self.curriculum_level = self.max_level - 1

class EvolutionEngine:
    """Main evolutionary loop for RV32I program synthesis.

    Maintains a champion program per curriculum level and iteratively
    generates challengers via mutation, evaluates them, and promotes
    fitter individuals.
    """

    def _init_archives(self) -> None:
        """Initialize strategy archive, QD archive, novelty archive, and champion archive."""
        from strategy_archive import StrategyArchive

        self.strategy_archive = StrategyArchive()
        if self.config.strategy_archive_path:
            self.strategy_archive.load(self.config.strategy_archive_path)
        self.novelty_archive = None
        if _HAS_NOVELTY and self.config.use_novelty_archive:
            try:
                from novelty import SparseArchive

                self.novelty_archive = SparseArchive(novelty_threshold=self.config.novelty_threshold)
            except (ImportError, Exception):
                pass
        self.qd_archive = None
        if _HAS_QD and self.config.use_qd_archive:
            try:
                from qd_archive import MAPElitesArchive

                self.qd_archive = MAPElitesArchive(bins=self.config.qd_bins)
            except (ImportError, Exception):
                pass
        from champion_archive import ChampionArchive

        archive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champions", "archive.jsonl")
        self.champion_archive = ChampionArchive(path=archive_path, max_size=500)

    def _init_state(self) -> None:
        """Initialize evolution state variables."""
        self.champion = None
        self.champion_trace = None
        self.champion_fitness = 0.0
        self.champion_level = self.config.curriculum_level
        self.current_generation = 0
        self.generations_stalled = 0
        self._qd_champion_reset_tried = False
        self.best_fitness_history = []
        self._max_history_len = 500
        self.task_instance = None
        self._consecutive_same_strategy = 0
        self._last_strategy_used = ""
        self._current_diversity = 0.5
        self._raw_task_score = 0.0
        self.circuit_breaker = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        self._seed_model_path = ""
        self._prefix_generator = None
        self._cycle_number = 1
        self._external_evolution_params = {}
        self.recurrent_examples_count = 0
        self._last_challenger_strategies = []
        self._last_challenger_programs = []
        self._last_challenger_fitnesses = []
        self._last_challengers = []
        self._last_results = []
        self._generation_state_consumed = True
        self._champion_raw_task_score = 0.0
        self._champion_partial_verify = 1.0
        self._champion_changed_since_last_qd = False
        self._champion_changed_since_last_novelty = False
        self._hall_of_fame = HallOfFame(capacity=10)
        self._unsolved_levels_set = frozenset(self.config.perturbation_capped_unsolved_levels)
        self._fitness_velocity = 0.0
        self._velocity_alpha = 0.3
        self._adaptive_mutation_floor = 0.15
        self._adaptive_mutation_ceil = 0.85
        self._original_mutation_rate = self.config.mutation_rate
        self._qd_coverage = 0.0
        self._champion_replacement_count = 0
        self._champion_replacement_history = []

        # Level circuit breaker state
        self._level_circuit_stats: dict[int, dict] = {}  # {level: {"cycles": N, "successes": M}}
        self._disabled_levels: set[int] = set()

    def __init__(self, config=None, rng=None, vm_farm=None, champion_anchor=None):
        self.config = config or EvolutionConfig()
        self.rng = rng or random.Random(self.config.seed)
        self.use_gpu = self.config.use_gpu
        self.vm_farm = vm_farm or VMFarm(num_workers=1, use_gpu=self.config.use_gpu)
        self.champion_anchor = champion_anchor or ChampionAnchor()
        self._init_archives()
        if _HAS_CURRICULUM and self.config.auto_curriculum:
            self.curriculum = CurriculumGenerator(
                CurriculumConfig(max_levels=len(LEVELS), stall_generations=self.config.curriculum_stall_threshold),
                seed=self.config.seed,
            )
        else:
            self.curriculum = None
        self._init_state()
        self._novelty_weight_cache = self.config.novelty_weight
        self._bonus_threshold_cache = None
        self._cached_eval_budget = self.config.eval_budget
        self._cached_expected_regs, self._cached_expected_total, self._cached_task_instance_id = {}, 0, -1
        self._compute_qd_axes_fn = None
        if _HAS_QD:
            try:
                # compute_qd_axes is defined in this module (line 348)
                self._compute_qd_axes_fn = compute_qd_axes
            except Exception:
                pass
        self._extract_behavioral_descriptor = None
        if _HAS_NOVELTY:
            try:
                from novelty import extract_behavioral_descriptor

                self._extract_behavioral_descriptor = extract_behavioral_descriptor
            except (ImportError, Exception):
                pass
        self._champion_qd_valid, self._cached_champion_qd_desc = False, None
        self._oscillation_window_size = 10
        self._champion_oscillation_window = __import__("collections").deque(maxlen=self._oscillation_window_size)
        self._oscillation_positive_count, self._oscillation_threshold, self._hysteresis_min_delta = 0, 3, 0.5
        self._base_hysteresis_min_delta = 0.5
        self.meta_controller, self.progression_tracker = None, None
        self._log_file = None
        self._log_path_cache = None
        self._cached_checkpoint_dir = None
        if self.config.log_dir:
            os.makedirs(self.config.log_dir, exist_ok=True)

    def apply_external_params(self, params: dict[str, Any]) -> None:
        """Apply externally-tuned evolution params (from BarrierTuner/MetaController).
        Call BEFORE run() to inject per-cycle tuning.
        Supported: mutation_rate, diversity_threshold, elite_ratio, etc.
        """
        for key, val in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, val)
        # Refresh cached config values that may have changed
        self._novelty_weight_cache = self.config.novelty_weight

    def load_archive_state(self, qd_data: dict | None = None, novelty_data: dict | None = None) -> None:
        """Load archive state from a previous cycle.

        Call BEFORE run() to restore QD and novelty archives persisted
        from an earlier evolution cycle.
        """
        if qd_data is not None:
            from qd_archive import MAPElitesArchive

            self.qd_archive = MAPElitesArchive.deserialize(qd_data)
        if novelty_data is not None:
            from novelty import SparseArchive

            self.novelty_archive = SparseArchive.deserialize(novelty_data)

    def save_archive_state(self) -> dict[str, Any]:
        """Extract serializable archive state for cross-cycle persistence.

        Returns a dict with 'qd' and 'novelty' keys, each containing
        the serialized archive (JSON-compatible) or None if unavailable.
        """
        return {
            "qd": self.qd_archive.serialize() if self.qd_archive is not None else None,
            "novelty": self.novelty_archive.serialize() if self.novelty_archive is not None else None,
        }

    # ------------------------------------------------------------------
    # R28: Adaptive mutation and oscillation control
    # ------------------------------------------------------------------

    def _compute_adaptive_mutation_rate(self) -> float:
        """Compute a mutation rate that responds to fitness landscape conditions.

        Uses both fitness velocity (EMA of fitness deltas) AND stall depth
        (consecutive generations without improvement) to adjust mutation:

        Velocity-based component:
        - High positive velocity (fast improvement) -> reduce mutation
          to exploit the current gradient.
        - Near-zero velocity (plateau) -> increase mutation to explore.
        - Negative velocity (regression) -> increase mutation strongly
          to escape the local basin.

        Stall-based component (R40):
        - When stalled for many generations, progressively increase mutation
          beyond what velocity alone would suggest. This helps escape deep
          local optima that velocity-based adaptation misses because the
          velocity EMA smooths out over many stagnant generations.
        - Progressive scaling: 10 stall -> 1.1x, 20 -> 1.2x, 30 -> 1.4x,
          50 -> 1.8x. This creates increasingly aggressive exploration
          pressure proportional to time spent stuck.

        The base mutation rate from config is blended with the adaptive
        rate (50/50 for velocity, additional 30% for stall boost) to
        prevent complete loss of the configured bias.

        R33: Uses the original config mutation rate as the base (stored
        in _original_mutation_rate on first adaptive call) instead of the
        current config value, which may already be the result of previous
        adaptive adjustments. This prevents progressive drift where each
        generation's adaptation stacks on top of the last.

        R42: When mutation scheduler is active, the scheduler updates
        _original_mutation_rate each generation to linearly decay from
        mutation_rate_start → mutation_rate_end. The adaptive system then
        adjusts around the scheduled base based on fitness velocity,
        combining broad exploration schedule with velocity-responsive tuning.

        Returns an effective mutation rate in [_adaptive_mutation_floor, _adaptive_mutation_ceil].
        """
        vel = self._fitness_velocity
        base = getattr(self, "_original_mutation_rate", self.config.mutation_rate)

        if vel > 5.0:
            # Fast improvement: reduce mutation to exploit
            adaptive = max(self._adaptive_mutation_floor, base * 0.5)
        elif vel > 1.0:
            # Moderate improvement: slight reduction
            adaptive = max(self._adaptive_mutation_floor, base * 0.75)
        elif vel > -1.0:
            # Plateau: increase mutation to explore
            adaptive = min(self._adaptive_mutation_ceil, base * 1.3)
        elif vel > -5.0:
            # Mild regression: increase mutation more
            adaptive = min(self._adaptive_mutation_ceil, base * 1.6)
        else:
            # Strong regression: maximum exploration
            adaptive = min(self._adaptive_mutation_ceil, base * 2.0)

        # Blend velocity-based adaptive with base config (50/50)
        blended = 0.5 * adaptive + 0.5 * base

        # R40: Stall-based boost -- progressively increase mutation when
        # the champion hasn't improved for many generations. This is
        # independent of velocity (which can be near-zero even when stall
        # is deep because delta=0 for many gens keeps velocity at 0).
        stall = self.generations_stalled
        if stall >= 10:
            # Progressive scaling: more aggressive as stall deepens
            if stall >= 50:
                stall_boost = 1.8
            elif stall >= 30:
                stall_boost = 1.4
            elif stall >= 20:
                stall_boost = 1.2
            else:
                stall_boost = 1.1
            # Apply stall boost blended at 30% weight with the velocity blend
            # This prevents stall boost from fully overriding velocity signal
            stall_adjusted = blended * stall_boost
            blended = 0.7 * blended + 0.3 * stall_adjusted

        return max(self._adaptive_mutation_floor, min(self._adaptive_mutation_ceil, blended))

    def _update_fitness_velocity(self, delta_fitness: float) -> None:
        """Update the exponential moving average of fitness improvement rate.

        Args:
            delta_fitness: change in champion fitness this generation
              (positive = improvement, negative = regression).
        """
        alpha = self._velocity_alpha
        self._fitness_velocity = alpha * delta_fitness + (1.0 - alpha) * self._fitness_velocity

    def get_generation_stats(self) -> dict:
        """Return per-generation statistics for the current run."""
        return {
            "current_generation": self.current_generation,
            "generations_stalled": self.generations_stalled,
            "champion_fitness": self.champion_fitness,
            "champion_level": self.champion_level,
            "mutation_rate": self.config.mutation_rate,
            "population_size": self.config.population_size,
            "novelty_weight": self.config.novelty_weight,
            "fitness_velocity": getattr(self, "_fitness_velocity", 0.0),
        }

    # ------------------------------------------------------------------
    # Fitness landscape diagnostics
    # ------------------------------------------------------------------

    def _get_bonus_config(self, lvl: int, champion_raw: float) -> dict:
        """Compute bonus configuration diagnostics."""
        mbf = getattr(self.config, "max_bonus_fraction", 1.0)
        norm_active = mbf < 1.0
        _sim_base = max(champion_raw, 100.0)
        sim_cap = mbf * _sim_base
        max_possible = {
            "loop": 5.0,
            "mem": 3.0,
            "div": 2.0,
            "level": 25.0 * lvl,
            "efficiency": 30.0,
            "multi_task": 15.0,
            "time_credit": 20.0,
            "smc": 25.0,
            "graduated": 50.0,
        }
        return {
            "normalization_active": norm_active,
            "max_bonus_fraction": mbf,
            "cap_at_simulated_base": sim_cap,
            "simulated_base": _sim_base,
            "bonus_log_interval": getattr(self.config, "bonus_log_interval", 50),
            "max_possible_contributions": max_possible,
            "total_max_bonus": sum(max_possible.values()),
        }

    def _get_bonus_thresholds(self, lvl: int, n_levels: int) -> dict:
        """Compute bonus threshold diagnostics."""
        cached = self._bonus_threshold_cache
        if cached is not None and cached[0] == lvl:
            current_threshold = cached[1]
        else:
            current_threshold = max(0.85, 0.95 - 0.10 * (lvl / max(1, n_levels - 1)))
        graduated_headroom = max(0.0, 1.0 - current_threshold)
        return {
            "current_level": lvl,
            "graduated_bonus_threshold": current_threshold,
            "graduated_headroom": graduated_headroom,
            "max_graduated_bonus": 50.0 * graduated_headroom,
            "level_advancement_bonus": lvl * 25.0,
            "max_level_bonus": (n_levels - 1) * 25.0,
        }

    def _get_structural_scaling(self, lvl: int) -> dict:
        """Compute structural scaling diagnostics."""
        try:
            _scales = self._get_structural_bonus_scale()
        except Exception:
            _scales = {"scale_factor": 1.0, "loop_pressure": 0.33, "mem_pressure": 0.33, "div_pressure": 0.34}
        return {
            "scale_factor": _scales.get("scale_factor", 1.0),
            "loop_pressure": _scales.get("loop_pressure", 0.33),
            "mem_pressure": _scales.get("mem_pressure", 0.33),
            "div_pressure": _scales.get("div_pressure", 0.34),
            "applies_to": ["loop", "mem", "div"] if lvl >= 5 else [],
            "level_gating": {
                "loop_bonus": "L5+",
                "mem_bonus": "L7+",
                "div_bonus": "L5+",
                "smc_bonus": "L12+",
                "efficiency_high_diversity": "L8+",
            },
        }

    def _get_fitness_history_stats(self) -> dict:
        """Compute fitness history trend statistics."""
        hist = self.best_fitness_history
        hist_len = len(hist)
        if hist_len >= 2:
            recent_n = min(hist_len, 20)
            recent = hist[-recent_n:]
            overall_min, overall_max = min(hist), max(hist)
            overall_mean = sum(hist) / hist_len
            recent_min, recent_max = min(recent), max(recent)
            recent_mean = sum(recent) / recent_n
            recent_trend = (recent[-1] - recent[0]) / max(1, recent_n - 1)
            improvements = sum(1 for i in range(1, hist_len) if hist[i] > hist[i - 1])
            improvement_rate = improvements / max(1, hist_len - 1)
            recent_var = sum((f - recent_mean) ** 2 for f in recent) / recent_n
        else:
            overall_min = hist[0] if hist_len == 1 else 0.0
            overall_max = overall_min
            overall_mean = overall_min
            recent_min = overall_max
            recent_max = overall_max
            recent_mean = overall_max
            recent_trend = 0.0
            improvements = 0
            improvement_rate = 0.0
            recent_var = 0.0
        return {
            "total_generations_with_data": hist_len,
            "overall_min": round(overall_min, 4),
            "overall_max": round(overall_max, 4),
            "overall_mean": round(overall_mean, 4),
            "overall_range": round(overall_max - overall_min, 4),
            "recent_window": min(hist_len, 20),
            "recent_min": round(recent_min, 4),
            "recent_max": round(recent_max, 4),
            "recent_mean": round(recent_mean, 4),
            "recent_trend_per_gen": round(recent_trend, 4),
            "improvement_count": improvements,
            "improvement_rate": round(improvement_rate, 4),
            "recent_variance": round(recent_var, 4),
        }

    def _get_population_dynamics(self) -> dict:
        """Compute population dynamics diagnostics."""
        vel = getattr(self, "_fitness_velocity", 0.0)
        stall = self.generations_stalled
        div = getattr(self, "_current_diversity", 0.5)
        qd_cov = getattr(self, "_qd_coverage", 0.0)
        osc_window = list(self._champion_oscillation_window)
        osc_count = getattr(self, "_oscillation_positive_count", 0)
        osc_threshold = getattr(self, "_oscillation_threshold", 3)
        rep_count = getattr(self, "_champion_replacement_count", 0)
        hysteresis_active = osc_count >= osc_threshold
        return {
            "fitness_velocity": round(vel, 4),
            "fitness_velocity_regime": (
                "fast_improvement"
                if vel > 5.0
                else (
                    "moderate_improvement"
                    if vel > 1.0
                    else "plateau" if vel > -1.0 else "mild_regression" if vel > -5.0 else "strong_regression"
                )
            ),
            "generations_stalled": stall,
            "stall_depth_category": (
                "none"
                if stall < 5
                else "shallow" if stall < 15 else "moderate" if stall < 30 else "deep" if stall < 50 else "critical"
            ),
            "diversity": round(div, 4),
            "diversity_category": (
                "critical" if div < 0.1 else "low" if div < 0.2 else "moderate" if div < 0.4 else "high"
            ),
            "qd_niche_coverage": round(qd_cov, 4),
            "qd_coverage_category": (
                "sparse" if qd_cov < 0.1 else "low" if qd_cov < 0.3 else "moderate" if qd_cov < 0.6 else "well_covered"
            ),
            "champion_replacements_total": rep_count,
            "oscillation_window": [round(d, 4) for d in osc_window],
            "oscillation_positive_count": osc_count,
            "oscillation_threshold": osc_threshold,
            "hysteresis_active": hysteresis_active,
        }

    def _get_hof_stats(self) -> dict:
        """Compute Hall of Fame diagnostics."""
        hof = self._hall_of_fame
        hof_heap = getattr(hof, "_heap", [])
        hof_size = len(hof_heap)
        if hof_size > 0:
            hof_fitnesses = [entry[0] for entry in hof_heap]
            hof_best, hof_worst = max(hof_fitnesses), min(hof_fitnesses)
            hof_mean = sum(hof_fitnesses) / hof_size
        else:
            hof_best = hof_worst = hof_mean = 0.0
        return {
            "capacity": hof.capacity,
            "size": hof_size,
            "best_fitness": round(hof_best, 4),
            "worst_fitness": round(hof_worst, 4),
            "mean_fitness": round(hof_mean, 4),
            "fitness_spread": round(hof_best - hof_worst, 4),
        }

    def _get_recommendations(
        self,
        lvl: int,
        mbf: float,
        norm_active: bool,
        stall: int,
        div: float,
        vel: float,
        qd_cov: float,
        qd_w: float,
        champion_raw: float,
        hist_len: int,
        recent_trend: float,
        improvement_rate: float,
        graduated_headroom: float,
        current_threshold: float,
        hof_size: int,
        hof_spread: float,
        hysteresis_active: bool,
        osc_count: int,
        population_dynamics: dict,
        fitness_history: dict,
    ) -> list[str]:
        """Generate actionable recommendations from diagnostic data."""
        recs: list[str] = []
        self._recommend_stall(recs, stall, mbf)
        self._recommend_diversity(recs, div)
        self._recommend_norm(recs, lvl, norm_active, champion_raw)
        self._recommend_graduated(recs, graduated_headroom, current_threshold)
        self._recommend_qd(recs, qd_cov, qd_w)
        self._recommend_oscillation(recs, hysteresis_active, osc_count)
        self._recommend_velocity(recs, vel, stall, population_dynamics)
        self._recommend_improvement(recs, hist_len, improvement_rate, hof_size, hof_spread)
        self._recommend_normalization_off(recs, norm_active, hist_len, recent_trend)
        if not recs:
            recs.append("No issues detected. Fitness landscape appears healthy.")
        return recs

    def _recommend_stall(self, recs: list[str], stall: int, mbf: float) -> None:
        """Add stall-related recommendations."""
        if stall >= 50:
            recs.append(
                f"CRITICAL STALL ({stall} gens): Consider enabling bonus normalization (max_bonus_fraction=0.30) or increasing novelty_weight to break out."
            )
        elif stall >= 20:
            recs.append(
                f"Deep stall ({stall} gens): Bonus normalization could help redistribute selection pressure. Current max_bonus_fraction={mbf:.2f}."
            )

    def _recommend_diversity(self, recs: list[str], div: float) -> None:
        """Add diversity-related recommendations."""
        if div < 0.1:
            recs.append(
                "CRITICAL LOW DIVERSITY (<0.1): Population nearly converged. Boost novelty_weight, increase mutation rate, or inject Hall of Fame entries."
            )
        elif div < 0.2:
            recs.append(
                f"Low diversity ({div:.3f}): Novelty weight will be adaptively boosted. Consider diversity_inject_ratio > {self.config.diversity_inject_ratio}."
            )

    def _recommend_norm(self, recs: list[str], lvl: int, norm_active: bool, champion_raw: float) -> None:
        """Add normalization-related recommendations."""
        if not norm_active:
            lvl_bon = lvl * 25.0
            if champion_raw > 0 and lvl_bon > champion_raw * 0.5:
                recs.append(
                    f"Level bonus ({lvl_bon:.0f}) dominates base score ({champion_raw:.0f}). Enable max_bonus_fraction to prevent level_bonus from overshadowing task-solving fitness at L{lvl}."
                )

    def _recommend_graduated(self, recs: list[str], graduated_headroom: float, current_threshold: float) -> None:
        """Add graduated bonus threshold recommendations."""
        if graduated_headroom < 0.10:
            recs.append(
                f"Graduated bonus threshold ({current_threshold:.3f}) leaves only {graduated_headroom:.3f} headroom. Near-perfect register match is required for any graduated bonus."
            )

    def _recommend_qd(self, recs: list[str], qd_cov: float, qd_w: float) -> None:
        """Add QD coverage recommendations."""
        if qd_cov < 0.1 and qd_w > 0:
            recs.append(
                f"QD niche coverage is sparse ({qd_cov:.3f}). The stall-scaled QD boost should help, but verify novelty_archive is populated."
            )

    def _recommend_oscillation(self, recs: list[str], hysteresis_active: bool, osc_count: int) -> None:
        """Add oscillation/hysteresis recommendations."""
        if hysteresis_active:
            recs.append(
                f"Oscillation detected: {osc_count} replacements in last {self._oscillation_window_size} gens (threshold={getattr(self, '_oscillation_threshold', 3)})."
            )

    def _recommend_velocity(self, recs: list[str], vel: float, stall: int, population_dynamics: dict) -> None:
        """Add velocity and fitness regression recommendations."""
        vel_regime = population_dynamics.get("fitness_velocity_regime", "")
        if vel_regime == "strong_regression":
            recs.append(
                f"Strong fitness regression (velocity={vel:.2f}). Mutation rate will be maximized automatically."
            )
        elif vel_regime == "fast_improvement" and stall > 5:
            recs.append(f"Positive velocity ({vel:.2f}) but still stalled ({stall} gens).")

    def _recommend_improvement(
        self, recs: list[str], hist_len: int, improvement_rate: float, hof_size: int, hof_spread: float
    ) -> None:
        """Add improvement rate and HoF spread recommendations."""
        if hist_len > 10 and improvement_rate < 0.05:
            recs.append(f"Very low improvement rate ({improvement_rate:.1%}) over {hist_len} generations.")
        if hof_size >= 2 and hof_spread < 1.0:
            recs.append(f"Hall of Fame fitness spread is narrow ({hof_spread:.2f}).")

    def _recommend_normalization_off(self, recs: list[str], norm_active: bool, hist_len: int, recent_trend: float) -> None:
        """Add normalization-off recommendations when fitness is flat."""
        if not norm_active and hist_len > 20 and recent_trend <= 0:
            recs.append(
                "Normalization is off and fitness is flat/declining. Enable bonus normalization (max_bonus_fraction=0.30)."
            )

    def get_fitness_diagnostic_summary(self) -> dict[str, Any]:
        """Return a comprehensive diagnostic of the current fitness landscape."""
        lvl = self.champion_level
        n_levels = len(LEVELS)
        raw_task = getattr(self, "_raw_task_score", 0.0)
        champion_raw = getattr(self, "_champion_raw_task_score", raw_task)
        bonus_config = self._get_bonus_config(lvl, champion_raw)
        bonus_thresholds = self._get_bonus_thresholds(lvl, n_levels)
        structural_scaling = self._get_structural_scaling(lvl)
        fitness_history = self._get_fitness_history_stats()
        population_dynamics = self._get_population_dynamics()
        hall_of_fame = self._get_hof_stats()
        qd_w = self.config.qd_bonus_weight
        div = getattr(self, "_current_diversity", 0.5)
        recommendations = self._get_recommendations(
            lvl,
            bonus_config["max_bonus_fraction"],
            bonus_config["normalization_active"],
            self.generations_stalled,
            div,
            getattr(self, "_fitness_velocity", 0.0),
            getattr(self, "_qd_coverage", 0.0),
            qd_w,
            champion_raw,
            len(self.best_fitness_history),
            fitness_history.get("recent_trend_per_gen", 0.0),
            fitness_history.get("improvement_rate", 0.0),
            bonus_thresholds["graduated_headroom"],
            bonus_thresholds["graduated_bonus_threshold"],
            hall_of_fame["size"],
            hall_of_fame["fitness_spread"],
            population_dynamics["hysteresis_active"],
            population_dynamics["oscillation_positive_count"],
            population_dynamics,
            fitness_history,
        )
        return {
            "generation": self.current_generation,
            "champion": {
                "fitness": round(self.champion_fitness, 4),
                "level": lvl,
                "raw_task_score": round(champion_raw, 4),
            },
            "bonus_config": bonus_config,
            "bonus_thresholds": bonus_thresholds,
            "structural_scaling": structural_scaling,
            "fitness_history": fitness_history,
            "population_dynamics": population_dynamics,
            "hall_of_fame": hall_of_fame,
            "novelty_qd": self._build_novelty_qd(div, qd_w),
            "recommendations": recommendations,
        }

    def _build_novelty_qd(self, div: float, qd_w: float) -> dict[str, Any]:
        """Build the novelty_qd sub-dict for diagnostic summary."""
        nw = getattr(self, "_novelty_weight_cache", self.config.novelty_weight)
        return {
            "novelty_weight": nw,
            "novelty_weight_effective": round(
                min(nw * (1.0 + (1.0 - div / 0.2)), 0.8) if div < 0.2 and nw > 0 else nw, 4
            ),
            "qd_bonus_weight": qd_w,
            "novelty_archive_size": len(self.novelty_archive) if self.novelty_archive is not None else 0,
            "qd_archive_occupied": getattr(self.qd_archive, "_occupied_count", 0) if self.qd_archive is not None else 0,
        }

    def _should_replace_champion(self, candidate_fitness: float) -> bool:
        """Decide whether a candidate should replace the champion.

        Uses hysteresis to prevent oscillation: when the champion has been
        replaced too many times recently (rapid A<->B cycling), require a
        minimum fitness delta to break the cycle.

        R33: Stall-adaptive hysteresis -- when deeply stalled, lower the
        minimum delta threshold to allow escape from local optima. At
        stall=0 the threshold is the base value (0.5). At stall=50+ it
        decays to 0.1 (20% of base), progressively relaxing the barrier
        to encourage exploration when stuck.

        Maintains a running count of replacement events in the oscillation
        window to avoid O(n) scan on every call.

        R35: Clamps _oscillation_positive_count to prevent underflow
        when evicting entries that were counted as positive but the
        running count has drifted due to boundary conditions. This
        prevents negative counts that would incorrectly suppress the
        oscillation threshold.

        Returns True if the candidate should replace the champion.
        """
        if candidate_fitness <= self.champion_fitness:
            return False

        delta = candidate_fitness - self.champion_fitness

        # Track replacement and oscillation window
        window = self._champion_oscillation_window
        # R31: deque with maxlen auto-evicts oldest on append (O(1) vs O(n) list.pop(0))
        # Track the evicted value to maintain _oscillation_positive_count.
        if len(window) >= self._oscillation_window_size:
            oldest = window[0]
            if oldest > 0:
                self._oscillation_positive_count -= 1
                # R35: Clamp to prevent underflow from counting drift
                self._oscillation_positive_count = max(self._oscillation_positive_count, 0)
        window.append(delta)

        # Update running count for the new delta
        if delta > 0:
            self._oscillation_positive_count += 1
            # R37: Clamp to window size to prevent overcount drift.
            # The count should never exceed the window size since the window
            # only holds _oscillation_window_size entries. Drift can occur
            # when multiple calls happen before the deque evicts, causing
            # the count to exceed the actual number of positive entries
            # in the window. Clamping ensures the oscillation threshold
            # check is always accurate.
            self._oscillation_positive_count = min(self._oscillation_positive_count, self._oscillation_window_size)

        # Check if oscillation is detected: many replacements in window
        if self._oscillation_positive_count >= self._oscillation_threshold:
            # R33: Stall-adaptive hysteresis -- lower the barrier when
            # deeply stalled to allow escape from local optima.
            # Linear decay: base at stall=0, 20% of base at stall=50+.
            stall = self.generations_stalled
            if stall > 0:
                stall_factor = max(0.2, 1.0 - 0.016 * stall)  # 0.016 * 50 = 0.8 -> 0.2
                effective_delta = self._base_hysteresis_min_delta * stall_factor
            else:
                effective_delta = self._hysteresis_min_delta
            # Hysteresis active: require minimum delta
            if delta < effective_delta:
                return False

        return True

    def _record_champion_replacement(self, delta_fitness: float) -> None:
        """Record that a champion replacement occurred (for oscillation tracking).

        R35: Caps _champion_replacement_count to prevent unbounded memory
        growth on extremely long runs where champions are replaced every
        generation. The count is used only for diagnostics (not algorithmic
        decisions), so capping at 10000 is more than sufficient.
        """
        self._champion_replacement_count += 1
        # R35: Cap replacement count for long-run memory safety
        self._champion_replacement_count = min(self._champion_replacement_count, 10000)
        history = getattr(self, "_champion_replacement_history", None)
        if history is not None:
            history.append((delta_fitness, self.current_generation))
        # Oscillation window is already maintained by _should_replace_champion

    def get_champion_history(self) -> list[dict]:
        """Return history of champion fitness improvements.

        Each entry is a dict with 'fitness' (delta_fitness) and 'cycle' (generation).
        """
        history = getattr(self, "_champion_replacement_history", None)
        if history is None:
            return []
        return [{"fitness": f, "cycle": c} for f, c in history]

    def enable_bonus_normalization(self, fraction: float = 0.30) -> None:
        """Enable bonus normalization with adaptive fitness landscape control.

        Sets max_bonus_fraction to cap each bonus dimension at a fraction
        of the base score, preventing any single fitness component from
        dominating the total.  This is intended to be called dynamically
        during evolution when the engine detects bonus imbalance (e.g. via
        the BonusDiag logging output).

        Args:
            fraction: Maximum fraction of base_score that any single bonus
                may contribute.  Must be in (0, 1].  Typical value is 0.30
                (no bonus exceeds 30% of base score).  Values > 1.0 are
                clamped to 1.0 (effectively disabling normalization).

        Side effects:
            - Updates self.config.max_bonus_fraction for all subsequent
              _compute_fitness calls.
            - Logs the configuration change.

        Raises:
            ValueError: If fraction is not in (0, 1].
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
        old = self.config.max_bonus_fraction
        self.config.max_bonus_fraction = fraction
        _logger.info(f"Bonus normalization enabled: fraction={fraction:.2f} " f"(was {old:.2f})")
        self._log(f"[BonusNorm] enabled fraction={fraction:.2f} " f"(was {old:.2f})")

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _initialize_run(self, ng: int) -> dict:
        """Initialize champion and return initial summary dict."""
        self._refresh_task_instance()
        if self.champion is None:
            self.champion = self._generate_initial_champion()
            is_valid, reason = self.validate_program(self.champion)
            if not is_valid:
                self._log(f"  Initial champion invalid ({reason}), generating replacement")
                for _retry in range(5):
                    self.champion = self._generate_initial_champion()
                    is_valid, reason = self.validate_program(self.champion)
                    if is_valid:
                        break
            self.champion_trace = self._evaluate(self.champion)
            if self.config.use_secondary_tasks:
                self._secondary_task_results = self._evaluate_secondary_tasks(self.champion)
            else:
                self._secondary_task_results = None
            self.champion_fitness = self._compute_fitness(self.champion, self.champion_trace)
            verified, verify_score = self._verify_champion()
            self._champion_verified = verified
            self._champion_verify_score = verify_score
        return {
            "generations_completed": 0,
            "final_fitness": self.champion_fitness,
            "final_level": self.champion_level,
            "champion_words": len(self.champion) if self.champion else 0,
            "elapsed_seconds": 0.0,
            "generations_stalled": self.generations_stalled,
            "total_champion_improvements": 0,
        }

    def _run_generation_init(self, gen: int, n_levels: int) -> None:
        """Initialize generation state."""
        self.current_generation = gen
        self._champion_changed_since_last_qd = False
        self._champion_changed_since_last_novelty = False
        lvl = self.champion_level
        self._bonus_threshold_cache = (lvl, max(0.85, 0.95 - 0.10 * (lvl / max(1, n_levels - 1))))
        if self.config.use_mutation_scheduler:
            sched_progress = min(1.0, gen / max(1, self.config.mutation_schedule_generations))
            scheduled_base = (
                self.config.mutation_rate_start
                + (self.config.mutation_rate_end - self.config.mutation_rate_start) * sched_progress
            )
            self._original_mutation_rate = scheduled_base
            self.config.mutation_rate = self._compute_adaptive_mutation_rate() if gen > 5 else scheduled_base
        elif gen > 5:
            self.config.mutation_rate = self._compute_adaptive_mutation_rate()

    def _run_generation_evaluate(self) -> tuple:
        """Generate challengers, evaluate, select elite."""
        challengers = self._generate_challengers()
        self._current_diversity = self._compute_population_diversity(challengers)
        results = self._evaluate_batch(challengers)
        self._last_challengers = challengers
        self._last_results = results
        elite, elite_fitnesses, _ = self._select_elite(challengers, results)
        detailed_results = self._evaluate_batch_detailed(elite)
        return challengers, results, elite, elite_fitnesses, detailed_results

    def _run_generation_update(
        self,
        gen: int,
        elite,
        detailed_results,
        elite_fitnesses,
        improved_before: bool,
        checkpoint_interval: int,
        auto_curriculum: bool,
    ) -> bool:
        """Update champion, archives, and curriculum. Returns improved flag."""
        prev_fitness = self.champion_fitness
        improved = self._update_champion(elite, detailed_results, stage1_fitnesses=elite_fitnesses)
        if improved and hasattr(self, "champion_archive") and self.champion_archive is not None:
            try:
                _pruned = self.champion_archive.quality_weighted_prune()
                if _pruned > 0:
                    self._log(f"[Champion] Quality prune: removed {_pruned}")
            except Exception:
                pass
        delta = self.champion_fitness - prev_fitness
        self._update_fitness_velocity(delta)
        if improved and delta > 0:
            self._record_champion_replacement(delta)
        try:
            from mutation_proposer import DirectedMutator as _DM

            for _res in self._last_results:
                _desc = _res.get("description", "") if isinstance(_res, dict) else ""
                if _desc:
                    _DM.record_mutation_outcome(
                        _desc, _res.get("fitness_delta", 0.0) if isinstance(_res, dict) else 0.0
                    )
        except Exception:
            pass
        return improved

    def _run_generation_maintenance(
        self,
        gen: int,
        improved: bool,
        task_refresh_interval: int,
        cuda_cleanup_interval: int,
        _cuda_cleanup,
        checkpoint_interval: int,
        auto_curriculum: bool,
    ) -> None:
        """Run post-generation maintenance tasks."""
        if gen > 0 and gen % task_refresh_interval == 0:
            self._refresh_task_instance()
        if _cuda_cleanup is not None and gen > 0 and gen % cuda_cleanup_interval == 0:
            try:
                _cuda_cleanup()
            except Exception:
                pass
        if gen > 0 and checkpoint_interval and gen % checkpoint_interval == 0:
            try:
                self._checkpoint()
            except Exception as e:
                self._log(f"[Checkpoint] Failed at gen {gen}: {e}")
        if improved:
            self.generations_stalled = 0
            self._qd_champion_reset_tried = False
        else:
            self.generations_stalled += 1
        self._try_recurrent_feedback()
        if auto_curriculum:
            self._try_curriculum_advance()
        # Level circuit breaker: record outcome after curriculum check
        if self.config.level_circuit_breaker_enabled:
            self._record_cycle_level_outcome()
        self._run_generation_meta_monitoring(gen)
        self._record_strategy_outcomes()
        self._update_qd_archive()
        if self.generations_stalled >= 50 and self._try_qd_champion_reset():
            self._log(f"[QD Reset] Champion replaced from QD archive at gen {gen}")
        self._update_novelty_archive()

    def _run_generation_meta_monitoring(self, gen: int) -> None:
        """Run meta-controller monitoring and diagnostics."""
        if gen > 0 and gen % 25 == 0 and hasattr(self, "meta_controller") and self.meta_controller is not None:
            try:
                _rc = self.meta_controller.get_recovery_confidence()
                if _rc["is_recovering"]:
                    self._log(f"[Recovery] confidence={_rc['confidence']:.2f} exit={_rc['should_exit_recovery']}")
                    if _rc["should_exit_recovery"]:
                        self.meta_controller.recovery_mode = False
            except Exception:
                pass
        if self.config.use_strategy_archive and self.strategy_archive is not None:
            try:
                _dom = self.strategy_archive.get_dominance_report()
                if _dom.get("is_monoculture") and gen > 0 and gen % 10 == 0:
                    self._log(f"[Strategy] Monoculture: {_dom['dominant_strategy']} at {_dom['dominance_ratio']:.0%}")
            except Exception:
                pass

    def _run_one_generation(
        self,
        gen: int,
        checkpoint_interval: int,
        auto_curriculum: bool,
        task_refresh_interval: int,
        cuda_cleanup_interval: int,
        _cuda_cleanup,
        n_levels: int,
    ) -> None:
        """Execute a single generation of the evolution loop."""
        gen_start = time.time()
        self._run_generation_init(gen, n_levels)
        _, _, elite, elite_fitnesses, detailed_results = self._run_generation_evaluate()
        improved = self._run_generation_update(
            gen, elite, detailed_results, elite_fitnesses, False, checkpoint_interval, auto_curriculum
        )
        self._log_generation(gen, gen_start)
        self._run_generation_maintenance(
            gen,
            improved,
            task_refresh_interval,
            cuda_cleanup_interval,
            _cuda_cleanup,
            checkpoint_interval,
            auto_curriculum,
        )

    def run(self, num_generations: int | None = None) -> dict[str, Any]:
        """Run the evolution loop for num_generations (default: config)."""
        ng = num_generations if num_generations is not None else self.config.max_generations
        start_time = time.time()
        checkpoint_interval = self.config.checkpoint_interval
        auto_curriculum = self.config.auto_curriculum
        task_refresh_interval = 25
        cuda_cleanup_interval = 10
        _levels_ref = LEVELS
        _cuda_cleanup = None
        if self.use_gpu if hasattr(self, "use_gpu") else self.config.use_gpu:
            try:
                import gc as _gc

                import torch as _torch

                if _torch.cuda.is_available():

                    def _cuda_cleanup():
                        _gc.collect()
                        _torch.cuda.empty_cache()

            except ImportError:
                pass
        summary = self._initialize_run(ng)
        n_levels = len(_levels_ref)
        try:
            if hasattr(self, "_raw_task_score"):
                self._champion_raw_task_score = self._raw_task_score
            vm_farm = self.vm_farm
            for gen in range(ng):
                self._run_one_generation(
                    gen,
                    checkpoint_interval,
                    auto_curriculum,
                    task_refresh_interval,
                    cuda_cleanup_interval,
                    _cuda_cleanup,
                    n_levels,
                )
        except Exception:
            pass  # Return partial summary on exception
        finally:
            self._close_log_handle()
        # Final checkpoint after all generations complete
        if checkpoint_interval:
            try:
                self._checkpoint()
            except Exception as e:
                self._log(f"[Checkpoint] Failed at final: {e}")
        summary["generations_completed"] = ng
        summary["final_fitness"] = self.champion_fitness
        summary["final_level"] = self.champion_level
        summary["champion_words"] = len(self.champion) if self.champion else 0
        summary["elapsed_seconds"] = time.time() - start_time
        return summary

    def _refresh_task_instance(self) -> None:
        """Generate a new task instance for the current curriculum level."""
        level_cls = get_task_level(self.champion_level)
        seed = self.rng.randint(0, 2**32 - 1)
        self.task_instance = level_cls(seed=seed).generate_instance()

    @staticmethod
    def validate_program(program: list[int]) -> tuple[bool, str]:
        """Cheap pre-check: is this program worth evaluating?

        Returns (is_valid, reason). Invalid programs are skipped to avoid
        wasting evaluation time on guaranteed-fitness-0 champions.

        Checks:
          - Non-empty
          - Contains at least one non-EBREAK instruction
          - At least 3 instructions (minimum for any useful program)
          - No instructions beyond 0xFFFFFFFF (32-bit overflow)
        """
        if not program:
            return False, "empty_program"

        # Check for overflow/underflow first (before EBREAK stripping)
        if any(w > 0xFFFFFFFF or w < 0 for w in program):
            return False, "overflow"

        # Remove trailing EBREAKs for length check
        live = program
        while live and live[-1] == 0x0010_0073:  # EBREAK
            live = live[:-1]

        if len(live) == 0:
            return False, "all_ebreak"
        if len(live) < 3:
            return False, f"too_short_{len(live)}"

        return True, "ok"

    def _evaluate(self, program: list[int]) -> dict[str, Any]:
        """Run a single evaluation and return a trace dict."""
        if self.task_instance is None:
            self._refresh_task_instance()
        inputs = self.task_instance.get("inputs", {}) if self.task_instance else {}
        budget = self.config.eval_budget
        result = self.vm_farm.evaluate(program, inputs, budget=budget)
        return self._eval_result_to_trace(result)

    def _evaluate_batch(self, programs: list[list[int]]) -> list[dict[str, Any]]:
        """Evaluate multiple programs (stage 1, lighter). Uses GPU batch when available.

        R45: Validates programs before evaluation. Invalid programs get a
        zero-trace result (status="invalid") instead of wasting GPU time.
        """
        if self.task_instance is None:
            self._refresh_task_instance()
        inputs = self.task_instance.get("inputs", {}) if self.task_instance else {}
        budget = self.config.eval_budget // 2

        # R45: Pre-validate programs — skip invalid ones
        valid_indices = []
        invalid_results = {}
        for i, prog in enumerate(programs):
            is_valid, reason = self.validate_program(prog)
            if is_valid:
                valid_indices.append(i)
            else:
                invalid_results[i] = {
                    "status": "invalid",
                    "trap_cause": reason,
                    "instructions_count": len(prog),
                    "final_regs": {},
                    "final_memory": {},
                    "pc_history": [],
                    "reg_history": [],
                    "memory_accesses": [],
                }

        if not valid_indices:
            # All programs invalid — return all invalid results
            return [invalid_results.get(i, {"status": "invalid"}) for i in range(len(programs))]

        # Evaluate only valid programs
        valid_programs = [programs[i] for i in valid_indices]
        inputs_list = [inputs] * len(valid_programs)
        results = self.vm_farm.evaluate_batch(valid_programs, inputs_list, budget=budget)
        traces = [self._eval_result_to_trace(r) for r in results]

        # Merge valid and invalid results back in original order
        final = [None] * len(programs)
        for vi, trace in zip(valid_indices, traces, strict=False):
            final[vi] = trace
        for i, result in invalid_results.items():
            final[i] = result

        return final

    def _evaluate_batch_detailed(self, programs: list[list[int]]) -> list[dict[str, Any]]:
        """Evaluate programs with full budget (stage 2). Uses GPU batch when available."""
        if self.task_instance is None:
            self._refresh_task_instance()
        inputs = self.task_instance.get("inputs", {}) if self.task_instance else {}
        budget = self.config.eval_budget
        inputs_list = [inputs] * len(programs)
        results = self.vm_farm.evaluate_batch(programs, inputs_list, budget=budget)
        return [self._eval_result_to_trace(r) for r in results]

    @staticmethod
    def _eval_result_to_trace(result: Any) -> dict[str, Any]:
        """Convert an EvalResult to a trace dict.

        Maps from the real vm_farm EvalResult fields (success, registers,
        memory, trace, cycles, error, blocked_writes) to the trace dict
        format expected by TraceAnalyzer and _compute_fitness.

        Uses direct attribute access instead of getattr (3-5x faster on
        namedtuples), shared empty tuples for unused fields, and a
        pre-built dict template to avoid per-call dict literal allocation
        overhead (~50 evaluations per generation, thousands over a run).

        R35: Handles None input gracefully (returns a crash trace instead
        of raising AttributeError on None.success).
        """
        if result is None:
            return {
                "status": "crash",
                "final_pc": 0,
                "final_regs": {},
                "final_memory": {},
                "instructions_count": 0,
                "pc_history": _EMPTY_TUPLE,
                "reg_history": _EMPTY_TUPLE,
                "memory_accesses": _EMPTY_TUPLE,
                "trap_cause": "null_eval_result",
            }
        try:
            # Direct attribute access -- EvalResult is always a namedtuple
            success = result.success
            return {
                "status": "ok" if success else "crash",
                "final_pc": 0,
                "final_regs": result.registers,
                "final_memory": result.memory,
                "instructions_count": result.cycles,
                "pc_history": _EMPTY_TUPLE,
                "reg_history": _EMPTY_TUPLE,
                "memory_accesses": _EMPTY_TUPLE,
                "trap_cause": result.error or "",
            }
        except AttributeError:
            # Fallback for non-EvalResult objects (dict, etc.)
            success = getattr(result, "success", True)
            return {
                "status": "ok" if success else "crash",
                "final_pc": 0,
                "final_regs": getattr(result, "registers", {}),
                "final_memory": getattr(result, "memory", {}),
                "instructions_count": getattr(result, "cycles", 0),
                "pc_history": _EMPTY_TUPLE,
                "reg_history": _EMPTY_TUPLE,
                "memory_accesses": _EMPTY_TUPLE,
                "trap_cause": getattr(result, "error", None) or "",
            }

    # ------------------------------------------------------------------
    # Secondary task evaluation (multi-task fitness extension)
    # ------------------------------------------------------------------

    def _generate_secondary_tasks(self) -> list[dict]:
        """Generate secondary task instances for multi-task evaluation.

        Returns a list of task instance dicts from BitManipulationTask,
        ArithmeticTask, and MemoryAccessTask. Each dict has the standard
        format with 'inputs', 'expected_output', and 'level'.
        """
        from tasks_rv32i import SECONDARY_TASKS

        rng = self.rng
        tasks = []
        for task_cls in SECONDARY_TASKS:
            inst = task_cls(seed=rng.randint(0, 2**31)).generate_instance()
            tasks.append(inst)
        return tasks

    def _evaluate_secondary_tasks(self, program: list[int]) -> list[tuple[dict, dict]]:
        """Evaluate a program against all secondary tasks.

        Returns list of (task_instance, trace) tuples. Each trace is the
        standard trace dict format. Programs that crash on the primary task
        still get evaluated here -- the per-task penalty in _compute_fitness
        handles the crash paths.

        Requires self.vm_farm to be set. If not, returns empty list.
        """
        vm = getattr(self, "vm_farm", None)
        if vm is None:
            return []
        secondary = self._generate_secondary_tasks()
        results = []
        budget = self.config.eval_budget
        for task_inst in secondary:
            inputs = task_inst.get("inputs", {})
            try:
                raw = vm.evaluate(program, inputs, budget=budget)
                trace = self._eval_result_to_trace(raw)
            except Exception:
                trace = {
                    "status": "crash",
                    "final_regs": {},
                    "final_memory": {},
                    "instructions_count": 0,
                    "pc_history": (),
                    "reg_history": (),
                    "memory_accesses": (),
                    "trap_cause": "secondary_eval_error",
                }
            results.append((task_inst, trace))
        return results

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _get_structural_bonus_scale(self) -> dict[str, Any]:
        """Return cached structural bonus scaling data for the current generation.

        Delegates to progression_tracker.analyze_population_patterns() when
        available, wrapped in try/except for safety.  The result is cached
        per-generation to avoid redundant analysis during batch evaluations.

        Returns a dict with keys:
            scale_factor: float -- multiplier for structural bonuses
            loop_pressure: float -- relative loop bonus weight [0, 1]
            mem_pressure: float -- relative memory bonus weight [0, 1]
            div_pressure: float -- relative diversity bonus weight [0, 1]

        Falls back to neutral defaults when the tracker is unavailable
        or raises an exception.
        """
        gen = self.current_generation
        cached = self._structural_patterns_cache
        if cached is not None and self._structural_patterns_gen == gen:
            return cached

        try:
            if hasattr(self, "progression_tracker") and self.progression_tracker is not None:
                patterns = self.progression_tracker.analyze_population_patterns()
                result = {
                    "scale_factor": float(patterns.get("scale_factor", 1.0)),
                    "loop_pressure": float(patterns.get("loop_pressure", 0.33)),
                    "mem_pressure": float(patterns.get("mem_pressure", 0.33)),
                    "div_pressure": float(patterns.get("div_pressure", 0.34)),
                }
                # Clamp scale_factor to [0.5, 3.0] for safety
                sf = result["scale_factor"]
                result["scale_factor"] = max(0.5, min(3.0, sf))
            else:
                result = {
                    "scale_factor": 1.0,
                    "loop_pressure": 0.33,
                    "mem_pressure": 0.33,
                    "div_pressure": 0.34,
                }
        except Exception:
            result = {
                "scale_factor": 1.0,
                "loop_pressure": 0.33,
                "mem_pressure": 0.33,
                "div_pressure": 0.34,
            }

        self._structural_patterns_cache = result
        self._structural_patterns_gen = gen
        return result

    def _score_register_match(self, expected_regs: dict, expected_total: int, final_regs: dict) -> float:
        """Score per-register match with distance-weighted partial credit."""
        return _score_register_match_fn(expected_regs, expected_total, final_regs)

    def _compute_structural_bonuses(
        self, status: str, instr_count: int, program: list[int], bonus_scale: float
    ) -> tuple[float, float, float]:
        """Compute loop, memory, and instruction diversity bonuses."""
        return _compute_structural_bonuses_fn(self, status, instr_count, program, bonus_scale, _OPCODE_CAT_ARRAY)

    def _compute_efficiency_score(self, program: list[int], instr_count: int, status: str) -> float:
        """Compute efficiency scoring (+15 points max)."""
        eff_weight = getattr(self.config, "efficiency_score_weight", 1.0)
        return _compute_efficiency_score_fn(program, instr_count, status, self.champion_level, eff_weight)

    def _compute_multi_task_score(self) -> float:
        """Compute multi-task scoring (+15 points max)."""
        mt_weight = getattr(self.config, "secondary_task_weight", 1.0)
        secondary_results = getattr(self, "_secondary_task_results", None)
        return _compute_multi_task_score_fn(secondary_results, mt_weight)

    def _compute_time_credit(self, status: str, reg_match: float, instr_count: int) -> float:
        """Compute time-to-solution credit (+20 points max)."""
        return _compute_time_credit_fn(status, reg_match, instr_count)

    def _compute_smc_bonus(self, status: str, program: list[int], instr_count: int) -> float:
        """Compute SMC pattern bonus (+25 points max) for L12+."""
        return _compute_smc_bonus_fn(status, program, instr_count, self.champion_level)

    def _log_bonus_diagnostics(self, base_score: float, bonuses: dict, max_single: float, norm_active: bool) -> None:
        """Log per-dimension bonus breakdown periodically."""
        _log_bonus_diagnostics_fn(self, base_score, bonuses, max_single, norm_active)

    def _apply_novelty_qd_blending(self, fitness, program, trace, base_score, _qd_fn, _prog_qd_desc):
        """Apply novelty-weighted fitness and QD archive bonus/penalty."""
        return _apply_novelty_qd_blending_fn(self, fitness, program, trace, base_score, _qd_fn, _prog_qd_desc)

    def _compute_fitness(
        self, program: list[int], trace: dict[str, Any], eval_budget: int | None = None, raw_only: bool = False
    ) -> float:
        """Compute crash-penalized fitness."""
        return _compute_fitness_fn(self, program, trace, eval_budget, raw_only)

    def _extract_trace_info(self, trace: dict) -> tuple:
        """Extract and cache trace information for fitness computation."""
        return _extract_trace_info_fn(self, trace)

    def _compute_base_score(self, reg_match: float, instr_count: int, status: str, budget: int) -> float:
        """Compute base fitness score from register match and execution metrics."""
        return _compute_base_score_fn(reg_match, instr_count, status, budget)

    def _compute_all_bonuses(
        self, status: str, instr_count: int, program: list[int], reg_match: float, base_score: float
    ) -> dict:
        """Compute all bonus components."""
        return _compute_all_bonuses_fn(self, status, instr_count, program, reg_match, base_score, _OPCODE_CAT_ARRAY)

    def _apply_bonuses_and_normalize(self, base_score: float, bonuses: dict) -> float:
        """Apply bonuses with normalization and store raw score."""
        return _apply_bonuses_and_normalize_fn(self, base_score, bonuses)

    def _finalize_with_blending(self, base_score: float, program: list[int], trace: dict) -> float:
        """Apply novelty/QD blending and partial verify scaling."""
        return _finalize_with_blending_fn(self, base_score, program, trace)

    def _batch_classify_programs(self, traces: list, n: int, budget: int):
        """Classify programs by status (crash/timeout/ok) for batch fitness."""
        return _batch_classify_programs_fn(traces, n, budget)

    def _batch_score_registers(self, ok_traces: list, expected_keys: list, expected_vals, n_ok: int):
        """Vectorized register scoring for batch fitness."""
        return _batch_score_registers_fn(ok_traces, expected_keys, expected_vals, n_ok)

    def _batch_compute_efficiency(self, ok_programs: list, n_ok: int, ok_instr, max_prog_len: int):
        """Vectorized efficiency scoring for batch fitness."""
        return _batch_compute_efficiency_fn(self, ok_programs, n_ok, ok_instr, max_prog_len)

    def _batch_compute_structural_bonuses(
        self, ok_programs: list, ok_indices, traces: list, n_ok: int, ok_instr, _bonus_scale: float, _max_single
    ):
        """Vectorized structural bonuses (loop, mem, div, smc, graduated, level)."""
        return _batch_compute_structural_bonuses_fn(
            self, ok_programs, ok_indices, traces, n_ok, ok_instr, _bonus_scale, _max_single, _OPCODE_CAT_ARRAY
        )

    def _batch_blend_novelty_qd(self, fitnesses, programs, traces, ok_indices, n_ok, expected_regs, base, nw):
        """Apply novelty/QD blending to batch fitness values."""
        _batch_blend_novelty_qd_fn(self, fitnesses, programs, traces, ok_indices, n_ok, expected_regs, base, nw)

    def batch_compute_fitness(
        self,
        programs: list[list[int]],
        traces: list[dict[str, Any]],
        eval_budget: int | None = None,
        raw_only: bool = False,
    ) -> tuple[list[float], list[float]]:
        """Vectorized batch fitness computation using numpy arrays."""
        return batch_compute_fitness_fn(self, programs, traces, eval_budget, raw_only)

    def _batch_get_expected(self):
        """Extract expected output arrays for batch computation."""
        return _batch_get_expected_fn(self)

    def _batch_apply_crash_penalties(self, n, crash_mask, illegal_mask, timeout_mask):
        """Apply crash/timeout penalties to fitness array."""
        return _batch_apply_crash_penalties_fn(n, crash_mask, illegal_mask, timeout_mask)

    def _batch_compute_base(
        self, programs, traces, ok_indices, ok_instr, n_ok, expected_keys, expected_vals, budget, expected_regs
    ):
        """Compute base fitness scores for batch."""
        return _batch_compute_base_fn(
            self, programs, traces, ok_indices, ok_instr, n_ok, expected_keys, expected_vals, budget, expected_regs
        )

    def _batch_apply_bonuses(self, base, programs, traces, ok_indices, n_ok, expected_regs, ok_instr):
        """Apply efficiency, structural, and graduated bonuses."""
        return _batch_apply_bonuses_fn(
            self, base, programs, traces, ok_indices, n_ok, expected_regs, ok_instr, _OPCODE_CAT_ARRAY
        )

    def _generate_initial_champion(self) -> list[int]:
        """Generate or load the initial champion."""
        # Try loading from champion dir
        if self.config.champion_dir:
            level = self.champion_level
            base_path = os.path.join(self.config.champion_dir, f"champion_l{level}")
            try:
                return self.champion_anchor.load(base_path)
            except (IntegrityError, FileNotFoundError, OSError):
                pass

        # Try loading the reference solution
        try:
            ref = reference_solution(self.champion_level)
            # Encode to words (using module-level _encode_inst import)
            encoded = []
            for inst in ref:
                w = _encode_inst(inst)
                if w is not None:
                    encoded.append(w)
            if encoded:
                # Append EBREAK if not present
                if encoded[-1] != _EBREAK_INST:
                    encoded.append(0x0010_0073)
                return encoded
        except (ValueError, ImportError):
            pass

        # Fall back to seed program
        prog = generate_seed_program(rng=self.rng)
        # Ensure EBREAK at end
        if prog[-1] != 0x0010_0073:
            prog.append(0x0010_0073)
        return prog

    def _trim_program(self, program: list[int]) -> list[int]:
        """Trim dead code from a program (instructions after last EBREAK)."""
        return trim_dead_code(program)

    def _generate_diversity_injection(self, challengers, pop_size):
        """Inject diversity if population is converging."""
        if len(challengers) < pop_size:
            return
        opcodes = set()
        for prog in challengers:
            for word in prog:
                if isinstance(word, int):
                    opcodes.add(word & 0x7F)
        if len(opcodes) >= 6:
            return
        n_inject = max(1, int(pop_size * self.config.diversity_inject_ratio))
        for i in range(min(n_inject, len(challengers))):
            idx = self.rng.randint(0, len(challengers) - 1)
            challengers[idx] = generate_seed_program(rng=self.rng)

    def _apply_model_guided_slots(self, challengers):
        """Reserve slots for model-guided generation."""
        if self.champion_level < 2 or self.config.model_guided_ratio is None:
            return
        n_model = max(1, int(len(challengers) * self.config.model_guided_ratio))
        replaced = self._replace_with_model_guided(challengers, model_ratio=n_model / len(challengers))

    def _apply_mutation(self, strategy) -> list[int]:
        """Apply a mutation strategy to the champion and return the mutated program."""
        task_inst = self.task_instance or {}
        expected = task_inst.get("expected_output", {})
        inputs = task_inst.get("inputs", {})
        trace_input = [self.champion_trace] if self.champion_trace is not None else []
        sa = self.strategy_archive if self.config.use_strategy_archive else None
        mutated, _desc, _analysis = propose_mutation(
            program=self.champion,
            trace=trace_input,
            expected=expected,
            inputs=inputs,
            rng=self.rng,
            task_level=self.champion_level,
            strategy=strategy,
            strategy_archive=sa,
        )
        return mutated

    def _generate_challengers(self) -> list[list[int]]:
        """Generate a population of challenger programs."""
        pop_size = self.config.population_size
        if self.champion is None:
            return [generate_seed_program(rng=self.rng) for _ in range(pop_size)]
        challengers = []
        for _ in range(pop_size):
            strategy = self._choose_mutation_strategy(len(challengers))
            prog = self._apply_mutation(strategy)
            challengers.append(prog)
        self._generate_diversity_injection(challengers, pop_size)
        self._apply_model_guided_slots(challengers)
        # Ensure EBREAK at end of each program
        for i, prog in enumerate(challengers):
            if prog and prog[-1] != 0x0010_0073:
                challengers[i] = prog + [0x0010_0073]
        return challengers

    @staticmethod
    def _compute_population_diversity(challengers: list[list[int]]) -> float:
        """Compute opcode diversity of the challenger population."""
        return _compute_population_diversity_fn(challengers, _OPCODE_CAT_ARRAY, NUM_OPCODE_CATEGORIES)

    def _inject_diversity(self, challengers: list[list[int]], n_inject: int) -> None:
        """Replace the least-diverse challengers with fresh random programs."""
        _inject_diversity_fn(
            challengers,
            n_inject,
            self.rng,
            self._last_challenger_strategies,
            _OPCODE_CAT_ARRAY,
            NUM_OPCODE_CATEGORIES,
            generate_seed_program,
        )

    def _check_variance_override(self):
        """Check variance-aware strategy override (priority 1)."""
        return _check_variance_override_fn(self)

    def _check_priority_overrides(self, current_priority: int) -> tuple[int, MutationStrategy | None, str]:
        """Check cross-cycle, mutation stats, and plateau overrides (priorities 2-4)."""
        return _check_priority_overrides_fn(self, current_priority)

    def _apply_cycle_phase_bias(self, chosen: MutationStrategy, override_priority: int) -> MutationStrategy:
        """Apply cycle-phase exploration-exploitation balance."""
        return _apply_cycle_phase_bias_fn(self, chosen, override_priority)

    def _choose_mutation_strategy(self, idx: int) -> MutationStrategy:
        """Choose a mutation strategy for challenger idx."""
        effective_cap = (
            max(self.config.max_consecutive_same_strategy // 2, 10)
            if self.champion_level >= 5
            else self.config.max_consecutive_same_strategy
        )
        skip_strat = None
        if self._consecutive_same_strategy >= effective_cap and self._last_strategy_used not in (
            "GROW",
            "MODEL_GUIDED",
        ):
            skip_strat = self._last_strategy_used
            self._consecutive_same_strategy = 0
        if (
            self.config.use_strategy_archive
            and self.strategy_archive is not None
            and self.strategy_archive.total_records() > 10
        ):
            rec = self.strategy_archive.recommend_strategy(self.champion_level, skip_strategy=skip_strat)
            chosen = _STRATEGY_ENUM_CACHE.get(rec, MutationStrategy.HYBRID)
        else:
            chosen = MutationStrategy.HYBRID
        if self.config.use_strategy_archive and self.strategy_archive is not None:
            try:
                _bonus = self.strategy_archive.adaptive_exploration_bonus(rec, self.champion_level)
                if _bonus > 0.15 and self.rng.random() < _bonus:
                    _alt = [
                        s for s in MutationStrategy if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)
                    ]
                    if _alt:
                        chosen = self.rng.choice(_alt)
            except Exception:
                pass
        chosen = self._apply_affinity_bias(chosen)
        op, os, ol = self._check_variance_override()
        op, os, ol = self._check_priority_overrides(op)
        if os is not None:
            chosen = os
            self._log(ol)
        chosen = self._apply_cycle_phase_bias(chosen, op)
        if self.champion_level in self._unsolved_levels_set and self.generations_stalled > 2:
            if self.rng.random() < self.config.perturbation_capped_probability_unsolved:
                chosen = MutationStrategy.PERTURBATION_CAPPED
        chosen = self._apply_level_aware_strategy_bias(chosen)
        sv = chosen.value
        if sv == self._last_strategy_used:
            self._consecutive_same_strategy += 1
        else:
            self._last_strategy_used = sv
            self._consecutive_same_strategy = 1
        return chosen

    def _apply_affinity_bias(self, chosen):
        """Apply per-program mutation affinity bias."""
        return _apply_affinity_bias_fn(self, chosen)

    def _apply_level_aware_strategy_bias(self, chosen: MutationStrategy) -> MutationStrategy:
        """Apply level-aware strategy weighting to bias selection by task difficulty."""
        return _apply_level_aware_strategy_bias_fn(self, chosen)

    # --------------------------------------------------------------
    # Model-guided generation (v12.5)
    # --------------------------------------------------------------

    def _get_model_guided_ratio(self) -> float:
        """Determine model-guided replacement ratio based on level and stall."""
        return _get_model_guided_ratio_fn(self.config, self.champion_level)

    def _get_model_guided_prefix(
        self, champion_hex: list[str], ref_prefix_hex: list[str] | None
    ) -> tuple[list[str], str]:
        """Select prefix source using Adaptive Prefix Scheduler (APS)."""
        return _get_model_guided_prefix_fn(champion_hex, ref_prefix_hex, self.champion_level, self.generations_stalled)

    def _compute_model_temperature(self) -> float:
        """Compute model temperature with optional annealing."""
        return _compute_model_temperature_fn(self.config, getattr(self, "current_generation", 0))

    def _run_model_inference(self, prefix_source, model_temp, num_replace, prefix_counts):
        """Run model inference loop and return replaced indices."""
        ri, reps = [], 0
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        for pc in prefix_counts:
            if reps >= num_replace:
                break
            try:
                comp = None
                for att in range(1, 3):
                    try:
                        comp = self._prefix_generator.complete(
                            prefix_source,
                            prefix_count=pc,
                            num_samples=min(3, max(1, num_replace // len(prefix_counts))),
                            temperature=model_temp,
                            top_k=40,
                            top_p=0.9,
                            repetition_penalty=1.1,
                        )
                        break
                    except RuntimeError:
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.synchronize(), torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if att == 2:
                            comp = None
                if not comp:
                    continue
                for ph in comp:
                    if reps >= num_replace:
                        break
                    pi = [int(w, 16) for w in ph]
                    if len(pi) >= 2:
                        idx = self.rng.randint(0, len(self._last_challengers) - 1)
                        self._last_challengers[idx] = pi
                        ri.append(idx)
                        reps += 1
            except Exception:
                continue
        return ri

    def _replace_with_model_guided(self, challengers: list[list[int]], model_ratio: float = 0.2) -> list[int]:
        """Replace a fraction of challengers with model-guided completions."""
        replaced_indices = []
        if not self.champion:
            return []
        if self.config.model_guided_ratio is not None and self.config.model_guided_ratio <= 0:
            return []
        ref_prefix_hex = None
        try:
            ref = reference_solution(self.champion_level)
            if ref and len(ref) > 0:
                encoded = [w for inst in ref if (w := _encode_inst(inst)) is not None]
                ref_prefix = encoded[: min(4, len(encoded))]
                if len(ref_prefix) >= 1:
                    ref_prefix_hex = [f"0x{w:08x}" for w in ref_prefix]
        except Exception:
            pass
        champion_hex = [f"0x{w:08x}" for w in self.champion]
        prefix_source, source_name = self._get_model_guided_prefix(champion_hex, ref_prefix_hex)
        if self._prefix_generator is None:
            try:
                from prefix_generator import PrefixGenerator

                ckpt = self._seed_model_path or ""
                if not ckpt or not os.path.isfile(ckpt):
                    base = os.path.dirname(os.path.abspath(__file__))
                    ckpt = os.path.join(base, "milestone1", "checkpoints", "checkpoint_best.pt")
                if os.path.isfile(ckpt):
                    self._prefix_generator = PrefixGenerator(ckpt, temperature=0.9, max_new_tokens=256, device=None)
            except Exception as e:
                self._log(f"[ModelGuided] Init failed: {e}")
                return []
        if self._prefix_generator is None:
            return []
        num_replace = max(1, int(len(challengers) * model_ratio))
        model_temp = self._compute_model_temperature()
        stalled = self.generations_stalled
        champion_level = self.champion_level
        if champion_level >= 8:
            prefix_counts = [12] if stalled >= 50 else [8] if stalled >= 20 else [6]
        else:
            prefix_counts = [6] if stalled >= 50 else [4] if stalled >= 20 else [2]
        replaced_indices = self._run_model_inference(prefix_source, model_temp, num_replace, prefix_counts)
        if replaced_indices:
            self._log(
                f"[ModelGuided] Replaced {len(replaced_indices)}/{num_replace} (prefix: {source_name}, L={champion_level}, temp={model_temp:.1f})"
            )
        return replaced_indices

    def _try_qd_crossover(self) -> list[int] | None:
        """Attempt to breed a hybrid program from the QD archive.

        When stalled >= 15 gens and the QD archive has 2+ distinct solutions,
        perform a crossover between the top two fitness-ranked programs.
        Returns the hybrid program (with EBREAK) or None.

        This brings dormant archive diversity back into the active population.

        R33: Uses heapq.nlargest(2) for O(N log 2) top-2 selection instead
        of full O(N log N) sort. For a 100-cell QD grid this avoids sorting
        98 unnecessary elements every call.
        """
        if self.qd_archive is None:
            return None
        if self.generations_stalled < 15:
            return None

        try:
            solutions = self.qd_archive.grid_solutions()
            if len(solutions) < 2:
                return None

            # R39: Use heapq.nlargest(3) to find top-3 by fitness.
            # The old code used solutions[2] as fallback, but grid_solutions()
            # returns cells in grid order, not fitness order — index 2 is
            # NOT guaranteed to be the 3rd-best. This caused the crossover
            # to sometimes use a low-fitness fallback, producing hybrids
            # that were worse than expected.
            top3 = heapq.nlargest(3, solutions, key=lambda s: s["fitness"])
            best = top3[0]
            second = top3[1] if len(top3) > 1 else None

            if best["program"] == (second["program"] if second else []):
                second = top3[2] if len(top3) > 2 else None
                if second is None or best["program"] == second["program"]:
                    return None

            hybrid = _crossover_mutation(
                program=list(best["program"]),
                task_level=self.champion_level,
                rng=self.rng,
                champion=list(second["program"]),
                champion_crossover_prob=self.config.champion_crossover_prob,
            )
            if not hybrid:
                return None
            if hybrid[-1] != 0x0010_0073:
                hybrid.append(0x0010_0073)
            return hybrid
        except Exception as e:
            self._log(f"[QD Cross] Crossover error: {e}")
            return None

    # ------------------------------------------------------------------
    # MAP-Elites QD archive (v13)
    # ------------------------------------------------------------------

    def _update_qd_archive(self) -> None:
        """Extract champion descriptor and add to MAP-Elites QD archive.

        Called every generation after the champion has been updated.
        Silently skips if the QD archive is not initialised.

        R21: Skips entirely when champion hasn't changed since the last
        update, avoiding redundant QD axes recomputation every generation.
        """
        qd = self.qd_archive
        if qd is None or not _HAS_QD:
            return
        # R21: Skip QD update when champion unchanged (avoids recomputing
        # QD axes for the same program every generation)
        if not self._champion_changed_since_last_qd:
            return
        champion = self.champion
        trace = self.champion_trace
        if champion is None or trace is None:
            return
        if self.config.qd_bonus_weight <= 0.0 and not self.config.use_qd_archive:
            return

        try:
            qd_desc = compute_qd_axes(champion, trace)
            # R31: Cache the computed descriptor for _compute_fitness niche penalty
            self._cached_champion_qd_desc = qd_desc
            self._champion_qd_valid = True
            if qd_desc is None:
                return
            was_added = qd.add_solution(
                descriptor=qd_desc,
                fitness=self.champion_fitness,
                program=champion,
                trace=trace,
            )

            # R30: Update niche coverage tracking for diagnostics
            self._qd_coverage = qd.coverage()

            # Log descriptor distribution for diagnostic (Phase K)
            gen = self.current_generation
            if gen % 25 == 0 or was_added:
                self._log(
                    f"[QD] Descriptor range gen={gen}: "
                    f"inst_div=[{qd._descriptor_min[0]:+.2f}, {qd._descriptor_max[0]:+.2f}], "
                    f"reg_press=[{qd._descriptor_min[1]:+.2f}, {qd._descriptor_max[1]:+.2f}], "
                    f"bounds=[{qd._per_dim_lower[0]:+.2f}..{qd._per_dim_upper[0]:+.2f}, "
                    f"{qd._per_dim_lower[1]:+.2f}..{qd._per_dim_upper[1]:+.2f}], "
                    f"coverage={qd.coverage():.0%}, "
                    f"rebins={qd._rebins}"
                )

            if was_added:
                self._log(
                    f"[QD] Added champion to grid cell "
                    f"(inst_div={qd_desc[0]:+.2f}, "
                    f"reg_press={qd_desc[1]:+.2f}, "
                    f"coverage={qd.coverage():.0%})"
                )
        except (TypeError, ValueError, RuntimeError) as exc:
            self._log(f"[QD] Skipped: {exc}")

    def _try_qd_champion_reset(self) -> bool:
        """Replace champion with best QD archive program when deeply stalled.

        When ``generations_stalled >= 50`` and the QD archive has at least
        one solution, finds the highest-fitness program in the archive and
        replaces the champion if the archive program (with a 0.9x task-
        specific discount) has higher discounted fitness.

        Only attempts once per stall episode (``_qd_champion_reset_tried``
        flag). Returns True if champion was replaced.

        This is a fallback for when the QD crossover and other diversity
        mechanisms have not been enough to escape deep stall.
        """
        if self.generations_stalled < 50:
            return False
        if self.qd_archive is None or not _HAS_QD:
            return False
        if self._qd_champion_reset_tried:
            return False

        self._qd_champion_reset_tried = True

        try:
            solutions = self.qd_archive.grid_solutions()
            if not solutions:
                self._log("[QD Reset] QD archive empty -- no champion to inject")
                return False

            best_qd = max(solutions, key=lambda s: s["fitness"])
            qd_prog = list(best_qd["program"])
            qd_fitness = best_qd["fitness"]

            self._log(
                f"[QD Reset] Found archive candidate: "
                f"fitness={qd_fitness:.4f}, "
                f"{len(qd_prog)} words (stalled={self.generations_stalled})"
            )

            # Apply task-specific discount (archive fitness may not match current task)
            qd_task_fitness = qd_fitness * 0.9

            if qd_task_fitness > self.champion_fitness:
                old_fitness = self.champion_fitness
                self.champion = qd_prog
                self.champion_fitness = qd_task_fitness
                # R33: Re-evaluate the new champion to get a fresh trace.
                # Without this, champion_trace still points to the OLD
                # champion's trace, causing _update_qd_archive and
                # _update_novelty_archive to compute descriptors from a
                # mismatched champion/trace pair, corrupting archive data.
                self.champion_trace = self._evaluate(qd_prog)
                self.generations_stalled = 0
                # R21: Mark champion changed so next QD update runs
                self._champion_changed_since_last_qd = True
                self._champion_changed_since_last_novelty = True
                self._log(
                    f"[QD Reset] Champion replaced: "
                    f"{old_fitness:.4f} -> {qd_task_fitness:.4f} "
                    f"(from QD archive, stall reset)"
                )
                # R31: Invalidate champion QD descriptor cache (new champion)
                self._champion_qd_valid = False
                self._cached_champion_qd_desc = None
                return True

            self._log(
                f"[QD Reset] Candidate not better: "
                f"current={self.champion_fitness:.4f}, "
                f"qd={qd_task_fitness:.4f} -- keeping champion"
            )
            return False
        except Exception as e:
            self._log(f"[QD Reset] Error: {e}")
            return False

    # ------------------------------------------------------------------
    # Novelty archive (v13)
    # ------------------------------------------------------------------

    def _update_novelty_archive(self) -> None:
        """Extract champion descriptor and add to the novelty archive.

        Called every generation after the champion has been updated.
        Silently skips if the novelty archive is not initialised.

        R21: Skips when champion unchanged since last update (avoids
        redundant behavioral descriptor extraction).
        R39: Uses dedicated _champion_changed_since_last_novelty flag
        instead of the QD flag, so novelty and QD archives update
        independently.
        """
        novelty_archive = self.novelty_archive
        if novelty_archive is None:
            return
        # R21: Skip novelty update when champion unchanged this generation
        if not self._champion_changed_since_last_novelty:
            return
        champion = self.champion
        trace = self.champion_trace
        if champion is None or trace is None:
            return

        _extract_fn = self._extract_behavioral_descriptor
        if _extract_fn is None:
            return

        try:
            descriptor = _extract_fn(
                trace,
                champion,
                budget=self.config.eval_budget,
            )
            metadata: dict[str, Any] = {
                "generation": self.current_generation,
                "level": self.champion_level,
                "fitness": self.champion_fitness,
                "champion_size": len(champion),
                "instructions_count": trace.get("instructions_count", 0),
                "status": trace.get("status", ""),
            }
            was_added, score = novelty_archive.add_to_archive(
                descriptor,
                metadata=metadata,
                check_threshold=novelty_archive.size > 0,
            )
            if was_added:
                self._log(f"[Novelty] Added descriptor (score={score:.4f}, archive_size={novelty_archive.size})")
            else:
                self._log(f"[Novelty] Below threshold (score={score:.4f})")
        except (TypeError, ValueError, RuntimeError) as exc:
            self._log(f"[Novelty] Skipped: {exc}")

        # Manage novelty archive capacity
        try:
            removed = self.novelty_archive.manage_capacity()
            if removed > 0:
                self._log(f"[Novelty] Capacity management: removed {removed} low-novelty entries")
        except Exception:
            pass

    def _compute_challenger_fitnesses(self, challengers, results, stage1_budget, raw_only):
        """Compute fitness for all challengers. Returns (scored, fitness_cache)."""
        n_challengers = len(challengers)
        scored = [None] * n_challengers
        fitness_cache = [0.0] * n_challengers
        _compute = self._compute_fitness
        _use_sec = self.config.use_secondary_tasks
        if _use_sec:
            sec_per_prog = {}
            for idx in range(n_challengers):
                trace = results[idx]
                status = trace.get("status", "")
                if status not in _CRASH_STATUSES and status != "timeout":
                    sec_per_prog[idx] = self._evaluate_secondary_tasks(challengers[idx])
            self._secondary_task_results_per_program = sec_per_prog
        else:
            self._secondary_task_results_per_program = None
        try:
            batch_fitnesses, _ = self.batch_compute_fitness(
                challengers, results, eval_budget=stage1_budget, raw_only=raw_only
            )
            for idx in range(n_challengers):
                scored[idx] = (batch_fitnesses[idx], idx, challengers[idx], results[idx])
                fitness_cache[idx] = batch_fitnesses[idx]
        except Exception:
            _status_get = dict.get
            for idx in range(n_challengers):
                trace = results[idx]
                status = _status_get(trace, "status", "")
                if status in _CRASH_STATUSES:
                    fit = -100.0
                elif status == "timeout" or _status_get(trace, "instructions_count", 0) >= stage1_budget:
                    fit = -25.0
                else:
                    if _use_sec:
                        self._secondary_task_results = self._evaluate_secondary_tasks(challengers[idx])
                    else:
                        self._secondary_task_results = None
                    fit = _compute(challengers[idx], trace, eval_budget=stage1_budget, raw_only=raw_only)
                scored[idx] = (fit, idx, challengers[idx], results[idx])
                fitness_cache[idx] = fit
        finally:
            self._secondary_task_results_per_program = None
        return scored, fitness_cache

    def _tournament_select(self, scored, elite_size, n_scored):
        """Fill remaining elite slots via adaptive tournament selection."""

        rng = self.rng
        base_tourney = self.config.tournament_size
        div = self._current_diversity
        if div >= 0.6:
            adaptive_tourney = base_tourney + 2
        elif div >= 0.4:
            adaptive_tourney = base_tourney + 1
        elif div >= 0.2:
            adaptive_tourney = base_tourney
        else:
            adaptive_tourney = max(2, base_tourney - 1)
        tourney_size = min(adaptive_tourney, n_scored)
        _scored_get = scored.__getitem__
        for _ in range(elite_size):
            best_i = rng.randint(0, n_scored - 1)
            best_f = _scored_get(best_i)[0]
            for _t in range(tourney_size - 1):
                ci = rng.randint(0, n_scored - 1)
                cf = _scored_get(ci)[0]
                if cf > best_f:
                    best_f = cf
                    best_i = ci
            yield _scored_get(best_i), best_i

    def _select_elite(self, challengers, results):
        """Select elite challengers for detailed evaluation."""
        import heapq

        stage1_budget = self.config.eval_budget // 2
        n_challengers = len(challengers)
        raw_only = self.champion_level >= self.config.max_level - 1
        scored, fitness_cache = self._compute_challenger_fitnesses(challengers, results, stage1_budget, raw_only)
        self._last_challenger_fitnesses = fitness_cache
        elite_size = max(1, int(self.config.population_size * self.config.elite_ratio))
        top_k = heapq.nsmallest(elite_size, scored, key=lambda x: -x[0])
        elites = [prog for _, _, prog, _ in top_k]
        elite_fits = [fit for fit, _, _, _ in top_k]
        elite_indices = [idx for _, idx, _, _ in top_k]
        seen_fps = {(len(p) << 32) ^ (p[0] if p else 0) ^ ((p[-1] << 16) if len(p) > 1 else 0) for p in elites}
        seen_indices = set(elite_indices)
        n_scored = len(scored)
        for (_, _, win_prog, _), best_i in self._tournament_select(scored, elite_size, n_scored):
            fp = (
                (len(win_prog) << 32)
                ^ (win_prog[0] if win_prog else 0)
                ^ ((win_prog[-1] << 16) if len(win_prog) > 1 else 0)
            )
            if fp not in seen_fps and best_i not in seen_indices:
                seen_fps.add(fp)
                seen_indices.add(best_i)
                elites.append(win_prog)
                elite_fits.append(scored[best_i][0])
                elite_indices.append(best_i)
        return elites, elite_fits, elite_indices

    def _find_best_challenger(self, elite, detailed_results, stage1_fitnesses):
        """Find the best challenger that beats the champion. Returns (best_fitness, best_program, best_trace, best_raw_score, improved)."""
        best_fitness = self.champion_fitness
        best_program = None
        best_trace = None
        best_raw_score = getattr(self, "_raw_task_score", 0.0)
        improved = False
        _compute = self._compute_fitness
        _use_sec = self.config.use_secondary_tasks
        for idx, (prog, trace) in enumerate(zip(elite, detailed_results, strict=False)):
            if stage1_fitnesses and idx < len(stage1_fitnesses) and stage1_fitnesses[idx] < best_fitness:
                continue
            if _use_sec:
                self._secondary_task_results = self._evaluate_secondary_tasks(prog)
            else:
                self._secondary_task_results = None
            fitness = _compute(prog, trace)
            if fitness > best_fitness and self._should_replace_champion(fitness):
                best_fitness = fitness
                best_program = prog
                best_trace = trace
                best_raw_score = getattr(self, "_raw_task_score", 0.0)
                improved = True
        return best_fitness, best_program, best_trace, best_raw_score, improved

    def _verify_and_replace_champion(self, best_program, best_trace, best_raw_score, best_fitness):
        """Verify challenger and replace champion if verified. Returns True if replaced."""
        _old_champion_trace = self.champion_trace
        self.champion_trace = best_trace
        verified, verify_score = self._verify_champion()
        self.champion_trace = _old_champion_trace
        if not verified:
            if verify_score > 0.0:
                self._log(f"[Champion] Partial verify: challenger passes {verify_score:.2f}. Allowing replacement.")
                self._champion_partial_verify = verify_score
            else:
                self._log(f"[Champion] BLOCKED (PM Rule #19): challenger verify failed ({verify_score:.2f} < 1.0).")
                self._champion_verified = False
                self._champion_verify_score = verify_score
                self._champion_partial_verify = 0.0
                return False
        qd = getattr(self, "qd_archive", None)
        if (
            qd is not None
            and _HAS_QD
            and self.champion is not None
            and self.champion_trace is not None
            and not self._champion_changed_since_last_qd
        ):
            try:
                old_desc = compute_qd_axes(self.champion, self.champion_trace)
                if old_desc is not None:
                    qd.add_solution(
                        descriptor=old_desc,
                        fitness=self.champion_fitness,
                        program=self.champion,
                        trace=self.champion_trace,
                    )
            except (TypeError, ValueError, RuntimeError):
                pass
        self.champion = trim_dead_code(list(best_program))
        self.champion_trace = best_trace
        if hasattr(self, "_champion_raw_task_score"):
            self._champion_raw_task_score = best_raw_score
        self.champion_fitness = best_fitness
        self._champion_verified = verified
        self._champion_verify_score = verify_score
        self._champion_changed_since_last_qd = True
        self._champion_changed_since_last_novelty = True
        self._champion_qd_valid = False
        self._cached_champion_qd_desc = None
        return True

    def _update_hall_of_fame(self, elite, stage1_fitnesses):
        """Update Hall of Fame with elite programs."""
        if not stage1_fitnesses or not elite:
            return
        n_f = len(stage1_fitnesses)
        hof_worst = self._hall_of_fame.worst_fitness
        if not isinstance(hof_worst, (int, float)):
            hof_worst = float("-inf")
        if max(stage1_fitnesses) < hof_worst:
            return
        k_top = max(1, n_f // 5)
        threshold_heap = []
        for fi in stage1_fitnesses:
            if len(threshold_heap) < k_top:
                heapq.heappush(threshold_heap, fi)
            elif fi > threshold_heap[0]:
                heapq.heapreplace(threshold_heap, fi)
        quality_threshold = threshold_heap[0] if threshold_heap else 0.0
        level = self.champion_level
        hof = self._hall_of_fame
        gen = getattr(self, "current_generation", 0)
        for idx, prog in enumerate(elite):
            if idx < n_f and stage1_fitnesses[idx] >= quality_threshold:
                hof.add(prog, stage1_fitnesses[idx], level, generation=gen)

    def _update_champion(self, elite, detailed_results, stage1_fitnesses=None):
        """Update champion if any elite challenger has higher fitness."""
        best_fitness, best_program, best_trace, best_raw_score, improved = self._find_best_challenger(
            elite, detailed_results, stage1_fitnesses
        )
        if improved and best_program is not None:
            self._verify_and_replace_champion(best_program, best_trace, best_raw_score, best_fitness)
            improved = self.champion_fitness == best_fitness
        self._update_hall_of_fame(elite, stage1_fitnesses)
        self.best_fitness_history.append(self.champion_fitness)
        max_hist = getattr(self, "_max_history_len", 500)
        if len(self.best_fitness_history) > max_hist:
            self.best_fitness_history = self.best_fitness_history[-max_hist:]
        return improved

    def _analyze_trace(self) -> TraceAnalysis | None:
        """Analyze the current champion trace against expected output."""
        if self.champion_trace is None or self.task_instance is None:
            return None
        expected = self.task_instance.get("expected_output", {})
        inputs = self.task_instance.get("inputs", {}) if self.task_instance else {}
        return TraceAnalyzer.analyze([self.champion_trace], expected, inputs, budget=self.config.eval_budget)

    # ------------------------------------------------------------------
    # Curriculum
    # ------------------------------------------------------------------

    def _try_curriculum_advance(self) -> bool:
        """Check curriculum advancement conditions.

        R50: Curriculum may only advance when the champion PASSES verify()
        for the current level. This prevents inflated fitness (from level_bonus)
        from advancing the curriculum when the champion doesn't actually solve
        the task.

        Returns True if the level was advanced.
        """
        if not self.config.auto_curriculum:
            return False
        if self.champion_level >= len(LEVELS) - 1:
            return False
        # Bisection: EE_MAX_LEVEL caps curriculum advancement
        _max_level = int(os.environ.get("EE_MAX_LEVEL", "999"))
        if self.champion_level >= _max_level:
            return False

        # Check if stalled long enough
        if self.generations_stalled >= self.config.curriculum_stall_threshold:
            # R50: Verify the champion actually passes the current level
            verified = False
            try:
                level_cls = get_task_level(self.champion_level)
                output = {
                    "registers": self.champion_trace.get("final_regs", {}),
                    "memory": self.champion_trace.get("final_memory", {}),
                }
                verify_score = level_cls(seed=self.champion_level).verify(output, self.task_instance)
                verified = verify_score >= 1.0
                self._champion_verified = verified
                self._champion_verify_score = verify_score
            except Exception:
                verified = False
                self._champion_verified = False
                self._champion_verify_score = 0.0

            if not verified:
                self._log(
                    f"[Curriculum] BLOCKED (R50): stalled={self.generations_stalled} "
                    f">= {self.config.curriculum_stall_threshold}, "
                    f"raw_score={self._champion_raw_task_score:.1f}, "
                    f"verify={self._champion_verify_score:.2f} < 1.0"
                )
                return False

            # R33: Use champion-specific raw task score instead of the instance
            # variable which reflects the last evaluated elite (not necessarily
            # the champion). This fixes curriculum advancing/not-advancing
            # based on the wrong program's score.
            # Threshold lowered from 140 to 120 to work with graduated bonus curve
            if self._champion_raw_task_score >= 120.0:
                self._log(
                    f"[Curriculum] Advancing: stalled={self.generations_stalled}, "
                    f"raw_score={self._champion_raw_task_score:.1f}, "
                    f"verify={self._champion_verify_score:.2f}, "
                    f"level={self.champion_level}->{self.champion_level + 1}"
                )
                self._advance_curriculum()
                return True
            else:
                self._log(
                    f"[Curriculum] BLOCKED: stalled={self.generations_stalled} "
                    f">= {self.config.curriculum_stall_threshold}, but "
                    f"raw_score={self._champion_raw_task_score:.1f} < 120.0"
                )

        return False

    def _verify_champion(self) -> tuple[bool, float]:
        """R50: Run verify() on the current champion against the current level.

        Returns (verified: bool, verify_score: float).
        """
        if self.champion is None or self.champion_trace is None:
            return False, 0.0
        try:
            level_cls = get_task_level(self.champion_level)
            output = {
                "registers": self.champion_trace.get("final_regs", {}),
                "memory": self.champion_trace.get("final_memory", {}),
            }
            score = level_cls(seed=self.champion_level).verify(output, self.task_instance)
            # Partial verify scoring: store the float score even when < 1.0
            # to create a fitness gradient for the L4 plateau escape.
            self._champion_partial_verify = score
            return score >= 1.0, score
        except Exception:
            return False, 0.0

    def _advance_curriculum(self) -> None:
        """Advance to the next curriculum level.

        Skips levels that have been disabled by the circuit breaker.
        """
        old_level = self.champion_level
        # Skip disabled levels
        new_level = old_level + 1
        max_level = min(len(LEVELS) - 1, self.config.max_level)
        while new_level <= max_level and self._is_level_disabled(new_level):
            self._log(f"[CircuitBreaker] Skipping disabled level {new_level}")
            new_level += 1
        if new_level > max_level:
            self._log(f"[CircuitBreaker] No more enabled levels after L{old_level}")
            return
        self.champion_level = new_level
        self.generations_stalled = 0
        self._qd_champion_reset_tried = False

        # D3: Reset consecutive strategy counter on level advance
        self._consecutive_same_strategy = 0
        self._last_strategy_used = ""

        # R21: Reset QD and novelty archive tracking on level advance (new champion)
        self._champion_changed_since_last_qd = True
        self._champion_changed_since_last_novelty = True
        # R31: Invalidate champion QD descriptor cache (new champion)
        self._champion_qd_valid = False
        self._cached_champion_qd_desc = None
        # R41: Invalidate bonus threshold cache on level change.
        # The threshold depends on champion_level; without invalidation,
        # the first generation at the new level uses the stale old-level
        # threshold in _compute_fitness, producing incorrect fitness scores.
        self._bonus_threshold_cache = None

        # D3: Decay archive weights for the old level so lower-level
        # strategy biases don't pollute the new level.  ~50% of success
        # records at the old level are flipped to False, giving the archive
        # a soft reset while preserving recency information for UCB.
        if self.strategy_archive is not None:
            n_records = self.strategy_archive.total_records()
            decayed = self.strategy_archive.decay_level_records(old_level, factor=0.5)
            self._log(
                f"[D3] Archive decay: {decayed} success records decayed at L{old_level} "
                f"(total records: {n_records})"
            )

        # Save champion for the old level
        self._save_champion(old_level)

        # Fresh task for the new level
        self._refresh_task_instance()

        # Generate new champion for the new level (or load reference)
        if self.champion is not None:
            self.champion = self._generate_initial_champion()
            self.champion_trace = self._evaluate(self.champion)
            if self.config.use_secondary_tasks:
                self._secondary_task_results = self._evaluate_secondary_tasks(self.champion)
            else:
                self._secondary_task_results = None
            self.champion_fitness = self._compute_fitness(self.champion, self.champion_trace)

        # Log
        self._log(f"Curriculum advanced: L{old_level} -> L{self.champion_level}")

    # ------------------------------------------------------------------
    # Level circuit breaker: auto-disable impossible levels
    # ------------------------------------------------------------------

    def _record_level_circuit_outcome(self, level: int, had_success: bool) -> None:
        """Record a cycle outcome for the given level."""
        if not self.config.level_circuit_breaker_enabled:
            return
        _record_level_circuit_outcome_fn(self._level_circuit_stats, level, had_success)

    def _check_level_circuit_breaker(self, level: int) -> bool:
        """Check if a level should be disabled by the circuit breaker."""
        if not self.config.level_circuit_breaker_enabled:
            return False
        return _check_level_circuit_breaker_fn(
            self._level_circuit_stats,
            self._disabled_levels,
            level,
            self.config.level_circuit_breaker_threshold,
            self.config.level_circuit_breaker_enabled,
            self._log,
        )

    def _is_level_disabled(self, level: int) -> bool:
        """Check if a level has been auto-disabled by the circuit breaker."""
        return _is_level_disabled_fn(self._disabled_levels, level)

    def _record_cycle_level_outcome(self) -> None:
        """Record the current cycle's outcome for the circuit breaker."""
        _record_cycle_level_outcome_fn(self)

    def get_disabled_levels(self) -> frozenset[int]:
        """Return the set of levels disabled by the circuit breaker."""
        return frozenset(self._disabled_levels)

    def get_level_circuit_stats(self) -> dict[int, dict]:
        """Return per-level circuit breaker statistics."""
        return {k: dict(v) for k, v in self._level_circuit_stats.items()}

    # ------------------------------------------------------------------
    # Recurrent feedback
    # ------------------------------------------------------------------

    def _collect_recurrent_candidates(self, fitnesses, n_fitnesses):
        """Phase 1: Collect eligible candidates for recurrent feedback."""
        _candidates = []
        rejected_count = 0
        _levels = LEVELS
        _verify_levels_sorted = None
        for idx in range(len(self._last_challengers)):
            prog = self._last_challengers[idx]
            if idx >= len(self._last_results):
                continue
            trace = self._last_results[idx]
            if trace.get("status", "") != "ok":
                continue
            sha256 = sha256_program(prog)
            if sha256 in self._saved_sha256s:
                continue
            challenger_fitness = fitnesses[idx] if idx < n_fitnesses else 0.0
            if challenger_fitness <= MIN_FITNESS_FOR_TRACE and self.generations_stalled < 30:
                rejected_count += 1
                continue
            if len(prog) < 3:
                continue
            _trace_instr_count = trace.get("instructions_count", 0)
            if _trace_instr_count <= 0:
                rejected_count += 1
                continue
            if _trace_instr_count > 0 and hasattr(self, "_saved_size_counts"):
                _size_bucket = (_trace_instr_count // 5) * 5
                _total_saved = sum(self._saved_size_counts.values()) if self._saved_size_counts else 0
                if _total_saved > 20:
                    _bucket_count = self._saved_size_counts.get(_size_bucket, 0)
                    if _bucket_count > _total_saved * 0.7:
                        rejected_count += 1
                        continue
                self._saved_size_counts[_size_bucket] = self._saved_size_counts.get(_size_bucket, 0) + 1
            _candidates.append((prog, challenger_fitness, trace, sha256))
        return _candidates, rejected_count, _levels, _verify_levels_sorted

    def _compute_level_weights(self, candidates, levels, verify_levels_sorted):
        """Phase 3.5 + 4: Compute level-diversity and fitness weights."""
        _level_data = []
        if candidates:
            if verify_levels_sorted is None:
                verify_levels_sorted = sorted(levels.keys())
            for _prog, _fit, _trace, _sha in candidates:
                _task_score = 0.0
                _solved_level = self.champion_level
                _output = {"registers": _trace.get("final_regs", {}), "memory": _trace.get("final_memory", {})}
                try:
                    for lvl in verify_levels_sorted:
                        level_cls = levels[lvl]
                        try:
                            score = level_cls(seed=lvl).verify(_output, self.task_instance)
                            if score > _task_score:
                                _task_score = score
                                _solved_level = lvl
                        except Exception:
                            continue
                except Exception:
                    pass
                _level_data.append((_solved_level, _task_score))
        _weights = None
        _n_levels_found = 1
        if candidates:
            _fits = [c[1] for c in candidates]
            _max_f, _min_f = max(_fits), min(_fits)
            _fitness_range = _max_f - _min_f
            _temperature = max(0.5, _fitness_range * 0.2) if _fitness_range > 0 else 1.0
            if _fitness_range > 1e-6:
                _logits = [((f - _min_f) / _fitness_range) * (1.0 / _temperature) for f in _fits]
                _max_logit = max(_logits)
                _exp_logits = [math.exp(l - _max_logit) for l in _logits]
                _sum_exp = sum(_exp_logits)
                _fitness_weights = (
                    [e / _sum_exp for e in _exp_logits] if _sum_exp > 0 else [1.0 / len(_logits)] * len(_logits)
                )
            else:
                _fitness_weights = [1.0 / len(_fits)] * len(_fits)
            _level_counter = {}
            for _sd, _ in _level_data:
                _level_counter[_sd] = _level_counter.get(_sd, 0) + 1
            _n_levels_found = len(_level_counter)
            _avg_per_level = len(candidates) / max(1, _n_levels_found)
            _level_div_mult = {lvl: _avg_per_level / max(1, cnt) for lvl, cnt in _level_counter.items()}
            _weights = [_fitness_weights[i] * _level_div_mult[_level_data[i][0]] for i in range(len(candidates))]
            _w_sum = sum(_weights)
            if _w_sum > 0:
                _weights = [w / _w_sum for w in _weights]
        return _level_data, _weights, _n_levels_found

    def _apply_novelty_weights(self, candidates, weights, jsonl_path):
        """Phase 6: Apply novelty weighting to training examples."""
        if not candidates:
            return weights, False
        if not hasattr(self, "_novelty_anchors"):
            self._novelty_anchors, self._novelty_anchors_file_size = [], 0
        try:
            cfs = os.path.getsize(jsonl_path) if os.path.isfile(jsonl_path) else 0
        except OSError:
            cfs = 0
        if cfs > self._novelty_anchors_file_size or not self._novelty_anchors:
            na = []
            try:
                if os.path.isfile(jsonl_path):
                    with open(jsonl_path) as f:
                        for line in f:
                            try:
                                e = json.loads(line)
                                ph = e.get("champion_program", [])
                                if ph:
                                    pw = [int(h, 16) for h in ph]
                                    if len(pw) >= 3:
                                        na.append(pw)
                                    if len(na) >= 200:
                                        break
                            except (json.JSONDecodeError, ValueError):
                                continue
            except Exception:
                pass
            self._novelty_anchors, self._novelty_anchors_file_size = na, cfs
        if not self._novelty_anchors:
            return weights, False
        ns = []
        for prog, *_ in candidates:
            md = float("inf")
            for a in self._novelty_anchors:
                xb = sum((prog[j] ^ a[j]).bit_count() for j in range(min(len(prog), len(a))))
                tb = (xb + abs(len(prog) - len(a)) * 8) / max(1, len(prog) + len(a))
                md = min(md, tb)
            ns.append(md if md != float("inf") else 0.5)
        mx = max(ns) if ns else 1.0
        nw = [max(0.1, math.sqrt(s / mx)) if mx > 1e-6 else 1.0 for s in ns]
        if weights is not None:
            for i in range(len(weights)):
                weights[i] *= nw[i]
            ws = sum(weights)
            if ws > 0:
                weights = [w / ws for w in weights]
        return weights, True

    def _write_recurrent_examples(self, candidates, weights, level_data, jsonl_path):
        """Phase 2: Write training examples with fitness+diversity+novelty weights."""
        saved_any = False
        for i, (prog, challenger_fitness, trace, sha256) in enumerate(candidates):
            solved_level = level_data[i][0] if level_data else self.champion_level
            task_score = level_data[i][1] if level_data else 0.0
            example = {
                "level": solved_level,
                "generation": self.current_generation,
                "fitness": challenger_fitness,
                "champion_sha256": sha256,
                "champion_words": len(prog),
                "champion_program": [hex(w) for w in prog],
                "trace_status": trace.get("status", ""),
                "trace_instr_count": trace.get("instructions_count", 0),
                "stalled_count": self.generations_stalled,
                "task_score": task_score,
                "timestamp": time.time(),
                "trace_quality_score": min(1.0, max(0.0, challenger_fitness / 100.0)),
                "training_weight": weights[i] if weights else 1.0,
            }
            self._saved_sha256s[sha256] = None
            if len(self._saved_sha256s) > 5000:
                keys = list(self._saved_sha256s.keys())
                self._saved_sha256s = {k: None for k in keys[-3000:]}
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(example) + "\n")
            self.recurrent_examples_count += 1
            saved_any = True
        return saved_any

    def _try_recurrent_feedback(self) -> bool:
        """Save model-generated programs that execute successfully to training data."""
        if not self._last_challengers:
            return False
        recur_dir = self.config.recurrent_feedback_dir
        if not recur_dir:
            recur_dir = MILESTONE1_DIR
        if getattr(self, "_recur_dir_made", None) != recur_dir:
            os.makedirs(recur_dir, exist_ok=True)
            self._recur_dir_made = recur_dir
        jsonl_path = os.path.join(recur_dir, "recurrent_data.jsonl")
        if not hasattr(self, "_saved_sha256s"):
            self._saved_sha256s = {}
            if os.path.isfile(jsonl_path):
                try:
                    with open(jsonl_path) as _f:
                        for _line in _f:
                            _e = json.loads(_line)
                            if "champion_sha256" in _e:
                                self._saved_sha256s[_e["champion_sha256"]] = None
                except Exception:
                    pass
        if not hasattr(self, "_saved_size_counts"):
            self._saved_size_counts = {}
        fitnesses = self._last_challenger_fitnesses
        n_fitnesses = len(fitnesses)
        _candidates, rejected_count, _levels, _verify_levels_sorted = self._collect_recurrent_candidates(
            fitnesses, n_fitnesses
        )
        _level_data, _weights, _n_levels_found = self._compute_level_weights(
            _candidates, _levels, _verify_levels_sorted
        )
        _weights, _novelty_weight_applied = self._apply_novelty_weights(_candidates, _weights, jsonl_path)
        saved_any = self._write_recurrent_examples(_candidates, _weights, _level_data, jsonl_path)
        if saved_any:
            self._log(
                f"Recurrent feedback saved ({self.recurrent_examples_count} total, {len(self._saved_sha256s)} unique, novelty={_novelty_weight_applied}, levels={_n_levels_found})"
            )
        if rejected_count > 0:
            self._log(f"Rejected {rejected_count} low-quality traces")
        self._evict_recurrent_buffer(jsonl_path)
        return self.recurrent_examples_count >= 50

    def _evict_recurrent_buffer(self, jsonl_path: str) -> None:
        """Evict lowest-fitness traces when recurrent buffer exceeds MAX_RECURRENT_ENTRIES.

        Reads all entries, sorts by fitness (ascending), removes the bottom
        entries to bring count below MAX_RECURRENT_ENTRIES, then rewrites.
        This ensures only high-quality traces remain for training.
        """
        if not os.path.isfile(jsonl_path):
            return
        try:
            entries = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            if len(entries) <= MAX_RECURRENT_ENTRIES:
                return

            # Sort by fitness ascending — lowest fitness first for eviction
            entries.sort(key=lambda e: e.get("fitness", 0.0))

            # Keep only the top MAX_RECURRENT_ENTRIES entries
            evicted_count = len(entries) - MAX_RECURRENT_ENTRIES
            entries = entries[evicted_count:]

            # Rewrite file with kept entries (atomic write)
            import tempfile

            dir_name = os.path.dirname(jsonl_path) or "."
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    for entry in entries:
                        f.write(json.dumps(entry) + "\n")
                os.replace(tmp_path, jsonl_path)
                self._log(
                    f"Evicted {evicted_count} low-fitness traces from recurrent buffer "
                    f"(kept {len(entries)} entries)"
                )
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            self._log(f"[RecurrentEvict] Failed: {e}")

    # ------------------------------------------------------------------
    # Strategy archive
    # ------------------------------------------------------------------

    def _record_strategy_outcomes(self) -> None:
        """Record strategy outcomes in the strategy archive.

        Matches each challenger's strategy (from _last_challenger_strategies)
        against its cached stage-1 fitness. Top 10% by fitness are recorded as
        successes, the rest as failures. This gives the archive fine-grained
        signal even when no challenger beats the champion.

        Uses cached stage-1 fitnesses from _select_elite to avoid recomputing
        fitness on every challenger a second time.

        R13: Single-pass threshold computation avoids heapq.nsmallest overhead
        and list slicing. For N=50, this is ~2x faster than the previous approach.
        """
        strategies = self._last_challenger_strategies
        fitnesses = self._last_challenger_fitnesses
        if not strategies or not fitnesses:
            return

        # Guard: truncate to shorter length
        pair_count = min(len(strategies), len(fitnesses))
        if pair_count == 0:
            return

        # Single-pass threshold: find the k-th largest fitness using
        # a partial sort (heapq.nlargest with a small k) without copying
        # the list. For N=50, k=5, heapq.nsmallest is ~3x faster than
        # a full sort.
        k = max(1, pair_count // 10)
        if k >= pair_count:
            # Edge case: all are "top"
            threshold = min(fitnesses[:pair_count]) if pair_count > 0 else 0.0
        else:
            # Use a heap of size k (avoids full list slice + copy)
            top_k = []
            for i in range(pair_count):
                fi = fitnesses[i]
                if len(top_k) < k:
                    heapq.heappush(top_k, fi)
                elif fi > top_k[0]:
                    heapq.heapreplace(top_k, fi)
            threshold = top_k[0]

        # R15: Batch record all outcomes in one call (single bulk append
        # instead of per-record append + per-record index update).
        sa = self.strategy_archive
        sa.refresh_timestamp()
        gen = self.current_generation
        lvl = self.champion_level
        batch = []
        batch_append = batch.append
        for i in range(pair_count):
            batch_append(
                {
                    "strategy": strategies[i],
                    "level": lvl,
                    "success": fitnesses[i] >= threshold,
                    "gen": gen,
                }
            )
        sa.record_batch(batch)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_champion(self, level: int | None = None) -> None:
        """Save the current champion to disk."""
        if self.champion is None:
            return
        lvl = level if level is not None else self.champion_level
        if self.config.champion_dir:
            base = os.path.join(self.config.champion_dir, f"champion_l{lvl}")
            self.champion_anchor.save(self.champion, base)
            self._log(f"Champion L{lvl} saved ({len(self.champion)} words)")

    def _checkpoint(self) -> None:
        """Save a full checkpoint of the evolution state.

        Caches the checkpoint directory path to avoid repeated
        os.path.join + os.makedirs on every checkpoint interval.

        R37: Uses stable filenames (strategy_archive.json, meta.json)
        instead of generation-numbered filenames to prevent disk space
        leaks. On 200-gen runs with checkpoint_interval=25, the old
        approach created 8 strategy archive + 8 metadata files per run;
        now it creates just 1 of each, overwritten each checkpoint.
        """
        if not self.config.checkpoint_interval:
            return

        cp_dir = self._cached_checkpoint_dir
        if cp_dir is None:
            cp_dir = os.path.join(SCRIPT_DIR, "..", "milestone1", "checkpoints")
            if self.config.log_dir:
                cp_dir = os.path.join(self.config.log_dir, "checkpoints")
            os.makedirs(cp_dir, exist_ok=True)
            self._cached_checkpoint_dir = cp_dir

        # Save champion
        self._save_champion()

        # Save strategy archive -- use stable filename (overwrites previous)
        if self.strategy_archive is not None:
            sa_path = self.config.strategy_archive_path
            if not sa_path:
                sa_path = os.path.join(cp_dir, "strategy_archive.json")
            self.strategy_archive.save(sa_path)

        # Save metadata -- use stable filename (overwrites previous)
        meta = {
            "generation": self.current_generation,
            "level": self.champion_level,
            "fitness": self.champion_fitness,
            "program_words": len(self.champion) if self.champion else 0,
            "generations_stalled": self.generations_stalled,
        }
        meta_path = os.path.join(cp_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Log a message via structured logger and optional log file.

        Uses lazy file handle caching to avoid opening/closing a file handle
        on every log call. The handle is closed on interpreter shutdown or
        when the log directory changes.  Caches the log file path to avoid
        repeated os.path.join on every call (~hundreds per run).

        Respects _log_handle_closed flag: after _close_log_handle() is called
        (end of run()), file logging is permanently disabled for this run.
        """
        line = f"Gen {self.current_generation:4d} | {message}"
        _logger.info(line)
        # After _close_log_handle() is called, don't reopen the file
        if getattr(self, "_log_handle_closed", False):
            return
        log_dir = self.config.log_dir
        if log_dir:
            # Cache the full log path (only computed once)
            log_path = self._log_path_cache
            if log_path is None:
                log_path = os.path.join(log_dir, "evolution.log")
                self._log_path_cache = log_path
            try:
                if self._log_file is None:
                    self._log_file = open(log_path, "a", encoding="utf-8")
                self._log_file.write(line + "\n")
                self._log_file.flush()
            except (OSError, ValueError):
                self._log_file = None
                try:
                    self._log_file = open(log_path, "a", encoding="utf-8")
                    self._log_file.write(line + "\n")
                    self._log_file.flush()
                except (OSError, ValueError):
                    pass  # best-effort: structured logger already captured it

    def _log_generation(self, gen: int, start_time: float) -> None:
        """Log per-generation stats."""
        elapsed = time.time() - start_time
        champion = self.champion
        self._log(
            f"Level={self.champion_level}  "
            f"Fitness={self.champion_fitness:7.1f}  "
            f"W={len(champion) if champion else 0:3d}  "
            f"Stalled={self.generations_stalled:2d}  "
            f"T={elapsed:.2f}s"
        )

    @staticmethod
    def _count_backward_branches(program: list[int]) -> int:
        """Count backward branches (loop structures) in RV32I bytecode."""
        return _count_backward_branches_fn(program)

    def get_champion_program_analysis(self) -> dict:
        """Analyze the current champion program structure."""
        if self.champion is None:
            return {"exists": False}
        program = self.champion
        backward_branches = self._count_backward_branches(program)
        n_ops = len(program)
        # Count distinct opcodes
        opcodes = set()
        for word in program:
            if isinstance(word, int):
                opcodes.add(word & 0x7F)
        return {
            "exists": True,
            "word_count": n_ops,
            "unique_opcodes": len(opcodes),
            "backward_branches": backward_branches,
            "has_ebreak": any(isinstance(w, int) and w == 0x00100073 for w in program),
            "estimated_complexity": "simple" if n_ops <= 5 else "moderate" if n_ops <= 15 else "complex",
        }

    def get_champion_summary(self) -> str:
        """Return a one-line champion summary string."""
        if self.champion is None:
            return "No champion"
        return (
            f"L{self.champion_level} F={self.champion_fitness} "
            f"W={len(self.champion)} verified={getattr(self, '_champion_verified', False)}"
        )

    def get_run_diagnostics(self) -> dict:
        """Return comprehensive diagnostics for the current run state."""
        # Graceful None handling for archives
        novelty_size = 0
        if self.novelty_archive is not None:
            try:
                novelty_size = self.novelty_archive.size
            except Exception:
                novelty_size = 0

        strategy_records = 0
        if self.strategy_archive is not None:
            try:
                strategy_records = self.strategy_archive.total_records()
            except Exception:
                strategy_records = 0

        strategy_div = 0.0
        if self.strategy_archive is not None:
            try:
                strategy_div = self.strategy_archive.get_strategy_diversity(self.champion_level)
                if strategy_div is None:
                    strategy_div = 0.0
            except Exception:
                strategy_div = 0.0

        qd_size = 0
        if self.qd_archive is not None:
            try:
                qd_size = self.qd_archive.size
            except Exception:
                qd_size = 0

        return {
            # Champion state
            "champion_level": self.champion_level,
            "champion_fitness": self.champion_fitness,
            "champion_words": len(self.champion) if self.champion else 0,
            "champion_verified": getattr(self, "_champion_verified", False),
            "generations_stalled": self.generations_stalled,
            "current_generation": self.current_generation,
            # Archive sizes
            "novelty_archive_size": novelty_size,
            "qd_archive_size": qd_size,
            "strategy_archive_records": strategy_records,
            "hall_of_fame_size": len(self._hall_of_fame._heap),
            # Strategy diversity
            "strategy_diversity": strategy_div,
            # Training data
            "recurrent_examples": self.recurrent_examples_count,
            # Fitness velocity
            "fitness_velocity": getattr(self, "_fitness_velocity", 0.0),
            "fitness_momentum": getattr(self, "_fitness_momentum", 0.0),
            # Mutation stats
            "mutation_rate": self.config.mutation_rate,
            "population_size": self.config.population_size,
            "elite_ratio": self.config.elite_ratio,
        }

    def get_performance_metrics(self) -> dict:
        """Return performance metrics for the current run."""
        return {
            "current_generation": self.current_generation,
            "population_size": self.config.population_size,
            "elite_ratio": self.config.elite_ratio,
            "eval_budget": self.config.eval_budget,
            "model_guided_ratio": self._get_model_guided_ratio(),
            "mutation_rate": self.config.mutation_rate,
            "champion_words": len(self.champion) if self.champion else 0,
            "novelty_weight": self.config.novelty_weight,
            "qd_bonus_weight": self.config.qd_bonus_weight,
            "total_recurrent_examples": self.recurrent_examples_count,
            "hof_size": len(self._hall_of_fame._heap),
        }

    def get_archive_summary(self) -> dict:
        """Aggregate summary of all archive states."""
        novelty = self.novelty_archive
        qd = self.qd_archive
        sa = self.strategy_archive
        return {
            "novelty": {
                "size": novelty.size if novelty else 0,
                "threshold": novelty.novelty_threshold if novelty else 0,
                "diversity": novelty.get_diversity_score() if novelty else 0.0,
            },
            "qd": {
                "coverage": qd.coverage() if qd else 0.0,
                "size": qd.size if qd else 0,
                "cells": qd.num_bins * qd.num_bins if qd else 0,
                "fitness_dist": qd.get_fitness_distribution() if qd else {},
            },
            "strategy": {
                "total_records": sa.total_records() if sa else 0,
                "diversity": sa.get_strategy_diversity(self.champion_level) if sa else 0.0,
            },
            "hall_of_fame": len(self._hall_of_fame._heap),
        }

    def get_training_summary(self) -> dict:
        """Return training-relevant metrics for the retrain pipeline."""
        return {
            "recurrent_examples": self.recurrent_examples_count,
            "champion_fitness": self.champion_fitness,
            "champion_level": self.champion_level,
            "generations_stalled": self.generations_stalled,
            "fitness_velocity": getattr(self, "_fitness_velocity", 0.0),
            "champion_verified": getattr(self, "_champion_verified", False),
            "champion_verify_score": getattr(self, "_champion_verify_score", 0.0),
            "strategy_diversity": (
                self.strategy_archive.get_strategy_diversity(self.champion_level) if self.strategy_archive else 0.0
            ),
            "novelty_diversity": self.novelty_archive.get_diversity_score() if self.novelty_archive else 0.0,
            "qd_coverage": self.qd_archive.coverage() if self.qd_archive else 0.0,
        }

    def get_run_summary(self) -> dict:
        """Single consolidated summary of the entire run state."""
        return {
            "diagnostics": self.get_run_diagnostics(),
            "performance": self.get_performance_metrics(),
            "training": self.get_training_summary(),
            "archives": self.get_archive_summary(),
            "champion_summary": self.get_champion_summary(),
        }

    def get_evolution_state(self) -> dict:
        """Full evolution state for serialization/debugging."""
        return {
            "champion": self.get_champion_program_analysis(),
            "run_summary": self.get_run_summary(),
            "state": {
                "current_generation": self.current_generation,
                "champion_level": self.champion_level,
                "champion_fitness": self.champion_fitness,
                "generations_stalled": self.generations_stalled,
            },
            "config": {
                "max_generations": self.config.max_generations,
                "population_size": self.config.population_size,
                "elite_ratio": self.config.elite_ratio,
                "mutation_rate": self.config.mutation_rate,
                "auto_curriculum": self.config.auto_curriculum,
                "use_gpu": self.config.use_gpu,
            },
        }

    def get_comprehensive_state(self) -> dict:
        """All-in-one state snapshot for dashboards, export, and debugging."""
        return {
            "evolution": self.get_evolution_state(),
            "champion_program": self.get_champion_program_analysis(),
            "run_diagnostics": self.get_run_diagnostics(),
            "performance": self.get_performance_metrics(),
            "training": self.get_training_summary(),
            "archives": self.get_archive_summary(),
            "champion_summary": self.get_champion_summary(),
        }

    def log_comprehensive_diagnostics(self) -> None:
        """Log comprehensive diagnostics for debugging and analysis."""
        state = self.get_comprehensive_state()
        evo = state.get("evolution", {})
        evo_state = evo.get("state", {})
        self._log("=== Comprehensive Diagnostics ===")
        self._log(f"  Champion: {self.get_champion_summary()}")
        self._log(f'  Generation: {evo_state.get("current_generation", 0)}')
        self._log(f'  Fitness velocity: {evo_state.get("fitness_velocity", 0):.4f}')
        archives = state.get("archives", {})
        novelty = archives.get("novelty", {})
        qd = archives.get("qd", {})
        strategy = archives.get("strategy", {})
        self._log(f'  Archives: novelty={novelty.get("size", 0)}, ' f'qd_coverage={qd.get("coverage", 0.0):.2%}')
        self._log(f'  Strategy diversity: {strategy.get("diversity", 0.0):.3f}')
        perf = state.get("performance", {})
        self._log(
            f'  Performance: mut_rate={perf.get("mutation_rate", 0.0):.2f}, ' f'pop={perf.get("population_size", 0)}'
        )
        self._log("=== End Diagnostics ===")

    def log_diagnostics(self) -> None:
        """Log run diagnostics in a readable format."""
        diag = self.get_run_diagnostics()
        lines = [
            "=== Run Diagnostics ===",
            (
                f"  Champion: L{diag['champion_level']} fitness={diag['champion_fitness']:.2f} "
                f"words={diag['champion_words']} verified={diag['champion_verified']}"
            ),
            f"  Generation: {diag['current_generation']} stalled={diag['generations_stalled']}",
            (
                f"  Archives: novelty={diag['novelty_archive_size']} qd={diag['qd_archive_size']} "
                f"strategy={diag['strategy_archive_records']} hof={diag['hall_of_fame_size']}"
            ),
            f"  Strategy diversity: {diag['strategy_diversity']:.3f}",
            f"  Recurrent examples: {diag['recurrent_examples']}",
            f"  Fitness velocity: {diag['fitness_velocity']:.4f} momentum={diag['fitness_momentum']:.4f}",
            f"  Mutation rate: {diag['mutation_rate']:.3f} pop={diag['population_size']} elite={diag['elite_ratio']:.2f}",
        ]
        self._log("\n".join(lines))

    def export_state_for_analysis(self) -> dict:
        """Export full engine state for external analysis.

        Returns a serializable dict with all key metrics, archive contents,
        and configuration for use in Jupyter notebooks or dashboards.
        """
        state = {
            "timestamp": __import__("time").time(),
            "generation": self.current_generation,
            "champion": {
                "level": self.champion_level,
                "fitness": self.champion_fitness,
                "words": len(self.champion) if self.champion else 0,
                "program_hex": [hex(w) for w in (self.champion or [])],
                "verified": getattr(self, "_champion_verified", False),
            },
            "config": {
                "population_size": self.config.population_size,
                "elite_ratio": self.config.elite_ratio,
                "mutation_rate": self.config.mutation_rate,
                "novelty_weight": self.config.novelty_weight,
            },
            "archives": {
                "novelty_size": self.novelty_archive.size if self.novelty_archive else 0,
                "qd_occupied": self.qd_archive.size if self.qd_archive else 0,
                "qd_coverage": self.qd_archive.coverage() if self.qd_archive else 0.0,
                "hof_size": len(self._hall_of_fame._heap),
            },
            "stall": {
                "generations_stalled": self.generations_stalled,
                "fitness_velocity": getattr(self, "_fitness_velocity", 0.0),
            },
            "diagnostics": self.get_run_diagnostics(),
        }
        return state

    def save_state_to_file(self, path: str) -> bool:
        """Save comprehensive state to a JSON file for analysis."""
        try:
            state = self.get_comprehensive_state()
            import json

            with open(path, "w") as f:
                json.dump(state, f, indent=2, default=str)
            return True
        except Exception as e:
            self._log(f"[StateExport] Failed to save to {path}: {e}")
            return False

    def _close_log_handle(self) -> None:
        """Close the lazy log file handle if open.

        Called at the end of run() and on exception to prevent fd exhaustion
        across multi-cycle runs. Safe to call multiple times.
        Sets _log_handle_closed flag and clears path cache so _log won't
        reopen the file after intentional close.
        """
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        self._log_handle_closed = True
        self._log_path_cache = None  # prevent _log from reopening

    # ------------------------------------------------------------------
    # Load / restore
    # ------------------------------------------------------------------

    def load_champion(self, level: int, path: str | None = None) -> bool:
        """Load a champion program from disk.

        Args:
            level: Curriculum level.
            path:  Optional explicit path.  Defaults to champion_dir.

        Returns:
            True if loaded successfully.
        """
        if path is None and self.config.champion_dir:
            path = os.path.join(self.config.champion_dir, f"champion_l{level}")
        if path is None:
            return False

        try:
            self.champion = self.champion_anchor.load(path)
            self.champion_level = level
            self._log(f"Champion L{level} loaded ({len(self.champion)} words)")
            return True
        except IntegrityError:
            self._log(f"Failed to load champion L{level}")
            return False


# ----------------------------------------------------------------------
# Main entrypoint
# ----------------------------------------------------------------------


def main() -> None:
    """Command-line entry point for the evolution engine."""
    import argparse

    parser = argparse.ArgumentParser(description="RV32I Evolution Engine -- Champion/Challenger Evolution")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations to run")
    parser.add_argument("--level", type=int, default=0, help="Starting curriculum level")
    parser.add_argument("--population", type=int, default=50, help="Population size per generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--champion-dir", type=str, default="", help="Directory for champion persistence")
    parser.add_argument("--log-dir", type=str, default="", help="Directory for logs")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Generations between checkpoints")
    parser.add_argument("--no-auto-curriculum", action="store_true", help="Disable automatic curriculum advancement")

    args = parser.parse_args()

    config = EvolutionConfig(
        population_size=args.population,
        max_generations=args.generations,
        curriculum_level=args.level,
        seed=args.seed,
        champion_dir=args.champion_dir,
        log_dir=args.log_dir,
        checkpoint_interval=args.checkpoint_interval,
        auto_curriculum=not args.no_auto_curriculum,
    )

    engine = EvolutionEngine(config=config)
    summary = engine.run()

    _logger.info("=" * 60)
    _logger.info("Evolution Run Complete")
    _logger.info("=" * 60)
    for k, v in summary.items():
        _logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
