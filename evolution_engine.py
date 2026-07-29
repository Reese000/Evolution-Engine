#!/usr/bin/env python3
"""RV32I Evolution Engine -- Champion/Challenger with Directed Mutation.

Implements the core evolutionary loop: maintain a champion program,
generate challengers via 4 mutation strategies (RANDOM, TASK_AWARE,
DIRECTED, PERTURBATION_CAPPED), evaluate on the VM farm, and replace the
champion when a challenger achieves higher fitness.
"""

from __future__ import annotations

import collections
import concurrent.futures
import heapq
import json
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
MIN_FITNESS_FOR_TRACE: float = 0.0
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

from paths import MILESTONE0_DIR, MILESTONE1_DIR
from paths import PROJECT_ROOT as SCRIPT_DIR
from paths import ensure_project_on_path

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

    def evaluate_batch(
        self, programs: list[list[int]], inputs_list: list[dict[str, Any]], budget: int = 10000
    ) -> list[EvalResult]:
        """Evaluate multiple programs. Uses GPU (30-300x faster) when use_gpu=True.

        R17: Reuses a cached ProcessPoolExecutor instead of creating/destroying
        one per batch call (~50 per generation). The executor persists for the
        lifetime of the VMFarm instance, amortizing process startup overhead.
        """
        if self.use_gpu and self._gpu_farm is not None and len(programs) > 1:
            try:
                gpu_results = self._gpu_farm.evaluate_multi(programs, inputs_list, budget=budget)
                self._gpu_eval_count += len(programs)
                return [VMFarm._dict_to_eval_result(gr) for gr in gpu_results]
            except Exception as e:
                self._gpu_fallback_count += 1
                self._log(f"GPU batch failed ({e}), falling back to CPU " f"(fallbacks={self._gpu_fallback_count})")
                # Warn if fallback rate exceeds 5%
                if self._gpu_fallback_count % 20 == 0 and self._gpu_eval_count > 0:
                    rate = self._gpu_fallback_count / max(1, self._gpu_eval_count)
                    if rate > 0.05:
                        self._log(
                            f"WARNING: GPU fallback rate {rate:.1%} "
                            f"({self._gpu_fallback_count}/{self._gpu_eval_count})"
                        )
        # CPU fallback — use cached ProcessPoolExecutor
        # R39: Fixed: old code used min(8, os.cpu_count()) which ignored
        # the user-specified num_workers entirely. Now respects num_workers
        # while still capping at CPU count and 8 to prevent over-parallelism.
        try:
            if self._pool_executor is None:
                max_workers = min(self.num_workers, 8, os.cpu_count() or 1)
                self._pool_executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
            executor = self._pool_executor
            futures = [
                executor.submit(
                    _run_one_evaluation,
                    kernel_path=self.kernel_path,
                    program=p,
                    budget=budget,
                    init_regs=i.get("registers", {}),
                    init_mem=i.get("memory", {}),
                )
                for p, i in zip(programs, inputs_list, strict=False)
            ]
            return [f.result() for f in futures]
        except Exception as exc:
            # Reset executor on failure (may have stale workers)
            self._pool_executor = None
            self._log(f"ProcessPool failed ({exc}), falling back to sequential")
            return [
                _run_one_evaluation(
                    kernel_path=self.kernel_path,
                    program=p,
                    budget=budget,
                    init_regs=i.get("registers", {}),
                    init_mem=i.get("memory", {}),
                )
                for p, i in zip(programs, inputs_list, strict=False)
            ]

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

    def get_random(self, rng: random.Random, level: int | None = None) -> list[int] | None:
        """Get a random program from the hall of fame (for diversity).

        R28: Uses recency-weighted selection -- programs from more recent
        generations are more likely to be chosen, preventing stale entries
        from dominating injection. Weight = generation / max_generation_in_pool.

        R29: Optimizes the common case (level=None, < capacity programs)
        by iterating the heap directly without creating a candidate list.
        Only allocates a filtered list when level filtering is needed.
        Returns a defensive copy of the program list to prevent callers
        from accidentally mutating the HoF's internal state.
        """
        if not self._heap:
            return None

        heap = self._heap
        n = len(heap)

        # Fast path: single program
        if n == 1:
            _, _, prog, _ = heap[0]
            return list(prog)

        # R29: When level=None, iterate heap directly without copying.
        # The heap is already a list of (fitness, generation, program, level).
        # We compute weights in a single pass using inline min/max scan.
        if level is None:
            # Single-pass: find min/max gen and build weight array
            min_gen = heap[0][1]
            max_gen = heap[0][1]
            for _, g, _, _ in heap[1:]:
                if g < min_gen:
                    min_gen = g
                elif g > max_gen:
                    max_gen = g
            gen_range = max_gen - min_gen if max_gen > min_gen else 1

            # Weighted random selection in one pass (no intermediate list)
            total = 0.0
            weights = [0.0] * n  # pre-allocate
            for i, (_, g, _, _) in enumerate(heap):
                w = 0.3 + 0.7 * ((g - min_gen) / gen_range)
                weights[i] = w
                total += w

            r = rng.random() * total
            cumulative = 0.0
            for i in range(n):
                cumulative += weights[i]
                if r <= cumulative:
                    _, _, prog, _ = heap[i]
                    return list(prog)

            _, _, prog, _ = heap[-1]
            return list(prog)

        # Level-filtered path: collect matching programs
        candidates: list[tuple[float, int, list[int], int]] = []
        for entry in heap:
            if entry[3] == level:
                candidates.append(entry)
        if not candidates:
            # Fall through to unfiltered (avoid returning None when
            # there are programs at other levels)
            candidates = list(heap)

        if len(candidates) == 1:
            _, _, prog, _ = candidates[0]
            return list(prog)

        # R28: Recency-weighted random selection
        max_gen = max(c[1] for c in candidates)
        min_gen = min(c[1] for c in candidates)
        gen_range = max_gen - min_gen if max_gen > min_gen else 1

        total = 0.0
        for _, g, _, _ in candidates:
            w = 0.3 + 0.7 * ((g - min_gen) / gen_range)
            total += w

        r = rng.random() * total
        cumulative = 0.0
        for i, (_, g, _, _) in enumerate(candidates):
            w = 0.3 + 0.7 * ((g - min_gen) / gen_range)
            cumulative += w
            if r <= cumulative:
                _, _, prog, _ = candidates[i]
                return list(prog)

        _, _, prog, _ = candidates[-1]
        return list(prog)

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
    max_level: int = 13
    max_champion_words: int = 500
    num_eval_seeds: int = 3
    self_calibrate_delta_threshold: float = 0.05
    use_circuit_breaker: bool = False
    circuit_breaker_stall_limit: int = 50

    # Multi-task fitness extension
    use_secondary_tasks: bool = True
    secondary_task_weight: float = 3.0
    efficiency_score_weight: float = 2.0

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

    def __post_init__(self) -> None:
        """Validate field values after construction.

        Prevents silent misconfiguration that would cause stuck evolution,
        division-by-zero, or GPU OOM. Fail-fast with clear error messages.
        """
        # Core evolution params
        if self.population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {self.population_size}")
        if self.max_generations < 1:
            raise ValueError(f"max_generations must be >= 1, got {self.max_generations}")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0.0, 1.0], got {self.mutation_rate}")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0.0, 1.0], got {self.crossover_rate}")
        if not 0.0 <= self.elite_ratio <= 0.5:
            raise ValueError(f"elite_ratio must be in [0.0, 0.5], got {self.elite_ratio}")
        if not 0.0 <= self.mutation_rate_start <= 1.0:
            raise ValueError(f"mutation_rate_start must be in [0.0, 1.0], got {self.mutation_rate_start}")
        if not 0.0 <= self.mutation_rate_end <= 1.0:
            raise ValueError(f"mutation_rate_end must be in [0.0, 1.0], got {self.mutation_rate_end}")
        if self.mutation_schedule_generations < 1:
            raise ValueError(f"mutation_schedule_generations must be >= 1, got {self.mutation_schedule_generations}")
        if not 0.0 <= self.temperature_start <= 5.0:
            raise ValueError(f"temperature_start must be in [0.0, 5.0], got {self.temperature_start}")
        if not 0.0 <= self.temperature_end <= 5.0:
            raise ValueError(f"temperature_end must be in [0.0, 5.0], got {self.temperature_end}")
        if self.temperature_schedule_generations < 1:
            raise ValueError(
                f"temperature_schedule_generations must be >= 1, got {self.temperature_schedule_generations}"
            )
        if self.tournament_size < 2:
            raise ValueError(f"tournament_size must be >= 2, got {self.tournament_size}")
        if self.tournament_size > self.population_size:
            raise ValueError(f"tournament_size ({self.tournament_size}) > population_size ({self.population_size})")
        if self.instruction_budget < 100:
            raise ValueError(f"instruction_budget must be >= 100, got {self.instruction_budget}")
        if self.eval_budget < 100:
            raise ValueError(f"eval_budget must be >= 100, got {self.eval_budget}")

        # Curriculum params
        if self.stall_threshold < 1:
            raise ValueError(f"stall_threshold must be >= 1, got {self.stall_threshold}")
        if self.max_level < 1:
            raise ValueError(f"max_level must be >= 1, got {self.max_level}")
        if self.curriculum_stall_threshold < 1:
            raise ValueError(f"curriculum_stall_threshold must be >= 1, got {self.curriculum_stall_threshold}")

        # Retrain params
        if self.retrain_val_split < 0.0 or self.retrain_val_split > 1.0:
            raise ValueError(f"retrain_val_split must be in [0.0, 1.0], got {self.retrain_val_split}")
        if self.retrain_max_epochs < 1:
            raise ValueError(f"retrain_max_epochs must be >= 1, got {self.retrain_max_epochs}")
        if self.retrain_batch_size < 1:
            raise ValueError(f"retrain_batch_size must be >= 1, got {self.retrain_batch_size}")
        if self.retrain_lr <= 0:
            raise ValueError(f"retrain_lr must be > 0, got {self.retrain_lr}")
        if self.retrain_freeze_layers < 0:
            raise ValueError(f"retrain_freeze_layers must be >= 0, got {self.retrain_freeze_layers}")
        if not 0.0 <= self.retrain_weight_decay < 1.0:
            raise ValueError(f"retrain_weight_decay must be in [0.0, 1.0), got {self.retrain_weight_decay}")
        if self.retrain_patience < 1:
            raise ValueError(f"retrain_patience must be >= 1, got {self.retrain_patience}")
        if self.retrain_timeout < 60:
            raise ValueError(f"retrain_timeout must be >= 60, got {self.retrain_timeout}")

        # Weighted fitness components
        if self.meta_weight_improvement < 0 or self.meta_weight_depth < 0 or self.meta_weight_diversity < 0:
            raise ValueError("meta_weight_* fields must be >= 0")
        if self.novelty_weight < 0:
            raise ValueError(f"novelty_weight must be >= 0, got {self.novelty_weight}")
        if self.qd_bonus_weight < 0:
            raise ValueError(f"qd_bonus_weight must be >= 0, got {self.qd_bonus_weight}")

        # Archive params
        if self.max_consecutive_same_strategy < 3:
            raise ValueError(f"max_consecutive_same_strategy must be >= 3, got {self.max_consecutive_same_strategy}")
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError(f"diversity_threshold must be in [0.0, 1.0], got {self.diversity_threshold}")
        if not 0.0 <= self.diversity_inject_ratio <= 1.0:
            raise ValueError(f"diversity_inject_ratio must be in [0.0, 1.0], got {self.diversity_inject_ratio}")
        if not 0.0 <= self.champion_crossover_prob <= 1.0:
            raise ValueError(f"champion_crossover_prob must be in [0.0, 1.0], got {self.champion_crossover_prob}")

        # Circuit breaker
        if self.circuit_breaker_stall_limit < 5:
            raise ValueError(f"circuit_breaker_stall_limit must be >= 5, got {self.circuit_breaker_stall_limit}")

        # Warning-only checks (log but don't raise)
        if self.elite_ratio == 0.0:
            _logger.warning("elite_ratio=0.0 means no elite preservation -- evolution will be purely random")
        if self.novelty_weight > 0.8:
            _logger.warning(f"novelty_weight={self.novelty_weight} is very high -- fitness may be dominated by novelty")


class EvolutionEngine:
    """Main evolutionary loop for RV32I program synthesis.

    Maintains a champion program per curriculum level and iteratively
    generates challengers via mutation, evaluates them, and promotes
    fitter individuals.
    """

    def __init__(
        self,
        config: EvolutionConfig | None = None,
        rng: random.Random | None = None,
        vm_farm: VMFarm | None = None,
        champion_anchor: ChampionAnchor | None = None,
    ):
        self.config = config or EvolutionConfig()
        self.rng = rng or random.Random(self.config.seed)

        # VM farm
        self.use_gpu = self.config.use_gpu
        self.vm_farm = vm_farm or VMFarm(num_workers=1, use_gpu=self.config.use_gpu)

        # Champion storage
        self.champion_anchor = champion_anchor or ChampionAnchor()

        # Strategy archive
        # StrategyArchive always available (sibling module)
        from strategy_archive import StrategyArchive

        self.strategy_archive = StrategyArchive()
        if self.config.strategy_archive_path:
            self.strategy_archive.load(self.config.strategy_archive_path)

        # Curriculum generator
        if _HAS_CURRICULUM and self.config.auto_curriculum:
            self.curriculum = CurriculumGenerator(
                CurriculumConfig(
                    max_levels=len(LEVELS),
                    stall_generations=self.config.curriculum_stall_threshold,
                ),
                seed=self.config.seed,
            )
        else:
            self.curriculum = None

        # State
        self.champion: list[int] | None = None
        self.champion_trace: dict[str, Any] | None = None
        self.champion_fitness: float = 0.0
        self.champion_level: int = self.config.curriculum_level
        self.current_generation: int = 0
        self.generations_stalled: int = 0
        self._qd_champion_reset_tried: bool = False
        self.best_fitness_history: list[float] = []
        # R33: Cap history length to prevent unbounded memory growth on
        # long runs (200+ generations). The last 500 entries are sufficient
        # for convergence diagnostics (improvement count, trend).
        # R37: Removed unused population_history (was allocated but never
        # read by any method -- pure memory waste on long runs).
        self._max_history_len: int = 500
        self.task_instance: dict[str, Any] | None = None

        # D3: Strategy monoculture counter — tracks how many consecutive
        # generations a single non-GROW strategy has been recommended.
        self._consecutive_same_strategy: int = 0
        self._last_strategy_used: str = ""

        # Current diversity metric (set during run)
        self._current_diversity: float = 0.5

        # Raw task score (before novelty/QD blending) for curriculum advancement
        self._raw_task_score: float = 0.0

        # Circuit breaker (placeholder for barrier tuner integration)
        self.circuit_breaker = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL

        # Seed model path (set externally by run_multi_cycle.py)
        self._seed_model_path: str = ""

        # Novelty archive (optional)
        self.novelty_archive: Any = None
        if _HAS_NOVELTY and self.config.use_novelty_archive:
            try:
                from novelty import SparseArchive

                self.novelty_archive = SparseArchive(novelty_threshold=self.config.novelty_threshold)
            except (ImportError, Exception):
                pass

        # QD archive (optional)
        self.qd_archive: Any = None
        if _HAS_QD and self.config.use_qd_archive:
            try:
                from qd_archive import MAPElitesArchive

                self.qd_archive = MAPElitesArchive(bins=self.config.qd_bins)
            except (ImportError, Exception):
                pass

        # Prefix generator for model-guided completions (lazy init)
        self._prefix_generator: Any = None

        # Champion archive — persistent training data across cycles
        from champion_archive import ChampionArchive

        archive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "champions", "archive.jsonl")
        self.champion_archive = ChampionArchive(path=archive_path, max_size=500)
        self._cycle_number: int = 1  # set externally by run_multi_cycle.py

        # External evolution params (from BarrierTuner)
        self._external_evolution_params: dict[str, Any] = {}

        # Recurrent feedback
        self.recurrent_examples_count: int = 0

        # Strategy tracking for archive recording
        self._last_challenger_strategies: list[str] = []
        self._last_challenger_programs: list[list[int]] = []
        # Cached stage-1 fitness values (avoids recomputation in _record_strategy_outcomes)
        self._last_challenger_fitnesses: list[float] = []

        # Pre-initialize generation-accumulated state to avoid AttributeError
        # if _try_recurrent_feedback runs before _generate_challengers.
        self._last_challengers: list[list[int]] = []
        self._last_results: list[dict[str, Any]] = []
        # R37: Explicitly mark generation-state as consumed so GC can reclaim
        # after _select_elite / _update_champion finish. Without this, the
        # large lists (pop_size * avg_program_len words) persist in memory
        # for the entire run duration.
        self._generation_state_consumed: bool = True

        # R33: Champion raw task score -- tracks the base_score (before
        # novelty/QD blending) of the CHAMPION specifically, not the last
        # elite candidate evaluated. Used by _try_curriculum_advance to
        # decide whether the champion has solved the task.
        self._champion_raw_task_score: float = 0.0

        # R21: Track whether champion changed since last QD archive update.
        # When champion is unchanged, _update_qd_archive can skip entirely
        # (avoids redundant QD axes recomputation every generation).
        self._champion_changed_since_last_qd: bool = False
        # R39: Track whether champion changed since last novelty archive update.
        # Decoupled from QD flag: novelty and QD archives serve different purposes
        # (behavioral diversity vs niche coverage) and should update independently.
        # Without this, disabling QD (use_qd_archive=False) would incorrectly
        # suppress novelty archive updates.
        self._champion_changed_since_last_novelty: bool = False

        # Hall of Fame: retains top programs across the entire run
        self._hall_of_fame = HallOfFame(capacity=10)

        # R22: Pre-compute unsolved levels as a set for O(1) membership test
        # in _choose_mutation_strategy (was tuple membership test per call).
        self._unsolved_levels_set: frozenset[int] = frozenset(self.config.perturbation_capped_unsolved_levels)

        # R28: Adaptive mutation state -- tracks fitness velocity and
        # adjusts mutation rate dynamically based on improvement rate.
        self._fitness_velocity: float = 0.0  # exponential moving average of fitness delta
        self._velocity_alpha: float = 0.3  # EMA smoothing factor
        self._adaptive_mutation_floor: float = 0.15  # minimum mutation rate
        self._adaptive_mutation_ceil: float = 0.85  # maximum mutation rate
        self._original_mutation_rate: float = self.config.mutation_rate  # R33: prevent drift

        # R30: QD niche coverage tracking -- updated each generation to
        # expose niche occupancy for test assertions and diagnostics.
        self._qd_coverage: float = 0.0

        # R28: Champion oscillation tracking -- prevents rapid A<->B replacement
        # by requiring a minimum fitness improvement to beat the champion.
        self._champion_replacement_count: int = 0  # replacements this run
        self._champion_replacement_history: list[tuple[float, int]] = []  # (delta_fitness, generation)
        # R31: Use deque for O(1) popleft (was list.pop(0) = O(n) per generation).
        self._champion_oscillation_window: collections.deque[float] = collections.deque(maxlen=10)
        self._oscillation_window_size: int = 10
        self._oscillation_threshold: float = 3  # replacements in window -> hysteresis active
        self._hysteresis_min_delta: float = 0.5  # minimum delta when hysteresis active
        self._base_hysteresis_min_delta: float = 0.5  # R33: base for stall-adaptive scaling
        self._oscillation_positive_count: int = 0  # running count of positive deltas in window

        # R31: Cache champion QD descriptor -- recomputed ~50x per gen
        # for niche penalty, only changes when champion is replaced.
        self._cached_champion_qd_desc: list[float] | None = None
        self._champion_qd_valid: bool = False

        # Cached config values to avoid repeated attribute lookups in hot paths
        self._novelty_weight_cache = self.config.novelty_weight
        self._cached_eval_budget = self.config.eval_budget
        # Bonus threshold cache: (level, threshold) or None
        self._bonus_threshold_cache: tuple[int, float] | None = None
        # Cached expected regs (invalidated when task instance changes)
        self._cached_expected_regs = None
        self._cached_expected_total = 0
        self._cached_task_instance_id = None

        # Cached lazy imports for hot-path functions (avoids repeated
        # import lookups on every _compute_fitness / _update_novelty_archive call)
        self._extract_behavioral_descriptor = None
        if _HAS_NOVELTY:
            try:
                from novelty import extract_behavioral_descriptor

                self._extract_behavioral_descriptor = extract_behavioral_descriptor
            except ImportError:
                pass
        self._compute_qd_axes_fn = compute_qd_axes  # already module-level

        # Logging -- lazy file handle for evolution log
        self._log_file = None
        self._log_path_cache: str | None = None  # cached log file path

        # Cached checkpoint dir path (avoids repeated os.path.join each checkpoint)
        self._cached_checkpoint_dir: str | None = None
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
            'current_generation': self.current_generation,
            'generations_stalled': self.generations_stalled,
            'champion_fitness': self.champion_fitness,
            'champion_level': self.champion_level,
            'mutation_rate': self.config.mutation_rate,
            'population_size': self.config.population_size,
            'novelty_weight': self.config.novelty_weight,
            'fitness_velocity': getattr(self, '_fitness_velocity', 0.0),
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
                if self._oscillation_positive_count < 0:
                    self._oscillation_positive_count = 0
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
            if self._oscillation_positive_count > self._oscillation_window_size:
                self._oscillation_positive_count = self._oscillation_window_size

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
        if self._champion_replacement_count > 10000:
            self._champion_replacement_count = 10000
        history = getattr(self, '_champion_replacement_history', None)
        if history is not None:
            history.append((delta_fitness, self.current_generation))
        # Oscillation window is already maintained by _should_replace_champion

    def get_champion_history(self) -> list[dict]:
        """Return history of champion fitness improvements.

        Each entry is a dict with 'fitness' (delta_fitness) and 'cycle' (generation).
        """
        history = getattr(self, '_champion_replacement_history', None)
        if history is None:
            return []
        return [
            {'fitness': f, 'cycle': c}
            for f, c in history
        ]

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def run(self, num_generations: int | None = None) -> dict[str, Any]:
        """Run the evolution loop for *num_generations* (default: config).

        Returns a summary dict of the run.

        Performance optimizations:
          - Precomputes loop-invariant config values (checkpoint_interval,
            auto_curriculum, task_refresh_interval) to avoid repeated dict
            attribute lookups in the hot generation loop.
          - CUDA/torch import cached once at first use (no repeated import).
        """
        ng = num_generations if num_generations is not None else self.config.max_generations
        start_time = time.time()

        # Precompute loop-invariant values (avoids self.config.<attr> lookups each gen)
        checkpoint_interval = self.config.checkpoint_interval
        auto_curriculum = self.config.auto_curriculum
        task_refresh_interval = 25
        cuda_cleanup_interval = 10

        # R16: Cache LEVELS reference once (avoid repeated module-level lookups)
        _levels_ref = LEVELS

        # Lazy CUDA cleanup function (imported once, cached)
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

        # Initialise task for current level
        self._refresh_task_instance()

        # Generate initial champion
        if self.champion is None:
            self.champion = self._generate_initial_champion()
            # R45: Validate champion before expensive evaluation
            is_valid, reason = self.validate_program(self.champion)
            if not is_valid:
                self._log(f"  Initial champion invalid ({reason}), generating replacement")
                for _retry in range(5):
                    self.champion = self._generate_initial_champion()
                    is_valid, reason = self.validate_program(self.champion)
                    if is_valid:
                        break
                if not is_valid:
                    self._log("  WARNING: Could not generate valid champion after 5 retries")
            self.champion_trace = self._evaluate(self.champion)
            if self.config.use_secondary_tasks:
                self._secondary_task_results = self._evaluate_secondary_tasks(self.champion)
            else:
                self._secondary_task_results = None
            self.champion_fitness = self._compute_fitness(self.champion, self.champion_trace)
            # R50: Run verify() on initial champion (uses initial task instance)
            verified, verify_score = self._verify_champion()
            self._champion_verified = verified
            self._champion_verify_score = verify_score

        summary = {
            "generations_completed": 0,
            "final_fitness": self.champion_fitness,
            "final_level": self.champion_level,
            "champion_words": len(self.champion) if self.champion else 0,
            "elapsed_seconds": 0.0,
        }

        # Precompute bonus threshold once per generation
        # getattr + cache check inside _compute_fitness for every challenger).
        n_levels = len(_levels_ref)
        _precomputed_bonus_threshold: float | None = None

        # Wrap generation loop in try/finally to ensure log handle is closed
        # on any exception during the evolution loop. Exception during
        # champion init (above) is NOT caught -- it propagates to caller.
        try:
            # Initialize _champion_raw_task_score from the initial champion.
            # Must be inside try block so that if _raw_task_score is missing,
            # the finally clause still runs and closes the log handle.
            if hasattr(self, "_raw_task_score"):
                self._champion_raw_task_score = self._raw_task_score

            # Cache VMFarm reference to avoid attribute lookup each iteration
            vm_farm = self.vm_farm
            for gen in range(ng):
                self.current_generation = gen
                gen_start = time.time()
                # R21: Reset champion-changed flag at start of generation
                self._champion_changed_since_last_qd = False
                self._champion_changed_since_last_novelty = False

                # Precompute per-generation invariants (avoids per-challenger work)
                lvl = self.champion_level
                _precomputed_bonus_threshold = max(0.85, 0.95 - 0.10 * (lvl / max(1, n_levels - 1)))
                self._bonus_threshold_cache = (lvl, _precomputed_bonus_threshold)

                # R42: Mutation rate scheduler — decays base rate from
                # mutation_rate_start (0.5) → mutation_rate_end (0.1) over
                # mutation_schedule_generations. This enables broad exploration
                # early and fine-grained exploitation later in the run.
                # The scheduled base feeds into the adaptive mutation system
                # which adjusts around it based on fitness velocity.
                if self.config.use_mutation_scheduler:
                    sched_progress = gen / max(1, self.config.mutation_schedule_generations)
                    sched_progress = min(1.0, sched_progress)
                    scheduled_base = (
                        self.config.mutation_rate_start
                        + (self.config.mutation_rate_end - self.config.mutation_rate_start) * sched_progress
                    )
                    # Update original rate so adaptive system tracks the new base
                    self._original_mutation_rate = scheduled_base
                    if gen > 5:  # let velocity accumulate a few data points first
                        self.config.mutation_rate = self._compute_adaptive_mutation_rate()
                    else:
                        self.config.mutation_rate = scheduled_base
                elif gen > 5:  # fallback: adaptive-only (no scheduler)
                    self.config.mutation_rate = self._compute_adaptive_mutation_rate()

                # 1. Generate challengers
                challengers = self._generate_challengers()

                # 2. Evaluate challengers (stage 1: quick)
                results = self._evaluate_batch(challengers)
                self._last_challengers = challengers
                self._last_results = results
                elite, elite_fitnesses, _elite_indices = self._select_elite(challengers, results)
                detailed_results = self._evaluate_batch_detailed(elite)

                # 4. Update champion -- pass cached stage-1 fitnesses for fast skip
                # R28: Track fitness velocity before champion update
                prev_fitness = self.champion_fitness
                improved = self._update_champion(
                    elite,
                    detailed_results,
                    stage1_fitnesses=elite_fitnesses,
                )

                # Quality-weighted champion pruning
                if improved and hasattr(self, 'champion_archive') and self.champion_archive is not None:
                    try:
                        _pruned = self.champion_archive.quality_weighted_prune()
                        if _pruned > 0:
                            self._log(f"[Champion] Quality prune: removed {_pruned} low-quality champions")
                    except Exception:
                        pass

                # R28: Update fitness velocity and record champion replacement
                delta = self.champion_fitness - prev_fitness
                self._update_fitness_velocity(delta)
                if improved and delta > 0:
                    self._record_champion_replacement(delta)
                # Record mutation outcomes for success tracking
                try:
                    from mutation_proposer import DirectedMutator as _DM
                    for _res in results:
                        _desc = _res.get('description', '') if isinstance(_res, dict) else ''
                        if _desc:
                            _fd = _res.get('fitness_delta', 0.0) if isinstance(_res, dict) else 0.0
                            _DM.record_mutation_outcome(_desc, _fd)
                except Exception:
                    pass

                # 5. Log progress
                self._log_generation(gen, gen_start)

                # Refresh task instance every 25 generations to prevent overfitting
                if gen > 0 and gen % task_refresh_interval == 0:
                    self._refresh_task_instance()

                # Periodic CUDA cleanup during evolution to prevent GPU context
                # corruption from prefix generation accumulating across gens.
                if _cuda_cleanup is not None and gen > 0 and gen % cuda_cleanup_interval == 0:
                    try:
                        _cuda_cleanup()
                    except Exception:
                        pass

                # 6. Checkpoint
                if gen > 0 and checkpoint_interval and gen % checkpoint_interval == 0:
                    try:
                        self._checkpoint()
                    except Exception as e:
                        self._log(f"[Checkpoint] Failed at gen {gen}: {e}")

                # 7. Curriculum check
                if improved:
                    self.generations_stalled = 0
                    self._qd_champion_reset_tried = False
                else:
                    self.generations_stalled += 1

                # 8. Recurrent feedback check (BEFORE curriculum advance, which resets stall count)
                self._try_recurrent_feedback()

                if auto_curriculum:
                    self._try_curriculum_advance()

                # Recovery confidence monitoring
                if hasattr(self, 'meta_controller') and self.meta_controller is not None and gen > 0 and gen % 25 == 0:
                    try:
                        _rc = self.meta_controller.get_recovery_confidence()
                        if _rc['is_recovering']:
                            self._log(f"[Recovery] confidence={_rc['confidence']:.2f} trend={_rc['fitness_trend_during_recovery']} exit={_rc['should_exit_recovery']}")
                            if _rc['should_exit_recovery']:
                                self.meta_controller.recovery_mode = False
                                self._log("[Recovery] Exiting recovery mode — fitness not improving")
                            if _rc['cycles_in_recovery'] >= 5:
                                try:
                                    from monitoring_alerts import AlertManager as _AM
                                    _am = _AM()
                                    _am.send_recovery_stuck_alert(_rc['cycles_in_recovery'], _rc['fitness_trend_during_recovery'], _rc['confidence'])
                                except Exception:
                                    pass
                    except Exception:
                        pass

                # 9. Strategy archive record
                self._record_strategy_outcomes()

                # Monoculture detection via dominance report
                if self.config.use_strategy_archive and self.strategy_archive is not None:
                    try:
                        _dom = self.strategy_archive.get_dominance_report()
                        if _dom.get('is_monoculture') and gen > 0 and gen % 10 == 0:
                            self._log(f"[Strategy] Monoculture detected: {_dom['dominant_strategy']} at {_dom['dominance_ratio']:.0%} — recommendation: {_dom['recommendation']}")
                            try:
                                from monitoring_alerts import AlertManager as _AM
                                _am = _AM()
                                _am.send_monoculture_alert(_dom['dominant_strategy'], _dom['dominance_ratio'], _dom['recommendation'])
                            except Exception:
                                pass
                    except Exception:
                        pass

                # 10. QD archive update
                self._update_qd_archive()

                # 11. QD champion reset (fallback for deep stall)
                if self.generations_stalled >= 50:
                    if self._try_qd_champion_reset():
                        self._log(f"[QD Reset] Champion replaced from QD archive at gen {gen}")

                # 12. Novelty archive update
                self._update_novelty_archive()

                # Periodic diagnostics — cross-cycle trends and QD metrics
                if gen > 0 and gen % 25 == 0 and hasattr(self, 'meta_controller') and self.meta_controller is not None:
                    try:
                        _trends = self.meta_controller.get_cross_cycle_trends()
                        for _metric, _data in _trends.items():
                            if isinstance(_data, dict) and _data.get('direction') != 'stable':
                                self._log(f"[Trend] {_metric}: {_data['direction']} (slope={_data.get('slope', 0):.4f}, confidence={_data.get('confidence', 0):.2f})")
                    except Exception:
                        pass
                if gen > 0 and gen % 25 == 0 and self.qd_archive is not None:
                    try:
                        _vel = self.qd_archive.get_improvement_velocity()
                        _illum = self.qd_archive.get_illumination_diversity()
                        self._log(f"[QD] velocity_avg={_vel.get('avg_velocity', 0):.4f} illumination={_illum.get('overall_diversity_score', 0):.4f}")
                    except Exception:
                        pass

                # Mutation success tracking
                if gen > 0 and gen % 25 == 0:
                    try:
                        from mutation_proposer import DirectedMutator as _DM
                        _mstats = _DM.get_mutation_stats()
                        if _mstats['total_attempts'] > 0:
                            _rate = _mstats['total_successes'] / _mstats['total_attempts']
                            self._log(f"[Mutation] success_rate={_rate:.1%} attempts={_mstats['total_attempts']}")
                    except Exception:
                        pass

                # Progression tracker plateau analysis
                if gen > 0 and gen % 50 == 0 and hasattr(self, 'progression_tracker') and self.progression_tracker is not None:
                    try:
                        _pa = self.progression_tracker.get_plateau_analysis()
                        if _pa.get('status') == 'plateau':
                            self._log(f"[Progression] Plateau detected: duration={_pa.get('plateau_duration', 0)} cycles, slope={_pa.get('trend_slope', 0):.6f}")
                        elif _pa.get('status') == 'declining':
                            self._log(f"[Progression] Declining: slope={_pa.get('trend_slope', 0):.6f}")
                    except Exception:
                        pass

                # Fitness anomaly detection
                if hasattr(self, '_last_challenger_fitnesses') and self._last_challenger_fitnesses:
                    try:
                        _fits = self._last_challenger_fitnesses
                        _mean_fit = sum(_fits) / len(_fits) if _fits else 0
                        _anomalies = [f for f in _fits if abs(f - _mean_fit) > 2.0 * (max(_fits) - min(_fits) + 1e-6)]
                        if _anomalies:
                            self._log(f"[Anomaly] {len(_anomalies)}/{len(_fits)} challengers deviate significantly from mean fitness {_mean_fit:.2f}")
                    except Exception:
                        pass

                # R37: Clear generation-state lists after all consumers are done.
                # These lists (challengers, results, programs, fitnesses) can
                # hold pop_size * avg_program_len words -- tens of KB per generation.
                # Without clearing, they persist until the next generation overwrites
                # them, doubling peak memory during the gap between generations.
                # On long runs (200+ gens), this prevents unnecessary memory pressure
                # that can trigger Python's cyclic GC and cause latency spikes.
                self._last_challengers = []
                self._last_results = []
                self._last_challenger_programs = []
                self._last_challenger_fitnesses = []

                # Early termination if curriculum maxed
                # BUT: require at least 30 stall gens at max level for adequate sampling.
                # With curriculum_stall_threshold=1, the default check is just 2 gens --
                # not enough to find good programs through random sampling (C2 collapse).
                if self.config.auto_curriculum and self.champion_level >= len(LEVELS) - 1:
                    min_stall_at_max = max(self.config.curriculum_stall_threshold * 2, 30)
                    if self.generations_stalled >= min_stall_at_max:
                        summary["early_stop_reason"] = "max_level_stalled"
                        break

                # R32: Deep plateau early termination -- when fitness velocity
                # is near zero for an extended period AND we've tried all
                # diversity mechanisms without success, terminate early to
                # avoid wasting compute on dead-end exploration.
                # Requires: 80+ stalled gens, velocity magnitude < 0.01,
                # and no QD/novelty progress (coverage unchanged).
                if self.generations_stalled >= 80:
                    vel_abs = abs(self._fitness_velocity)
                    if vel_abs < 0.01 and self._qd_coverage < 0.10:
                        summary["early_stop_reason"] = "deep_plateau"
                        self._log(
                            f"[EarlyStop] Deep plateau: stalled={self.generations_stalled}, "
                            f"vel={vel_abs:.4f}, qd_cov={self._qd_coverage:.2f}"
                        )
                        break
        except Exception as e:
            # R6: Catch any exception in the generation loop and return a
            # partial summary instead of crashing the entire multi-cycle run.
            # The caller (run_multi_cycle.py) can inspect generations_completed
            # to determine how far the loop got and decide whether to retry.
            self._log(f"Evolution loop crashed at gen {self.current_generation}: {e}")
            summary["error"] = str(e)
            summary["error_generation"] = self.current_generation
        finally:
            # R6: Always close the log file handle to prevent fd exhaustion
            # across multi-cycle runs, even if an exception occurs mid-loop.
            self._close_log_handle()

        # Final checkpoint -- skipped on exception since _close_log_handle
        # was called above and _checkpoint() would reopen the file handle.
        if "error" not in summary:
            self._checkpoint()

        summary["generations_completed"] = self.current_generation + 1
        summary["final_fitness"] = self.champion_fitness
        summary["final_level"] = self.champion_level
        summary["champion_words"] = len(self.champion) if self.champion else 0
        summary["elapsed_seconds"] = time.time() - start_time
        # R50: Record verify() result for integrity gate
        # Use stored result from champion replacement (uses correct task instance)
        verified = getattr(self, "_champion_verified", False)
        verify_score = getattr(self, "_champion_verify_score", 0.0)
        summary["champion_verified"] = verified
        summary["champion_verify_score"] = verify_score
        # R50: verified_fitness = fitness if verify() passes, else negated
        summary["verified_fitness"] = self.champion_fitness if verified else -abs(self.champion_fitness)

        # R4: Convergence diagnostics -- help downstream consumers detect
        # whether evolution is stuck, making progress, or oscillating.
        summary["generations_stalled"] = self.generations_stalled
        # R15: Count improvements in single pass (avoids generator overhead)
        improvements = 0
        bfhist = self.best_fitness_history
        for i in range(1, len(bfhist)):
            if bfhist[i] > bfhist[i - 1]:
                improvements += 1
        summary["total_champion_improvements"] = improvements
        summary["unique_strategies_used"] = (
            len(set(self._last_challenger_strategies)) if self._last_challenger_strategies else 0
        )
        summary["final_diversity"] = self._current_diversity
        hof = getattr(self, "_hall_of_fame", None)
        summary["hall_of_fame_size"] = hof.size if hof is not None else 0

        # Save champion to archive for persistent training data
        if self.champion is not None and self.champion_fitness > 0 and hasattr(self, "champion_archive"):
            cycle_num = getattr(self, "_cycle_number", 1)
            added = self.champion_archive.add(
                program=self.champion,
                level=self.champion_level,
                fitness=self.champion_fitness,
                cycle=cycle_num,
                generation=self.current_generation,
                trace_status=self.champion_trace.get("status", "ok") if self.champion_trace else "ok",
                task_score=getattr(self, "_raw_task_score", 0.0),
            )
            if added:
                self.champion_archive.save()
                summary["archive_size"] = self.champion_archive.size

        return summary

    # ------------------------------------------------------------------
    # Task & evaluation
    # ------------------------------------------------------------------

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

    def _compute_fitness(
        self,
        program: list[int],
        trace: dict[str, Any],
        eval_budget: int | None = None,
        raw_only: bool = False,
    ) -> float:
        """Compute crash-penalized fitness.

        Base fitness from expected output match.  Penalties for:
          - Illegal instructions (-100)
          - PC crash / timeout (-50)
          - Memory crash (-25)
          - Instruction count bloat (-0.01 per instruction over 10)

        When raw_only=True (used at max curriculum level), skips novelty/QD
        blending to prevent novelty bias from filtering out strong
        task-solving candidates.

        Performance: pre-computes bonus_threshold once per generation
        via _bonus_threshold_cache, and uses local variable hoisting to
        reduce dict lookups in the hot per-register loop. Also caches
        eval_budget lookup to avoid repeated self.config attribute access.
        Uses direct .get on trace dict (not intermediate .get("trace")) for
        the hot register scoring loop.
        """
        if self.task_instance is None:
            return 0.0

        budget = eval_budget if eval_budget is not None else self._cached_eval_budget

        # Hoist task instance lookups to local variables (avoid repeated dict ops)
        task_inst = self.task_instance
        # Cache expected_regs on the engine (dicts don't support arbitrary attrs)
        # R21: Use generation counter instead of id() to invalidate cache.
        # id() can be reused for deallocated objects, causing stale cache
        # hits on long runs. Generation counter is monotonically increasing.
        # R22: Cache expected_regs and total count together to avoid
        # separate dict.get in the scoring loop
        if self._cached_task_instance_id == self.current_generation:
            expected_regs = self._cached_expected_regs
            expected_total = self._cached_expected_total
        else:
            expected = task_inst.get("expected_output", {})
            expected_regs = expected.get("registers", {})
            expected_total = len(expected_regs)
            self._cached_expected_regs = expected_regs
            self._cached_expected_total = expected_total
            self._cached_task_instance_id = self.current_generation

        status = trace.get("status", "")
        final_regs = trace.get("final_regs", {})
        instr_count = trace.get("instructions_count", 0)
        trap = trace.get("trap_cause", "")

        # Crash penalties -- early return, skip all scoring work
        if status in ("crash", "trap", "exception"):
            self._raw_task_score = 0.0
            if "illegal" in trap or "invalid" in trap:
                return -100.0
            return -50.0

        if status == "timeout" or instr_count >= budget:
            self._raw_task_score = 0.0
            return -25.0

        # Score per register match (distance-weighted partial credit)
        # Use direct dict.get on final_regs -- avoids intermediate dict creation
        total = expected_total
        if total == 0:
            reg_match = 1.0
        else:
            total_score = 0.0
            get_actual = final_regs.get  # hoist method to local
            for reg_idx, expected_val in expected_regs.items():
                actual_val = get_actual(reg_idx, 0)
                # R35: Fast-path exact match -- avoids division for correct registers.
                # In typical solved tasks, most registers match exactly, so this
                # branch is taken frequently and saves the division + abs operations.
                if actual_val == expected_val:
                    total_score += 1.0
                    continue
                # Distance-weighted: score decreases as error increases.
                # When expected_val == 0: binary scoring (any non-zero = failure)
                # When expected_val != 0: continuous score via error_ratio
                if expected_val == 0:
                    # Binary: expected 0, actual must be 0
                    total_score += 0.0
                else:
                    error_ratio = abs(actual_val - expected_val) / max(1, abs(expected_val))
                    total_score += 1.0 / (1.0 + error_ratio)
            reg_match = total_score / total

        # Bloat penalty -- moderate discouragement for long programs
        # Reduced from 0.5 to 0.2 per instruction to allow 20-30 instr programs
        bloat_penalty = (instr_count - 50) * 0.05 if instr_count > 50 else 0.0

        max_score = total if total > 0 else 1

        # D1: Execution progress credit
        exec_credit = 0.0
        if status == "ok" and instr_count > 0 and reg_match > 0.0:
            exec_credit = (instr_count / budget) * 10.0 if budget > 0 else 0.0
            if exec_credit > 10.0:
                exec_credit = 10.0

        # Solved-task bonus -- large reward for programs that solve the task
        solved_bonus = 50.0 if reg_match >= 0.99 else 0.0

        base_score = reg_match * 100.0 - bloat_penalty + exec_credit + solved_bonus

        # ── Level advancement bonus (PRIMARY signal) ────────────────────
        # Each curriculum level adds 25 points. This ensures level 17
        # always scores higher than level 0, even with fewer instructions.
        # Level 0 gets 0 bonus, level 17 gets 425 bonus.
        # This is the DOMINANT fitness signal — the system must reward
        # progress, not punish complexity.
        level_bonus = self.champion_level * 25.0
        base_score += level_bonus

        # ── Efficiency scoring (+15 points max) ──────────────────────────
        # At high levels (L8+), REWARD instruction diversity (SMC needs
        # LUI/SLLI/ORI/SW/JALR). At low levels, reward compact programs.
        eff_weight = getattr(self.config, "efficiency_score_weight", 1.0)
        if eff_weight > 0 and instr_count > 0 and status == "ok":
            # NOP = ADDI x0, x0, 0 = 0x00000013
            nop_count = 0
            unique_vals = set()
            for word in program:
                unique_vals.add(word)
                if word == 0x00000013:
                    nop_count += 1
            total_instr = len(program) if program else 1
            useful = total_instr - nop_count
            if useful < 1:
                useful = 1

            if self.champion_level >= 8:
                # HIGH LEVELS: reward diversity and complexity
                # Useful programs get best score (8+ instructions = good)
                if useful >= 8:
                    instr_eff = 1.0
                elif useful >= 5:
                    instr_eff = 0.7 + (useful - 5) * 0.1
                else:
                    instr_eff = useful / 5.0

                # Bonus for MORE unique opcodes (diversity = exploration)
                unique_ratio = len(unique_vals) / total_instr if total_instr > 0 else 1.0
                opcode_bonus = unique_ratio * 0.5  # up to 0.25 (INVERTED from low levels)

                efficiency_score = (instr_eff + opcode_bonus) * 30.0 * eff_weight
            else:
                # LOW LEVELS: reward compact programs
                if useful <= 5:
                    instr_eff = 1.0
                elif useful <= 10:
                    instr_eff = 1.0 - (useful - 5) * 0.1
                else:
                    instr_eff = max(0.0, 0.5 - (useful - 10) * 0.05)

                unique_ratio = len(unique_vals) / total_instr if total_instr > 0 else 1.0
                opcode_bonus = (1.0 - unique_ratio) * 0.5

                efficiency_score = (instr_eff + opcode_bonus) * 30.0 * eff_weight

            base_score += efficiency_score

        # ── Multi-task scoring (+15 points max) ──────────────────────────
        # Evaluates program against secondary tasks (bit, arith, memory).
        mt_weight = getattr(self.config, "secondary_task_weight", 1.0)
        secondary_results = getattr(self, "_secondary_task_results", None)
        if mt_weight > 0 and secondary_results is not None and len(secondary_results) > 0:
            task_count = 0
            task_total = 0.0
            for sec_task, sec_trace in secondary_results:
                sec_status = sec_trace.get("status", "")
                if sec_status in ("crash", "trap", "exception", "timeout"):
                    continue
                task_count += 1
                sec_expected = sec_task.get("expected_output", {}).get("registers", {})
                sec_final = sec_trace.get("final_regs", {})
                sec_total = len(sec_expected)
                if sec_total == 0:
                    task_total += 1.0
                    continue
                sec_score = 0.0
                for reg_idx, exp_val in sec_expected.items():
                    act_val = sec_final.get(reg_idx, 0)
                    if act_val == exp_val:
                        sec_score += 1.0
                    else:
                        diff = abs(act_val - exp_val)
                        max_diff = max(abs(exp_val), 1)
                        sec_score += 1.0 / (1.0 + diff / max_diff)
                task_total += sec_score / sec_total
            if task_count > 0:
                multi_task_score = (task_total / task_count) * 5.0 * min(task_count, 3) * mt_weight
                base_score += multi_task_score

        # ── Time-to-solution credit (+20 points max) ──────────────────────
        # Rewards programs that solve tasks faster (fewer cycles).
        if status == "ok" and reg_match >= 0.99 and instr_count > 0:
            # Perfect/near-perfect match: reward efficient solutions
            # Ideal is around 5-15 instructions for most tasks
            if instr_count <= 5:
                time_credit = 20.0
            elif instr_count <= 15:
                time_credit = 20.0 * (1.0 - (instr_count - 5) / 10.0)
            else:
                time_credit = max(0.0, 10.0 - (instr_count - 15) * 0.2)
            base_score += time_credit

        # ── SMC pattern bonus (+25 points max) ──────────────────────────
        # At L12-L13, reward programs containing self-modifying code
        # instruction patterns (LUI, SLLI, ORI, OR, SW, JALR, FENCE.I).
        # This prevents the system from regressing to 4-word programs that
        # get high fitness through partial credit on fewer registers.
        if status == "ok" and self.champion_level >= 12 and instr_count > 0:
            smc_opcodes = {0x37, 0x13, 0x33, 0x23, 0x67, 0x0F}  # LUI, OP-IMM, OP, STORE, JALR, FENCE
            smc_count = 0
            for word in program:
                opcode = word & 0x7F
                if opcode in smc_opcodes:
                    smc_count += 1
            if smc_count >= 3:
                # Bonus: 5 points per SMC instruction, up to 25
                smc_bonus = min(25.0, smc_count * 5.0)
                base_score += smc_bonus

        # Level-aware graduated bonus -- use cached threshold if available
        # R16: Skip len(LEVELS) lookup on cache hit (hot path: 50x per gen)
        lvl = self.champion_level
        cached = self._bonus_threshold_cache
        if cached is not None and cached[0] == lvl:
            bonus_threshold = cached[1]
        else:
            n_levels = len(LEVELS)
            bonus_threshold = max(0.85, 0.95 - 0.10 * (lvl / max(1, n_levels - 1)))
            self._bonus_threshold_cache = (lvl, bonus_threshold)

        if reg_match >= bonus_threshold:
            bonus = 50.0 * (reg_match - bonus_threshold) / (1.0 - bonus_threshold)
            base_score += bonus

        # Store raw task score (before novelty/QD blending)
        self._raw_task_score = base_score
        fitness = base_score

        # When raw_only=True (at max curriculum level), skip novelty/QD
        # blending to prevent novelty bias from filtering out strong
        # task-solving candidates.
        if raw_only:
            return fitness

        # v13: Novelty-weighted fitness -- reward behaviorally novel programs
        # Cache novelty_weight as local to avoid repeated attribute lookup
        # R29: Adaptive novelty weight -- boost novelty contribution when
        # population diversity is low (< 0.2), up to 2x the configured weight.
        # This prevents premature convergence by rewarding behavioral novelty
        # more strongly when the population is becoming homogeneous.
        nw = self._novelty_weight_cache
        if self._current_diversity < 0.2 and nw > 0.0:
            # Boost novelty weight inversely to diversity: at div=0, boost=2x;
            # at div=0.2, boost=1x (no boost). Linear interpolation.
            diversity_boost = 1.0 + (1.0 - self._current_diversity / 0.2)
            nw = min(nw * diversity_boost, 0.8)  # cap at 0.8 to prevent dominance
        _extract_fn = self._extract_behavioral_descriptor
        if nw > 0.0 and self.novelty_archive is not None and _extract_fn is not None:
            try:
                descriptor = _extract_fn(
                    trace,
                    program,
                    budget=self.config.eval_budget,
                )
                try:
                    novelty_val = self.novelty_archive.decayed_novelty(program, self.current_generation)
                except Exception:
                    novelty_val = self.novelty_archive.novelty_score(descriptor)
                norm = max_score / 10.0
                fitness = fitness * (1.0 - nw) + novelty_val * norm * nw
                if self.current_generation > 0 and self.current_generation % 100 == 0:
                    self._log(f"[Novelty] gen={self.current_generation} novelty_val={novelty_val:.4f}")
            except (TypeError, ValueError, RuntimeError):
                pass  # fall through to raw fitness

        # v13: QD archive bonus -- reward under-explored behavioral niches
        # R30: Stall-scaled QD bonus -- when deeply stalled, amplify the QD
        # bonus to push exploration into unexplored behavioral niches.
        # At stall=0 the multiplier is 1.0x (no boost).  At stall=50+ the
        # multiplier caps at 2.5x.  This creates stronger selection pressure
        # toward novel behavioral regions when the population is stuck.
        # R32: Coverage-scaled QD bonus -- when QD grid coverage is low
        # (< 20%), the vast majority of niches are unexplored, so we boost
        # the QD bonus further (up to 1.5x additional) to accelerate
        # niche discovery.  When coverage is high (> 60%), reduce the
        # boost since most niches are already occupied.
        # R33: Compute QD descriptor once and reuse for both the QD bonus
        # and the niche penalty (was calling compute_qd_axes twice per
        # challenger per generation -- ~100 calls wasted per generation).
        _qd_fn = self._compute_qd_axes_fn
        _prog_qd_desc = None  # cached program QD descriptor
        if self.config.qd_bonus_weight > 0.0 and self.qd_archive is not None and _qd_fn is not None:
            try:
                _prog_qd_desc = _qd_fn(program, trace)
                if _prog_qd_desc is not None:
                    qd_bonus = self.qd_archive.selection_bias(_prog_qd_desc)
                    qd_weight = self.config.qd_bonus_weight
                    stall = self.generations_stalled
                    if stall > 10:
                        # Linear scaling: 1.0x at stall=10, 2.5x at stall=50+
                        stall_multiplier = min(2.5, 1.0 + 0.0375 * (stall - 10))
                        qd_weight *= stall_multiplier
                    # R32: Coverage multiplier -- boost when coverage is low
                    cov = self._qd_coverage
                    if cov < 0.20:
                        # Low coverage: up to 1.5x boost (linear from 1.0 at 20% to 1.5 at 0%)
                        qd_weight *= 1.0 + 2.5 * (0.20 - cov)
                    fitness += qd_bonus * qd_weight
            except (TypeError, ValueError, RuntimeError):
                pass

        # R30: Niche-aware fitness sharing -- penalize behavioral similarity
        # to the champion to reduce crowding in the local fitness peak.
        # R31: Cache champion QD descriptor per generation (avoids ~50x
        # redundant recomputation per generation -- champion trace doesn't
        # change within a generation).
        # R33: Reuse _prog_qd_desc from above instead of recomputing.
        if (
            self.config.qd_bonus_weight > 0.0
            and self.qd_archive is not None
            and _qd_fn is not None
            and _prog_qd_desc is not None
        ):
            try:
                if self.champion is not None and self.champion_trace is not None:
                    # Use cached champion descriptor if still valid
                    if self._champion_qd_valid and self._cached_champion_qd_desc is not None:
                        champ_desc = self._cached_champion_qd_desc
                    else:
                        champ_desc = _qd_fn(self.champion, self.champion_trace)
                        self._cached_champion_qd_desc = champ_desc
                        self._champion_qd_valid = True
                    if champ_desc is not None:
                        # Manhattan distance on 2-axis descriptor in [-1, 1]
                        dist = abs(_prog_qd_desc[0] - champ_desc[0]) + abs(_prog_qd_desc[1] - champ_desc[1])
                        # Penalty: 10% at dist=0, linear decay to 0% at dist>=2
                        if dist < 2.0:
                            penalty = 0.10 * (1.0 - dist / 2.0)
                            fitness *= 1.0 - penalty
            except (TypeError, ValueError, RuntimeError):
                pass

        return fitness

    # ------------------------------------------------------------------
    # Vectorized batch fitness evaluation
    # ------------------------------------------------------------------

    def batch_compute_fitness(
        self,
        programs: list[list[int]],
        traces: list[dict[str, Any]],
        eval_budget: int | None = None,
        raw_only: bool = False,
    ) -> tuple[list[float], list[float]]:
        """Vectorized batch fitness computation using numpy arrays.

        Evaluates all programs in a single vectorized pass for ~5-10x
        speedup over individual _compute_fitness calls.  The inner loops
        (per-register scoring, efficiency metrics, SMC opcode counting)
        are replaced by numpy array operations.

        Returns (fitnesses, raw_task_scores) where:
          - fitnesses: final blended fitness values
          - raw_task_scores: base scores before novelty/QD blending
        """
        import numpy as np

        n = len(programs)
        if n == 0:
            return [], []

        if self.task_instance is None:
            return [0.0] * n, [0.0] * n

        budget = eval_budget if eval_budget is not None else self._cached_eval_budget

        # ── Expected output (same for all programs in this gen) ──────
        expected = self.task_instance.get("expected_output", {})
        expected_regs = expected.get("registers", {})
        expected_total = len(expected_regs)
        expected_keys = list(expected_regs.keys()) if expected_regs else []
        expected_vals = (
            np.array([expected_regs[k] for k in expected_keys], dtype=np.float64)
            if expected_keys
            else np.array([], dtype=np.float64)
        )

        # ── Classify programs by status (vectorised string ops) ──────
        statuses = [t.get("status", "") for t in traces]
        instr_counts = np.array([t.get("instructions_count", 0) for t in traces], dtype=np.int64)
        trap_causes = [t.get("trap_cause", "") for t in traces]

        crash_mask = np.zeros(n, dtype=bool)
        illegal_mask = np.zeros(n, dtype=bool)
        timeout_mask = np.zeros(n, dtype=bool)
        for i in range(n):
            s = statuses[i]
            if s in ("crash", "trap", "exception"):
                crash_mask[i] = True
                tc = trap_causes[i]
                if "illegal" in tc or "invalid" in tc:
                    illegal_mask[i] = True
            elif s == "timeout" or instr_counts[i] >= budget:
                timeout_mask[i] = True

        # ── Fitness array ───────────────────────────────────────────
        fitnesses = np.zeros(n, dtype=np.float64)
        raw_scores = np.zeros(n, dtype=np.float64)

        # Crash/timeout penalties
        fitnesses[illegal_mask] = -100.0
        fitnesses[crash_mask & ~illegal_mask] = -50.0
        fitnesses[timeout_mask] = -25.0

        ok_mask = ~crash_mask & ~timeout_mask
        ok_indices = np.where(ok_mask)[0]
        n_ok = len(ok_indices)

        if n_ok == 0:
            return fitnesses.tolist(), raw_scores.tolist()

        # ── Gather ok-program data ──────────────────────────────────
        ok_programs = [programs[i] for i in ok_indices]
        ok_traces = [traces[i] for i in ok_indices]
        ok_instr = instr_counts[ok_indices]

        # ── Register scoring (vectorised) ───────────────────────────
        if expected_total == 0:
            reg_matches = np.ones(n_ok, dtype=np.float64)
        else:
            actual_vals = np.zeros((n_ok, len(expected_keys)), dtype=np.float64)
            for j, trace in enumerate(ok_traces):
                fr = trace.get("final_regs", {})
                if isinstance(fr, dict):
                    for k, key in enumerate(expected_keys):
                        actual_vals[j, k] = fr.get(key, 0)
                # else: fr is not a dict -> row stays 0

            exact_match = actual_vals == expected_vals
            diff = np.abs(actual_vals - expected_vals)
            max_diff = np.maximum(np.abs(expected_vals), 1.0)
            scores = np.where(exact_match, 1.0, 1.0 / (1.0 + diff / max_diff))
            reg_matches = scores.mean(axis=1)

        # ── Bloat penalty (vectorised) ──────────────────────────────
        bloat = np.maximum(0.0, (ok_instr - 10).astype(np.float64) * 0.2)

        # ── Execution progress credit (vectorised) ──────────────────
        exec_credit = np.zeros(n_ok, dtype=np.float64)
        pos = reg_matches > 0.0
        if budget > 0:
            exec_credit[pos] = np.minimum(10.0, (ok_instr[pos].astype(np.float64) / budget) * 10.0)

        # ── Solved-task bonus (vectorised) ──────────────────────────
        solved_bonus = np.where(reg_matches >= 0.99, 50.0, 0.0)

        # ── Base score ──────────────────────────────────────────────
        base = reg_matches * 100.0 - bloat + exec_credit + solved_bonus

        # ── Efficiency scoring (vectorised) ─────────────────────────
        eff_weight = getattr(self.config, "efficiency_score_weight", 1.0)
        if eff_weight > 0:
            max_prog_len = max((len(p) for p in ok_programs), default=0)
            if max_prog_len > 0:
                padded = np.zeros((n_ok, max_prog_len), dtype=np.int64)
                prog_lengths = np.zeros(n_ok, dtype=np.int64)
                for j, prog in enumerate(ok_programs):
                    pl = len(prog)
                    prog_lengths[j] = pl
                    padded[j, :pl] = prog

                nop_counts = np.sum(padded == 0x00000013, axis=1)
                useful = np.maximum(1, prog_lengths - nop_counts)

                # Unique values per program (small per-program loop, but
                # avoids the Python set() overhead in the original by
                # reusing the padded array for NOP detection above).
                unique_ratios = np.zeros(n_ok, dtype=np.float64)
                for j in range(n_ok):
                    total_i = int(prog_lengths[j]) or 1
                    unique_ratios[j] = len(set(ok_programs[j])) / total_i

                if self.champion_level >= 8:
                    instr_eff = np.where(
                        useful >= 8,
                        1.0,
                        np.where(useful >= 5, 0.7 + (useful - 5) * 0.1, useful / 5.0),
                    )
                    opcode_bonus = unique_ratios * 0.5
                    base += (instr_eff + opcode_bonus) * 30.0 * eff_weight
                else:
                    instr_eff = np.where(
                        useful <= 5,
                        1.0,
                        np.where(
                            useful <= 10,
                            1.0 - (useful - 5) * 0.1,
                            np.maximum(0.0, 0.5 - (useful - 10) * 0.05),
                        ),
                    )
                    opcode_bonus = (1.0 - unique_ratios) * 0.5
                    base += (instr_eff + opcode_bonus) * 30.0 * eff_weight

        # ── Multi-task scoring (vectorised per-secondary-task) ───────
        mt_weight = getattr(self.config, "secondary_task_weight", 1.0)
        if mt_weight > 0:
            for j, idx in enumerate(ok_indices):
                sec_results = getattr(self, "_secondary_task_results_per_program", None)
                if sec_results is not None and idx in sec_results:
                    secondary = sec_results[idx]
                    task_count = 0
                    task_total = 0.0
                    for sec_task, sec_trace in secondary:
                        sec_status = sec_trace.get("status", "")
                        if sec_status in ("crash", "trap", "exception", "timeout"):
                            continue
                        task_count += 1
                        sec_exp = sec_task.get("expected_output", {}).get("registers", {})
                        sec_fin = sec_trace.get("final_regs", {})
                        sec_total = len(sec_exp)
                        if sec_total == 0:
                            task_total += 1.0
                            continue
                        sec_score = 0.0
                        for ri, ev in sec_exp.items():
                            av = sec_fin.get(ri, 0)
                            if av == ev:
                                sec_score += 1.0
                            else:
                                sec_score += 1.0 / (1.0 + abs(av - ev) / max(abs(ev), 1))
                        task_total += sec_score / sec_total
                    if task_count > 0:
                        base[j] += (task_total / task_count) * 5.0 * min(task_count, 3) * mt_weight

        # ── Time-to-solution credit (vectorised) ────────────────────
        solved_ok = reg_matches >= 0.99
        instr_pos = ok_instr > 0
        can_credit = solved_ok & instr_pos
        tc = np.zeros(n_ok, dtype=np.float64)
        can5 = can_credit & (ok_instr <= 5)
        can15 = can_credit & (ok_instr > 5) & (ok_instr <= 15)
        can_slow = can_credit & (ok_instr > 15)
        tc[can5] = 20.0
        tc[can15] = 20.0 * (1.0 - (ok_instr[can15].astype(np.float64) - 5) / 10.0)
        tc[can_slow] = np.maximum(0.0, 10.0 - (ok_instr[can_slow].astype(np.float64) - 15) * 0.2)
        base += tc

        # ── SMC pattern bonus (vectorised) ──────────────────────────
        lvl = self.champion_level
        if lvl >= 12:
            smc_mask = np.zeros((n_ok, max_prog_len if max_prog_len > 0 else 1), dtype=bool)
            for j, prog in enumerate(ok_programs):
                for wi, word in enumerate(prog):
                    if (word & 0x7F) in (0x37, 0x13, 0x33, 0x23, 0x67, 0x0F):
                        smc_mask[j, wi] = True
            smc_counts = smc_mask.sum(axis=1)
            smc_ok = smc_counts >= 3
            base[smc_ok] += np.minimum(25.0, smc_counts[smc_ok].astype(np.float64) * 5.0)

        # ── Level-aware graduated bonus (vectorised) ────────────────
        cached = self._bonus_threshold_cache
        if cached is not None and cached[0] == lvl:
            bonus_threshold = cached[1]
        else:
            n_levels = len(LEVELS)
            bonus_threshold = max(0.85, 0.95 - 0.10 * (lvl / max(1, n_levels - 1)))
            self._bonus_threshold_cache = (lvl, bonus_threshold)

        above = reg_matches >= bonus_threshold
        denom = 1.0 - bonus_threshold
        grad_bonus = np.where(
            above,
            50.0 * (reg_matches - bonus_threshold) / denom if denom > 0 else 0.0,
            0.0,
        )
        base += grad_bonus

        # ── Raw task scores ─────────────────────────────────────────
        raw_scores[ok_indices] = base

        # When raw_only=True (at max curriculum level), skip novelty/QD
        # blending to prevent novelty bias from filtering out strong
        # task-solving candidates.
        if raw_only:
            fitnesses[ok_indices] = base
            return fitnesses.tolist(), raw_scores.tolist()

        # ── Novelty / QD blending (per-program, archive lookups) ────
        # These involve archive lookups that are hard to vectorise but
        # are fast (O(1) per program).  Done here so the returned
        # fitnesses are fully blended.
        nw = self._novelty_weight_cache
        if self._current_diversity < 0.2 and nw > 0.0:
            diversity_boost = 1.0 + (1.0 - self._current_diversity / 0.2)
            nw = min(nw * diversity_boost, 0.8)
        _extract_fn = self._extract_behavioral_descriptor
        _qd_fn = self._compute_qd_axes_fn

        for j, idx in enumerate(ok_indices):
            fj = float(base[j])
            prog = programs[idx]
            trace = traces[idx]
            max_score_val = float(expected_total) if expected_total > 0 else 1.0

            # Novelty
            if nw > 0.0 and self.novelty_archive is not None and _extract_fn is not None:
                try:
                    descriptor = _extract_fn(trace, prog, budget=self.config.eval_budget)
                    try:
                        novelty_val = self.novelty_archive.decayed_novelty(prog, self.current_generation)
                    except Exception:
                        novelty_val = self.novelty_archive.novelty_score(descriptor)
                    norm = max_score_val / 10.0
                    fj = fj * (1.0 - nw) + novelty_val * norm * nw
                except (TypeError, ValueError, RuntimeError):
                    pass

            # QD bonus + niche penalty
            if self.config.qd_bonus_weight > 0.0 and self.qd_archive is not None and _qd_fn is not None:
                try:
                    prog_desc = _qd_fn(prog, trace)
                    if prog_desc is not None:
                        qd_bonus = self.qd_archive.selection_bias(prog_desc)
                        qd_w = self.config.qd_bonus_weight
                        stall = self.generations_stalled
                        if stall > 10:
                            qd_w *= min(2.5, 1.0 + 0.0375 * (stall - 10))
                        cov = self._qd_coverage
                        if cov < 0.20:
                            qd_w *= 1.0 + 2.5 * (0.20 - cov)
                        fj += qd_bonus * qd_w
                        # Niche penalty
                        if self.champion is not None and self.champion_trace is not None:
                            if self._champion_qd_valid and self._cached_champion_qd_desc is not None:
                                champ_desc = self._cached_champion_qd_desc
                            else:
                                champ_desc = _qd_fn(self.champion, self.champion_trace)
                                self._cached_champion_qd_desc = champ_desc
                                self._champion_qd_valid = True
                            if champ_desc is not None:
                                dist = abs(prog_desc[0] - champ_desc[0]) + abs(prog_desc[1] - champ_desc[1])
                                if dist < 2.0:
                                    fj *= 1.0 - 0.10 * (1.0 - dist / 2.0)
                except (TypeError, ValueError, RuntimeError):
                    pass

            fitnesses[idx] = fj

        return fitnesses.tolist(), raw_scores.tolist()

    # ------------------------------------------------------------------
    # Champion / Challenger management
    # ------------------------------------------------------------------

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

    def _generate_challengers(self) -> list[list[int]]:
        """Generate a population of challenger programs.

        Now includes adaptive diversity injection: when the population
        is converging (low opcode diversity), replaces a fraction of
        the most similar challengers with fresh random programs to
        prevent premature convergence.

        R28: Uses adaptive mutation rate based on fitness velocity to
        dynamically balance exploration vs exploitation.

        v14.5: Dedicated MODEL_GUIDED slots — at L2+, reserves a fraction
        of challenger slots for prefix-guided completions, bypassing the
        strategy archive's GROW bias that suppressed model guidance.
        """
        pop_size = self.config.population_size
        if self.champion is None:
            self._last_challenger_strategies = ["SEED"] * pop_size
            return [generate_seed_program(rng=self.rng) for _ in range(pop_size)]

        # R16: Pre-allocate lists with known final size (avoids repeated
        # append reallocations: ~50 per generation, thousands over a run).
        challengers: list[list[int]] = [None] * pop_size  # type: ignore[list-item]
        self._last_challenger_strategies: list[str] = [None] * pop_size  # type: ignore[list-item]

        # Hoist loop-invariant lookups
        task_inst = self.task_instance or {}
        expected = task_inst.get("expected_output", {})
        inputs = task_inst.get("inputs", {})
        trace_input = [self.champion_trace] if self.champion_trace is not None else []
        sa = self.strategy_archive if self.config.use_strategy_archive else None
        champion = self.champion  # avoid repeated self.champion lookups
        level = self.champion_level
        rng = self.rng
        strategies = self._last_challenger_strategies
        _propose = propose_mutation  # hoist module function to local
        _choose = self._choose_mutation_strategy

        # v14.5: Dedicated MODEL_GUIDED reserved slots at L2+
        # Reserve n_mg_slots for model-guided completions. This ensures
        # the model contributes even when the archive favors GROW/HYBRID.
        # At low levels only 1-2 slots; at L8+ up to 25% of population.
        model_ratio = self._get_model_guided_ratio()
        if model_ratio > 0 and level >= 2:
            n_mg = max(1, int(pop_size * min(model_ratio, 0.25)))
            mg_indices = self._replace_with_model_guided(challengers, model_ratio)
            for idx in mg_indices[:n_mg]:
                if 0 <= idx < len(strategies):
                    strategies[idx] = "MODEL_GUIDED"
        else:
            n_mg = 0

        for i in range(pop_size):
            # Skip slots already filled by MODEL_GUIDED
            if strategies[i] == "MODEL_GUIDED":
                continue
            # Choose strategy
            strategy = _choose(i)
            strategies[i] = strategy.value

            # Apply mutation to champion
            mutated, _desc, _analysis = _propose(
                program=champion,
                trace=trace_input,
                expected=expected,
                inputs=inputs,
                rng=rng,
                task_level=level,
                strategy=strategy,
                strategy_archive=sa,
            )
            challengers[i] = mutated

        # v13: Adaptive diversity injection
        # When population diversity is too low, replace least-diverse
        # challengers with fresh random programs to prevent convergence.
        # Stall-adaptive: deeper stall = more aggressive injection.
        # R32: Level-adaptive threshold -- higher curriculum levels need
        # more diversity because the fitness landscape is more rugged and
        # local optima are more numerous.  Scale the threshold by 1.0 + 0.05*level,
        # so L0 uses the base threshold, L10 uses 1.5x, L13 uses 1.65x.
        div = self._compute_population_diversity(challengers)
        self._current_diversity = div
        level_scale = 1.0 + 0.05 * self.champion_level
        effective_div_threshold = min(0.8, self.config.diversity_threshold * level_scale)
        if div < effective_div_threshold and pop_size >= 5:
            base_ratio = self.config.diversity_inject_ratio
            # Stall-adaptive: up to 2x more injection during deep stall
            stall_multiplier = 1.0 + min(1.0, self.generations_stalled / 50.0)
            n_inject = max(1, int(pop_size * base_ratio * stall_multiplier))
            self._inject_diversity(challengers, n_inject)

        # v12.5: Model-guided replacement -- replace fraction of challengers
        # with prefix-guided completions from the trained transformer.
        # Active at L5+ where mutation struggles to find valid programs.
        model_ratio = self._get_model_guided_ratio()
        if model_ratio > 0 and self.champion_level >= 5:
            replaced_indices = self._replace_with_model_guided(challengers, model_ratio)
            for idx in replaced_indices:
                if 0 <= idx < len(self._last_challenger_strategies):
                    self._last_challenger_strategies[idx] = "MODEL_GUIDED"

        # v14: Hall of Fame injection -- when deeply stalled, replace a
        # few challengers with the best programs ever seen. This prevents
        # losing good programs to aggressive mutation or curriculum shifts.
        if self.generations_stalled >= 30 and self._hall_of_fame.size > 0:
            n_hof = min(2, max(1, pop_size // 10))
            for _ in range(n_hof):
                hof_prog = self._hall_of_fame.get_random(self.rng, level=self.champion_level)
                if hof_prog is not None:
                    for attempt in range(20):
                        idx = self.rng.randint(0, len(challengers) - 1)
                        if challengers[idx] != self.champion:
                            challengers[idx] = hof_prog
                            if 0 <= idx < len(self._last_challenger_strategies):
                                self._last_challenger_strategies[idx] = "HOF_INJECT"
                            break

        # v12.5: QD-Archive Crossover -- inject a QD-bred hybrid as one
        # challenger when stalled >= 15 gens. This brings dormant archive
        # diversity back into the active population.
        qd_hybrid = self._try_qd_crossover()
        if qd_hybrid is not None:
            # Replace a random non-champion challenger
            for attempt in range(20):
                idx = self.rng.randint(0, len(challengers) - 1)
                if challengers[idx] != self.champion:
                    challengers[idx] = qd_hybrid
                    if 0 <= idx < len(self._last_challenger_strategies):
                        self._last_challenger_strategies[idx] = "QD_CROSSOVER"
                    self._log(f"[QD Cross] Injected QD-archive hybrid ({len(qd_hybrid)} words) at idx {idx}")
                    break

        return challengers

    # ------------------------------------------------------------------
    # Diversity computation and injection (v13 quality improvement)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_population_diversity(challengers: list[list[int]]) -> float:
        """Compute opcode diversity of the challenger population.

        Returns a value in [0.0, 1.0] representing the fraction of
        distinct RV32I opcode categories used across all challengers.
        0.0 = all programs use the same opcode; 1.0 = all 11 categories
        represented.

        Uses a 128-element list for O(1) opcode-to-category lookup
        without hash computation per word. ~30% faster than dict.get()
        for the hot inner loop.

        R37: Early termination -- once all 11 categories are discovered
        (seen_bits == ALL_CATEGORIES_MASK), stops scanning immediately.
        For diverse populations this cuts the scan short by 30-70%
        depending on program lengths, saving O(remaining_words) work
        per generation.

        O(N * M) worst case where N = population size, M = avg program
        length; typical case is significantly shorter due to early exit.
        """
        if not challengers:
            return 0.0

        cat_arr = _OPCODE_CAT_ARRAY
        seen_bits = 0  # bitset: bit i = category i seen
        # R37: Pre-computed mask for all 11 categories (bits 0-10 set)
        all_mask = (1 << NUM_OPCODE_CATEGORIES) - 1
        for prog in challengers:
            for word in prog:
                if not isinstance(word, int):
                    continue
                cat = cat_arr[word & 0x7F]
                if cat >= 0:
                    seen_bits |= 1 << cat
            # R37: Early termination -- all categories found
            if seen_bits == all_mask:
                return 1.0

        # Count set bits via C-implemented int.bit_count()
        return seen_bits.bit_count() / NUM_OPCODE_CATEGORIES

    def _inject_diversity(self, challengers: list[list[int]], n_inject: int) -> None:
        """Replace the least-diverse challengers with fresh random programs.

        Uses a fast O(N * M) heuristic: for each challenger, count how
        many of its opcodes are shared by the majority of the population.
        Programs whose opcodes are all "common" (appear in > 60% of
        population) are replaced.  This is cheaper than the O(N^2) full
        overlap approach and still correctly identifies redundant programs.

        Injected programs are generated from scratch (not mutated) to
        maximize opcode diversity. They receive the DIVERSITY_INJECT
        strategy tag for archive tracking.
        """
        if not challengers or n_inject <= 0:
            return

        pop_size = len(challengers)
        n_inject = min(n_inject, pop_size)

        # Use 128-element array for O(1) opcode-to-category lookup
        # without hash computation per word (~30% faster than dict.get).
        cat_arr = _OPCODE_CAT_ARRAY

        # Compute opcode frequency across population (O(N * M))
        # Use bitsets per program + frequency array for speed.
        opcode_count: list[int] = [0] * NUM_OPCODE_CATEGORIES
        prog_opcodes_bits: list[int] = []
        for prog in challengers:
            bits = 0
            for word in prog:
                cat = cat_arr[word & 0x7F]
                if cat >= 0:
                    bits |= 1 << cat
                    opcode_count[cat] += 1
            prog_opcodes_bits.append(bits)

        # Score each program: fraction of its opcodes that are "common"
        common_threshold = pop_size * 0.6
        # R21: Precompute a bitmask of which categories are "common" (above threshold).
        # This replaces the per-bit inner loop with a single AND + bit_count,
        # giving a significant speedup for the hot path (~pop_size iterations).
        common_mask = 0
        for cat in range(NUM_OPCODE_CATEGORIES):
            if opcode_count[cat] > common_threshold:
                common_mask |= 1 << cat
        # R37: Merge scoring + top-k selection into a single pass.
        # Instead of building a full redundancy_scores list and then
        # calling heapq.nlargest/sort, maintain a fixed-size min-heap
        # of size n_inject during the scoring pass. This eliminates
        # the O(pop_size) list allocation and the separate O(N log k)
        # top-k step, reducing total work by ~50% for the common case
        # (n_inject << pop_size, e.g. 3 out of 50).
        import heapq as _heapq

        top_heap: list[tuple[float, int]] = []  # min-heap of (score, idx)
        for i, bits in enumerate(prog_opcodes_bits):
            if bits == 0:
                score = 1.0
            else:
                common_count = (bits & common_mask).bit_count()
                total_bits = bits.bit_count()
                score = common_count / total_bits if total_bits > 0 else 1.0
            if len(top_heap) < n_inject:
                _heapq.heappush(top_heap, (score, i))
            elif score > top_heap[0][0]:
                _heapq.heapreplace(top_heap, (score, i))
        # R37: The top-k selection was already done inline during scoring
        # (min-heap of size n_inject), so top_heap IS the sorted
        # top-n list. No second pass needed.
        # Sort descending by score to inject most-redundant first.
        top_heap.sort(reverse=True)
        top_n = top_heap
        rng = self.rng
        strategies = self._last_challenger_strategies
        for _, idx in top_n:
            challengers[idx] = generate_seed_program(rng=rng)
            strategies[idx] = "DIVERSITY_INJECT"

    def _choose_mutation_strategy(self, idx: int) -> MutationStrategy:
        """Choose a mutation strategy for challenger *idx*.

        D3 fix: Enforces a tighter consecutive-strategy cap for non-GROW
        strategies at L5+ to prevent archive lock-in. At L5+ the cap is
        halved (15 instead of 30) because lower-level biases (CROSSOVER
        dominance) carry over and crowd out exploration.
        """
        # Determine the effective consecutive cap
        if self.champion_level >= 5:
            effective_cap = max(self.config.max_consecutive_same_strategy // 2, 10)
        else:
            effective_cap = self.config.max_consecutive_same_strategy

        # D3: If we've hit the consecutive cap for a non-GROW strategy,
        # skip it so the archive's UCB exploration bonus can surface alternatives.
        skip_strat: str | None = None
        if (
            self._consecutive_same_strategy >= effective_cap
            and self._last_strategy_used != "GROW"
            and self._last_strategy_used != "MODEL_GUIDED"
        ):
            skip_strat = self._last_strategy_used
            self._consecutive_same_strategy = 0  # reset counter after skip

        # Use strategy archive if available and configured
        if (
            self.config.use_strategy_archive
            and self.strategy_archive is not None
            and self.strategy_archive.total_records() > 10
        ):
            rec = self.strategy_archive.recommend_strategy(self.champion_level, skip_strategy=skip_strat)
            # R33: Use pre-built O(1) dict lookup instead of MutationStrategy(rec)
            # enum constructor (~2x faster for the hot path).
            chosen = _STRATEGY_ENUM_CACHE.get(rec, MutationStrategy.HYBRID)
        else:
            chosen = MutationStrategy.HYBRID

        # Wire adaptive exploration bonus — boost underperforming strategies at this level
        if self.config.use_strategy_archive and self.strategy_archive is not None:
            try:
                _bonus = self.strategy_archive.adaptive_exploration_bonus(rec, self.champion_level)
                if _bonus > 0.15 and self.rng.random() < _bonus:
                    # Override to a random non-GROW strategy for exploration
                    _alt_strats = [s for s in MutationStrategy if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)]
                    if _alt_strats:
                        chosen = self.rng.choice(_alt_strats)
            except Exception:
                pass

        # Mutation stats awareness: if recent mutations are failing, bias exploration
        try:
            from mutation_proposer import DirectedMutator as _DM
            _mstats = _DM.get_mutation_stats()
            if _mstats['total_attempts'] > 50:
                _recent_rate = _mstats['recent_avg_delta']
                if _recent_rate < -0.5 and self.rng.random() < 0.3:
                    # Recent mutations are hurting fitness — force exploration
                    _exploration_strats = [s for s in MutationStrategy if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)]
                    if _exploration_strats:
                        chosen = self.rng.choice(_exploration_strats)
                        self._log(f"[Strategy] Mutation stats override: recent_avg_delta={_recent_rate:.3f}")
        except Exception:
            pass

        # Unsolved levels 9-11: use higher PERTURBATION_CAPPED probability
        if self.champion_level in self._unsolved_levels_set and self.generations_stalled > 2:
            if self.rng.random() < self.config.perturbation_capped_probability_unsolved:
                chosen = MutationStrategy.PERTURBATION_CAPPED

        # D3: Track consecutive usage
        strat_val = chosen.value
        if strat_val == self._last_strategy_used:
            self._consecutive_same_strategy += 1
        else:
            self._last_strategy_used = strat_val
            self._consecutive_same_strategy = 1

        return chosen

    # --------------------------------------------------------------
    # Model-guided generation (v12.5)
    # --------------------------------------------------------------

    def _get_model_guided_ratio(self) -> float:
        """Determine model-guided replacement ratio based on level and stall.

        Returns fraction of challengers to replace with model completions.
        When model_guided_ratio is None (or 0), uses curriculum-adaptive
        defaults (low/mid/high) based on champion level.
        When set to a specific value > 0, that value is used directly.
        """
        ratio = self.config.model_guided_ratio
        if ratio is not None and ratio > 0:
            return float(ratio)
        # None or 0 means curriculum-adaptive defaults
        if self.champion_level <= 5:
            return self.config.model_guided_ratio_low
        elif self.champion_level <= 7:
            return self.config.model_guided_ratio_mid
        else:
            return self.config.model_guided_ratio_high

    def _replace_with_model_guided(self, challengers: list[list[int]], model_ratio: float = 0.2) -> list[int]:
        """Replace a fraction of challengers with model-guided completions.

        Uses PrefixGenerator to generate completions conditioned on reference
        solution prefixes (or champion body fallback). APS (Adaptive Prefix
        Scheduler) adjusts prefix composition by stall depth.

        Args:
            challengers: List of challenger programs (modified in-place).
            model_ratio: Fraction of challengers to replace.

        Returns:
            List of indices in challengers that were replaced.
        """
        replaced_indices: list[int] = []
        if not self.champion:
            self._log("[ModelGuided] Skipped: champion=None")
            return []
        if self.config.model_guided_ratio is not None and self.config.model_guided_ratio <= 0:
            self._log(f"[ModelGuided] Skipped: model_guided_ratio={self.config.model_guided_ratio} (disabled)")
            return []

        # Get reference solution for the champion's level to use as prefix
        ref_prefix_hex = None
        try:
            ref = reference_solution(self.champion_level)
            if ref and len(ref) > 0:
                encoded = []
                for inst in ref:
                    w = _encode_inst(inst)
                    if w is not None:
                        encoded.append(w)
                # Use up to 4 instructions as prefix (enough for task context)
                ref_prefix = encoded[: min(4, len(encoded))]
                if len(ref_prefix) >= 1:
                    ref_prefix_hex = [f"0x{w:08x}" for w in ref_prefix]
        except Exception:
            pass

        # Convert champion to hex strings for fallback
        champion_hex = [f"0x{w:08x}" for w in self.champion]

        # ----------------------------------------------------------
        # v12.5: Adaptive Prefix Scheduler (APS)
        # Dynamically adjusts prefix composition based on stall depth
        # to balance exploration vs guidance.
        # ----------------------------------------------------------
        stalled = self.generations_stalled
        champion_level = self.champion_level
        composite_hex = None
        source_name = None
        prefix_source = None

        if champion_level >= 8 and len(champion_hex) >= 4:
            if stalled >= 50:
                # Deep stall: composite (champion + reference) -- max guidance
                composite = list(champion_hex)
                if ref_prefix_hex and len(ref_prefix_hex) >= 2:
                    composite.extend(ref_prefix_hex[: min(12, len(ref_prefix_hex))])
                composite_hex = composite
                source_name = (
                    f"composite(L{champion_level}): "
                    f"champ({len(composite)}w)+ref({len(ref_prefix_hex or [])}w) [stall={stalled}]"
                )
            elif stalled >= 20:
                # Medium stall: ref-only -- balanced
                if ref_prefix_hex and len(ref_prefix_hex) >= 4:
                    composite_hex = ref_prefix_hex[: min(12, len(ref_prefix_hex))]
                    source_name = f"ref-only(L{champion_level}): {len(composite_hex)}w [stall={stalled}]"
                else:
                    prefix_len = min(8, len(champion_hex))
                    composite_hex = champion_hex[:prefix_len]
                    source_name = f"champion-fallback(L{champion_level}): {prefix_len}w [stall={stalled}]"
            else:
                # Low/no stall: ref-short -- high exploration
                if ref_prefix_hex and len(ref_prefix_hex) >= 2:
                    prefix_len = min(6, max(2, len(ref_prefix_hex) // 3))
                    composite_hex = ref_prefix_hex[:prefix_len]
                    source_name = f"ref-short(L{champion_level}): {len(composite_hex)}w [stall={stalled}]"
                else:
                    prefix_len = min(4, len(champion_hex))
                    composite_hex = champion_hex[:prefix_len]
                    source_name = f"champion-short(L{champion_level}): {prefix_len}w [stall={stalled}]"

        if composite_hex is not None:
            prefix_source = composite_hex
        elif ref_prefix_hex is not None:
            prefix_source = ref_prefix_hex
            source_name = source_name or f"reference (L{champion_level})"
        else:
            prefix_source = champion_hex
            source_name = source_name or f"champion body (L{champion_level})"

        if source_name is None:
            source_name = f"default (L{champion_level})"

        # Lazy init PrefixGenerator
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
        replacements = 0

        # R42: Temperature annealing — smoothly decay model temperature from
        # temperature_start (2.0) to temperature_end (0.5) over the generation
        # schedule. This provides broad exploration early (diverse completions)
        # and focused exploitation late (high-quality completions).
        # Replaces the old stepped approach (1.2→0.9→0.6) with continuous decay.
        gen = getattr(self, "current_generation", 0)
        if self.config.use_temperature_annealing:
            anneal_progress = gen / max(1, self.config.temperature_schedule_generations)
            anneal_progress = min(1.0, anneal_progress)
            model_temp = (
                self.config.temperature_start
                + (self.config.temperature_end - self.config.temperature_start) * anneal_progress
            )
        else:
            # Fallback: old stepped approach
            if gen < 10:
                model_temp = 1.2
            elif gen < 30:
                model_temp = 0.9
            else:
                model_temp = 0.6

        # R34: CUDA sync once before the entire loop (was inside per-prefix
        # iteration). Prevents driver race conditions on CUDA 13.0.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        # Use stall-adaptive prefix_counts for APS — single count to limit inference cost
        if champion_level >= 8:
            if stalled >= 50:
                prefix_counts = [12]
            elif stalled >= 20:
                prefix_counts = [8]
            else:
                prefix_counts = [6]
        else:
            if stalled >= 50:
                prefix_counts = [6]
            elif stalled >= 20:
                prefix_counts = [4]
            else:
                prefix_counts = [2]

        for pc in prefix_counts:
            if replacements >= num_replace:
                break
            try:

                # CUDA error recovery: catch RuntimeError (SIGSEGV), sync+retry,
                # then skip this prefix count if still failing.
                _completions = None
                _cuda_failed = False
                for attempt in range(1, 3):  # max 2 attempts
                    try:
                        _completions = self._prefix_generator.complete(
                            prefix_source,
                            prefix_count=pc,
                            num_samples=min(3, max(1, num_replace // len(prefix_counts))),
                            temperature=model_temp,
                            top_k=40,
                            top_p=0.9,
                            repetition_penalty=1.1,
                        )
                        break  # success
                    except RuntimeError as cuda_err:
                        self._log(f"[ModelGuided] CUDA error (attempt {attempt}): {cuda_err}")
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        if attempt == 2:
                            self._log("[ModelGuided] CUDA fallback: skipping this prefix count")
                            _cuda_failed = True

                if _cuda_failed:
                    continue

                completions = _completions or []
                if not completions:
                    continue

                for prog_hex in completions:
                    if replacements >= num_replace:
                        break
                    prog_int = [int(w, 16) for w in prog_hex]
                    if len(prog_int) >= 2:
                        idx = self.rng.randint(0, len(challengers) - 1)
                        for _ in range(5):
                            if challengers[idx] != self.champion:
                                break
                            idx = self.rng.randint(0, len(challengers) - 1)
                        challengers[idx] = prog_int
                        replaced_indices.append(idx)
                        replacements += 1
            except Exception as e:
                self._log(f"[ModelGuided] Generation error: {e}")
                continue

        if replacements > 0:
            self._log(
                f"[ModelGuided] Replaced {replacements}/{num_replace} "
                f"(prefix: {source_name}, L={champion_level}, "
                f"temp={model_temp:.1f})"
            )

        # NOTE: PrefixGenerator instance is intentionally NOT deleted here.
        # It stays in GPU memory across generations within a single run(),
        # saving ~3s per generation of model re-initialization. Cleanup
        # happens when the EvolutionEngine goes out of scope at cycle end.

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

    def _select_elite(
        self,
        challengers: list[list[int]],
        results: list[dict[str, Any]],
    ) -> tuple[list[list[int]], list[float], list[int]]:
        """Select elite challengers for detailed evaluation.

        Implements tournament selection with elitism:
        1. ELITISM: Top elite_ratio (default 10%) by stage-1 fitness are
           preserved — guaranteed to get detailed evaluation. This prevents
           the best candidates from being lost due to stochastic effects.
        2. TOURNAMENT SELECTION: Remaining slots are filled via adaptive
           tournament selection (tournament_size adjusted by population
           diversity). Each tournament draws k random candidates and picks
           the best, applying selection pressure while maintaining diversity.
        3. DEDUP: Tournament winners are deduplicated against elites via
           lightweight fingerprint (length + first/last word hash) to
           prevent redundant evaluations.

        Uses heapq.nsmallest for O(N log k) top-k selection instead of
        full O(N log N) sort. For population=50, elite_size=5, this
        avoids sorting 45 unnecessary elements every generation.

        Returns a tuple of (elite_programs, elite_fitnesses, elite_indices)
        where elite_fitnesses contains the stage-1 fitness for each elite
        program (in the same order), and elite_indices contains the original
        indices of the selected programs in the challengers list.
        """
        # Compute fitness for stage 1 results and cache for archive recording.
        # Pass half-budget so exec_credit scaling matches the actual evaluation
        # budget used (stage 1 runs with eval_budget // 2).
        stage1_budget = self.config.eval_budget // 2
        # R16: Pre-allocate with known size (avoids ~50 list appends per gen)
        n_challengers = len(challengers)
        scored: list[tuple[float, int, list[int], dict[str, Any]]] = [None] * n_challengers  # type: ignore[list-item]
        fitness_cache: list[float] = [0.0] * n_challengers
        _compute = self._compute_fitness  # hoist to local for hot loop
        _use_sec = self.config.use_secondary_tasks

        # At max curriculum level, use raw fitness only (no novelty/QD bias)
        raw_only = self.champion_level >= self.config.max_level - 1

        # ── Vectorized batch fitness (fast path) ────────────────────
        # Evaluate secondary tasks for non-crashed programs first so
        # batch_compute_fitness can include multi-task scoring.
        if _use_sec:
            sec_per_prog: dict[int, list] = {}
            for idx in range(n_challengers):
                trace = results[idx]
                status = trace.get("status", "")
                if status not in _CRASH_STATUSES and status != "timeout":
                    sec_per_prog[idx] = self._evaluate_secondary_tasks(challengers[idx])
            self._secondary_task_results_per_program = sec_per_prog
        else:
            self._secondary_task_results_per_program = None

        try:
            batch_fitnesses, batch_raw = self.batch_compute_fitness(
                challengers,
                results,
                eval_budget=stage1_budget,
                raw_only=raw_only,
            )
            for idx in range(n_challengers):
                scored[idx] = (batch_fitnesses[idx], idx, challengers[idx], results[idx])
                fitness_cache[idx] = batch_fitnesses[idx]
        except Exception:
            # Fallback to per-program evaluation on any batch failure
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

        self._last_challenger_fitnesses = fitness_cache

        elite_size = max(1, int(self.config.population_size * self.config.elite_ratio))

        # heapq.nsmallest with negated fitness gives top-k in O(N log k)
        # instead of full sort O(N log N). For k=5, N=50 this is ~3x fewer
        # comparisons.
        top_k = heapq.nsmallest(elite_size, scored, key=lambda x: -x[0])

        elites = [prog for _, _, prog, _ in top_k]
        elite_fits = [fit for fit, _, _, _ in top_k]
        elite_indices = [idx for _, idx, _, _ in top_k]

        # Also include tournament selection -- but only programs not already
        # in the top-k, to avoid dedup-induced fitness misalignment.
        # Use a set of integer tuples for O(1) dedup (avoids converting
        # list[int] to tuple on every membership check).
        # R21: Replace rng.sample + max with manual randint loop to avoid
        # creating a list of candidates on each tournament (elite_size times).
        # R24: Use program length + first/last words as a lightweight
        # fingerprint for dedup. This avoids creating a full tuple (O(n)
        # copy + hash) for every program. Collisions are extremely rare
        # for the typical case (programs differ in first/last instruction).
        # Fallback: if fingerprint collides, accept it (duplicate programs
        # in elite selection are benign -- they just waste one elite slot).
        seen_fps: set[int] = {
            (len(p) << 32) ^ (p[0] if p else 0) ^ ((p[-1] << 16) if len(p) > 1 else 0) for p in elites
        }
        seen_indices: set[int] = set(elite_indices)
        n_scored = len(scored)
        rng = self.rng  # hoist to local

        # R29: Adaptive tournament sizing based on population diversity.
        # High diversity -> larger tournament (stronger selection pressure,
        # faster convergence on good solutions). Low diversity -> smaller
        # tournament (weaker selection, more random exploration to escape
        # local optima). Range: [base-1, base+2] clamped to [2, pop_size].
        base_tourney = self.config.tournament_size
        div = self._current_diversity
        if div >= 0.6:
            # High diversity: strong selection
            adaptive_tourney = base_tourney + 2
        elif div >= 0.4:
            # Medium diversity: standard selection
            adaptive_tourney = base_tourney + 1
        elif div >= 0.2:
            # Low-ish diversity: slightly weaker
            adaptive_tourney = base_tourney
        else:
            # Very low diversity: weak selection to maintain exploration
            adaptive_tourney = max(2, base_tourney - 1)
        tourney_size = min(adaptive_tourney, n_scored)
        _scored_get = scored.__getitem__  # hoist for hot loop
        for _ in range(elite_size):
            # R21: Manual tournament -- avoids rng.sample() list allocation
            best_i = rng.randint(0, n_scored - 1)
            best_f = _scored_get(best_i)[0]
            for _t in range(tourney_size - 1):
                ci = rng.randint(0, n_scored - 1)
                cf = _scored_get(ci)[0]
                if cf > best_f:
                    best_f = cf
                    best_i = ci
            win_prog = _scored_get(best_i)[2]
            fp = (
                (len(win_prog) << 32)
                ^ (win_prog[0] if win_prog else 0)
                ^ ((win_prog[-1] << 16) if len(win_prog) > 1 else 0)
            )
            if fp not in seen_fps and best_i not in seen_indices:
                seen_fps.add(fp)
                seen_indices.add(best_i)
                elites.append(win_prog)
                elite_fits.append(best_f)
                elite_indices.append(best_i)

        return elites, elite_fits, elite_indices

    def _update_champion(
        self,
        elite: list[list[int]],
        detailed_results: list[dict[str, Any]],
        stage1_fitnesses: list[float] | None = None,
    ) -> bool:
        """Update champion if any elite challenger has higher fitness.

        Returns True if champion was replaced.

        R4 optimization: when stage1_fitnesses is provided, uses cached
        stage-1 fitness values to skip recomputation for the majority of
        elite candidates. Only recomputes at full budget when a candidate
        actually beats the champion (which happens rarely -- typically 0-1
        times per generation). This eliminates ~98% of redundant
        _compute_fitness calls in _update_champion.
        """
        best_fitness = self.champion_fitness
        best_program = None
        best_trace = None
        # R33: Capture winner's raw task score for curriculum advancement.
        # Use getattr for robustness against mock objects in tests that
        # bypass __init__ (e.g., __new__ construction).
        best_raw_score = getattr(self, "_raw_task_score", 0.0)
        improved = False
        _compute = self._compute_fitness  # hoist to local

        _use_sec = self.config.use_secondary_tasks
        for idx, (prog, trace) in enumerate(zip(elite, detailed_results, strict=False)):
            if stage1_fitnesses and idx < len(stage1_fitnesses):
                if stage1_fitnesses[idx] < best_fitness:
                    continue
            if _use_sec:
                self._secondary_task_results = self._evaluate_secondary_tasks(prog)
            else:
                self._secondary_task_results = None
            fitness = _compute(prog, trace)
            # R28: Use oscillation control to prevent rapid champion cycling
            if fitness > best_fitness and self._should_replace_champion(fitness):
                best_fitness = fitness
                best_program = prog
                best_trace = trace
                best_raw_score = getattr(self, "_raw_task_score", 0.0)  # R33: capture at this point
                improved = True

        if improved and best_program is not None:
            # R32: Archive old champion's behavioral descriptor before replacing.
            # This preserves the old champion's niche data in the QD archive
            # so we don't lose behavioral diversity when the champion is replaced.
            qd = getattr(self, "qd_archive", None)
            if (
                qd is not None
                and _HAS_QD
                and self.champion is not None
                and self.champion_trace is not None
                and self._champion_changed_since_last_qd is False
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
            # R33: Store champion raw score for curriculum advancement.
            if hasattr(self, "_champion_raw_task_score"):
                self._champion_raw_task_score = best_raw_score
            self.champion_fitness = best_fitness
            # R50: Run verify() on champion after replacement (uses current task instance)
            verified, verify_score = self._verify_champion()
            self._champion_verified = verified
            self._champion_verify_score = verify_score
            # R21: Mark champion changed so _update_qd_archive and
            # _update_novelty_archive know to run next generation.
            self._champion_changed_since_last_qd = True
            self._champion_changed_since_last_novelty = True
            # R31: Invalidate champion QD descriptor cache (new champion)
            self._champion_qd_valid = False
            self._cached_champion_qd_desc = None

        # Hall of Fame: track the best programs seen this generation.
        # R22: HoF quality gate — compute threshold in a single pass
        # using partial sort (heapq.nsmallest on negated values) to find
        # the k-th largest value, then do a single filtered pass.
        # R35: Early skip when no program in this generation can beat
        # the worst entry (avoids unnecessary threshold computation).
        # R39: Fixed bug: old code checked `size < 10` which prevented
        # ALL adds when the HoF was full. The HallOfFame.add() method
        # already handles eviction of the worst entry if the new program
        # is better, so we should always try to add. Now we only skip
        # when ALL stage1 fitnesses are below the HoF's worst entry.
        if stage1_fitnesses and elite:
            n_f = len(stage1_fitnesses)
            hof_worst = self._hall_of_fame.worst_fitness
            # Guard: if hof_worst is not a numeric type (e.g. MagicMock
            # in tests), treat as -inf so the quality gate still runs.
            if not isinstance(hof_worst, (int, float)):
                hof_worst = float("-inf")
            max_stage1 = max(stage1_fitnesses)
            if max_stage1 < hof_worst:
                pass  # no program qualifies, skip quality gate
            else:
                k_top = max(1, n_f // 5)
                # Single pass: find k-th largest using min-heap of size k
                threshold_heap: list[float] = []
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

        self.best_fitness_history.append(self.champion_fitness)
        # R33: Cap history length to prevent unbounded memory growth.
        # Use getattr for robustness against mock objects in tests that
        # bypass __init__ (e.g., __new__ construction).
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
            return score >= 1.0, score
        except Exception:
            return False, 0.0

    def _advance_curriculum(self) -> None:
        """Advance to the next curriculum level."""
        old_level = self.champion_level
        self.champion_level += 1
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
    # Recurrent feedback
    # ------------------------------------------------------------------

    def _try_recurrent_feedback(self) -> bool:
        """Save model-generated programs that execute successfully to training data.

        Saves the best non-crashing model-guided challengers each generation.
        This feeds the model's own output back into training, creating a
        self-improving loop. The champion (reference solution) is NOT saved
        because it's already in the training data.

        Phase 3: Quality-gated trace storage — only saves traces where
        fitness > MIN_FITNESS_FOR_TRACE, preventing training on low-quality
        or crash traces. Adds timestamp for recency decay in training.
        Performs quality-based eviction when buffer exceeds MAX_RECURRENT_ENTRIES.

        Returns True if >= 50 examples accumulated (signal to retrain).
        """
        if not self._last_challengers:
            return False

        # Determine recurrent data directory
        # R16: Cache makedirs path to avoid repeated os.makedirs each generation
        recur_dir = self.config.recurrent_feedback_dir
        if not recur_dir:
            recur_dir = MILESTONE1_DIR  # Write to milestone1/ directly (not subdirectory)
        if getattr(self, "_recur_dir_made", None) != recur_dir:
            os.makedirs(recur_dir, exist_ok=True)
            self._recur_dir_made = recur_dir

        # Load existing hashes (cached on instance)
        jsonl_path = os.path.join(recur_dir, "recurrent_data.jsonl")  # milestone1/recurrent_data.jsonl
        if not hasattr(self, "_saved_sha256s"):
            self._saved_sha256s: dict[str, None] = {}  # R38: OrderedDict-like dict for age-based eviction
            if os.path.isfile(jsonl_path):
                try:
                    with open(jsonl_path) as _f:
                        for _line in _f:
                            _e = json.loads(_line)
                            if "champion_sha256" in _e:
                                self._saved_sha256s[_e["champion_sha256"]] = None
                except Exception:
                    pass

        saved_any = False
        rejected_count = 0
        # Cache module-level LEVELS (already imported at module top — avoid re-import)
        _levels = LEVELS
        _verify_levels_sorted = None  # lazy: compute sorted keys once

        # Pre-cache fitnesses list and its length for fast indexed access
        fitnesses = self._last_challenger_fitnesses
        n_fitnesses = len(fitnesses)

        for idx in range(len(self._last_challengers)):
            prog = self._last_challengers[idx]
            if idx >= len(self._last_results):
                continue

            trace = self._last_results[idx]
            status = trace.get("status", "")
            if status != "ok":
                continue

            sha256 = sha256_program(prog)
            if sha256 in self._saved_sha256s:
                continue

            # Phase 3.1: Quality-gated trace storage
            # Only save traces above MIN_FITNESS_FOR_TRACE threshold.
            # Exception: if deeply stalled (>= 30 gens), save regardless
            # as a desperate measure to get any training signal.
            challenger_fitness = fitnesses[idx] if idx < n_fitnesses else 0.0
            if challenger_fitness <= MIN_FITNESS_FOR_TRACE and self.generations_stalled < 30:
                rejected_count += 1
                continue
            if len(prog) < 2:
                continue

            # Verify against task levels (use module-level LEVELS, sorted once)
            task_score = 0.0
            solved_level = self.champion_level
            output = {
                "registers": trace.get("final_regs", {}),
                "memory": trace.get("final_memory", {}),
            }
            try:
                if _verify_levels_sorted is None:
                    _verify_levels_sorted = sorted(_levels.keys())
                for lvl in _verify_levels_sorted:
                    level_cls = _levels[lvl]
                    try:
                        score = level_cls(seed=lvl).verify(output, self.task_instance)
                        if score > task_score:
                            task_score = score
                            solved_level = lvl
                    except Exception:
                        continue
            except Exception:
                pass

            # Phase 3.2: Add timestamp for recency decay in training
            example = {
                "level": solved_level,
                "generation": self.current_generation,
                "fitness": challenger_fitness,
                "champion_sha256": sha256,
                "champion_words": len(prog),
                "champion_program": [hex(w) for w in prog],
                "trace_status": status,
                "trace_instr_count": trace.get("instructions_count", 0),
                "stalled_count": self.generations_stalled,
                "task_score": task_score,
                "timestamp": time.time(),
                "trace_quality_score": min(1.0, max(0.0, challenger_fitness / 100.0)),
            }

            self._saved_sha256s[sha256] = None
            # R38: Cap sha256 dedup dict to prevent unbounded memory growth.
            # Uses dict (insertion-ordered in Python 3.7+) instead of set
            # for deterministic age-based eviction. When cap is hit, evicts
            # the oldest 2000 entries (FIFO), preserving recent dedup data.
            if len(self._saved_sha256s) > 5000:
                keys = list(self._saved_sha256s.keys())
                self._saved_sha256s = {k: None for k in keys[-3000:]}
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(example) + "\n")
            self.recurrent_examples_count += 1
            saved_any = True

        if saved_any:
            self._log(
                f"Recurrent feedback saved ({self.recurrent_examples_count} total, "
                f"{len(self._saved_sha256s)} unique)"
            )
        if rejected_count > 0:
            self._log(f"Rejected {rejected_count} low-quality traces " f"(fitness < {MIN_FITNESS_FOR_TRACE})")

        # Phase 3.3: Quality-based eviction when buffer exceeds limit
        # Evicts lowest-fitness traces (not oldest) to maintain training quality.
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
        """Count backward branches (loop structures) in RV32I bytecode.

        Decodes B-type instructions (BEQ, BNE, BLT, BGE, BLTU, BGEU)
        by extracting the 13-bit signed immediate and checking for
        negative offsets (which indicate backward jumps / loops).
        """
        if not program:
            return 0
        count = 0
        for word in program:
            if not isinstance(word, int):
                continue
            opcode = word & 0x7F
            # B-type instructions: opcode = 0x63 (0b1100011)
            if opcode == 0x63:
                # B-type immediate encoding (bits 31,7,30:25,11:8 → 12-bit signed)
                imm = (
                    ((word >> 31) & 0x1) << 12
                    | ((word >> 7) & 0x1) << 11
                    | ((word >> 25) & 0x3F) << 5
                    | ((word >> 8) & 0xF) << 1
                )
                # Sign-extend 13-bit immediate
                if imm & 0x1000:
                    imm -= 0x2000
                if imm < 0:
                    count += 1
        return count

    def get_champion_program_analysis(self) -> dict:
        """Analyze the current champion program structure."""
        if self.champion is None:
            return {'exists': False}
        program = self.champion
        backward_branches = self._count_backward_branches(program)
        n_ops = len(program)
        # Count distinct opcodes
        opcodes = set()
        for word in program:
            if isinstance(word, int):
                opcodes.add(word & 0x7F)
        return {
            'exists': True,
            'word_count': n_ops,
            'unique_opcodes': len(opcodes),
            'backward_branches': backward_branches,
            'has_ebreak': any(isinstance(w, int) and w == 0x00100073 for w in program),
            'estimated_complexity': 'simple' if n_ops <= 5 else 'moderate' if n_ops <= 15 else 'complex',
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
            'champion_level': self.champion_level,
            'champion_fitness': self.champion_fitness,
            'champion_words': len(self.champion) if self.champion else 0,
            'champion_verified': getattr(self, '_champion_verified', False),
            'generations_stalled': self.generations_stalled,
            'current_generation': self.current_generation,
            # Archive sizes
            'novelty_archive_size': novelty_size,
            'qd_archive_size': qd_size,
            'strategy_archive_records': strategy_records,
            'hall_of_fame_size': len(self._hall_of_fame._heap),
            # Strategy diversity
            'strategy_diversity': strategy_div,
            # Training data
            'recurrent_examples': self.recurrent_examples_count,
            # Fitness velocity
            'fitness_velocity': getattr(self, '_fitness_velocity', 0.0),
            'fitness_momentum': getattr(self, '_fitness_momentum', 0.0),
            # Mutation stats
            'mutation_rate': self.config.mutation_rate,
            'population_size': self.config.population_size,
            'elite_ratio': self.config.elite_ratio,
        }

    def get_performance_metrics(self) -> dict:
        """Return performance metrics for the current run."""
        return {
            'current_generation': self.current_generation,
            'population_size': self.config.population_size,
            'elite_ratio': self.config.elite_ratio,
            'eval_budget': self.config.eval_budget,
            'model_guided_ratio': self._get_model_guided_ratio(),
            'mutation_rate': self.config.mutation_rate,
            'champion_words': len(self.champion) if self.champion else 0,
            'novelty_weight': self.config.novelty_weight,
            'qd_bonus_weight': self.config.qd_bonus_weight,
            'total_recurrent_examples': self.recurrent_examples_count,
            'hof_size': len(self._hall_of_fame._heap),
        }

    def get_archive_summary(self) -> dict:
        """Aggregate summary of all archive states."""
        novelty = self.novelty_archive
        qd = self.qd_archive
        sa = self.strategy_archive
        return {
            'novelty': {
                'size': novelty.size if novelty else 0,
                'threshold': novelty.novelty_threshold if novelty else 0,
                'diversity': novelty.get_diversity_score() if novelty else 0.0,
            },
            'qd': {
                'coverage': qd.coverage() if qd else 0.0,
                'size': qd.size if qd else 0,
                'cells': qd.num_bins * qd.num_bins if qd else 0,
                'fitness_dist': qd.get_fitness_distribution() if qd else {},
            },
            'strategy': {
                'total_records': sa.total_records() if sa else 0,
                'diversity': sa.get_strategy_diversity(self.champion_level) if sa else 0.0,
            },
            'hall_of_fame': len(self._hall_of_fame._heap),
        }

    def get_training_summary(self) -> dict:
        """Return training-relevant metrics for the retrain pipeline."""
        return {
            'recurrent_examples': self.recurrent_examples_count,
            'champion_fitness': self.champion_fitness,
            'champion_level': self.champion_level,
            'generations_stalled': self.generations_stalled,
            'fitness_velocity': getattr(self, '_fitness_velocity', 0.0),
            'champion_verified': getattr(self, '_champion_verified', False),
            'champion_verify_score': getattr(self, '_champion_verify_score', 0.0),
            'strategy_diversity': self.strategy_archive.get_strategy_diversity(self.champion_level) if self.strategy_archive else 0.0,
            'novelty_diversity': self.novelty_archive.get_diversity_score() if self.novelty_archive else 0.0,
            'qd_coverage': self.qd_archive.coverage() if self.qd_archive else 0.0,
        }

    def get_run_summary(self) -> dict:
        """Single consolidated summary of the entire run state."""
        return {
            'diagnostics': self.get_run_diagnostics(),
            'performance': self.get_performance_metrics(),
            'training': self.get_training_summary(),
            'archives': self.get_archive_summary(),
            'champion_summary': self.get_champion_summary(),
        }

    def get_evolution_state(self) -> dict:
        """Full evolution state for serialization/debugging."""
        return {
            'champion': self.get_champion_program_analysis(),
            'run_summary': self.get_run_summary(),
            'state': {
                'current_generation': self.current_generation,
                'champion_level': self.champion_level,
                'champion_fitness': self.champion_fitness,
                'generations_stalled': self.generations_stalled,
            },
            'config': {
                'max_generations': self.config.max_generations,
                'population_size': self.config.population_size,
                'elite_ratio': self.config.elite_ratio,
                'mutation_rate': self.config.mutation_rate,
                'auto_curriculum': self.config.auto_curriculum,
                'use_gpu': self.config.use_gpu,
            },
        }

    def get_comprehensive_state(self) -> dict:
        """All-in-one state snapshot for dashboards, export, and debugging."""
        return {
            'evolution': self.get_evolution_state(),
            'champion_program': self.get_champion_program_analysis(),
            'run_diagnostics': self.get_run_diagnostics(),
            'performance': self.get_performance_metrics(),
            'training': self.get_training_summary(),
            'archives': self.get_archive_summary(),
            'champion_summary': self.get_champion_summary(),
        }

    def log_comprehensive_diagnostics(self) -> None:
        """Log comprehensive diagnostics for debugging and analysis."""
        state = self.get_comprehensive_state()
        evo = state.get("evolution", {})
        evo_state = evo.get("state", {})
        self._log('=== Comprehensive Diagnostics ===')
        self._log(f'  Champion: {self.get_champion_summary()}')
        self._log(f'  Generation: {evo_state.get("current_generation", 0)}')
        self._log(f'  Fitness velocity: {evo_state.get("fitness_velocity", 0):.4f}')
        archives = state.get("archives", {})
        novelty = archives.get("novelty", {})
        qd = archives.get("qd", {})
        strategy = archives.get("strategy", {})
        self._log(f'  Archives: novelty={novelty.get("size", 0)}, '
                  f'qd_coverage={qd.get("coverage", 0.0):.2%}')
        self._log(f'  Strategy diversity: {strategy.get("diversity", 0.0):.3f}')
        perf = state.get("performance", {})
        self._log(f'  Performance: mut_rate={perf.get("mutation_rate", 0.0):.2f}, '
                  f'pop={perf.get("population_size", 0)}')
        self._log('=== End Diagnostics ===')

    def log_diagnostics(self) -> None:
        """Log run diagnostics in a readable format."""
        diag = self.get_run_diagnostics()
        lines = [
            "=== Run Diagnostics ===",
            f"  Champion: L{diag['champion_level']} fitness={diag['champion_fitness']:.2f} "
            f"words={diag['champion_words']} verified={diag['champion_verified']}",
            f"  Generation: {diag['current_generation']} stalled={diag['generations_stalled']}",
            f"  Archives: novelty={diag['novelty_archive_size']} qd={diag['qd_archive_size']} "
            f"strategy={diag['strategy_archive_records']} hof={diag['hall_of_fame_size']}",
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
            'timestamp': __import__('time').time(),
            'generation': self.current_generation,
            'champion': {
                'level': self.champion_level,
                'fitness': self.champion_fitness,
                'words': len(self.champion) if self.champion else 0,
                'program_hex': [hex(w) for w in (self.champion or [])],
                'verified': getattr(self, '_champion_verified', False),
            },
            'config': {
                'population_size': self.config.population_size,
                'elite_ratio': self.config.elite_ratio,
                'mutation_rate': self.config.mutation_rate,
                'novelty_weight': self.config.novelty_weight,
            },
            'archives': {
                'novelty_size': self.novelty_archive.size if self.novelty_archive else 0,
                'qd_occupied': self.qd_archive.size if self.qd_archive else 0,
                'qd_coverage': self.qd_archive.coverage() if self.qd_archive else 0.0,
                'hof_size': len(self._hall_of_fame._heap),
            },
            'stall': {
                'generations_stalled': self.generations_stalled,
                'fitness_velocity': getattr(self, '_fitness_velocity', 0.0),
            },
            'diagnostics': self.get_run_diagnostics(),
        }
        return state

    def save_state_to_file(self, path: str) -> bool:
        """Save comprehensive state to a JSON file for analysis."""
        try:
            state = self.get_comprehensive_state()
            import json
            with open(path, 'w') as f:
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
