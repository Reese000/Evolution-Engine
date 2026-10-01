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
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
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


from paths import MILESTONE0_DIR, MILESTONE1_DIR
from paths import PROJECT_ROOT as SCRIPT_DIR
from paths import ensure_project_on_path

ensure_project_on_path()


# ----------------------------------------------------------------------

# Imports from the real vm_farm module

# ----------------------------------------------------------------------


# EE-W5: per-generation phase timing -- splits generation wall time into
# VM-farm fitness evaluation vs Python orchestration so the
# phase_evolution_seconds band can be attributed to a dominant phase.
from evolution.phase_timing import PhaseTimer, default_log_path, timed_vm_eval, timers_enabled
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

            except (RuntimeError, ValueError) as e:

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

            except RuntimeError as e:

                self._log(f"Bit-exact check failed: {e} -- disabling GPU")

                self._gpu_farm = None

                self.use_gpu = False

                return

        self._log(f"GPU bit-exact: {n_samples}/{n_samples} samples match CPU")

    def evaluate(self, program: list[int], inputs: dict[str, Any], budget: int = 10000) -> EvalResult:

        init_regs = inputs.get("registers", {})

        init_mem = inputs.get("memory", {})

        with timed_vm_eval():
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

        except RuntimeError as e:

            return {"status": "error", "error": str(e), "instructions_count": 0}

    def evaluate_batch(self, programs, task, budget=None):
        """Evaluate a batch of programs using the VM farm.

        Wall time is recorded here, at the outermost call, so a batch
        that fans out to :meth:`evaluate` is counted exactly once (EE-W5).
        """
        with timed_vm_eval():
            return self._evaluate_batch_untimed(programs, task, budget)

    def _evaluate_batch_untimed(self, programs, task, budget=None):
        """Batch body; wall time is accounted by the evaluate_batch wrapper."""
        if not programs:

            return []

        budget = budget or self.config.eval_budget

        # R32: Try GPU farm first when available

        if self.use_gpu and self._gpu_farm is not None and len(programs) > 1:

            try:

                gpu_results = self._gpu_farm.evaluate_multi(programs, task, budget=budget)

                if gpu_results is not None:

                    self._gpu_eval_count += 1

                    return [self._dict_to_eval_result(r) if isinstance(r, dict) else r for r in gpu_results]

            except (RuntimeError, ValueError, TypeError, IndexError) as exc:

                self._gpu_fallback_count += 1

                self._log(f"GPU batch failed ({type(exc).__name__}) — falling back to CPU")

        if self.num_workers <= 1:
            if isinstance(task, list):
                return [self._eval_single_worker(p, t, budget) for p, t in zip(programs, task)]
            return [self._eval_single_worker(p, task, budget) for p in programs]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            if isinstance(task, list):
                futures = [executor.submit(self._eval_single_worker, p, t, budget) for p, t in zip(programs, task)]
            else:
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

from evolution.curriculum_advance import curriculum_fitness_gate as _curriculum_fitness_gate_fn
from evolution.curriculum_advance import curriculum_verify_threshold as _curriculum_verify_threshold_fn
from evolution.curriculum_advance import forced_advance_verify_floor as _forced_advance_verify_floor_fn
from evolution.diagnostics_mixin import DiagnosticsMixin
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
from evolution.qd_archive_mixin import QDArchiveMixin
from evolution.recurrent_mixin import RecurrentMixin

# --- Mixin classes extracted from EvolutionEngine (P2 decomposition) ---
from evolution.selection_mixin import SelectionMixin
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

    def programs_at_level(self, level: int) -> list[list[int]]:
        """Distinct programs stored at a curriculum level (defensive copies)."""
        out: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for _f, _g, prog, lvl in self._heap:
            if int(lvl) != int(level):
                continue
            key = tuple(prog)
            if key in seen:
                continue
            seen.add(key)
            out.append(list(prog))
        return out

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

            for f, _g, p, lvl in self._heap:

                if lvl == level and f > best_fit:

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

            candidates = [(f, g, p, lvl) for f, g, p, lvl in heap if lvl == level]

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

        if self.curriculum_level > self.max_level:

            self.curriculum_level = self.max_level


class EvolutionEngine(SelectionMixin, QDArchiveMixin, RecurrentMixin, DiagnosticsMixin):
    """Main evolutionary loop for RV32I program synthesis.



    Maintains a champion program per curriculum level and iteratively

    generates challengers via mutation, evaluates them, and promotes

    fitter individuals.

    """

    def _init_archives(self) -> None:
        """Initialize strategy archive, QD archive, novelty archive, and champion archive."""

        from strategy_archive import StrategyArchive

        self.strategy_archive = StrategyArchive(
            max_records=int(
                os.environ.get(
                    "EE_STRATEGY_ARCHIVE_MAX_RECORDS",
                    os.environ.get("EE_TERMINAL_STRATEGY_ARCHIVE_CAP", "3200"),
                )
            )
        )

        if self.config.strategy_archive_path:

            self.strategy_archive.load(self.config.strategy_archive_path)

        self.novelty_archive = None

        if _HAS_NOVELTY and self.config.use_novelty_archive:

            try:

                from novelty import SparseArchive

                self.novelty_archive = SparseArchive(novelty_threshold=self.config.novelty_threshold)

            except ImportError:

                pass

        self.qd_archive = None

        if _HAS_QD and self.config.use_qd_archive:

            try:

                from qd_archive import MAPElitesArchive

                self.qd_archive = MAPElitesArchive(bins=self.config.qd_bins)

            except ImportError:

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

        self._champion_stage1_fitness = None

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

        # Level-aware mutation strategy profiles (used by mutation_helpers)
        self._LEVEL_WEIGHT_PROFILES = {
            "L1_L4": {
                "RANDOM": 2.0,
                "TASK_AWARE": 3.0,
                "DIRECTED": 2.5,
                "HYBRID": 1.5,
                "GROW": 0.5,
            },
            "L5_L6": {
                "GROW": 2.5,
                "PERTURBATION_CAPPED": 2.0,
                "HYBRID": 2.0,
                "TASK_AWARE": 1.5,
                "DIRECTED": 1.5,
            },
            "L7_L11": {
                "GROW": 2.0,
                "PERTURBATION_CAPPED": 2.5,
                "HYBRID": 2.0,
                "TASK_AWARE": 1.5,
                "DIRECTED": 1.5,
                "MODEL_GUIDED": 1.0,
            },
            "L12_plus": {
                "TASK_AWARE": 3.5,
                "DIRECTED": 3.5,
                "HYBRID": 2.5,
                "MODEL_GUIDED": 2.0,
                "GROW": 1.0,
                "PERTURBATION_CAPPED": 1.0,
                "RANDOM": 0.5,
            },
            "L17_plus": {
                "TASK_AWARE": 4.0,
                "DIRECTED": 4.0,
                "HYBRID": 2.0,
                "MODEL_GUIDED": 2.5,
                "GROW": 0.5,
                "PERTURBATION_CAPPED": 0.5,
                "RANDOM": 0.25,
            },
        }
        self._LEVEL_OVERRIDE_PROBS = {
            "L1_L4": 0.35,
            "L5_L6": 0.40,
            "L7_L11": 0.45,
            "L12_plus": 0.55,
            "L17_plus": 0.70,
        }

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

        self._refresh_eval_budget()

        self._cached_expected_regs, self._cached_expected_total, self._cached_task_instance_id = {}, 0, -1

        self._compute_qd_axes_fn = None

        if _HAS_QD:

            try:

                # compute_qd_axes is defined in this module (line 348)

                self._compute_qd_axes_fn = compute_qd_axes

            except NameError:

                pass

        self._extract_behavioral_descriptor = None

        if _HAS_NOVELTY:

            try:

                from novelty import extract_behavioral_descriptor

                self._extract_behavioral_descriptor = extract_behavioral_descriptor

            except ImportError:

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

        ceil = params.get("adaptive_mutation_ceil")
        if ceil is not None:
            self._adaptive_mutation_ceil = max(float(self._adaptive_mutation_ceil), float(ceil))

        # Refresh cached config values that may have changed

        self._novelty_weight_cache = self.config.novelty_weight

    def load_archive_state(
        self,
        qd_data: dict | None = None,
        novelty_data: dict | None = None,
        strategy_data: dict | None = None,
    ) -> None:
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

        if strategy_data is not None:

            from strategy_archive import StrategyArchive

            self.strategy_archive = StrategyArchive.deserialize(strategy_data)

    def save_archive_state(self) -> dict[str, Any]:
        """Extract serializable archive state for cross-cycle persistence.



        Returns a dict with 'qd', 'novelty', and 'strategy' keys, each containing

        the serialized archive (JSON-compatible) or None if unavailable.

        """

        return {
            "qd": self.qd_archive.serialize() if self.qd_archive is not None else None,
            "novelty": self.novelty_archive.serialize() if self.novelty_archive is not None else None,
            "strategy": self.strategy_archive.serialize() if self.strategy_archive is not None else None,
        }

    # ------------------------------------------------------------------

    # R28: Adaptive mutation and oscillation control

    # ------------------------------------------------------------------

    def _velocity_based_mutation_rate(self, base: float, vel: float) -> float:
        """Blend velocity-responsive adaptive rate 50/50 with config base."""

        if vel > 5.0:

            adaptive = max(self._adaptive_mutation_floor, base * 0.5)

        elif vel > 1.0:

            adaptive = max(self._adaptive_mutation_floor, base * 0.75)

        elif vel > -1.0:

            adaptive = min(self._adaptive_mutation_ceil, base * 1.3)

        elif vel > -5.0:

            adaptive = min(self._adaptive_mutation_ceil, base * 1.6)

        else:

            adaptive = min(self._adaptive_mutation_ceil, base * 2.0)

        return 0.5 * adaptive + 0.5 * base

    def _stall_mutation_boost_factor(self, stall: int) -> float | None:
        """Return stall boost multiplier when stall >= 10, else None."""

        if stall < 10:

            return None

        if stall >= 50:

            return 1.8

        if stall >= 30:

            return 1.4

        if stall >= 20:

            return 1.2

        return 1.1

    def _apply_stall_mutation_boost(self, blended: float, stall: int) -> float:
        """Apply R40 stall-based mutation boost blended at 30% weight."""

        stall_boost = self._stall_mutation_boost_factor(stall)

        if stall_boost is None:

            return blended

        stall_adjusted = blended * stall_boost

        return 0.7 * blended + 0.3 * stall_adjusted

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

        base = getattr(self, "_original_mutation_rate", self.config.mutation_rate)

        blended = self._velocity_based_mutation_rate(base, self._fitness_velocity)

        blended = self._apply_stall_mutation_boost(blended, self.generations_stalled)

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

        except (RuntimeError, AttributeError):

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

    def _recommend_normalization_off(
        self, recs: list[str], norm_active: bool, hist_len: int, recent_trend: float
    ) -> None:
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

    def _record_oscillation_delta(self, delta: float) -> None:
        """Append delta to oscillation window and maintain positive-count tracker."""

        window = self._champion_oscillation_window

        if len(window) >= self._oscillation_window_size:

            oldest = window[0]

            if oldest > 0:

                self._oscillation_positive_count -= 1

                self._oscillation_positive_count = max(self._oscillation_positive_count, 0)

        window.append(delta)

        if delta > 0:

            self._oscillation_positive_count += 1

            self._oscillation_positive_count = min(self._oscillation_positive_count, self._oscillation_window_size)

    def _oscillation_hysteresis_blocks_replacement(self, delta: float) -> bool:
        """Return True when stall-adaptive hysteresis rejects a small fitness gain."""

        if self._oscillation_positive_count < self._oscillation_threshold:

            return False

        stall = self.generations_stalled

        if stall > 0:

            stall_factor = max(0.2, 1.0 - 0.016 * stall)

            effective_delta = self._base_hysteresis_min_delta * stall_factor

        else:

            effective_delta = self._hysteresis_min_delta

        return delta < effective_delta

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

        if self._champion_is_timeout_penalized():

            return candidate_fitness > self.champion_fitness

        champ_verify = float(getattr(self, "_champion_verify_score", 0.0))
        if champ_verify >= 0.99 and self.champion is not None and self.champion_trace is not None:
            from evolution.fitness_helpers import selection_fitness

            champion_fitness = float(selection_fitness(self, self.champion, self.champion_trace))
        else:
            champion_fitness = float(self.champion_fitness)

        if candidate_fitness <= champion_fitness + 1e-6:

            return False

        delta = candidate_fitness - champion_fitness

        self._record_oscillation_delta(delta)

        if champ_verify >= 0.99 - 1e-6:
            from evolution.terminal_stagnation import terminal_certified_plateau, terminal_verify_promote_active

            if terminal_verify_promote_active() and terminal_certified_plateau(
                int(self.champion_level),
                champ_verify,
                champion_fitness,
            ):
                return True

        return not self._oscillation_hysteresis_blocks_replacement(delta)

    def _champion_is_timeout_penalized(self) -> bool:
        """Champion hit instruction-budget / timeout fitness floor (-25)."""

        return self.champion_fitness <= -25.0

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

        self._refresh_eval_budget()

        if self._apply_champion_rollback_from_tick():
            pass
        elif self.champion is None:
            if self._apply_carry_forward_champion():
                pass
            elif self._apply_best_certified_champion_seed():
                pass
            else:
                self.champion = self._generate_initial_champion()

        if self.champion is not None:

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

            self._try_repair_terminal_champion_verify()

        self._apply_forced_reseed_from_tick_state()
        self._enforce_champion_word_cap()
        self._maybe_reseed_stuck_champion()
        if self.champion is not None:
            if hasattr(self, "_enforce_l30_certified_champion"):
                self._enforce_l30_certified_champion()
            else:
                self._maybe_restore_best_certified_champion()
            self._maybe_anchor_champion_at_scan_floor()
            self._bootstrap_scan_floor_growth_material()
            self._reconcile_initialized_champion_fitness()

        self._max_challenger_words_cycle = len(self.champion) if self.champion else 0

        return {
            "generations_completed": 0,
            "final_fitness": self.champion_fitness,
            "final_level": self.champion_level,
            "champion_words": len(self.champion) if self.champion else 0,
            "elapsed_seconds": 0.0,
            "generations_stalled": self.generations_stalled,
            "total_champion_improvements": 0,
        }

    def _sync_champion_after_reseed(
        self,
        words: list[int],
        desc: str,
        label: str,
        trusted_verify: float | None = None,
        trusted_fitness: float | None = None,
    ) -> None:
        """Apply reference program and sync verify/fitness for EPI-positive state."""
        self.champion = list(words)
        self._refresh_eval_budget()
        self.champion_trace = self._evaluate(self.champion)
        verified, verify_score = self._verify_champion()
        if trusted_verify is not None and float(verify_score) < float(trusted_verify) - 1e-6:
            verify_score = float(trusted_verify)
            verified = verify_score >= 0.99 - 1e-6
        self._champion_verified = verified
        self._champion_verify_score = verify_score
        self._champion_partial_verify = verify_score
        if self.config.use_secondary_tasks:
            self._secondary_task_results = self._evaluate_secondary_tasks(self.champion)
        else:
            self._secondary_task_results = None
        if trusted_verify is not None:
            from evolution.fitness_helpers import reporting_champion_fitness

            self.champion_fitness = float(
                reporting_champion_fitness(
                    self,
                    self.champion,
                    self.champion_trace,
                    float(verify_score),
                )
            )
            if trusted_fitness is not None and float(trusted_fitness) > self.champion_fitness:
                self.champion_fitness = float(trusted_fitness)
            self._certified_fitness_floor = max(
                float(getattr(self, "_certified_fitness_floor", 0.0)),
                float(self.champion_fitness),
            )
        else:
            self.champion_fitness = self._compute_fitness(self.champion, self.champion_trace)
        self._log(
            f"[RegisterRepair] {label} L{self.champion_level} from {desc} "
            f"(fit={self.champion_fitness:.1f}, verify={verify_score:.3f})"
        )

    def _reconcile_initialized_champion_fitness(self) -> None:
        """Align in-cycle champion_fitness with certified reporting at scan floor."""
        if self.champion is None or self.champion_trace is None:
            return
        verify = float(getattr(self, "_champion_verify_score", 0.0))
        if verify < 0.99 - 1e-6:
            return
        try:
            from evolution.fitness_helpers import end_of_cycle_champion_fitness, terminal_certified_fitness_context

            with terminal_certified_fitness_context(self):
                reconciled = float(
                    end_of_cycle_champion_fitness(
                        self,
                        self.champion,
                        self.champion_trace,
                        verify,
                    )
                )
        except (ImportError, RuntimeError, TypeError, ValueError, AttributeError):
            return
        if reconciled <= float(self.champion_fitness) + 1e-6:
            return
        self.champion_fitness = reconciled
        self._certified_fitness_floor = max(
            float(getattr(self, "_certified_fitness_floor", 0.0)),
            reconciled,
        )

    def _restore_certified_on_curriculum_verify_regression(
        self,
        verify: float,
        level: int,
    ) -> bool:
        """Swap transient bytecode regression back to the last certified champion."""
        if verify >= 0.99 - 1e-6:
            return False
        max_level = int(getattr(self.config, "max_level", 30))
        if int(level) < max_level:
            return False
        project_root = os.path.dirname(os.path.abspath(__file__))
        from evolution.champion_rollback import find_best_certified_champion
        from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS

        results = os.path.join(project_root, "multi_cycle_results.jsonl")
        clean = find_best_certified_champion(results, int(level), min_verify=0.99, min_fitness=100.0)
        if clean:
            prog = [int(x, 16) if isinstance(x, str) else int(x) for x in clean["champion_program"]]
            if 12 <= len(prog) <= L30_COMPACT_MAX_WORDS:
                self._sync_champion_after_reseed(
                    prog,
                    f"certified_cycle_{clean['cycle']}",
                    (f"Restore certified after curriculum verify regression " f"{verify:.3f}"),
                    trusted_verify=float(clean.get("champion_verify_score", 1.0)),
                    trusted_fitness=float(clean.get("final_fitness", 0.0)),
                )
                return True
        from evolution.champion_carry_forward import load_carry_forward_champion

        loaded = load_carry_forward_champion(project_root, int(level))
        if not loaded:
            return False
        words, desc = loaded
        if not (12 <= len(words) <= L30_COMPACT_MAX_WORDS):
            return False
        self._sync_champion_after_reseed(
            words,
            desc,
            f"Restore carry-forward after curriculum verify regression {verify:.3f}",
            trusted_verify=0.989,
        )
        return True

    def _apply_carry_forward_champion(self) -> bool:
        """Restore champion saved from the previous clean cycle."""
        project_root = os.path.dirname(os.path.abspath(__file__))
        from evolution.champion_carry_forward import load_carry_forward_champion

        loaded = load_carry_forward_champion(project_root, int(self.champion_level))
        if loaded is None:
            return False
        words, desc = loaded
        self._sync_champion_after_reseed(words, desc, "Carry-forward champion")
        return True

    def _apply_best_certified_champion_seed(self) -> bool:
        """Seed terminal cycles from the best certified row in results history."""
        max_level = int(getattr(self.config, "max_level", 30))
        if int(self.champion_level) < max_level:
            return False
        project_root = os.path.dirname(os.path.abspath(__file__))
        from evolution.champion_rollback import find_best_certified_champion

        results = os.path.join(project_root, "multi_cycle_results.jsonl")
        clean = find_best_certified_champion(results, int(self.champion_level), min_verify=0.99, min_fitness=100.0)
        if not clean:
            return False
        words = [int(x, 16) if isinstance(x, str) else int(x) for x in clean["champion_program"]]
        if len(words) < 8:
            return False
        logged_verify = float(clean.get("champion_verify_score", 0.0))
        logged_fit = float(clean.get("final_fitness", 0.0))
        self._sync_champion_after_reseed(
            words,
            f"certified_cycle_{clean['cycle']}",
            f"Certified seed C{clean['cycle']}",
            trusted_verify=logged_verify,
            trusted_fitness=logged_fit,
        )
        return True

    def _bootstrap_scan_floor_growth_material(self) -> None:
        """Seed growth-material archive from ranked fills (cross-cycle breeding stock)."""
        if self.champion is None or int(self.champion_level) < 30:
            return
        try:
            from evolution.champion_rollback import record_scan_floor_growth_material
            from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills, scan_floor_plateau_armed

            armed, _floor = scan_floor_plateau_armed()
            if not armed:
                return
            cycle_no = int(getattr(self, "_cycle_number", 0))
            for fill in collect_scan_floor_growth_crossover_fills(list(self.champion), max_fills=12):
                record_scan_floor_growth_material(
                    list(self.champion),
                    list(fill),
                    cycle=cycle_no,
                )
        except (ImportError, TypeError, ValueError, OSError):
            return

    def _maybe_anchor_champion_at_scan_floor(self) -> bool:
        """Restore certified scan-floor bytes when live champion shrank below the floor."""
        if self.champion is None or int(self.champion_level) < 30:
            return False
        verify = float(getattr(self, "_champion_verify_score", 0.0))
        if verify < 0.99 - 1e-6:
            return False
        from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
        from evolution.mutation_helpers import scan_floor_anchor_program, scan_floor_plateau_armed

        armed, floor_w = scan_floor_plateau_armed()
        if not armed or len(self.champion) >= int(floor_w):
            return False
        anchor = scan_floor_anchor_program(list(self.champion))
        if len(anchor) != int(floor_w) or tuple(anchor) == tuple(self.champion):
            return False
        anchor_pr = float(compact_l30_promotion_raw_fitness(anchor))
        live_pr = float(compact_l30_promotion_raw_fitness(list(self.champion)))
        if anchor_pr + 1e-6 < live_pr:
            return False
        self._sync_champion_after_reseed(
            anchor,
            "scan_floor_anchor",
            f"Anchor {floor_w}w prom_raw {anchor_pr:.3f} over live {len(self.champion)}w {live_pr:.3f}",
            trusted_verify=verify,
            trusted_fitness=self.champion_fitness,
        )
        return True

    def _enforce_l30_certified_champion(self) -> bool:
        """Replace degraded L30 champion with best certified row from history."""
        level = int(self.champion_level)
        max_level = int(getattr(self.config, "max_level", 30))
        if level < max_level:
            return False
        verify = float(getattr(self, "_champion_verify_score", 0.0))
        if self.champion_trace is not None:
            live = float(self._verify_score_from_trace(self.champion_trace))
            if live + 1e-6 < verify:
                verify = live
        fit = float(getattr(self, "champion_fitness", 0.0))
        champ_words = len(self.champion) if self.champion else 0
        from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS

        if 12 <= champ_words <= L30_COMPACT_MAX_WORDS:
            from evolution.champion_carry_forward import load_carry_forward_champion
            from evolution.terminal_stagnation import heldout_l30_verify_mean

            live_h = heldout_l30_verify_mean(list(self.champion)) if self.champion else 0.0
            project_root = os.path.dirname(os.path.abspath(__file__))
            loaded = load_carry_forward_champion(project_root, level)
            if loaded:
                words, desc = loaded
                if 12 <= len(words) <= L30_COMPACT_MAX_WORDS:
                    carry_h = heldout_l30_verify_mean(words)
                    from evolution.terminal_stagnation import compact_curriculum_beats, compact_curriculum_side_score

                    live_side = compact_curriculum_side_score(list(self.champion)) if self.champion else 0.0
                    carry_side = compact_curriculum_side_score(words)
                    better_l30 = live_h + 0.05 < carry_h
                    better_side = carry_h + 1e-6 >= live_h and compact_curriculum_beats(list(self.champion), words)
                    if better_l30 or better_side:
                        self._sync_champion_after_reseed(
                            words,
                            desc,
                            (
                                f"Restore carry held-out {carry_h:.3f}/{carry_side:.3f} "
                                f"over live {live_h:.3f}/{live_side:.3f}"
                            ),
                            trusted_verify=min(carry_h, 0.989),
                        )
                        return True
            from evolution.terminal_stagnation import cached_best_compact_l30_fill

            cached = cached_best_compact_l30_fill(tuple(int(x) for x in self.champion)) if self.champion else None
            fill = list(cached) if cached else None
            if fill and fill != list(self.champion) and len(fill) <= champ_words:
                fill_h = heldout_l30_verify_mean(fill)
                if fill_h + 1e-6 >= live_h:
                    self._sync_champion_after_reseed(
                        fill,
                        "best_compact_l30_fill",
                        f"Install compact fill {len(fill)}w held-out {fill_h:.3f}",
                        trusted_verify=min(fill_h, 0.989),
                    )
                    return True
            if live_h + 0.05 < 1.0:
                from evolution.champion_rollback import find_best_honest_compact_l30

                results = os.path.join(project_root, "multi_cycle_results.jsonl")
                honest = find_best_honest_compact_l30(results, level)
                if honest:
                    prog = [int(x, 16) if isinstance(x, str) else int(x) for x in honest["champion_program"]]
                    honest_h = heldout_l30_verify_mean(prog)
                    if live_h + 0.05 < honest_h:
                        self._sync_champion_after_reseed(
                            prog,
                            f"honest_compact_cycle_{honest['cycle']}",
                            f"Restore honest held-out {honest_h:.3f} over live {live_h:.3f}",
                            trusted_verify=min(honest_h, 0.989),
                        )
                        return True
            if self._restore_certified_on_curriculum_verify_regression(verify, level):
                return True
            return False
        if verify >= 0.99 - 1e-6 and fit >= 200.0 and champ_words >= 12:
            return False
        if self._restore_certified_on_curriculum_verify_regression(verify, level):
            return True
        if self._maybe_restore_best_certified_champion():
            return True
        return self._apply_best_certified_champion_seed()

    def _apply_champion_rollback_from_tick(self) -> bool:
        """Restore champion from last clean cycle after same-level collapse."""
        tick_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agents", "tick_state.json")
        try:
            if not os.path.isfile(tick_path):
                return False
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        rollback = tick.get("champion_rollback_program")
        if not rollback or not tick.get("apply_champion_rollback"):
            return False
        words: list[int] = []
        for item in rollback:
            if isinstance(item, str):
                words.append(int(item, 16))
            else:
                words.append(int(item))
        if len(words) < 8:
            return False
        src_cycle = tick.get("champion_rollback_cycle", "?")
        self._sync_champion_after_reseed(
            words,
            f"rollback_cycle_{src_cycle}",
            f"Champion rollback from cycle {src_cycle}",
        )
        for key in (
            "champion_rollback_program",
            "champion_rollback_cycle",
            "apply_champion_rollback",
        ):
            tick.pop(key, None)
        try:
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump(tick, f, indent=2)
        except OSError:
            pass
        return True

    def _apply_forced_reseed_from_tick_state(self) -> None:
        """Force reference reseed when EPI governor or level clamp requested it."""
        tick_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agents", "tick_state.json")
        try:
            if not os.path.isfile(tick_path):
                return
            with open(tick_path, encoding="utf-8") as f:
                tick = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        if not tick.get("force_champion_reseed") and not tick.get("force_reference_champion"):
            return
        if tick.get("force_reference_champion") and not tick.get("force_champion_reseed"):
            tick.pop("force_reference_champion", None)
            try:
                with open(tick_path, "w", encoding="utf-8") as f:
                    json.dump(tick, f, indent=2)
            except OSError:
                pass
            return
        reseed_level = int(tick.get("reseed_level", self.champion_level))
        cur = int(self.champion_level)
        if reseed_level < cur:
            reseed_level = cur
        elif reseed_level > cur + 1:
            reseed_level = cur
        if reseed_level > cur:
            self.champion_level = reseed_level
            self.config.curriculum_level = reseed_level
            tick["reseed_level"] = reseed_level
            try:
                with open(tick_path, "w", encoding="utf-8") as f:
                    json.dump(tick, f, indent=2)
            except OSError:
                pass
        from evolution.register_repair import reference_champion_program

        words, desc = reference_champion_program(reseed_level)
        if not words:
            tick["force_champion_reseed"] = False
            tick["force_reference_champion"] = False
            try:
                with open(tick_path, "w", encoding="utf-8") as f:
                    json.dump(tick, f, indent=2)
            except OSError:
                pass
            return
        self._sync_champion_after_reseed(words, desc, "Forced tick-state reseed")
        tick["force_champion_reseed"] = False
        tick["force_reference_champion"] = False
        try:
            with open(tick_path, "w", encoding="utf-8") as f:
                json.dump(tick, f, indent=2)
        except OSError:
            pass

    def _enforce_champion_word_cap(self) -> None:
        """Rollback graft bloat that bypassed challenger gates (timeout/archive paths)."""
        if self.champion is None or self.champion_level < 21:
            return
        from evolution.champion_bloat_guard import level_word_cap

        cap = level_word_cap(self.champion_level)
        if cap is None or len(self.champion) <= cap:
            return
        if self._maybe_restore_best_certified_champion():
            return
        from evolution.register_repair import reference_champion_program

        words, desc = reference_champion_program(self.champion_level)
        if not words:
            return
        self._log(
            f"[Champion] Word-cap rollback L{self.champion_level}: " f"{len(self.champion)}w -> {len(words)}w ({desc})"
        )
        self._sync_champion_after_reseed(words, desc, "Word-cap rollback")

    def _maybe_restore_l30_working_skeleton(self) -> bool:
        """Undo immediate-only 24w collapse back to the last 13w working skeleton."""
        if self.champion is None or int(self.champion_level) < int(self.config.max_level):
            return False
        if not (15 <= len(self.champion) <= 28):
            return False
        live = float(self._verify_score_from_trace(self.champion_trace or {}))
        if live >= 0.25:
            return False
        n_li = 0
        for word in self.champion:
            if (int(word) & 0x7F) == 0x13 and ((int(word) >> 15) & 0x1F) == 0:
                n_li += 1
        if n_li < 5:
            return False
        import json
        from pathlib import Path

        results = Path(__file__).resolve().parent / "multi_cycle_results.jsonl"
        skeleton = None
        with results.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("champion_words", 0)) != 13:
                    continue
                if int(row.get("final_level", 0)) != 30:
                    continue
                prog = row.get("champion_program")
                if isinstance(prog, list) and len(prog) == 13:
                    skeleton = [int(x, 16) if isinstance(x, str) else int(x) for x in prog]
        if not skeleton:
            return False
        prev_n = len(self.champion)
        self.champion = list(skeleton)
        self.champion_trace = self._evaluate(self.champion)
        verify = float(self._verify_score_from_trace(self.champion_trace))
        self._champion_verify_score = verify
        self._champion_partial_verify = verify
        self._log(f"[Champion] Restore 13w skeleton from {prev_n}w li-collapse")
        return True

    def _maybe_apply_l30_easy_fill(self) -> bool:
        """Install the compact L30 missing-reg bundle when it is strictly longer."""
        import traceback
        from pathlib import Path

        marker = Path(__file__).resolve().parent / ".agents" / "easy_fill_apply.log"
        try:
            if self.champion is None or int(self.champion_level) < int(self.config.max_level):
                marker.write_text(
                    f"skip level={getattr(self, 'champion_level', None)} "
                    f"max={getattr(self.config, 'max_level', None)} "
                    f"n={len(self.champion) if self.champion else 0}\n",
                    encoding="utf-8",
                )
                return False
            from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS, l30_easy_fill_variants

            fills = l30_easy_fill_variants(list(self.champion))
            if not fills:
                marker.write_text("skip no fills\n", encoding="utf-8")
                return False
            fill = fills[0]
            if not (12 <= len(fill) <= L30_COMPACT_MAX_WORDS) or len(fill) <= len(self.champion):
                marker.write_text(
                    f"skip size champ={len(self.champion)} fill={len(fill)}\n",
                    encoding="utf-8",
                )
                return False
            trace = self._evaluate(fill)
            prev_n = len(self.champion)
            self.champion = list(fill)
            self.champion_trace = trace
            verify = float(self._verify_score_from_trace(trace))
            self._champion_verify_score = verify
            self._champion_partial_verify = verify
            self._champion_verified = verify >= 1.0 - 1e-6
            self.champion_fitness = float(self._compute_fitness(fill, trace))
            marker.write_text(
                f"applied {prev_n}->{len(fill)} verify={verify:.3f}\n",
                encoding="utf-8",
            )
            self._log(f"[Champion] L30 easy-fill {prev_n}w -> {len(fill)}w verify={verify:.3f}")
            return True
        except Exception as exc:  # noqa: BLE001
            marker.write_text(f"error {exc}\n{traceback.format_exc()}", encoding="utf-8")
            return False

    def _maybe_reseed_stuck_champion(self) -> None:
        """Replace stuck L17+ champion with reference solution when verify/fitness plateau."""

        from evolution.register_repair import reference_champion_program, should_reseed_champion

        verify = float(getattr(self, "_champion_verify_score", 0.0))

        if not should_reseed_champion(self.champion_level, verify, self.champion_fitness):

            return

        words, desc = reference_champion_program(self.champion_level)

        if not words:

            return

        prev_champion = list(self.champion) if self.champion else None
        prev_trace = self.champion_trace
        prev_fitness = self.champion_fitness
        prev_verify = verify
        self._sync_champion_after_reseed(words, desc, "Reseeding stuck champion")
        new_verify = float(getattr(self, "_champion_verify_score", 0.0))
        if prev_champion and new_verify < prev_verify - 0.05:
            self.champion = prev_champion
            self.champion_trace = prev_trace
            self.champion_fitness = prev_fitness
            self._champion_verify_score = prev_verify
            self._champion_partial_verify = prev_verify
            self._log(f"[RegisterRepair] Reseed rolled back verify {prev_verify:.3f} -> " f"{new_verify:.3f}")

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

        if gen % 10 == 0:
            self._maybe_reseed_stuck_champion()

    def _run_generation_evaluate(self) -> tuple:
        """Generate challengers, evaluate, select elite."""

        with self._pt("mutation_propagation"), self._sub("generate_challengers"):
            with self._pt_sub("challenger_generation"):
                challengers = self._generate_challengers()

            with self._sub("topup_diversity"), self._pt_sub("diversity_topup"):
                self._maybe_topup_scan_floor_diversity(challengers)

            with self._pt_sub("length_capping"):
                try:
                    from evolution.mutation_helpers import (
                        cap_scan_floor_challenger_lengths,
                        scan_floor_stalled_challenger_max_words,
                    )
                    from evolution.terminal_stagnation import (
                        L30_COMPACT_MAX_WORDS,
                        _scan_floor_probe_stall_mode,
                        get_scan_floor_gate,
                    )

                    champ = list(self.champion or [])
                    if champ and _scan_floor_probe_stall_mode(len(champ)):
                        inject_gen = int(
                            getattr(
                                self,
                                "_scan_floor_inject_generation",
                                max(0, int(self.config.max_generations) - 1),
                            )
                        )
                        word_cap = scan_floor_stalled_challenger_max_words(
                            len(champ),
                            generation=int(getattr(self, "current_generation", 0)),
                            mg_inject_generation=inject_gen,
                            default_max=int(L30_COMPACT_MAX_WORDS),
                            stalled=True,
                        )
                        challengers = cap_scan_floor_challenger_lengths(
                            challengers,
                            champ,
                            max_words=word_cap,
                            gate=get_scan_floor_gate(champ),
                            cycle_no=int(getattr(self, "_cycle_number", 0)),
                        )
                except ImportError:
                    pass

            # Deliberately UNMARKED (EE-W6 limitation 3): bookkeeping only,
            # its time falls into the `unattributed` residual of the
            # mutation_subphase_seconds partition.
            if challengers:
                gen_max = max(len(p) for p in challengers)
                self._max_challenger_words_cycle = max(int(getattr(self, "_max_challenger_words_cycle", 0)), gen_max)

        with self._pt("diagnostics_logging"):
            self._current_diversity = self._compute_population_diversity(challengers)

        with self._pt("mutation_propagation"), self._sub("dedupe"), self._pt_sub("dedupe"):
            from evolution.mutation_helpers import dedupe_programs_for_eval, expand_eval_results

            unique_challengers, dedupe_map = dedupe_programs_for_eval(challengers)
        with self._pt("vm_eval"):
            unique_results = self._evaluate_batch(unique_challengers)
            results = expand_eval_results(unique_results, dedupe_map)

        with self._pt("diagnostics_logging"):
            self._last_challengers = challengers

            self._last_results = results

            try:
                from evolution.mutation_helpers import (
                    record_l10_lift_growth_discoveries,
                    scan_floor_late_inject_generation,
                )

                if self.champion is not None:
                    gen_idx = int(getattr(self, "current_generation", 0))
                    growth_inject_gen = getattr(self, "_scan_floor_growth_inject_generation", None)
                    mg_inject_gen = getattr(self, "_scan_floor_inject_generation", None)
                    late_inject = scan_floor_late_inject_generation(
                        gen_idx,
                        growth_inject_generation=growth_inject_gen,
                        mg_inject_generation=mg_inject_gen,
                    )
                    sample_discovery = gen_idx == 0 or late_inject or gen_idx % 10 == 0
                    if sample_discovery:
                        state = getattr(self, "_l10_lift_discovery_state", None)
                        if not isinstance(state, dict):
                            state = {}
                            self._l10_lift_discovery_state = state
                        record_l10_lift_growth_discoveries(
                            list(self.champion),
                            challengers,
                            cycle=int(getattr(self, "_cycle_number", 0)),
                            state=state,
                        )
            except (ImportError, TypeError, ValueError, OSError):
                pass

        with self._pt("selection_archives"):
            self._fill_empty_qd_cells_from_evaluated(challengers, results, self._last_challenger_strategies)

            elite, elite_fitnesses, _ = self._select_elite(challengers, results)

        with self._pt("vm_eval"):
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

        self._stall_prev_program = tuple(self.champion) if self.champion else ()

        self._stall_prev_level = int(self.champion_level)
        self._stall_prev_fitness = float(prev_fitness or 0.0)

        improved = self._update_champion(elite, detailed_results, stage1_fitnesses=elite_fitnesses)
        if self._maybe_restore_l30_working_skeleton():
            improved = True

        self._enforce_champion_word_cap()

        if improved and hasattr(self, "champion_archive") and self.champion_archive is not None:

            try:

                _pruned = self.champion_archive.quality_weighted_prune()

                if _pruned > 0:

                    self._log(f"[Champion] Quality prune: removed {_pruned}")

            except Exception:  # noqa: BLE001, S110

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

        except Exception:  # noqa: BLE001, S110

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

            except Exception:  # noqa: BLE001, S110

                pass

        if gen > 0 and checkpoint_interval and gen % checkpoint_interval == 0:

            try:

                self._checkpoint()

            except (RuntimeError, OSError) as e:

                self._log(f"[Checkpoint] Failed at gen {gen}: {e}")

        prev_prog = getattr(self, "_stall_prev_program", ())
        prev_lvl = getattr(self, "_stall_prev_level", int(self.champion_level))
        prev_fit = float(getattr(self, "_stall_prev_fitness", 0.0) or 0.0)
        prev_words = len(prev_prog) if prev_prog else 0
        fitter = float(self.champion_fitness) > prev_fit + 1e-6
        shorter = len(self.champion or []) < prev_words
        level_up = int(self.champion_level) > int(prev_lvl)

        if fitter or shorter or level_up:

            self.generations_stalled = 0

            self._qd_champion_reset_tried = False

        else:

            self.generations_stalled += 1

        self._try_recurrent_feedback()

        if auto_curriculum:

            self._run_progress_controller_generation()

        # Level circuit breaker: record outcome after curriculum check

        if self.config.level_circuit_breaker_enabled:

            self._record_cycle_level_outcome()

        self._run_generation_meta_monitoring(gen)

        self._record_strategy_outcomes()

        self._update_qd_archive()

        pc = getattr(self, "progress_controller", None)

        if pc is None and self.generations_stalled >= 35 and self._try_qd_champion_reset():

            self._log(f"[QD Reset] Champion replaced from QD archive at gen {gen}")

        self._update_novelty_archive()

    def _run_progress_controller_generation(self) -> None:
        """Unified per-generation progress decisions."""

        pc = getattr(self, "progress_controller", None)

        if pc is None:

            self._try_curriculum_advance()

            self._anti_plateau_escalation()

            if self.generations_stalled >= 35:

                self._try_qd_champion_reset()

            return

        from evolution.progress_controller import ProgressAction

        _verified, verify_score = self._verify_champion()

        self._champion_verify_score = verify_score

        plan = pc.on_generation_end(
            self.generations_stalled,
            self.champion_level,
            verify_score,
            self._champion_raw_task_score,
            self.champion_fitness,
        )

        for line in plan.log_lines:

            self._log(line)

        if ProgressAction.ADVANCE_LEVEL in plan.actions and not self._try_curriculum_advance():
            self._try_forced_curriculum_escape()

        if ProgressAction.REFRESH_TASK in plan.actions:

            self._refresh_task_instance()

        if ProgressAction.QD_RESET in plan.actions:

            self._try_qd_champion_reset()

    def _run_generation_meta_monitoring(self, gen: int) -> None:
        """Run meta-controller monitoring and diagnostics."""

        if gen > 0 and gen % 25 == 0 and hasattr(self, "meta_controller") and self.meta_controller is not None:

            try:

                _rc = self.meta_controller.get_recovery_confidence()

                if _rc["is_recovering"]:

                    self._log(f"[Recovery] confidence={_rc['confidence']:.2f} exit={_rc['should_exit_recovery']}")

                    if _rc["should_exit_recovery"]:

                        self.meta_controller.recovery_mode = False

            except Exception:  # noqa: BLE001, S110

                pass

        if self.config.use_strategy_archive and self.strategy_archive is not None:

            try:

                _dom = self.strategy_archive.get_dominance_report()

                if _dom.get("is_monoculture") and gen > 0 and gen % 10 == 0:

                    self._log(f"[Strategy] Monoculture: {_dom['dominant_strategy']} at {_dom['dominance_ratio']:.0%}")

            except Exception:  # noqa: BLE001, S110

                pass

    def _pt(self, name: str):
        """Return the phase-timing context for ``name`` (no-op when idle)."""
        _timer = getattr(self, "_phase_timer", None)
        if _timer is None or not _timer.enabled:
            return nullcontext()
        return _timer.phase(name)

    def _pt_sub(self, name: str):
        """Return the ``mutation_propagation`` sub-phase context for ``name``.

        EE-W6: sub-phases split the mutation/propagation bucket into named,
        additive blocks (``evolution.phase_timing.MUTATION_SUBPHASES``) so a
        dominant block is visible instead of hiding inside one bucket.  Like
        ``_pt``, it is a no-op when the timer is absent or disabled.

        Distinct from ``_sub`` below: this one is VM-subtracted and the
        sub-phases it opens partition their phase (they sum to it, plus the
        ``unattributed`` residual), so it writes ``mutation_subphase_seconds``.
        """
        _timer = getattr(self, "_phase_timer", None)
        if _timer is None or not _timer.enabled:
            return nullcontext()
        return _timer.sub_phase(name)

    def _sub(self, name: str):
        """Return the sub-block marker context for ``name`` (no-op when idle).

        Diagnostic hotspot markers (growth pool rebuild, locus probe,
        enrich, pair-drop trials, ...) recorded into each generation's
        ``subphase_seconds`` and surfaced as a ``[PhaseSub]`` log line.

        Deliberately NOT the same accounting as ``_pt_sub``: markers nest
        inside each other and do not subtract VM time, so they sum to more
        than their phase -- a hotspot map, not a partition.  The name
        mapping between the two systems lives in ``MUTATION_SUBPHASES``.
        """
        _timer = getattr(self, "_phase_timer", None)
        if _timer is None or not _timer.enabled:
            return nullcontext()
        from evolution.phase_timing import subphase

        return subphase(name)

    def _log_subphase_marker(self, record: dict | None) -> None:
        """Emit a ``[PhaseSub]`` line when a sub-block dominates a generation."""
        if not record:
            return
        try:
            subs = record.get("subphase_seconds") or {}
            if not subs:
                return
            worst = max(subs.values())
            if worst < float(os.environ.get("EE_PHASE_SUB_LOG_MIN", "3")):
                return
            top = ", ".join(f"{k}={v:.1f}" for k, v in list(subs.items())[:5])
            self._log(
                f"[PhaseSub] gen={record.get('generation')} "
                f"total={record.get('generation_seconds', 0.0):.1f}s top: {top}"
            )
        except Exception:  # noqa: BLE001 -- markers must never break evolution
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

        timer = getattr(self, "_phase_timer", None)
        if timer is not None:
            timer.begin_generation(gen)
        with self._pt("mutation_propagation"), self._sub("gen_init"):
            with self._pt_sub("generation_init"):
                self._run_generation_init(gen, n_levels)
        try:
            from evolution.mutation_helpers import clear_shrink_challenger_cache

            clear_shrink_challenger_cache()
        except ImportError:
            pass

        with self._sub("generation_evaluate"):
            _, _, elite, elite_fitnesses, detailed_results = self._run_generation_evaluate()

        with self._pt("selection_archives"):
            improved = self._run_generation_update(
                gen, elite, detailed_results, elite_fitnesses, False, checkpoint_interval, auto_curriculum
            )

        with self._pt("diagnostics_logging"):
            self._log_generation(gen, gen_start)

        with self._pt("selection_archives"):
            self._run_generation_maintenance(
                gen,
                improved,
                task_refresh_interval,
                cuda_cleanup_interval,
                _cuda_cleanup,
                checkpoint_interval,
                auto_curriculum,
            )
        if timer is not None:
            _gen_record = timer.end_generation(
                {
                    "generations_stalled": int(self.generations_stalled),
                    "champion_level": int(self.champion_level),
                }
            )
            self._log_subphase_marker(_gen_record)

    def _resolve_cuda_cleanup_callback(self) -> Callable[[], None] | None:
        """Return a CUDA cache cleanup callback when GPU mode is active."""

        use_gpu = self.use_gpu if hasattr(self, "use_gpu") else self.config.use_gpu

        if not use_gpu:

            return None

        try:

            import gc as _gc

            import torch as _torch

            if _torch.cuda.is_available():

                def _cuda_cleanup():

                    _gc.collect()

                    _torch.cuda.empty_cache()

                return _cuda_cleanup

        except ImportError:

            pass

        return None

    def _execute_generation_loop(
        self,
        ng: int,
        checkpoint_interval: int,
        auto_curriculum: bool,
        task_refresh_interval: int,
        cuda_cleanup_interval: int,
        cuda_cleanup: Callable[[], None] | None,
        n_levels: int,
    ) -> None:
        """Run all generations for the current evolution session."""

        if hasattr(self, "_raw_task_score"):

            self._champion_raw_task_score = self._raw_task_score

        self._scan_floor_inject_generation = max(0, int(ng) - 1)
        growth_offset = max(1, int(os.environ.get("EE_SCAN_FLOOR_GROWTH_INJECT_OFFSET", "2")))
        self._scan_floor_growth_inject_generation = max(0, int(ng) - growth_offset)

        for gen in range(ng):

            self._run_one_generation(
                gen,
                checkpoint_interval,
                auto_curriculum,
                task_refresh_interval,
                cuda_cleanup_interval,
                cuda_cleanup,
                n_levels,
            )

    def _finalize_run_summary(self, summary: dict[str, Any], ng: int, start_time: float) -> dict[str, Any]:
        """Checkpoint, populate summary fields, and return the run summary."""

        try:
            self._finalize_recurrent_buffer()
        except Exception:  # noqa: BLE001, S110
            pass

        checkpoint_interval = self.config.checkpoint_interval

        if checkpoint_interval:

            try:

                self._checkpoint()

            except (RuntimeError, OSError) as e:

                self._log(f"[Checkpoint] Failed at final: {e}")

        summary["generations_completed"] = ng

        summary["final_fitness"] = self.champion_fitness

        summary["final_level"] = self.champion_level

        summary["champion_words"] = len(self.champion) if self.champion else 0

        summary["elapsed_seconds"] = time.time() - start_time

        verified, verify_score = self._verify_champion()

        max_pop_verify = float(getattr(self, "_last_max_pop_verify", 0.0))
        champion_verify = float(verify_score)
        verified = champion_verify >= 1.0 - 1e-6

        self._champion_verified = verified

        self._champion_verify_score = champion_verify

        summary["champion_verified"] = verified

        summary["champion_verify_score"] = champion_verify

        summary["max_pop_verify"] = max_pop_verify

        effective_verify = max(champion_verify, max_pop_verify)

        if self.champion is not None and self.champion_trace is not None:
            try:
                from evolution.fitness_helpers import end_of_cycle_champion_fitness

                recomputed = end_of_cycle_champion_fitness(
                    self,
                    self.champion,
                    self.champion_trace,
                    champion_verify,
                )
                self.champion_fitness = recomputed
                summary["final_fitness"] = recomputed
            except (RuntimeError, TypeError, ValueError, AttributeError):
                pass

        summary["verified_fitness"] = self.champion_fitness if verified else 0.0

        if self.config.auto_curriculum:

            pc = getattr(self, "progress_controller", None)

            if pc is not None:

                raw = self._champion_raw_task_score

                fit = self.champion_fitness

                plan = pc.on_run_end(self.champion_level, effective_verify, raw, fit)

                for line in plan.log_lines:

                    self._log(line)

                from evolution.progress_controller import ProgressAction

                if ProgressAction.ADVANCE_LEVEL in plan.actions:

                    saved_stall = self.generations_stalled

                    self.generations_stalled = max(saved_stall, self.config.curriculum_stall_threshold)

                    try:

                        self._try_end_of_run_curriculum_push()

                    finally:

                        self.generations_stalled = saved_stall

            elif self._try_end_of_run_curriculum_push():

                pass

            summary["final_level"] = self.champion_level

            summary["champion_verified"] = verified

            summary["champion_verify_score"] = champion_verify

            summary["verified_fitness"] = self.champion_fitness if verified else 0.0

        if self.champion is not None:
            if hasattr(self, "_enforce_l30_certified_champion"):
                self._enforce_l30_certified_champion()
            else:
                self._maybe_restore_best_certified_champion()

        # Terminal efficiency promotion runs once post-cycle in run_multi_cycle.py
        # (calling here too doubled held-out variant screening every cycle).
        self._maybe_promote_max_pop_verify_champion()
        verified, verify_score = self._verify_champion()
        champion_verify = max(
            float(verify_score),
            float(getattr(self, "_champion_verify_score", 0.0)),
        )
        verified = champion_verify >= 1.0 - 1e-6
        self._champion_verified = verified
        self._champion_verify_score = champion_verify
        summary["champion_verified"] = verified
        summary["champion_verify_score"] = champion_verify
        summary["verified_fitness"] = self.champion_fitness if verified else 0.0
        if self.champion is not None and self.champion_trace is not None:
            try:
                from evolution.fitness_helpers import end_of_cycle_champion_fitness, terminal_certified_fitness_context

                with terminal_certified_fitness_context(self):
                    recomputed = end_of_cycle_champion_fitness(
                        self,
                        self.champion,
                        self.champion_trace,
                        champion_verify,
                    )
                self.champion_fitness = recomputed
                summary["final_fitness"] = recomputed
            except (RuntimeError, TypeError, ValueError, AttributeError):
                pass

        if self.champion is not None:
            verified, verify_score = self._verify_champion()
            champion_verify = float(verify_score)
            verified = champion_verify >= 1.0 - 1e-6
            self._champion_verified = verified
            self._champion_verify_score = champion_verify
            summary["champion_verified"] = verified
            summary["champion_verify_score"] = champion_verify
            summary["champion_words"] = len(self.champion) if self.champion else 0
            if self.champion_trace is not None:
                try:
                    from evolution.fitness_helpers import (
                        end_of_cycle_champion_fitness,
                        terminal_certified_fitness_context,
                    )

                    with terminal_certified_fitness_context(self):
                        recomputed = end_of_cycle_champion_fitness(
                            self,
                            self.champion,
                            self.champion_trace,
                            champion_verify,
                        )
                    self.champion_fitness = recomputed
                    summary["final_fitness"] = recomputed
                except (RuntimeError, TypeError, ValueError, AttributeError):
                    pass
            try:
                from evolution.champion_rollback import apply_l30_certified_metrics_if_degraded

                project_root = os.path.dirname(os.path.abspath(__file__))
                apply_l30_certified_metrics_if_degraded(project_root, self, summary)
            except (ImportError, RuntimeError, TypeError, ValueError, OSError):
                pass
            summary["verified_fitness"] = self.champion_fitness if verified else 0.0

        return summary

    def run(self, num_generations: int | None = None) -> dict[str, Any]:
        """Run the evolution loop for num_generations (default: config)."""

        ng = num_generations if num_generations is not None else self.config.max_generations

        start_time = time.time()
        timer = PhaseTimer(enabled=timers_enabled())
        timer.set_cycle(int(getattr(self, "_cycle_number", 0)))
        timer.set_log_path(default_log_path(os.path.dirname(os.path.abspath(__file__))))
        self._phase_timer = timer
        run_perf_start = time.perf_counter()

        try:
            from evolution.register_repair import register_repair_boost_active

            if register_repair_boost_active():
                os.environ["EE_REGISTER_REPAIR_BOOST"] = "1"
        except ImportError:
            pass

        checkpoint_interval = self.config.checkpoint_interval

        auto_curriculum = self.config.auto_curriculum

        task_refresh_interval = 25

        cuda_cleanup_interval = 10

        n_levels = len(LEVELS)

        cuda_cleanup = self._resolve_cuda_cleanup_callback()

        _init_perf = time.perf_counter()
        summary = self._initialize_run(ng)
        timer.run_init_seconds = time.perf_counter() - _init_perf
        if hasattr(self, "_clear_scan_floor_growth_crossover_cache"):
            self._clear_scan_floor_growth_crossover_cache()
        from evolution.mutation_helpers import clear_scan_floor_inject_caches

        clear_scan_floor_inject_caches()
        from evolution.mutation_helpers import prime_scan_floor_growth_pool_cache

        self._scan_floor_growth_pool_cache = prime_scan_floor_growth_pool_cache(
            list(self.champion or []),
            getattr(self, "_scan_floor_growth_pool_cache", None),
        )

        try:

            self._execute_generation_loop(
                ng,
                checkpoint_interval,
                auto_curriculum,
                task_refresh_interval,
                cuda_cleanup_interval,
                cuda_cleanup,
                n_levels,
            )

        except Exception as exc:  # noqa: BLE001
            import traceback as _tb

            detail = f"[Run] Generation loop aborted: {type(exc).__name__}: {exc}"
            try:
                self._log(detail)
            except Exception:  # noqa: BLE001
                pass
            try:
                sys.stderr.write(detail + "\n" + _tb.format_exc())
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass

        finally:

            self._close_log_handle()

        _fin_perf = time.perf_counter()
        summary = self._finalize_run_summary(summary, ng, start_time)
        timer.run_finalize_seconds = time.perf_counter() - _fin_perf
        for _key, _value in timer.summary(time.perf_counter() - run_perf_start).items():
            summary[_key] = _value
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

        coerced: list[int] = []
        for w in program:
            try:
                if isinstance(w, str):
                    val = int(w, 0)
                else:
                    val = int(w)
            except (TypeError, ValueError):
                return False, "non_integer_word"
            coerced.append(val)

        # Check for overflow/underflow first (before EBREAK stripping)

        if any(w > 0xFFFFFFFF or w < 0 for w in coerced):

            return False, "overflow"

        # Remove trailing EBREAKs for length check

        live = coerced

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

        budget = self._cached_eval_budget

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

        budget = self._cached_eval_budget // 2

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

        budget = self._cached_eval_budget

        inputs_list = [inputs] * len(programs)

        results = self.vm_farm.evaluate_batch(programs, inputs_list, budget=budget)

        return [self._eval_result_to_trace(r) for r in results]

    @staticmethod
    def _null_eval_crash_trace() -> dict[str, Any]:
        """Trace dict for a missing/null EvalResult (R35)."""

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

    @staticmethod
    def _build_eval_trace_dict(
        success: bool,
        registers: Any,
        memory: Any,
        cycles: int,
        error: str | None,
    ) -> dict[str, Any]:
        """Build standardized trace dict from evaluation outcome fields."""

        return {
            "status": "ok" if success else "crash",
            "final_pc": 0,
            "final_regs": registers,
            "final_memory": memory,
            "instructions_count": cycles,
            "pc_history": _EMPTY_TUPLE,
            "reg_history": _EMPTY_TUPLE,
            "memory_accesses": _EMPTY_TUPLE,
            "trap_cause": error or "",
        }

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

            return EvolutionEngine._null_eval_crash_trace()

        try:

            return EvolutionEngine._build_eval_trace_dict(
                result.success,
                result.registers,
                result.memory,
                result.cycles,
                result.error,
            )

        except AttributeError:

            return EvolutionEngine._build_eval_trace_dict(
                getattr(result, "success", True),
                getattr(result, "registers", {}),
                getattr(result, "memory", {}),
                getattr(result, "cycles", 0),
                getattr(result, "error", None),
            )

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

        packed = self._evaluate_secondary_tasks_many([program])

        return packed[0] if packed else []

    def _evaluate_secondary_tasks_many(self, programs: list[list[int]]) -> list[list[tuple[dict, dict]]]:
        """Evaluate each program on freshly generated secondary instances.

        One VM batch for all (program, task) pairs. RNG order matches calling
        ``_generate_secondary_tasks`` once per program in the given order.
        """
        vm = getattr(self, "vm_farm", None)
        if vm is None or not programs:
            return [[] for _ in programs]
        budget = self._cached_eval_budget
        task_lists: list[list[dict]] = []
        flat_progs: list[list[int]] = []
        flat_inputs: list[dict] = []
        for prog in programs:
            tasks = self._generate_secondary_tasks()
            task_lists.append(tasks)
            for task_inst in tasks:
                flat_progs.append(prog)
                flat_inputs.append(task_inst.get("inputs", {}))
        if not flat_progs:
            return [[] for _ in programs]
        try:
            raw_list = vm.evaluate_batch(flat_progs, flat_inputs, budget=budget)
        except RuntimeError:
            raw_list = [None] * len(flat_progs)
        out: list[list[tuple[dict, dict]]] = []
        offset = 0
        for tasks in task_lists:
            chunk: list[tuple[dict, dict]] = []
            for task_inst in tasks:
                raw = raw_list[offset] if offset < len(raw_list) else None
                offset += 1
                try:
                    trace = self._eval_result_to_trace(raw)
                except RuntimeError:
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
                chunk.append((task_inst, trace))
            out.append(chunk)
        return out

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

        except RuntimeError:

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

    def _refresh_eval_budget(self) -> None:
        """Scale eval budget for high-curriculum tasks (meta-JALR needs more steps)."""
        from evolution.register_repair import scaled_eval_budget

        self._cached_eval_budget = scaled_eval_budget(self.config.eval_budget, self.champion_level)

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

        if os.environ.get("EE_FORCE_REFERENCE_CHAMPION", "0") == "1":
            try:
                ref = reference_solution(self.champion_level)
                encoded = []
                for inst in ref:
                    w = _encode_inst(inst)
                    if w is not None:
                        encoded.append(w)
                if encoded:
                    if encoded[-1] != _EBREAK_INST:
                        encoded.append(0x0010_0073)
                    self._log(
                        f"[Champion] Forced reference champion for L{self.champion_level} " f"({len(encoded)} words)"
                    )
                    return encoded
            except (ValueError, ImportError):
                pass

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

    def _effective_diversity_threshold(self) -> float:
        """Opcode-category floor; higher at held-out scan floor when probes are gated."""
        base = float(self.config.diversity_threshold)
        if self.champion is None or int(self.champion_level) < int(self.config.max_level):
            return base
        from evolution.terminal_stagnation import scan_floor_diversity_threshold, terminal_efficiency_at_scan_floor

        if terminal_efficiency_at_scan_floor(len(self.champion)):
            return max(base, scan_floor_diversity_threshold())
        return base

    def _maybe_topup_scan_floor_diversity(self, challengers: list[list[int]]) -> None:
        """Final opcode-category top-up when probe gating leaves a monoculture tail."""
        if not challengers or self.champion is None:
            return
        from evolution.diversity_helpers import population_category_bits
        from evolution.terminal_stagnation import scan_floor_diversity_threshold, terminal_efficiency_at_scan_floor

        if not terminal_efficiency_at_scan_floor(len(self.champion)):
            return
        from evolution.mutation_helpers import (
            scan_floor_l10_lift_preserve_indices,
            scan_floor_late_inject_generation,
            swap_inject_overlength_for_eq_seeds,
        )
        from evolution.terminal_stagnation import get_scan_floor_gate

        inject_gate = get_scan_floor_gate(list(self.champion))
        preserve_slots = scan_floor_l10_lift_preserve_indices(
            list(self.champion),
            challengers,
            gate=inject_gate,
        )

        growth_inject_gen = getattr(self, "_scan_floor_growth_inject_generation", None)
        mg_inject_gen = getattr(self, "_scan_floor_inject_generation", None)
        gen_idx = int(getattr(self, "current_generation", 0))
        late_inject = scan_floor_late_inject_generation(
            gen_idx,
            growth_inject_generation=growth_inject_gen,
            mg_inject_generation=mg_inject_gen,
        )
        floor_min = scan_floor_diversity_threshold()
        all_mask = (1 << NUM_OPCODE_CATEGORIES) - 1
        n_inject = max(2, int(len(challengers) * max(self.config.diversity_inject_ratio, 0.2)))
        for _ in range(6):
            div = self._compute_population_diversity(challengers)
            bits = population_category_bits(challengers, _OPCODE_CAT_ARRAY)
            if div >= floor_min and bits == all_mask:
                if not late_inject:
                    return
                break
            self._inject_diversity(challengers, n_inject, preserve_indices=preserve_slots)
        from evolution.diversity_helpers import generate_category_seed

        bits = population_category_bits(challengers, _OPCODE_CAT_ARRAY)
        missing = [cat for cat in range(NUM_OPCODE_CATEGORIES) if not (bits & (1 << cat))]
        if missing:
            slots = [slot for slot in range(len(challengers)) if slot not in preserve_slots]
            self.rng.shuffle(slots)
            for cat, slot in zip(missing, slots, strict=False):
                challengers[slot] = generate_category_seed(cat, self.rng)
                if slot < len(self._last_challenger_strategies):
                    self._last_challenger_strategies[slot] = "DIVERSITY_INJECT"
        category_saturated = bits == all_mask
        if late_inject:
            swap_inject_overlength_for_eq_seeds(
                list(self.champion),
                challengers,
                cycle_no=int(getattr(self, "_cycle_number", 0)),
                generation=gen_idx,
                growth_inject_generation=growth_inject_gen,
                mg_inject_generation=mg_inject_gen,
                strategies=getattr(self, "_last_challenger_strategies", None),
                category_saturated=category_saturated,
            )

    def _generate_diversity_injection(self, challengers, pop_size):
        """Inject diversity when opcode-category diversity falls below threshold."""

        if len(challengers) < pop_size:

            return

        div = self._compute_population_diversity(challengers)

        if div >= self._effective_diversity_threshold():

            return

        n_inject = max(1, int(pop_size * self.config.diversity_inject_ratio))

        self._inject_diversity(challengers, n_inject)

    def _terminal_archive_parents(self) -> list[list[int]]:
        """Gate-neutral archive parents for scan-floor crossover (HoF + QD cells)."""
        if self.champion is None:
            return []
        champ_n = len(self.champion)
        seen: set[tuple[int, ...]] = {tuple(self.champion)}
        parents: list[list[int]] = []
        hof = getattr(self, "_hall_of_fame", None)
        if hof is not None:
            for prog in hof.programs_at_level(int(self.champion_level)):
                if len(prog) != champ_n:
                    continue
                key = tuple(prog)
                if key in seen:
                    continue
                seen.add(key)
                parents.append(list(prog))
        if hasattr(self, "_distinct_qd_cell_programs"):
            for prog in self._distinct_qd_cell_programs():
                if len(prog) != champ_n:
                    continue
                key = tuple(prog)
                if key in seen:
                    continue
                seen.add(key)
                parents.append(list(prog))
        try:
            from evolution.champion_rollback import collect_jsonl_archive_parents
            from evolution.terminal_stagnation import (
                _scan_floor_probe_stall_mode,
                scan_floor_compression_hot,
                terminal_efficiency_at_scan_floor,
            )

            results = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "multi_cycle_results.jsonl",
            )
            word_tol = 2 if terminal_efficiency_at_scan_floor(champ_n) else 0
            max_parents = 12
            rank_by_cap = False

            if terminal_efficiency_at_scan_floor(champ_n) and _scan_floor_probe_stall_mode(champ_n):
                from evolution.terminal_stagnation import _skip_empty_compression_rescan

                compression_exhausted = scan_floor_compression_hot(list(self.champion))
                if not compression_exhausted:
                    compression_exhausted = _skip_empty_compression_rescan(
                        tuple(int(x) for x in self.champion),
                        champ_n,
                    )
                if not compression_exhausted:
                    max_parents = max(
                        24,
                        int(os.environ.get("EE_SCAN_FLOOR_ARCHIVE_PARENTS_EXHAUSTED", "24")),
                    )
                    word_tol = max(
                        6,
                        int(os.environ.get("EE_SCAN_FLOOR_ARCHIVE_WORD_TOL_EXHAUSTED", "12")),
                    )
                    rank_by_cap = True
            for prog in collect_jsonl_archive_parents(
                results,
                list(self.champion),
                int(self.champion_level),
                max_parents=max_parents,
                word_tolerance=word_tol,
                rank_by_promotion_raw=rank_by_cap,
            ):
                key = tuple(prog)
                if key in seen:
                    continue
                seen.add(key)
                parents.append(list(prog))
        except (ImportError, OSError, TypeError, ValueError):
            pass
        return parents

    def _terminal_probe_pool(self, pop_size: int) -> list[list[int]]:
        """Qualified terminal probes for this cycle/champion (built once, sampled per gen)."""
        if self.champion is None:
            return []
        from evolution.fitness_helpers import compact_l30_promotion_raw_fitness
        from evolution.terminal_stagnation import select_terminal_probes_for_batch

        cycle_no = int(getattr(self, "_cycle_number", 0))
        champ_pr = round(float(compact_l30_promotion_raw_fitness(list(self.champion))), 4)
        champ_key = (champ_pr, len(self.champion))
        cache = getattr(self, "_terminal_probe_pool_cache", None)
        if cache and cache[0] == cycle_no and cache[1] == champ_key:
            return list(cache[2])
        from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor, terminal_probe_escalation_ratio

        ratio = terminal_probe_escalation_ratio()
        n_probe = max(2, int(pop_size * ratio))
        pool_size = max(12, pop_size // 4)
        at_floor = terminal_efficiency_at_scan_floor(len(self.champion))
        if at_floor:
            pool_size = max(pool_size, n_probe)
        archive_parents = self._terminal_archive_parents() if at_floor else None
        pool = select_terminal_probes_for_batch(
            self.champion,
            self.champion_level,
            self.rng,
            pool_size,
            trace=self.champion_trace,
            task_instance=self.task_instance,
            pop_size=pop_size,
            archive_parents=archive_parents,
        )
        self._terminal_probe_pool_cache = (cycle_no, champ_key, tuple(pool))
        return list(pool)

    def _inject_terminal_verify_probes(self, challengers: list[list[int]], pop_size: int) -> None:
        """Replace a slice of challengers with reference-tail variants at max depth."""
        if self.champion is None or self.champion_level < self.config.max_level:
            return
        from evolution.register_repair import terminal_incremental_repair_active
        from evolution.terminal_stagnation import (
            terminal_efficiency_at_scan_floor,
            terminal_probe_escalation_ratio,
            terminal_verify_promote_active,
        )

        at_floor = terminal_efficiency_at_scan_floor(len(self.champion))
        warm_steady = False
        if at_floor and os.environ.get("EE_SCAN_FLOOR_WARM_STEADY_LIGHT", "1") == "1":
            try:
                from evolution.mutation_helpers import should_skip_mg_stall_collect_for_warm_eq_refine

                warm_steady = bool(should_skip_mg_stall_collect_for_warm_eq_refine(list(self.champion or [])))
            except ImportError:
                warm_steady = False

        champ_verify = float(getattr(self, "_champion_verify_score", 0.0))
        verify_saturated = champ_verify >= 0.99 - 1e-6
        if not (
            terminal_incremental_repair_active(self.champion_level)
            or terminal_verify_promote_active()
            or verify_saturated
        ):
            return
        ratio = terminal_probe_escalation_ratio()
        if warm_steady:
            probe_cap = float(os.environ.get("EE_SCAN_FLOOR_WARM_STEADY_PROBE_CAP", "0.12"))
            ratio = min(ratio, max(0.02, probe_cap))
        n_probe = max(2, int(pop_size * ratio))
        pool = self._terminal_probe_pool(pop_size)
        variants = list(pool)
        if variants:
            if at_floor and not warm_steady and len(variants) > n_probe:
                from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                if _scan_floor_probe_stall_mode(len(self.champion or [])):
                    # Pool is prom_raw-ranked; always inject the head every generation.
                    variants = variants[:n_probe]
                else:
                    gen = int(getattr(self, "current_generation", 0))
                    offset = (gen * n_probe) % len(variants)
                    variants = [variants[(offset + i) % len(variants)] for i in range(n_probe)]
            elif not at_floor or warm_steady:
                self.rng.shuffle(variants)
                variants = variants[:n_probe]
            else:
                variants = variants[:n_probe]
        if variants:
            slots = self.rng.sample(range(len(challengers)), min(len(variants), len(challengers)))
            for slot, variant in zip(slots, variants, strict=False):
                challengers[slot] = variant
                if slot < len(self._last_challenger_strategies):
                    self._last_challenger_strategies[slot] = "TERMINAL_VERIFY_PROBE"
        elif at_floor and not warm_steady:
            n_inject = max(2, int(pop_size * max(self.config.diversity_inject_ratio, 0.2)))
            self._inject_diversity(challengers, n_inject)
        from evolution.terminal_stagnation import _l11_fib_gate_imm, l30_easy_fill_variants, semantic_compress

        if not at_floor or warm_steady:
            compact = semantic_compress(list(self.champion))
            if compact and tuple(compact) != tuple(self.champion) and challengers and len(self.champion) > 24:
                challengers[0] = compact
                if self._last_challenger_strategies:
                    self._last_challenger_strategies[0] = "SEMANTIC_COMPRESS"
            if _l11_fib_gate_imm(list(self.champion)) != 13:
                fills = l30_easy_fill_variants(list(self.champion))
                if fills and len(challengers) > 1:
                    challengers[1] = fills[-1]
                    if len(self._last_challenger_strategies) > 1:
                        self._last_challenger_strategies[1] = "L30_EASY_FILL"
        from evolution.mutation_helpers import (
            scan_floor_anchor_program,
            scan_floor_capability_growth_allowed,
            scan_floor_compression_injection_generation,
            scan_floor_growth_injection_generation,
            scan_floor_late_inject_generation,
            scan_floor_stalled_growth_injection_generation,
        )
        from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

        growth_once_per_cycle = os.environ.get("EE_SCAN_FLOOR_GROWTH_ONCE_PER_CYCLE", "1") != "0"
        growth_gen = int(getattr(self, "current_generation", 0))
        growth_inject_gen = getattr(self, "_scan_floor_growth_inject_generation", None)
        mg_inject_gen = getattr(self, "_scan_floor_inject_generation", None)
        late_inject_probe = scan_floor_late_inject_generation(
            growth_gen,
            growth_inject_generation=growth_inject_gen,
            mg_inject_generation=mg_inject_gen,
        )
        stall_growth_inject = _scan_floor_probe_stall_mode(len(self.champion or []))
        growth_inject_ok = (
            scan_floor_stalled_growth_injection_generation(
                growth_gen,
                inject_generation=growth_inject_gen,
                max_generations=int(self.config.max_generations),
                once_per_cycle=growth_once_per_cycle,
            )
            if stall_growth_inject
            else scan_floor_growth_injection_generation(
                growth_gen,
                inject_generation=growth_inject_gen,
                max_generations=int(self.config.max_generations),
                once_per_cycle=growth_once_per_cycle,
            )
        )
        if growth_inject_ok and stall_growth_inject and growth_gen == 0:
            try:
                from evolution.mutation_helpers import should_skip_scan_floor_gen0_growth_inject

                if should_skip_scan_floor_gen0_growth_inject(list(self.champion or [])):
                    growth_inject_ok = False
            except ImportError:
                pass
        if warm_steady and os.environ.get("EE_SCAN_FLOOR_WARM_STEADY_SKIP_GROWTH_INJECT", "1") == "1":
            growth_inject_ok = False
        if scan_floor_capability_growth_allowed(len(self.champion)) and growth_inject_ok:
            anchor = scan_floor_anchor_program(list(self.champion))
            if challengers:
                from evolution.epi_momentum_governor import (
                    promotion_raw_capability_stalled,
                    scan_floor_length_prom_stalled,
                )
                from evolution.terminal_stagnation import terminal_growth_tail_perturb_variants

                root = os.path.dirname(os.path.abspath(__file__))
                champ_n_slots = len(self.champion or [])
                growth_stalled = bool(stall_growth_inject)
                default_slots = (
                    "8"
                    if growth_stalled and scan_floor_capability_growth_allowed(champ_n_slots)
                    else ("2" if scan_floor_capability_growth_allowed(champ_n_slots) else "1")
                )
                cycle_no = int(getattr(self, "_cycle_number", 0))
                from evolution.fitness_helpers import compact_l30_promotion_raw_fitness

                pool_cache_pr = round(
                    float(compact_l30_promotion_raw_fitness(list(self.champion or anchor))),
                    4,
                )
                pool_cache_key = (
                    pool_cache_pr,
                    len(self.champion or anchor),
                )
                cached_pool = getattr(self, "_scan_floor_growth_pool_cache", None)
                pool_cache_hit = cached_pool is not None and cached_pool[0] == pool_cache_key
                if pool_cache_hit:
                    pool = [list(p) for p in cached_pool[1]]
                    cross_keys = {tuple(k) for k in cached_pool[2]}
                elif growth_stalled:
                    from evolution.mutation_helpers import load_scan_floor_growth_pool_cache

                    disk_pool = load_scan_floor_growth_pool_cache(list(self.champion or anchor))
                    if disk_pool is not None and disk_pool[0] == pool_cache_key:
                        pool = [list(p) for p in disk_pool[1]]
                        cross_keys = {tuple(k) for k in disk_pool[2]}
                        pool_cache_hit = True
                        self._scan_floor_growth_pool_cache = disk_pool
                if not pool_cache_hit:
                    from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills

                    growth = collect_scan_floor_growth_crossover_fills(anchor)
                    if not growth:
                        pool = []
                    else:
                        pool = list(growth)
                        cross_keys: set[tuple[int, ...]] = set()
                if pool:
                    n_slots = min(
                        len(pool) + 8,
                        max(
                            1,
                            int(os.environ.get("EE_SCAN_FLOOR_GROWTH_SLOTS", default_slots)),
                        ),
                    )
                    if growth_stalled and growth_gen == 0:
                        gen0_slots = max(
                            1,
                            int(os.environ.get("EE_SCAN_FLOOR_GROWTH_SLOTS_GEN0", "2")),
                        )
                        from evolution.mutation_helpers import _curriculum_l10_cached

                        if _curriculum_l10_cached(list(self.champion)) < 0.85 - 1e-6:
                            gen0_slots = max(
                                gen0_slots,
                                int(
                                    os.environ.get(
                                        "EE_SCAN_FLOOR_L10_LIFT_GROWTH_SLOTS_GEN0",
                                        "3",
                                    )
                                ),
                            )
                        n_slots = min(n_slots, gen0_slots)
                if pool and not pool_cache_hit:
                    if os.environ.get("EE_SCAN_FLOOR_GROWTH_CROSSOVER", "1") != "0":
                        from evolution.mutation_helpers import build_scan_floor_growth_crossover_pool

                        cross_attempts = max(
                            3,
                            int(os.environ.get("EE_SCAN_FLOOR_GROWTH_CROSSOVER_ATTEMPTS", "6")),
                        )
                        if growth_stalled:
                            cross_attempts = max(
                                cross_attempts,
                                int(
                                    os.environ.get(
                                        "EE_SCAN_FLOOR_GROWTH_CROSSOVER_ATTEMPTS_STALLED",
                                        "24",
                                    )
                                ),
                            )
                        cross_pool = build_scan_floor_growth_crossover_pool(
                            anchor,
                            growth,
                            self.rng,
                            cycle_no=cycle_no,
                            max_attempts=cross_attempts,
                        )
                        if cross_pool:
                            seen_pool = {tuple(p) for p in pool}
                            for hybrid in cross_pool:
                                key = tuple(hybrid)
                                if key in seen_pool:
                                    continue
                                seen_pool.add(key)
                                cross_keys.add(key)
                                pool.append(list(hybrid))
                            self._log(
                                f"[GrowthCross] Merged {len(cross_pool)} hybrid(s) "
                                f"into growth inject (cycle {cycle_no})"
                            )
                    near_fills = sorted(
                        growth,
                        key=lambda f: (abs(len(f) - len(anchor)), len(f)),
                    )
                    anchor_fill = near_fills[0] if near_fills else growth[cycle_no % len(growth)]
                    tail_budget = max(
                        4,
                        int(os.environ.get("EE_SCAN_FLOOR_GROWTH_TAIL_PERTURB", "8")),
                    )
                    pool.extend(
                        terminal_growth_tail_perturb_variants(
                            anchor,
                            anchor_fill,
                            self.rng,
                            max_evals=tail_budget,
                            start_offset=cycle_no * tail_budget,
                        )
                    )
                    if pool:
                        from evolution.mutation_helpers import (
                            compress_growth_probe_pool,
                            scan_floor_capability_growth_allowed,
                        )
                        from evolution.terminal_stagnation import get_scan_floor_gate

                        if scan_floor_capability_growth_allowed(len(self.champion or [])):
                            from evolution.epi_momentum_governor import (
                                promotion_raw_capability_stalled,
                                scan_floor_length_prom_stalled,
                            )
                            from evolution.mutation_helpers import (
                                collect_scan_floor_fresh_compression_variants,
                                scan_floor_growth_compress_sources,
                            )

                            compress_cap = max(
                                4,
                                int(os.environ.get("EE_SCAN_FLOOR_GROWTH_COMPRESS_CAP", "6")),
                            )
                            champ_n = len(self.champion or [])
                            long_pool = [list(p) for p in pool if len(p) > champ_n]
                            short_pool = [list(p) for p in pool if len(p) <= champ_n]
                            root = os.path.dirname(os.path.abspath(__file__))
                            stalled = promotion_raw_capability_stalled(root) or scan_floor_length_prom_stalled(
                                root, champ_n
                            )
                            if stalled:
                                fresh_compress = collect_scan_floor_fresh_compression_variants(
                                    list(self.champion),
                                    long_sources=long_pool,
                                    cycle_no=cycle_no,
                                    gen_idx=growth_gen,
                                    stall_seed_offset=1,
                                )
                                if fresh_compress:
                                    seen_long = {tuple(p) for p in long_pool}
                                    for variant in fresh_compress:
                                        key = tuple(variant)
                                        if key in seen_long:
                                            continue
                                        seen_long.add(key)
                                        long_pool.insert(0, list(variant))
                            compress_src = scan_floor_growth_compress_sources(
                                list(self.champion),
                                long_pool,
                                stalled=stalled,
                                max_trials=compress_cap,
                            )
                            if compress_src:
                                compressed = compress_growth_probe_pool(
                                    list(self.champion),
                                    compress_src,
                                    gate=get_scan_floor_gate(list(self.champion)),
                                    cycle_no=cycle_no,
                                )
                                compressed_keys = {tuple(p) for p in compressed}
                                remainder = [p for p in long_pool if tuple(p) not in compressed_keys]
                                pool = compressed + short_pool + remainder
                            else:
                                pool = short_pool + long_pool
                        from evolution.mutation_helpers import scan_floor_growth_inject_rank
                        from evolution.terminal_stagnation import get_scan_floor_gate, scan_floor_min_raw_gain

                        champ_n = len(self.champion or [])
                        inject_gate = get_scan_floor_gate(list(self.champion))
                        champ_pr = float(compact_l30_promotion_raw_fitness(list(self.champion)))
                        min_gain = float(scan_floor_min_raw_gain(champ_n))
                        if growth_stalled:
                            from evolution.champion_rollback import collect_growth_material_compress_parents
                            from evolution.mutation_helpers import (
                                growth_material_diff_graft_variants,
                                load_scan_floor_eq_refine_seeds,
                            )

                            graft_parent_cap = max(
                                12,
                                int(
                                    os.environ.get(
                                        "EE_GROWTH_MATERIAL_GRAFT_PARENTS_STALLED",
                                        "16",
                                    )
                                ),
                            )
                            graft_material = collect_growth_material_compress_parents(
                                anchor, max_parents=graft_parent_cap
                            )
                            graft_variant_cap = max(
                                12,
                                int(
                                    os.environ.get(
                                        "EE_GROWTH_MATERIAL_GRAFT_VARIANTS_STALLED",
                                        "16",
                                    )
                                ),
                            )
                            graft_eq_seeds = []
                            for graft in growth_material_diff_graft_variants(
                                list(self.champion),
                                graft_material,
                                max_variants=graft_variant_cap,
                                cycle_no=cycle_no,
                            ):
                                key = tuple(graft)
                                if key in {tuple(p) for p in pool}:
                                    continue
                                pool.insert(0, list(graft))
                                if len(graft) == len(anchor) and inject_gate.qualifies(list(graft)):
                                    graft_eq_seeds.append(list(graft))
                            eq_pre = load_scan_floor_eq_refine_seeds(anchor, max_seeds=8, probe_material=False)
                            if eq_pre:
                                seen_pre = {tuple(p) for p in pool}
                                for seed_prog in eq_pre:
                                    key = tuple(seed_prog)
                                    if key in seen_pre:
                                        continue
                                    seen_pre.add(key)
                                    pool.insert(0, list(seed_prog))
                        pool.sort(
                            key=lambda prog: scan_floor_growth_inject_rank(
                                list(self.champion),
                                prog,
                                gate=inject_gate,
                                champ_pr=champ_pr,
                                min_gain=min_gain,
                                stalled=growth_stalled,
                            ),
                            reverse=True,
                        )
                        if growth_stalled:
                            from evolution.mutation_helpers import (
                                collect_growth_material_projection_seeds,
                                record_scan_floor_eq_refine_seeds,
                            )

                            champ_live = list(self.champion or anchor)
                            eq_from_pool = [
                                list(p)
                                for p in pool
                                if len(p) == len(anchor)
                                and inject_gate.qualifies(list(p))
                                and tuple(p) != tuple(champ_live)
                                and tuple(p) != tuple(anchor)
                            ]
                            eq_record = list(eq_from_pool)
                            if graft_eq_seeds:
                                seen_eq = {tuple(p) for p in eq_record}
                                for graft_prog in graft_eq_seeds:
                                    key = tuple(graft_prog)
                                    if key in seen_eq:
                                        continue
                                    seen_eq.add(key)
                                    eq_record.append(list(graft_prog))
                            if eq_record:
                                record_scan_floor_eq_refine_seeds(anchor, eq_record, cycle=cycle_no)
                    cache_entry = (
                        pool_cache_key,
                        tuple(tuple(int(w) for w in p) for p in pool),
                        frozenset(cross_keys),
                    )
                    self._scan_floor_growth_pool_cache = cache_entry
                    if growth_stalled and pool:
                        from evolution.mutation_helpers import save_scan_floor_growth_pool_cache

                        save_scan_floor_growth_pool_cache(
                            list(self.champion or anchor),
                            pool,
                            cross_keys=cross_keys,
                            cycle=cycle_no,
                        )
                if pool and growth_stalled:
                    from evolution.mutation_helpers import (
                        enrich_growth_inject_equal_length_pool,
                        filter_scan_floor_growth_programs_by_max_words,
                        scan_floor_stalled_challenger_max_words,
                        select_growth_inject_slot_variants,
                    )
                    from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS, get_scan_floor_gate

                    inject_gen = int(
                        mg_inject_gen if mg_inject_gen is not None else max(0, int(self.config.max_generations) - 1)
                    )
                    long_word_cap = scan_floor_stalled_challenger_max_words(
                        len(self.champion or anchor),
                        generation=growth_gen,
                        mg_inject_generation=inject_gen,
                        default_max=int(L30_COMPACT_MAX_WORDS),
                        stalled=True,
                    )
                    pool = filter_scan_floor_growth_programs_by_max_words(
                        pool,
                        max_words=long_word_cap,
                    )
                    if late_inject_probe:
                        slot_gate = get_scan_floor_gate(list(self.champion or anchor))
                        champ_live = list(self.champion or anchor)
                        enrich_skip_eq = max(
                            6,
                            int(os.environ.get("EE_SCAN_FLOOR_ENRICH_SKIP_EQ_POOL", "8")),
                        )
                        eq_in_pool = sum(1 for p in pool if len(p) == len(champ_live) and slot_gate.qualifies(list(p)))
                        skip_enrich_warm = False
                        try:
                            from evolution.mutation_helpers import should_skip_scan_floor_enrich_equal_length_pool

                            skip_enrich_warm = should_skip_scan_floor_enrich_equal_length_pool(
                                champ_live,
                                seed_cap=enrich_skip_eq,
                            )
                        except ImportError:
                            pass
                        if (
                            not pool_cache_hit
                            and not skip_enrich_warm
                            and (
                                eq_in_pool < enrich_skip_eq
                                or os.environ.get("EE_SCAN_FLOOR_SKIP_ENRICH_RICH_EQ", "0") != "0"
                            )
                        ):
                            pool = enrich_growth_inject_equal_length_pool(
                                champ_live,
                                pool,
                                gate=slot_gate,
                                cycle_no=cycle_no,
                            )
                    rotated = select_growth_inject_slot_variants(
                        list(self.champion or anchor),
                        pool,
                        n_slots=n_slots,
                        cycle_no=cycle_no,
                        stalled=True,
                        max_long_words=long_word_cap,
                    )
                    slots = self.rng.sample(range(len(challengers)), min(n_slots, len(challengers)))
                    for slot, variant in zip(slots, rotated, strict=False):
                        challengers[slot] = variant
                        if slot < len(self._last_challenger_strategies):
                            label = "GROWTH_CROSSOVER" if tuple(variant) in cross_keys else "L30_EASY_FILL"
                            self._last_challenger_strategies[slot] = label
                elif pool:
                    rotated = [pool[(cycle_no + i) % len(pool)] for i in range(n_slots)]
                    slots = self.rng.sample(range(len(challengers)), min(n_slots, len(challengers)))
                    for slot, variant in zip(slots, rotated, strict=False):
                        challengers[slot] = variant
                        if slot < len(self._last_challenger_strategies):
                            label = "GROWTH_CROSSOVER" if tuple(variant) in cross_keys else "L30_EASY_FILL"
                            self._last_challenger_strategies[slot] = label
        compression_once_per_cycle = os.environ.get("EE_SCAN_FLOOR_COMPRESSION_ONCE_PER_CYCLE", "1") != "0"
        compression_inject_gen = getattr(self, "_scan_floor_compression_inject_generation", None)
        from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

        stall_compress_inject = _scan_floor_probe_stall_mode(len(self.champion or []))
        from evolution.mutation_helpers import (
            collect_scan_floor_fresh_compression_variants,
            scan_floor_stalled_compression_injection_generation,
        )

        compression_inject_ok = (
            scan_floor_stalled_compression_injection_generation(
                growth_gen,
                inject_generation=compression_inject_gen,
                max_generations=int(self.config.max_generations),
                once_per_cycle=compression_once_per_cycle,
            )
            if stall_compress_inject
            else scan_floor_compression_injection_generation(
                growth_gen,
                inject_generation=compression_inject_gen,
                max_generations=int(self.config.max_generations),
                once_per_cycle=compression_once_per_cycle,
            )
        )
        if scan_floor_capability_growth_allowed(len(self.champion)) and compression_inject_ok:
            from evolution.terminal_stagnation import ranked_scan_floor_compression_variants

            if stall_compress_inject:
                cycle_no = int(getattr(self, "_cycle_number", 0))
                compress = ranked_scan_floor_compression_variants(list(self.champion), cycle_no=cycle_no)
                compression_heavy_ok = late_inject_probe and not growth_inject_ok
                skip_compress_heavy = False
                if compression_heavy_ok:
                    try:
                        from evolution.mutation_helpers import (
                            should_skip_mg_stall_collect_for_warm_eq_refine,
                            should_skip_scan_floor_enrich_equal_length_pool,
                        )

                        warm_skip = os.environ.get("EE_SCAN_FLOOR_WARM_SKIP_COMPRESS_HEAVY", "1") == "1"
                        skip_compress_heavy = warm_skip and (
                            should_skip_scan_floor_enrich_equal_length_pool(
                                list(self.champion),
                            )
                            or should_skip_mg_stall_collect_for_warm_eq_refine(list(self.champion))
                        )
                    except ImportError:
                        pass
                if compression_heavy_ok and not skip_compress_heavy:
                    from evolution.mutation_helpers import (
                        collect_scan_floor_fresh_compression_variants,
                        collect_scan_floor_growth_crossover_fills,
                    )

                    growth_fills = collect_scan_floor_growth_crossover_fills(list(self.champion))
                    long_for_fresh = [list(f) for f in growth_fills if len(f) > len(self.champion or [])]
                    fresh_compress = collect_scan_floor_fresh_compression_variants(
                        list(self.champion),
                        long_sources=long_for_fresh,
                        cycle_no=cycle_no,
                        gen_idx=growth_gen,
                    )
                    if fresh_compress:
                        if not compress:
                            compress = []
                        seen_c = {tuple(p) for p in compress}
                        for variant in fresh_compress:
                            key = tuple(variant)
                            if key in seen_c:
                                continue
                            seen_c.add(key)
                            compress.insert(0, list(variant))
                if (
                    compression_heavy_ok
                    and not skip_compress_heavy
                    and compress
                    and hasattr(self, "_terminal_archive_parents")
                ):
                    from evolution.terminal_stagnation import (
                        get_scan_floor_gate,
                        terminal_deterministic_archive_crossover_probe_variants,
                    )

                    gate = get_scan_floor_gate(list(self.champion))
                    champ_n = len(self.champion or [])
                    parents = [
                        list(p)
                        for p in self._terminal_archive_parents()
                        if p and len(p) == champ_n and gate.qualifies(list(p))
                    ]
                    if len(parents) >= 2:
                        cross_budget = max(
                            4,
                            int(
                                os.environ.get(
                                    "EE_SCAN_FLOOR_COMPRESS_ARCHIVE_CROSSOVER",
                                    "8",
                                )
                            ),
                        )
                        cross = terminal_deterministic_archive_crossover_probe_variants(
                            list(self.champion),
                            parents,
                            max_evals=cross_budget,
                            start_offset=cycle_no * cross_budget,
                            gate=gate,
                        )
                        if cross:
                            seen_c = {tuple(p) for p in compress}
                            for variant in cross:
                                key = tuple(variant)
                                if key in seen_c:
                                    continue
                                seen_c.add(key)
                                compress.insert(0, list(variant))
                if compress and challengers:
                    default_compress_slots = "4" if _scan_floor_probe_stall_mode(len(self.champion or [])) else "2"
                    n_slots = max(
                        1,
                        min(
                            len(compress),
                            int(
                                os.environ.get(
                                    "EE_SCAN_FLOOR_COMPRESSION_SLOTS",
                                    default_compress_slots,
                                )
                            ),
                        ),
                    )
                    rotated = [compress[(cycle_no + i) % len(compress)] for i in range(n_slots)]
                    slots = self.rng.sample(range(len(challengers)), min(n_slots, len(challengers)))
                    for slot, variant in zip(slots, rotated, strict=False):
                        challengers[slot] = variant
                        if slot < len(self._last_challenger_strategies):
                            self._last_challenger_strategies[slot] = "SEMANTIC_COMPRESS"
        if stall_compress_inject and challengers:
            from evolution.mutation_helpers import inject_scan_floor_eq_refine_seed_challengers

            inject_scan_floor_eq_refine_seed_challengers(
                list(self.champion),
                challengers,
                self.rng,
                cycle_no=int(getattr(self, "_cycle_number", 0)),
                generation=int(getattr(self, "current_generation", 0)),
                growth_inject_generation=growth_inject_gen,
                mg_inject_generation=mg_inject_gen,
                strategies=getattr(self, "_last_challenger_strategies", None),
            )

    def _apply_model_guided_slots(self, challengers):
        """Reserve slots for model-guided generation."""

        if self.champion_level < 2:

            return

        from evolution.mutation_helpers import scan_floor_model_guided_slot_allowed, skip_model_guided_at_terminal
        from evolution.terminal_stagnation import terminal_cycles_at_level

        verify = float(getattr(self, "_champion_verify_score", 0.0))
        if skip_model_guided_at_terminal(self.champion_level, verify):
            if scan_floor_model_guided_slot_allowed(
                self.champion_level,
                verify,
                len(self.champion or []),
                terminal_cycles_at_level(),
                generation=getattr(self, "current_generation", 0),
                max_generations=int(self.config.max_generations),
                inject_generation=getattr(self, "_scan_floor_inject_generation", None),
                cycle_number=int(getattr(self, "_cycle_number", 0)),
            ):
                from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                n_slots = 1
                stalled = _scan_floor_probe_stall_mode(len(self.champion or []))
                if stalled:
                    n_slots = max(
                        1,
                        int(os.environ.get("EE_SCAN_FLOOR_MODEL_GUIDED_SLOTS_STALLED", "2")),
                    )
                slot_ratio = min(1.0, n_slots / max(1, len(challengers)))
                replaced: list[int] = []
                if stalled and n_slots >= 2:
                    inject_gen = int(getattr(self, "_scan_floor_inject_generation", -1))
                    gen_now = int(getattr(self, "current_generation", 0))
                    replaced.extend(
                        self._replace_with_model_guided(
                            challengers,
                            model_ratio=slot_ratio,
                            scan_floor_growth_mode=False,
                            num_replace_cap=1,
                        )
                    )
                    replaced.extend(
                        self._replace_with_model_guided(
                            challengers,
                            model_ratio=slot_ratio,
                            scan_floor_growth_mode=True,
                            num_replace_cap=1,
                            exclude_indices=set(replaced),
                        )
                    )
                    if gen_now == inject_gen and os.environ.get("EE_GROWTH_MG_DUAL_ANCHOR_INJECT", "1") != "0":
                        replaced.extend(
                            self._replace_with_model_guided(
                                challengers,
                                model_ratio=slot_ratio,
                                scan_floor_growth_mode=True,
                                num_replace_cap=1,
                                exclude_indices=set(replaced),
                                growth_anchor_rank_offset=1,
                            )
                        )
                    if gen_now == inject_gen and os.environ.get("EE_GROWTH_MG_TRIPLE_ANCHOR_INJECT", "1") != "0":
                        replaced.extend(
                            self._replace_with_model_guided(
                                challengers,
                                model_ratio=slot_ratio,
                                scan_floor_growth_mode=True,
                                num_replace_cap=1,
                                exclude_indices=set(replaced),
                                growth_anchor_rank_offset=2,
                            )
                        )
                elif stalled:
                    replaced.extend(
                        self._replace_with_model_guided(
                            challengers,
                            model_ratio=slot_ratio,
                            scan_floor_growth_mode=False,
                        )
                    )
                else:
                    replaced.extend(
                        self._replace_with_model_guided(
                            challengers,
                            model_ratio=slot_ratio,
                        )
                    )
                for idx in replaced:
                    if idx < len(self._last_challenger_strategies):
                        self._last_challenger_strategies[idx] = MutationStrategy.MODEL_GUIDED.value
            return

        if self.config.model_guided_ratio is None:

            return

        n_model = max(1, int(len(challengers) * self.config.model_guided_ratio))

        replaced = self._replace_with_model_guided(challengers, model_ratio=n_model / len(challengers))

        for idx in replaced:

            if idx < len(self._last_challenger_strategies):

                self._last_challenger_strategies[idx] = MutationStrategy.MODEL_GUIDED.value

    def _apply_mutation(self, strategy, parent=None, parent_traces=None) -> list[int]:
        """Apply a mutation strategy to a parent program (champion by default)."""

        program = parent if parent is not None else self.champion

        task_inst = self.task_instance or {}

        expected = task_inst.get("expected_output", {})

        inputs = task_inst.get("inputs", {})

        if parent_traces is not None:
            trace_input = parent_traces
        else:
            trace_input = [self.champion_trace] if self.champion_trace is not None else []

        sa = self.strategy_archive if self.config.use_strategy_archive else None

        mutated, _desc, _analysis = propose_mutation(
            program=program,
            trace=trace_input,
            expected=expected,
            inputs=inputs,
            rng=self.rng,
            task_level=self.champion_level,
            strategy=strategy,
            strategy_archive=sa,
        )

        if (
            int(self.champion_level) >= 29
            and float(getattr(self, "_champion_verify_score", 0.0)) >= 0.99 - 1e-6
            and mutated
        ):
            from evolution.champion_bloat_guard import level_word_cap
            from evolution.terminal_stagnation import (
                maybe_drop_interior_instruction,
                semantic_compress,
                terminal_length_drop_params,
            )

            cap = level_word_cap(int(self.champion_level))
            p, n_drop = terminal_length_drop_params(len(mutated), cap)
            mutated = maybe_drop_interior_instruction(mutated, self.rng, p=p, n_drop=n_drop)
            mutated = semantic_compress(mutated)

        if parent is not None and self.champion is not None and mutated and len(parent) > len(self.champion):
            try:
                from evolution.mutation_helpers import maybe_l10_lift_post_mutation_compress
                from evolution.terminal_stagnation import terminal_efficiency_at_scan_floor

                if terminal_efficiency_at_scan_floor(len(self.champion)):
                    mutated = maybe_l10_lift_post_mutation_compress(
                        list(self.champion),
                        list(parent),
                        list(mutated),
                    )
            except (ImportError, TypeError, ValueError, OSError):
                pass

        return mutated

    def _generate_challengers(self) -> list[list[int]]:
        """Generate a population of challenger programs."""

        pop_size = self.config.population_size

        if self.champion is None:

            seeds = [generate_seed_program(rng=self.rng) for _ in range(pop_size)]

            self._last_challenger_strategies = ["SEED"] * pop_size

            return seeds

        challengers = []

        self._last_challenger_strategies = []

        with self._sub("mutate_loop"):
            for _ in range(pop_size):

                strategy = self._choose_mutation_strategy(len(challengers))

                self._last_challenger_strategies.append(strategy.value)

                parent, parent_traces = self._pick_mutation_parent()

                prog = self._apply_mutation(strategy, parent=parent, parent_traces=parent_traces)

                challengers.append(prog)

        with self._sub("diversity_inject"):
            self._generate_diversity_injection(challengers, pop_size)

        with self._sub("terminal_probes"):
            self._inject_terminal_verify_probes(challengers, pop_size)

        with self._sub("model_guided_slots"):
            self._apply_model_guided_slots(challengers)

        with self._sub("qd_hybrids"):
            self._inject_qd_archive_hybrids(challengers)

        # Ensure EBREAK at end of each program

        for i, prog in enumerate(challengers):

            if prog and prog[-1] != 0x0010_0073:

                challengers[i] = prog + [0x0010_0073]

        return challengers

    @staticmethod
    def _compute_population_diversity(challengers: list[list[int]]) -> float:
        """Compute opcode diversity of the challenger population."""

        return _compute_population_diversity_fn(challengers, _OPCODE_CAT_ARRAY, NUM_OPCODE_CATEGORIES)

    def _inject_diversity(
        self,
        challengers: list[list[int]],
        n_inject: int,
        *,
        preserve_indices: set[int] | None = None,
    ) -> None:
        """Replace the least-diverse challengers with fresh random programs."""

        _inject_diversity_fn(
            challengers,
            n_inject,
            self.rng,
            self._last_challenger_strategies,
            _OPCODE_CAT_ARRAY,
            NUM_OPCODE_CATEGORIES,
            generate_seed_program,
            preserve_indices=preserve_indices,
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

    def _resolve_strategy_consecutive_cap(self) -> int:
        """Cap consecutive same-strategy runs; tighter at high curriculum levels."""

        if self.champion_level >= 5:

            return max(self.config.max_consecutive_same_strategy // 2, 10)

        return self.config.max_consecutive_same_strategy

    def _resolve_skip_strategy_after_cap(self, effective_cap: int) -> str | None:
        """Reset consecutive counter and return strategy to skip when cap is hit."""

        if self._consecutive_same_strategy >= effective_cap and self._last_strategy_used not in (
            "GROW",
            "MODEL_GUIDED",
        ):

            self._consecutive_same_strategy = 0

            return self._last_strategy_used

        return None

    def _recommend_strategy_from_archive(self, skip_strat: str | None) -> tuple[MutationStrategy, Any]:
        """Return archive-recommended strategy and record (or HYBRID when archive is sparse)."""

        if (
            self.config.use_strategy_archive
            and self.strategy_archive is not None
            and self.strategy_archive.total_records() > 10
        ):

            rec = self.strategy_archive.recommend_strategy(self.champion_level, skip_strategy=skip_strat)

            return _STRATEGY_ENUM_CACHE.get(rec, MutationStrategy.HYBRID), rec

        return MutationStrategy.HYBRID, None

    def _maybe_apply_exploration_bonus(self, chosen: MutationStrategy, rec: Any) -> MutationStrategy:
        """Occasionally swap to an alternate strategy when exploration bonus is high."""

        if rec is None or not (self.config.use_strategy_archive and self.strategy_archive is not None):

            return chosen

        try:

            _bonus = self.strategy_archive.adaptive_exploration_bonus(rec, self.champion_level)

            if _bonus > 0.15 and self.rng.random() < _bonus:

                _alt = [s for s in MutationStrategy if s not in (MutationStrategy.GROW, MutationStrategy.MODEL_GUIDED)]

                if _alt:

                    return self.rng.choice(_alt)

        except Exception:  # noqa: BLE001, S110

            pass

        return chosen

    def _apply_strategy_override_pipeline(self, chosen: MutationStrategy) -> MutationStrategy:
        """Apply variance, priority, and cycle-phase strategy overrides."""

        op, os, ol = self._check_variance_override()

        op, os, ol = self._check_priority_overrides(op)

        if os is not None:

            chosen = os

            self._log(ol)

        return self._apply_cycle_phase_bias(chosen, op)

    def _track_consecutive_strategy(self, chosen: MutationStrategy) -> None:
        """Update consecutive-same-strategy counters after a choice."""

        sv = chosen.value

        if sv == self._last_strategy_used:

            self._consecutive_same_strategy += 1

        else:

            self._last_strategy_used = sv

            self._consecutive_same_strategy = 1

    def _choose_mutation_strategy(self, idx: int) -> MutationStrategy:
        """Choose a mutation strategy for challenger idx."""

        effective_cap = self._resolve_strategy_consecutive_cap()

        skip_strat = self._resolve_skip_strategy_after_cap(effective_cap)

        chosen, rec = self._recommend_strategy_from_archive(skip_strat)

        chosen = self._maybe_apply_exploration_bonus(chosen, rec)

        chosen = self._apply_affinity_bias(chosen)

        chosen = self._apply_strategy_override_pipeline(chosen)

        if (
            self.champion_level in self._unsolved_levels_set
            and self.generations_stalled > 2
            and self.rng.random() < self.config.perturbation_capped_probability_unsolved
        ):

            chosen = MutationStrategy.PERTURBATION_CAPPED

        chosen = self._apply_level_aware_strategy_bias(chosen)

        self._track_consecutive_strategy(chosen)

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

    def _run_model_inference(
        self,
        prefix_source,
        model_temp,
        num_replace,
        prefix_counts,
        *,
        gate=None,
        normalize_to_champion: bool = False,
        growth_mode: bool = False,
        growth_max_words: int = 72,
        growth_prefix_count: int | None = None,
        exclude_indices: set[int] | None = None,
    ):
        """Run model inference loop and return replaced indices."""

        ri, reps = [], 0
        from evolution.mutation_helpers import (
            lock_model_guided_champion_prefix,
            normalize_model_guided_program,
            normalize_model_guided_tail_growth,
            repair_model_guided_for_gate,
            resolve_growth_mg_completion,
            scan_floor_growth_heldout_qualifies,
        )

        champ = list(self.champion or [])
        growth_projection_cap = 0
        growth_projection_used = 0
        if growth_mode:
            growth_projection_cap = max(
                0,
                int(os.environ.get("EE_SCAN_FLOOR_GROWTH_MG_PROJECTION_CAP", "3")),
            )
        collect_stalled_gate = bool(gate is not None and not growth_mode and champ)
        if collect_stalled_gate:
            try:
                from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                collect_stalled_gate = bool(_scan_floor_probe_stall_mode(len(champ)))
            except ImportError:
                collect_stalled_gate = False
        if collect_stalled_gate:
            try:
                from evolution.mutation_helpers import should_skip_mg_stall_collect_for_warm_eq_refine

                if should_skip_mg_stall_collect_for_warm_eq_refine(champ):
                    collect_stalled_gate = False
            except ImportError:
                pass
        inject_gen = getattr(
            self,
            "_scan_floor_inject_generation",
            max(0, int(self.config.max_generations) - 1),
        )
        if collect_stalled_gate and int(getattr(self, "current_generation", 0)) != int(inject_gen):
            # Gen-0 dual-MG is lightweight slot fill only; bulk stall collect
            # runs on the last generation to avoid 5+ minute cycle spikes.
            collect_stalled_gate = False
        stall_collect_cap = max(
            8,
            int(os.environ.get("EE_SCAN_FLOOR_MG_STALL_COLLECT_CAP", "16")),
        )
        collected_gate: list[list[int]] = []
        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.synchronize()

        except Exception:  # noqa: BLE001, S110

            pass

        for pc in prefix_counts:

            if not collect_stalled_gate and reps >= num_replace:

                break
            if collect_stalled_gate and len(collected_gate) >= stall_collect_cap:

                break

            try:

                comp = None

                for att in range(1, 3):

                    try:

                        if growth_mode:
                            sample_n = max(
                                2,
                                int(os.environ.get("EE_SCAN_FLOOR_GROWTH_MODEL_SAMPLES", "6")),
                            )
                            try:
                                from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                                if _scan_floor_probe_stall_mode(len(champ)):
                                    sample_n = max(
                                        sample_n,
                                        int(
                                            os.environ.get(
                                                "EE_SCAN_FLOOR_GROWTH_MODEL_SAMPLES_STALLED",
                                                "8",
                                            )
                                        ),
                                    )
                            except ImportError:
                                pass
                        elif gate is not None:
                            sample_n = min(4, max(2, num_replace * 2))
                            try:
                                from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                                if _scan_floor_probe_stall_mode(len(champ)):
                                    sample_n = max(
                                        sample_n,
                                        int(
                                            os.environ.get(
                                                "EE_SCAN_FLOOR_MODEL_GUIDED_GATE_SAMPLES",
                                                "12",
                                            )
                                        ),
                                    )
                            except ImportError:
                                pass
                        else:
                            sample_n = min(3, max(1, num_replace // len(prefix_counts)))
                        comp = self._prefix_generator.complete(
                            prefix_source,
                            prefix_count=pc,
                            num_samples=sample_n,
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

                        except Exception:  # noqa: BLE001, S110

                            pass

                        if att == 2:

                            comp = None

                if not comp:

                    continue

                for ph in comp:

                    if not collect_stalled_gate and reps >= num_replace:

                        break
                    if collect_stalled_gate and len(collected_gate) >= stall_collect_cap:

                        break

                    pi = [int(w, 16) for w in ph]

                    if len(pi) < 2:
                        continue
                    if normalize_to_champion and champ:
                        if growth_mode:
                            pi = normalize_model_guided_tail_growth(
                                pi,
                                champ,
                                prefix_count=int(pc),
                                max_words=int(growth_max_words),
                            )
                            if pi is None:
                                continue
                            if not scan_floor_growth_heldout_qualifies(champ, pi):
                                continue
                            grown = list(pi)
                            cycle_key = int(getattr(self, "_cycle_number", 0))
                            if getattr(self, "_mg_material_buffer_cycle", -1) != cycle_key:
                                self._mg_material_buffer_cycle = cycle_key
                                self._mg_material_buffer = []
                            buffer = getattr(self, "_mg_material_buffer", None)
                            if buffer is not None:
                                buffer.append(list(grown))
                                self._mg_material_buffer = buffer
                            from evolution.terminal_stagnation import get_scan_floor_gate

                            growth_gate = gate or get_scan_floor_gate(champ)
                            allow_projection = growth_projection_used < growth_projection_cap
                            resolved, projection_attempted = resolve_growth_mg_completion(
                                champ,
                                grown,
                                gate=growth_gate,
                                prefix_count=int(pc),
                                allow_projection=allow_projection,
                            )
                            if projection_attempted:
                                growth_projection_used += 1
                            if resolved is None:
                                continue
                            pi = resolved
                            if len(pi) > len(champ) and not collect_stalled_gate:
                                try:
                                    from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                                    if _scan_floor_probe_stall_mode(len(champ)):
                                        gen_now = int(getattr(self, "current_generation", 0))
                                        if gen_now != int(
                                            getattr(
                                                self,
                                                "_scan_floor_inject_generation",
                                                -1,
                                            )
                                        ):
                                            continue
                                except ImportError:
                                    pass
                        else:
                            pi = normalize_model_guided_program(pi, champ)
                            if pi is None:
                                continue
                            locked = lock_model_guided_champion_prefix(pi, champ, pc)
                            if locked is not None:
                                pi = locked
                            if gate is not None:
                                if not gate.qualifies(pi):
                                    if collect_stalled_gate:
                                        continue
                                    repair_evals = 12
                                    try:
                                        from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                                        if _scan_floor_probe_stall_mode(len(champ)):
                                            repair_evals = max(
                                                repair_evals,
                                                int(
                                                    os.environ.get(
                                                        "EE_SCAN_FLOOR_GATE_REPAIR_EVALS",
                                                        "48",
                                                    )
                                                ),
                                            )
                                    except ImportError:
                                        pass
                                    repaired = repair_model_guided_for_gate(
                                        pi,
                                        gate,
                                        max_evals=repair_evals,
                                        champion=champ,
                                        prefix_count=pc,
                                    )
                                    if repaired is None:
                                        continue
                                    pi = repaired
                            if champ and tuple(pi) == tuple(champ):
                                continue
                    elif gate is not None and not gate.qualifies(pi):
                        continue

                    if collect_stalled_gate:
                        collected_gate.append(list(pi))
                        continue

                    idx = self.rng.randint(0, len(self._last_challengers) - 1)
                    if exclude_indices:
                        for _ in range(12):
                            if idx not in exclude_indices:
                                break
                            idx = self.rng.randint(0, len(self._last_challengers) - 1)
                        if idx in exclude_indices:
                            continue

                    self._last_challengers[idx] = pi

                    ri.append(idx)

                    reps += 1

            except (RuntimeError, ValueError):

                continue

        if collect_stalled_gate and collected_gate:
            from evolution.mutation_helpers import score_stall_collect_gate_programs

            score_cap = max(
                2,
                int(os.environ.get("EE_SCAN_FLOOR_MG_STALL_PROM_SCORE_CAP", "4")),
            )
            collected_gate = score_stall_collect_gate_programs(
                champ,
                collected_gate,
                cycle_no=int(getattr(self, "_cycle_number", 0)),
                score_cap=score_cap,
            )
            seen_gate: set[tuple[int, ...]] = set()
            champ_t = tuple(champ)
            for pi in collected_gate:
                key = tuple(pi)
                if key in seen_gate or key == champ_t:
                    continue
                seen_gate.add(key)
                idx = self.rng.randint(0, len(self._last_challengers) - 1)
                if exclude_indices:
                    for _ in range(12):
                        if idx not in exclude_indices:
                            break
                        idx = self.rng.randint(0, len(self._last_challengers) - 1)
                    if idx in exclude_indices:
                        continue
                self._last_challengers[idx] = list(pi)
                ri.append(idx)
                reps += 1
                if reps >= num_replace:
                    break

        if growth_mode and champ:
            buffer = getattr(self, "_mg_material_buffer", None)
            if buffer:
                try:
                    from evolution.mutation_helpers import flush_growth_mg_material_candidates

                    record_cap = max(
                        1,
                        int(os.environ.get("EE_GROWTH_MG_MATERIAL_RECORD_CAP", "4")),
                    )
                    inject_gen_now = int(getattr(self, "_scan_floor_inject_generation", -1))
                    if int(getattr(self, "current_generation", 0)) == inject_gen_now:
                        record_cap = max(
                            record_cap,
                            int(
                                os.environ.get(
                                    "EE_GROWTH_MG_MATERIAL_RECORD_CAP_INJECT",
                                    "8",
                                )
                            ),
                        )
                    flush_growth_mg_material_candidates(
                        champ,
                        buffer,
                        cycle=int(getattr(self, "_cycle_number", 0)),
                        record_cap=record_cap,
                    )
                except (ImportError, TypeError, ValueError, OSError):
                    pass
                self._mg_material_buffer = []

        return ri

    def _replace_with_model_guided(
        self,
        challengers: list[list[int]],
        model_ratio: float = 0.2,
        *,
        scan_floor_growth_mode: bool | None = None,
        num_replace_cap: int | None = None,
        exclude_indices: set[int] | None = None,
        growth_anchor_rank_offset: int = 0,
    ) -> list[int]:
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

        except Exception:  # noqa: BLE001, S110

            pass

        champion_hex = [f"0x{w:08x}" for w in self.champion]

        if self._prefix_generator is None:

            try:

                from prefix_generator import PrefixGenerator

                ckpt = self._seed_model_path or ""

                if not ckpt or not os.path.isfile(ckpt):

                    base = os.path.dirname(os.path.abspath(__file__))

                    ckpt = os.path.join(base, "milestone1", "checkpoints", "checkpoint_best.pt")

                if os.path.isfile(ckpt):

                    self._prefix_generator = PrefixGenerator(ckpt, temperature=0.9, max_new_tokens=256, device=None)

            except (ImportError, RuntimeError) as e:

                self._log(f"[ModelGuided] Init failed: {e}")

                return []

        if self._prefix_generator is None:

            return []

        num_replace = max(1, int(len(challengers) * model_ratio))
        if num_replace_cap is not None:
            num_replace = max(1, min(num_replace, int(num_replace_cap)))

        model_temp = self._compute_model_temperature()

        stalled = self.generations_stalled

        champion_level = self.champion_level

        from evolution.mutation_helpers import (
            scan_floor_anchor_program,
            scan_floor_capability_growth_allowed,
            scan_floor_model_guided_inference_params,
            scan_floor_model_guided_slot_allowed,
        )
        from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS, terminal_cycles_at_level

        verify = float(getattr(self, "_champion_verify_score", 0.0))
        scan_floor_slot = scan_floor_model_guided_slot_allowed(
            champion_level,
            verify,
            len(self.champion),
            terminal_cycles_at_level(),
            generation=getattr(self, "current_generation", 0),
            max_generations=int(self.config.max_generations),
            inject_generation=getattr(self, "_scan_floor_inject_generation", None),
            cycle_number=int(getattr(self, "_cycle_number", 0)),
        )
        scan_floor_gate = None
        growth_mode = False
        growth_max_words = int(L30_COMPACT_MAX_WORDS)
        growth_prefix_count: int | None = None
        inject_gen = int(
            getattr(
                self,
                "_scan_floor_inject_generation",
                max(0, int(self.config.max_generations) - 1),
            )
        )
        if scan_floor_slot:
            if scan_floor_growth_mode is not None:
                growth_mode = bool(scan_floor_growth_mode)
            else:
                growth_mode = scan_floor_capability_growth_allowed(len(self.champion))
            growth_anchor_hex: list[str] | None = None
            if growth_mode:
                from evolution.mutation_helpers import collect_scan_floor_growth_crossover_fills

                anchor_prog = scan_floor_anchor_program(list(self.champion))
                growth_fills = collect_scan_floor_growth_crossover_fills(anchor_prog)
                if growth_fills:
                    from evolution.mutation_helpers import (
                        filter_scan_floor_growth_programs_by_max_words,
                        scan_floor_stalled_challenger_max_words,
                    )
                    from evolution.terminal_stagnation import L30_COMPACT_MAX_WORDS, _scan_floor_probe_stall_mode

                    cycle_no = int(getattr(self, "_cycle_number", 0))
                    gen_idx = int(getattr(self, "current_generation", 0))
                    inject_gen = int(
                        getattr(
                            self,
                            "_scan_floor_inject_generation",
                            max(0, int(self.config.max_generations) - 1),
                        )
                    )
                    if _scan_floor_probe_stall_mode(len(self.champion or [])):
                        mg_cap = scan_floor_stalled_challenger_max_words(
                            len(self.champion or []),
                            generation=gen_idx,
                            mg_inject_generation=inject_gen,
                            default_max=int(L30_COMPACT_MAX_WORDS),
                            stalled=True,
                        )
                        growth_fills = filter_scan_floor_growth_programs_by_max_words(
                            growth_fills,
                            max_words=mg_cap,
                        )
                    if growth_fills:
                        from evolution.mutation_helpers import select_growth_mg_anchor_fill

                        fill = select_growth_mg_anchor_fill(
                            list(self.champion or anchor_prog),
                            growth_fills,
                            cycle_no=cycle_no,
                            generation=gen_idx,
                            inject_generation=inject_gen,
                            rank_offset=int(growth_anchor_rank_offset),
                        )
                        growth_anchor_hex = [f"0x{int(w):08x}" for w in fill]
            if not growth_mode:
                from evolution.terminal_stagnation import get_scan_floor_gate

                scan_floor_gate = get_scan_floor_gate(list(self.champion))
            archive_hex: list[list[str]] = []
            if hasattr(self, "_terminal_archive_parents"):
                for prog in self._terminal_archive_parents()[:6]:
                    archive_hex.append([f"0x{w:08x}" for w in prog])
            prefix_source, prefix_counts, source_name = scan_floor_model_guided_inference_params(
                champion_hex,
                archive_hex,
                cycle_offset=terminal_cycles_at_level(),
                generation=getattr(self, "current_generation", 0),
                growth_mode=growth_mode,
                growth_anchor_hex=growth_anchor_hex,
            )
            if scan_floor_slot and not growth_mode and scan_floor_gate is not None:
                from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                if _scan_floor_probe_stall_mode(len(self.champion or [])):
                    champ_n = len(self.champion or [])
                    tail_opts = [6, 8, 10, 12, 14, 16, 20, 24, 28, 32]
                    cycle_no = int(getattr(self, "_cycle_number", 0))
                    gen_idx = int(getattr(self, "current_generation", 0))
                    extra_pcs: list[int] = []
                    for k in range(4):
                        tail = tail_opts[(cycle_no + gen_idx + k) % len(tail_opts)]
                        pc = max(1, min(champ_n - 1, champ_n - tail))
                        if pc not in extra_pcs:
                            extra_pcs.append(pc)
                    if extra_pcs:
                        prefix_counts = extra_pcs
            if growth_mode:
                from evolution.mutation_helpers import scan_floor_growth_mg_max_words
                from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                if _scan_floor_probe_stall_mode(len(self.champion or [])):
                    growth_max_words = scan_floor_growth_mg_max_words(
                        len(self.champion or []),
                        generation=int(getattr(self, "current_generation", 0)),
                        inject_generation=inject_gen,
                        default_max=int(L30_COMPACT_MAX_WORDS),
                        stalled=True,
                    )
            if growth_mode and prefix_counts and growth_anchor_hex:
                from evolution.mutation_helpers import expand_growth_anchor_prefix_counts_stalled
                from evolution.terminal_stagnation import _scan_floor_probe_stall_mode

                if _scan_floor_probe_stall_mode(len(self.champion or [])):
                    champ_words = [int(h, 16) for h in champion_hex]
                    anchor_words = [int(h, 16) for h in growth_anchor_hex]
                    cycle_no = int(getattr(self, "_cycle_number", 0))
                    gen_idx = int(getattr(self, "current_generation", 0))
                    prefix_max = int(os.environ.get("EE_SCAN_FLOOR_GROWTH_PREFIX_COUNTS_STALLED", "3"))
                    if gen_idx == inject_gen:
                        prefix_max = max(
                            prefix_max,
                            int(
                                os.environ.get(
                                    "EE_SCAN_FLOOR_GROWTH_PREFIX_COUNTS_INJECT",
                                    "5",
                                )
                            ),
                        )
                    prefix_counts = expand_growth_anchor_prefix_counts_stalled(
                        champ_words,
                        anchor_words,
                        cycle_offset=cycle_no + gen_idx,
                        max_counts=prefix_max,
                        generation=gen_idx,
                        inject_generation=inject_gen,
                    )
            if growth_mode and prefix_counts:
                growth_prefix_count = int(prefix_counts[0])
            model_temp = min(float(model_temp), 0.85)
        else:
            prefix_source, source_name = self._get_model_guided_prefix(champion_hex, ref_prefix_hex)
            if champion_level >= 8:
                prefix_counts = [12] if stalled >= 50 else [8] if stalled >= 20 else [6]
            else:
                prefix_counts = [6] if stalled >= 50 else [4] if stalled >= 20 else [2]

        # Inference mutates ``_last_challengers``; bind the live batch under construction.
        self._last_challengers = challengers

        replaced_indices = self._run_model_inference(
            prefix_source,
            model_temp,
            num_replace,
            prefix_counts,
            gate=scan_floor_gate,
            normalize_to_champion=bool(scan_floor_slot),
            growth_mode=growth_mode,
            growth_max_words=growth_max_words,
            growth_prefix_count=growth_prefix_count,
            exclude_indices=exclude_indices,
        )

        if replaced_indices:
            self._log(
                f"[ModelGuided] Replaced {len(replaced_indices)}/{num_replace} "
                f"(prefix: {source_name}, L={champion_level}, temp={model_temp:.1f})"
            )
        elif scan_floor_slot:
            mode = "growth" if growth_mode else "gate"
            self._log(f"[ModelGuided] scan-floor {mode} slot: 0/{num_replace} qualified " f"(prefix: {source_name})")

        return replaced_indices

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

    def _curriculum_verify_threshold(self) -> float:
        """Minimum partial-verify score to advance curriculum at champion_level."""

        return _curriculum_verify_threshold_fn(
            self.champion_level,
            self.generations_stalled,
            self.config.curriculum_stall_threshold,
        )

    def _curriculum_fitness_gate_passes(self) -> bool:

        return _curriculum_fitness_gate_fn(
            self.champion_level,
            self.generations_stalled,
            self._champion_raw_task_score,
            self.config.curriculum_stall_threshold,
        )

    def _curriculum_verify_passes(self, verify_score: float) -> bool:

        return verify_score >= self._curriculum_verify_threshold()

    def _curriculum_advance_preconditions_met(self) -> bool:
        """Return True when stall threshold and level caps allow curriculum check."""

        if not self.config.auto_curriculum:

            return False

        if self.champion_level >= len(LEVELS) - 1:

            return False

        _max_level = int(os.environ.get("EE_MAX_LEVEL", "999"))

        if self.champion_level >= _max_level:

            return False

        if self._champion_is_timeout_penalized():

            return False

        return self.generations_stalled >= self.config.curriculum_stall_threshold

    def _try_curriculum_advance(self) -> bool:
        """Check curriculum advancement conditions.



        R50: Curriculum advances when verify score meets a level-aware threshold.

        Low levels require full verify (1.0). High levels allow near-perfect

        partial scores (continuous_verify), with extra slack after extended stall.



        Returns True if the level was advanced.

        """

        if not self._curriculum_advance_preconditions_met():

            return False

        verified, verify_score = self._verify_champion()

        self._champion_verified = verified

        self._champion_verify_score = verify_score

        threshold = self._curriculum_verify_threshold()

        if not self._curriculum_verify_passes(verify_score):

            self._log(
                f"[Curriculum] BLOCKED (R50): stalled={self.generations_stalled} "
                f">= {self.config.curriculum_stall_threshold}, "
                f"raw_score={self._champion_raw_task_score:.1f}, "
                f"verify={verify_score:.2f} < {threshold:.3f}"
            )

            return False

        if self._curriculum_fitness_gate_passes():

            self._log(
                f"[Curriculum] Advancing: stalled={self.generations_stalled}, "
                f"raw_score={self._champion_raw_task_score:.1f}, "
                f"verify={verify_score:.2f}, "
                f"level={self.champion_level}->{self.champion_level + 1}"
            )

            self._advance_curriculum()

            return True

        self._log(
            f"[Curriculum] BLOCKED: stalled={self.generations_stalled} "
            f">= {self.config.curriculum_stall_threshold}, but "
            f"raw_score={self._champion_raw_task_score:.1f} below fitness gate"
        )

        return False

    def _try_forced_curriculum_escape(self) -> bool:
        """Deep-stall escape: advance when verify meets progressive floor."""

        if not self._curriculum_advance_preconditions_met():

            return False

        stall = self.generations_stalled

        if stall < 4 * self.config.curriculum_stall_threshold:

            return False

        verified, verify_score = self._verify_champion()

        self._champion_verified = verified

        self._champion_verify_score = verify_score

        floor = _forced_advance_verify_floor_fn(
            self.champion_level,
            stall,
            self.config.curriculum_stall_threshold,
        )

        if verify_score < floor:

            return False

        if not self._curriculum_fitness_gate_passes():

            return False

        self._log(
            f"[AntiPlateau] Forced curriculum escape at stall={stall}: "
            f"verify={verify_score:.2f} >= floor={floor:.3f}, "
            f"L{self.champion_level}->L{self.champion_level + 1}"
        )

        self._advance_curriculum()

        return True

    def _anti_plateau_escalation(self) -> None:
        """Progressive responses so fitness/curriculum cannot flatline indefinitely."""

        stall = self.generations_stalled

        if stall < self.config.curriculum_stall_threshold:

            return

        if stall >= 4 * self.config.curriculum_stall_threshold and self._try_forced_curriculum_escape():

            return

        if stall >= 25 and stall % 10 == 0:

            self._refresh_task_instance()

            self._log(f"[AntiPlateau] Task instance refresh at stall={stall}")

        if stall >= 35 and self._try_qd_champion_reset():

            self._log(f"[AntiPlateau] QD champion reset at stall={stall}")

    def _verify_champion(self) -> tuple[bool, float]:
        """R50: Run verify() on the current champion against the current level.

        Re-evaluates the champion on the current task instance so the trace

        matches the instance passed to verify() (stale traces fail verify).

        Returns (verified: bool, verify_score: float).

        """

        if self.champion is None:

            return False, 0.0

        cache_key = (tuple(self.champion), int(self.champion_level), id(self.task_instance))
        hit = getattr(self, "_verify_champion_cache", None)
        if hit is not None and hit[0] == cache_key:
            return hit[1], hit[2]

        try:

            if self.task_instance is None:

                self._refresh_task_instance()

            fresh_trace = self._evaluate(self.champion)

            if fresh_trace.get("status") in ("invalid", "error"):
                self._verify_champion_cache = (cache_key, False, 0.0)
                return False, 0.0

            self.champion_trace = fresh_trace

            level_cls = get_task_level(self.champion_level)

            output = {
                "registers": fresh_trace.get("final_regs", {}),
                "memory": fresh_trace.get("final_memory", {}),
            }

            score = level_cls(seed=self.champion_level).verify(output, self.task_instance)

            # Partial verify scoring: store the float score even when < 1.0

            # to create a fitness gradient for the L4 plateau escape.

            self._champion_partial_verify = score

            verified = score >= 1.0
            self._verify_champion_cache = (cache_key, verified, score)
            return verified, score

        except RuntimeError:

            return False, 0.0

    def _find_next_enabled_level(self, old_level: int) -> int | None:
        """Return the next enabled curriculum level, skipping circuit-breaker disabled levels."""

        new_level = old_level + 1

        max_level = min(len(LEVELS) - 1, self.config.max_level)

        while new_level <= max_level and self._is_level_disabled(new_level):

            self._log(f"[CircuitBreaker] Skipping disabled level {new_level}")

            new_level += 1

        if new_level > max_level:

            self._log(f"[CircuitBreaker] No more enabled levels after L{old_level}")

            return None

        return new_level

    def _reset_curriculum_advance_state(self, old_level: int) -> None:
        """Reset stall counters, archive tracking, and strategy state after level advance."""

        self.generations_stalled = 0

        self._qd_champion_reset_tried = False

        self._consecutive_same_strategy = 0

        self._last_strategy_used = ""

        self._champion_changed_since_last_qd = True

        self._champion_changed_since_last_novelty = True

        self._champion_qd_valid = False

        self._cached_champion_qd_desc = None

        self._bonus_threshold_cache = None

        if self.strategy_archive is not None:

            n_records = self.strategy_archive.total_records()

            decayed = self.strategy_archive.decay_level_records(old_level, factor=0.5)

            self._log(
                f"[D3] Archive decay: {decayed} success records decayed at L{old_level} "
                f"(total records: {n_records})"
            )

    def _reseed_champion_for_new_level(self, old_level: int) -> None:
        """Save old-level champion and bootstrap a fresh champion for the new level."""

        self._save_champion(old_level)

        self._refresh_task_instance()

        if self.champion is not None:

            self.champion = self._generate_initial_champion()

            self.champion_trace = self._evaluate(self.champion)

            if self.config.use_secondary_tasks:

                self._secondary_task_results = self._evaluate_secondary_tasks(self.champion)

            else:

                self._secondary_task_results = None

            self.champion_fitness = self._compute_fitness(self.champion, self.champion_trace)

    def _advance_curriculum(self) -> None:
        """Advance to the next curriculum level.



        Skips levels that have been disabled by the circuit breaker.

        """

        old_level = self.champion_level

        new_level = self._find_next_enabled_level(old_level)

        if new_level is None:

            return

        self.champion_level = new_level

        self._refresh_eval_budget()

        self._reset_curriculum_advance_state(old_level)

        self._reseed_champion_for_new_level(old_level)

        self._log(f"Curriculum advanced: L{old_level} -> L{self.champion_level}")

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
