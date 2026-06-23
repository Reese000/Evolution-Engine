#!/usr/bin/env python3
"""RV32I Evolution Engine -- Champion/Challenger with Directed Mutation.

Implements the core evolutionary loop: maintain a champion program,
generate challengers via 4 mutation strategies (RANDOM, TASK_AWARE,
DIRECTED, CROSSOVER), evaluate on the VM farm, and replace the
champion when a challenger achieves higher fitness.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any

# ----------------------------------------------------------------------
# Path setup -- import from sibling modules and milestone0
# ----------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MILESTONE0_DIR = os.path.join(SCRIPT_DIR, "..", "milestone0")
MILESTONE3_DIR = os.path.join(SCRIPT_DIR, "..", "milestone3")
MILESTONE1_DIR = os.path.join(SCRIPT_DIR, "..", "milestone1")

for _p in [
    os.path.join(MILESTONE0_DIR, "logit-masking"),
    os.path.join(MILESTONE0_DIR, "vm-farm"),
    os.path.join(MILESTONE0_DIR, "gpu-vm"),  # GPU acceleration (CuPy RV32I kernel)
    MILESTONE3_DIR,
    SCRIPT_DIR,
    MILESTONE1_DIR,
]:
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

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

        Raises:
            FileNotFoundError: If no safety-kernel binary can be found at
                any searched location.
        """
        if os.path.isfile(kernel_path):
            return kernel_path
        # Auto-detect relative to milestone2/
        candidates = [
            os.path.join(SCRIPT_DIR, "..", "milestone0", "safety-kernel", "target", "release", "safety-kernel"),
            "/root/milestone0/safety-kernel/target/release/safety-kernel",
        ]
        for c in candidates:
            if os.path.isfile(c):
                print(f"[VMFarm] Kernel binary resolved: {c}")
                return c
        raise FileNotFoundError(
            f"Safety-kernel binary not found. Searched:\n"
            f"  - {kernel_path} (given)\n"
            f"  - {candidates[0]}\n"
            f"  - {candidates[1]}\n"
            f"Build it: cd /root/milestone0/safety-kernel && cargo build --release"
        )

    def __init__(self, num_workers: int = 1, kernel_path: str = "safety_kernel",
                 use_gpu: bool = False, **kwargs):
        # Input validation
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError(f"num_workers must be a positive int, got {num_workers!r}")
        self.num_workers = num_workers
        self.kernel_path = self._resolve_kernel_path(kernel_path)
        self.use_gpu = bool(use_gpu)
        self._gpu_farm = None
        if self.use_gpu:
            try:
                from gpu_vm_farm import GpuVMFarm, HAS_CUDA
                if not HAS_CUDA:
                    self._log("CUDA not available -- GPU acceleration disabled")
                    self.use_gpu = False
                else:
                    self._gpu_farm = GpuVMFarm(batch_size=128, budget=5000)
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
        print(f"[VMFarm] {msg}")

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
        """Evaluate multiple programs. Uses GPU (30-300x faster) when use_gpu=True."""
        if self.use_gpu and self._gpu_farm is not None and len(programs) > 1:
            try:
                gpu_results = self._gpu_farm.evaluate_multi(programs, inputs_list, budget=budget)
                return [VMFarm._dict_to_eval_result(gr) for gr in gpu_results]
            except Exception as e:
                self._log(f"GPU batch failed: {e}, falling back to CPU")
        # CPU fallback
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
    MutationStrategy,
    TraceAnalysis,
    TraceAnalyzer,
    generate_seed_program,
    propose_mutation,
)
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
    """
    num_categories = 11
    if program and isinstance(program, list) and len(program) > 0:
        seen: set[str] = set()
        opcode_map = {
            0x37: "LUI",
            0x17: "AUIPC",
            0x6F: "JAL",
            0x67: "JALR",
            0x63: "BRANCH",
            0x03: "LOAD",
            0x23: "STORE",
            0x13: "ALU_IMM",
            0x33: "ALU_REG",
            0x0F: "FENCE",
            0x73: "SYSTEM",
        }
        for word in program:
            cat = opcode_map.get(word & 0x7F, "UNKNOWN")
            if cat != "UNKNOWN":
                seen.add(cat)
        inst_div = (len(seen) / num_categories) * 2.0 - 1.0
    else:
        inst_div = -1.0

    if trace and isinstance(trace, dict) and "final_regs" in trace:
        regs = trace.get("final_regs", {})
        if isinstance(regs, dict):
            active = sum(
                1
                for idx, val in regs.items()
                if isinstance(idx, int) and 1 <= idx <= 31 and isinstance(val, (int, float)) and val != 0
            )
            reg_pressure = (active / 31.0) * 2.0 - 1.0
        else:
            reg_pressure = -1.0
    else:
        reg_pressure = -1.0

    return [inst_div, reg_pressure]


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
    novelty_weight: float = 0.3
    use_novelty_archive: bool = True
    novelty_threshold: float = 0.15
    novelty_archive_path: str = ""

    # QD archive
    use_qd_archive: bool = True
    qd_bonus_weight: float = 0.1
    qd_archive_path: str = ""

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
    ref_high_augment: int = 50

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
    circuit_breaker_stall_limit: int = 20

    diversity_threshold: float = 0.3
    diversity_inject_ratio: float = 0.1
    champion_crossover_prob: float = 0.7
    crossover_dedicated_prob: float = 0.05
    max_consecutive_same_strategy: int = 30

    # Model-guided generation
    model_guided_ratio: float | None = None
    model_guided_ratio_low: float = 0.1
    model_guided_ratio_mid: float = 0.3
    model_guided_ratio_high: float = 0.6

    # CROSSOVER configuration
    crossover_unsolved_levels: tuple[int, ...] = (9, 10, 11)
    crossover_probability_unsolved: float = 0.20
    use_strategy_archive: bool = True

    def __post_init__(self) -> None:
        """Validate field values after construction."""
        if self.retrain_val_split < 0.0 or self.retrain_val_split > 1.0:
            raise ValueError(f"retrain_val_split must be in [0.0, 1.0], got {self.retrain_val_split}")
        if self.retrain_max_epochs < 1:
            raise ValueError(f"retrain_max_epochs must be >= 1, got {self.retrain_max_epochs}")
        if self.stall_threshold < 1:
            raise ValueError(f"stall_threshold must be >= 1, got {self.stall_threshold}")
        if self.retrain_batch_size < 1:
            raise ValueError(f"retrain_batch_size must be >= 1, got {self.retrain_batch_size}")
        if self.max_level < 1:
            raise ValueError(f"max_level must be >= 1, got {self.max_level}")
        if self.meta_weight_improvement < 0 or self.meta_weight_depth < 0 or self.meta_weight_diversity < 0:
            raise ValueError("meta_weight_* fields must be >= 0")
        if self.novelty_weight < 0:
            raise ValueError(f"novelty_weight must be >= 0, got {self.novelty_weight}")


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
        self.population_history: list[dict[str, Any]] = []
        self.task_instance: dict[str, Any] | None = None

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

                self.novelty_archive = SparseArchive()
            except (ImportError, Exception):
                pass

        # QD archive (optional)
        self.qd_archive: Any = None
        if _HAS_QD and self.config.use_qd_archive:
            try:
                from qd_archive import MAPElitesArchive

                self.qd_archive = MAPElitesArchive()
            except (ImportError, Exception):
                pass

        # Prefix generator for model-guided completions (lazy init)
        self._prefix_generator: Any = None

        # External evolution params (from BarrierTuner)
        self._external_evolution_params: dict[str, Any] = {}

        # Recurrent feedback
        self.recurrent_examples_count: int = 0

        # Strategy tracking for archive recording
        self._last_challenger_strategies: list[str] = []
        self._last_challenger_programs: list[list[int]] = []

        # Logging
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

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def run(self, num_generations: int | None = None) -> dict[str, Any]:
        """Run the evolution loop for *num_generations* (default: config).

        Returns a summary dict of the run.
        """
        ng = num_generations if num_generations is not None else self.config.max_generations
        start_time = time.time()

        # Initialise task for current level
        self._refresh_task_instance()

        # Generate initial champion
        if self.champion is None:
            self.champion = self._generate_initial_champion()
            self.champion_trace = self._evaluate(self.champion)
            self.champion_fitness = self._compute_fitness(self.champion, self.champion_trace)

        summary = {
            "generations_completed": 0,
            "final_fitness": self.champion_fitness,
            "final_level": self.champion_level,
            "champion_words": len(self.champion) if self.champion else 0,
            "elapsed_seconds": 0.0,
        }

        for gen in range(ng):
            self.current_generation = gen
            gen_start = time.time()

            # 1. Generate challengers
            challengers = self._generate_challengers()

            # 2. Evaluate challengers (stage 1: quick)
            results = self._evaluate_batch(challengers)

            # 3. Select elite challengers for detailed eval (stage 2)
            elite = self._select_elite(challengers, results)
            detailed_results = self._evaluate_batch_detailed(elite)

            # 4. Update champion
            improved = self._update_champion(elite, detailed_results)

            # 5. Log progress
            self._log_generation(gen, gen_start)

            # Refresh task instance every 25 generations to prevent overfitting
            if gen > 0 and gen % 25 == 0:
                self._refresh_task_instance()

            # 6. Checkpoint
            if gen > 0 and self.config.checkpoint_interval and gen % self.config.checkpoint_interval == 0:
                try:
                    self._checkpoint()
                except Exception as e:
                    self._log(f"[Checkpoint] Failed at gen {gen}: {e}")

            # 7. Curriculum check
            if improved:
                self.generations_stalled = 0
            else:
                self.generations_stalled += 1

            if self.config.auto_curriculum:
                self._try_curriculum_advance()

            # 8. Recurrent feedback check
            self._try_recurrent_feedback()

            # 9. Strategy archive record
            self._record_strategy_outcomes(results)

            # 10. QD archive update
            self._update_qd_archive()

            # 11. QD champion reset (fallback for deep stall)
            if self.generations_stalled >= 50:
                if self._try_qd_champion_reset():
                    self._log(f"[QD Reset] Champion replaced from QD archive at gen {gen}")

            # 12. Novelty archive update
            self._update_novelty_archive()

            # Early termination if curriculum maxed
            # BUT: require at least 30 stall gens at max level for adequate sampling.
            # With curriculum_stall_threshold=1, the default check is just 2 gens --
            # not enough to find good programs through random sampling (C2 collapse).
            if self.config.auto_curriculum and self.champion_level >= len(LEVELS) - 1:
                min_stall_at_max = max(self.config.curriculum_stall_threshold * 2, 30)
                if self.generations_stalled >= min_stall_at_max:
                    summary["early_stop_reason"] = "max_level_stalled"
                    break

        # Final checkpoint
        self._checkpoint()

        summary["generations_completed"] = self.current_generation + 1
        summary["final_fitness"] = self.champion_fitness
        summary["final_level"] = self.champion_level
        summary["champion_words"] = len(self.champion) if self.champion else 0
        summary["elapsed_seconds"] = time.time() - start_time

        return summary

    # ------------------------------------------------------------------
    # Task & evaluation
    # ------------------------------------------------------------------

    def _refresh_task_instance(self) -> None:
        """Generate a new task instance for the current curriculum level."""
        level_cls = get_task_level(self.champion_level)
        seed = self.rng.randint(0, 2**32 - 1)
        self.task_instance = level_cls(seed=seed).generate_instance()

    def _evaluate(self, program: list[int]) -> dict[str, Any]:
        """Run a single evaluation and return a trace dict."""
        if self.task_instance is None:
            self._refresh_task_instance()
        inputs = self.task_instance.get("inputs", {}) if self.task_instance else {}
        budget = self.config.eval_budget
        result = self.vm_farm.evaluate(program, inputs, budget=budget)
        return self._eval_result_to_trace(result)

    def _evaluate_batch(self, programs: list[list[int]]) -> list[dict[str, Any]]:
        """Evaluate multiple programs (stage 1, lighter). Uses GPU batch when available."""
        if self.task_instance is None:
            self._refresh_task_instance()
        inputs = self.task_instance.get("inputs", {}) if self.task_instance else {}
        budget = self.config.eval_budget // 2
        # Use batch evaluation (GPU when available, CPU fallback)
        inputs_list = [inputs] * len(programs)
        results = self.vm_farm.evaluate_batch(programs, inputs_list, budget=budget)
        return [self._eval_result_to_trace(r) for r in results]

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
        """
        success = getattr(result, "success", True)
        error = getattr(result, "error", None) or ""
        return {
            "status": "ok" if success else "crash",
            "final_pc": 0,
            "final_regs": getattr(result, "registers", {}),
            "final_memory": getattr(result, "memory", {}),
            "instructions_count": getattr(result, "cycles", 0),
            "pc_history": [],
            "reg_history": [],
            "memory_accesses": [],
            "trap_cause": error,
        }

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _compute_fitness(
        self,
        program: list[int],
        trace: dict[str, Any],
    ) -> float:
        """Compute crash-penalized fitness.

        Base fitness from expected output match.  Penalties for:
          - Illegal instructions (-100)
          - PC crash / timeout (-50)
          - Memory crash (-25)
          - Instruction count bloat (-0.01 per instruction over 10)
        """
        if self.task_instance is None:
            return 0.0

        expected = self.task_instance.get("expected_output", {})
        expected_regs = expected.get("registers_checked", {})

        status = trace.get("status", "")
        final_regs = trace.get("final_regs", {})
        instr_count = trace.get("instructions_count", 0)
        trap = trace.get("trap_cause", "")

        # Crash penalties
        if status in ("crash", "trap", "exception"):
            self._raw_task_score = 0.0
            if "illegal" in trap.lower() or "invalid" in trap.lower():
                return -100.0
            return -50.0

        if status == "timeout" or instr_count >= self.config.eval_budget:
            self._raw_task_score = 0.0
            return -25.0

        # Score per register match (distance-weighted partial credit)
        total = len(expected_regs)
        if total == 0:
            reg_match = 1.0
        else:
            total_score = 0.0
            for reg_idx, expected_val in expected_regs.items():
                actual_val = final_regs.get(reg_idx, 0)
                diff = abs(actual_val - expected_val)
                max_diff = max(1, abs(expected_val))
                total_score += 1.0 / (1.0 + diff / max_diff)
            reg_match = total_score / total

        # Bloat penalty
        bloat_penalty = max(0, (instr_count - 10) * 0.01)

        max_score = max(total, 1)
        base_score = reg_match * 100.0 - bloat_penalty

        # Level-aware graduated bonus for near-exact matches
        # Relaxes from 0.95 at L0 to 0.85 at L13 so high-level programs
        # (which are more complex) can still access the bonus gradient.
        bonus_threshold = max(0.85, 0.95 - 0.10 * (self.champion_level / max(1, len(LEVELS) - 1)))
        if reg_match >= bonus_threshold:
            bonus = 50.0 * (reg_match - bonus_threshold) / (1.0 - bonus_threshold)
            base_score += bonus

        # Store raw task score (before novelty/QD blending)
        self._raw_task_score = base_score
        fitness = base_score

        # v13: Novelty-weighted fitness -- reward behaviorally novel programs
        if self.config.novelty_weight > 0.0 and self.novelty_archive is not None and _HAS_NOVELTY:
            try:
                from novelty import extract_behavioral_descriptor

                descriptor = extract_behavioral_descriptor(
                    trace,
                    program,
                    budget=self.config.eval_budget,
                )
                novelty_val = self.novelty_archive.novelty_score(descriptor)
                norm = max_score / 10.0
                nw = self.config.novelty_weight
                fitness = fitness * (1.0 - nw) + novelty_val * norm * nw
            except (TypeError, ValueError, RuntimeError):
                pass  # fall through to raw fitness

        # v13: QD archive bonus -- reward under-explored behavioral niches
        if self.config.qd_bonus_weight > 0.0 and self.qd_archive is not None and _HAS_QD:
            try:
                qd_desc = compute_qd_axes(program, trace)
                if qd_desc is not None:
                    qd_bonus = self.qd_archive.selection_bias(qd_desc)
                    fitness += qd_bonus * self.config.qd_bonus_weight
            except (TypeError, ValueError, RuntimeError):
                pass

        return fitness

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
            # Encode to words
            from mutation_proposer import _encode_inst

            encoded = []
            for inst in ref:
                w = _encode_inst(inst)
                if w is not None:
                    encoded.append(w)
            if encoded:
                # Append EBREAK if not present
                if encoded[-1] != 0x0010_0073:
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

    def _generate_challengers(self) -> list[list[int]]:
        """Generate a population of challenger programs."""
        if self.champion is None:
            self._last_challenger_strategies = ["SEED"] * self.config.population_size
            return [generate_seed_program(rng=self.rng) for _ in range(self.config.population_size)]

        challengers: list[list[int]] = []
        self._last_challenger_strategies = []
        self._last_challenger_programs = []

        for i in range(self.config.population_size):
            # Choose strategy
            strategy = self._choose_mutation_strategy(i)
            self._last_challenger_strategies.append(strategy.value)

            # Apply mutation to champion
            task_inst = self.task_instance or {}
            expected = task_inst.get("expected_output", {})
            inputs = task_inst.get("inputs", {})
            # champion_trace is always a dict from _eval_result_to_trace.
            # Wrap in a list because TraceAnalyzer.analyze expects a list.
            trace_input = [self.champion_trace] if self.champion_trace is not None else []
            mutated, _desc, _analysis = propose_mutation(
                program=self.champion,
                trace=trace_input,
                expected=expected,
                inputs=inputs,
                rng=self.rng,
                task_level=self.champion_level,
                strategy=strategy,
                strategy_archive=self.strategy_archive if self.config.use_strategy_archive else None,
            )
            challengers.append(mutated)
            self._last_challenger_programs.append(mutated)

        # v12.5: Model-guided replacement -- replace fraction of challengers
        # with prefix-guided completions from the trained transformer.
        # Active at L5+ where mutation struggles to find valid programs.
        model_ratio = self._get_model_guided_ratio()
        if model_ratio > 0 and self.champion_level >= 5:
            replaced_indices = self._replace_with_model_guided(challengers, model_ratio)
            for idx in replaced_indices:
                if 0 <= idx < len(self._last_challenger_strategies):
                    self._last_challenger_strategies[idx] = "MODEL_GUIDED"

        # v12.5: QD-Archive Crossover -- inject a QD-bred hybrid as one
        # challenger when stalled >= 30 gens. This brings dormant archive
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

    def _choose_mutation_strategy(self, idx: int) -> MutationStrategy:
        """Choose a mutation strategy for challenger *idx*."""
        # Use strategy archive if available and configured
        if (
            self.config.use_strategy_archive
            and self.strategy_archive is not None
            and self.strategy_archive.total_records() > 10
        ):
            rec = self.strategy_archive.recommend_strategy(self.champion_level)
            try:
                return MutationStrategy(rec)
            except ValueError:
                pass

        # Unsolved levels 9-11: use higher CROSSOVER probability
        if self.champion_level in self.config.crossover_unsolved_levels and self.generations_stalled > 2:
            if self.rng.random() < self.config.crossover_probability_unsolved:
                return MutationStrategy.CROSSOVER

        # Default: HYBRID (mix of strategies)
        return MutationStrategy.HYBRID

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
            from mutation_proposer import _encode_inst

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
                    ckpt = os.path.join(base, "..", "milestone1", "checkpoints", "checkpoint_best.pt")
                if os.path.isfile(ckpt):
                    self._prefix_generator = PrefixGenerator(ckpt, temperature=0.9, max_new_tokens=64)
            except Exception as e:
                self._log(f"[ModelGuided] Init failed: {e}")
                return []

        if self._prefix_generator is None:
            return []

        num_replace = max(1, int(len(challengers) * model_ratio))
        replacements = 0
        model_temp = 0.9

        # Use stall-adaptive prefix_counts for APS
        if champion_level >= 8:
            if stalled >= 50:
                prefix_counts = [8, 10, 12, min(16, max(1, len(prefix_source) // 2))]
            elif stalled >= 20:
                prefix_counts = [6, 8, min(10, max(1, len(prefix_source) // 2))]
            else:
                prefix_counts = [4, 6, min(8, max(1, len(prefix_source) // 3))]
        else:
            if stalled >= 50:
                prefix_counts = [4, 6, min(8, max(1, len(prefix_source) // 2))]
            elif stalled >= 20:
                prefix_counts = [3, 5, min(6, max(1, len(prefix_source) // 2))]
            else:
                prefix_counts = [2, 4, min(6, max(1, len(prefix_source) // 3))]

        for pc in prefix_counts:
            if replacements >= num_replace:
                break
            try:
                # Temperature scheduling
                gen = getattr(self, "current_generation", 0)
                if gen < 10:
                    model_temp = 1.2
                elif gen < 30:
                    model_temp = 0.9
                else:
                    model_temp = 0.6

                # CUDA sync before model inference prevents driver race conditions
                # on CUDA 13.0. Without this, concurrent kernel launches crash (SIGSEGV).
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass

                # CUDA error recovery: catch RuntimeError (SIGSEGV), sync+retry,
                # then skip this prefix count if still failing.
                _completions = None
                _cuda_failed = False
                for attempt in range(1, 3):  # max 2 attempts
                    try:
                        _completions = self._prefix_generator.complete(
                            prefix_source,
                            prefix_count=pc,
                            num_samples=max(1, num_replace // len(prefix_counts)),
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

        return replaced_indices

    def _try_qd_crossover(self) -> list[int] | None:
        """Attempt to breed a hybrid program from the QD archive.

        When stalled >= 30 gens and the QD archive has 2+ distinct solutions,
        perform a crossover between the top two fitness-ranked programs.
        Returns the hybrid program (with EBREAK) or None.

        This brings dormant archive diversity back into the active population.
        """
        if self.qd_archive is None:
            return None
        if self.generations_stalled < 15:
            return None

        try:
            solutions = self.qd_archive.grid_solutions()
            if len(solutions) < 2:
                return None

            solutions.sort(key=lambda s: s["fitness"], reverse=True)
            best = solutions[0]
            second = solutions[1]

            if best["program"] == second["program"]:
                if len(solutions) >= 3:
                    second = solutions[2]
                else:
                    return None
                if best["program"] == second["program"]:
                    return None

            from mutation_proposer import _crossover_mutation

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
        """
        if self.qd_archive is None or not _HAS_QD:
            return
        if self.champion is None or self.champion_trace is None:
            return
        if self.config.qd_bonus_weight <= 0.0 and not self.config.use_qd_archive:
            return

        try:
            qd_desc = compute_qd_axes(self.champion, self.champion_trace)
            if qd_desc is None:
                return
            was_added = self.qd_archive.add_solution(
                descriptor=qd_desc,
                fitness=self.champion_fitness,
                program=self.champion,
                trace=self.champion_trace,
            )

            # Log descriptor distribution for diagnostic (Phase K)
            if self.current_generation % 25 == 0 or was_added:
                d0_min = self.qd_archive._descriptor_min[0]
                d0_max = self.qd_archive._descriptor_max[0]
                d1_min = self.qd_archive._descriptor_min[1]
                d1_max = self.qd_archive._descriptor_max[1]
                pd_lower = self.qd_archive._per_dim_lower
                pd_upper = self.qd_archive._per_dim_upper
                self._log(
                    f"[QD] Descriptor range gen={self.current_generation}: "
                    f"inst_div=[{d0_min:+.2f}, {d0_max:+.2f}], "
                    f"reg_press=[{d1_min:+.2f}, {d1_max:+.2f}], "
                    f"bounds=[{pd_lower[0]:+.2f}..{pd_upper[0]:+.2f}, "
                    f"{pd_lower[1]:+.2f}..{pd_upper[1]:+.2f}], "
                    f"coverage={self.qd_archive.coverage():.0%}, "
                    f"rebins={self.qd_archive._rebins}"
                )

            if was_added:
                self._log(
                    f"[QD] Added champion to grid cell "
                    f"(inst_div={qd_desc[0]:+.2f}, "
                    f"reg_press={qd_desc[1]:+.2f}, "
                    f"coverage={self.qd_archive.coverage():.0%})"
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
                self.generations_stalled = 0
                self._log(
                    f"[QD Reset] Champion replaced: "
                    f"{old_fitness:.4f} -> {qd_task_fitness:.4f} "
                    f"(from QD archive, stall reset)"
                )
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
        """
        if self.novelty_archive is None or not _HAS_NOVELTY:
            return
        if self.champion is None or self.champion_trace is None:
            return

        try:
            from novelty import extract_behavioral_descriptor

            descriptor = extract_behavioral_descriptor(
                self.champion_trace,
                self.champion,
                budget=self.config.eval_budget,
            )
            metadata: dict[str, Any] = {
                "generation": self.current_generation,
                "level": self.champion_level,
                "fitness": self.champion_fitness,
                "champion_size": len(self.champion),
                "instructions_count": self.champion_trace.get("instructions_count", 0),
                "status": self.champion_trace.get("status", ""),
            }
            check = self.novelty_archive.size > 0
            was_added, score = self.novelty_archive.add_to_archive(
                descriptor,
                metadata=metadata,
                check_threshold=check,
            )
            if was_added:
                self._log(f"[Novelty] Added descriptor (score={score:.4f}, archive_size={self.novelty_archive.size})")
            else:
                self._log(f"[Novelty] Below threshold (score={score:.4f})")
        except (TypeError, ValueError, RuntimeError) as exc:
            self._log(f"[Novelty] Skipped: {exc}")

    def _select_elite(
        self,
        challengers: list[list[int]],
        results: list[dict[str, Any]],
    ) -> list[list[int]]:
        """Select elite challengers for detailed evaluation."""
        # Compute fitness for stage 1 results
        scored = [
            (self._compute_fitness(prog, trace), prog, trace) for prog, trace in zip(challengers, results, strict=False)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        elite_size = max(1, int(self.config.population_size * self.config.elite_ratio))
        elites = [prog for _, prog, _ in scored[:elite_size]]

        # Also include tournament selection
        for _ in range(elite_size):
            contestants = self.rng.sample(range(len(scored)), min(self.config.tournament_size, len(scored)))
            winner = max(contestants, key=lambda i: scored[i][0])
            elites.append(scored[winner][1])

        # Deduplicate before the expensive stage-2 detailed evaluation.
        # The top-k slice and the tournament winners overlap heavily (a
        # high-fitness program is both ranked top-k AND wins tournaments), so
        # `elites` routinely contains the same program several times. An exact
        # duplicate evaluates identically at full budget, so dropping it cannot
        # change which program becomes champion -- it only removes redundant
        # full-budget VM evaluations (a whole safety-kernel subprocess spawn in
        # CPU mode, extra warp work on GPU) and redundant _compute_fitness calls
        # in _update_champion. Order is preserved (top-k first) for determinism.
        seen: set[tuple[int, ...]] = set()
        unique_elites: list[list[int]] = []
        for prog in elites:
            key = tuple(prog)
            if key not in seen:
                seen.add(key)
                unique_elites.append(prog)

        return unique_elites

    def _update_champion(
        self,
        elite: list[list[int]],
        detailed_results: list[dict[str, Any]],
    ) -> bool:
        """Update champion if any elite challenger has higher fitness.

        Returns True if champion was replaced.
        """
        best_fitness = self.champion_fitness
        best_program = None
        best_trace = None
        improved = False

        for prog, trace in zip(elite, detailed_results, strict=False):
            fitness = self._compute_fitness(prog, trace)
            if fitness > best_fitness:
                best_fitness = fitness
                best_program = prog
                best_trace = trace
                improved = True

        if improved and best_program is not None:
            self.champion = list(best_program)
            self.champion_trace = best_trace
            self.champion_fitness = best_fitness

        self.best_fitness_history.append(self.champion_fitness)
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

        Returns True if the level was advanced.
        """
        if not self.config.auto_curriculum:
            return False
        if self.champion_level >= len(LEVELS) - 1:
            return False

        # Check if stalled long enough
        if self.generations_stalled >= self.config.curriculum_stall_threshold:
            # Check raw task score (before novelty/QD blending) is high enough
            # Threshold lowered from 140 to 120 to work with graduated bonus curve
            if self._raw_task_score >= 120.0:
                self._advance_curriculum()
                return True

        return False

    def _advance_curriculum(self) -> None:
        """Advance to the next curriculum level."""
        old_level = self.champion_level
        self.champion_level += 1
        self.generations_stalled = 0

        # Save champion for the old level
        self._save_champion(old_level)

        # Fresh task for the new level
        self._refresh_task_instance()

        # Generate new champion for the new level (or load reference)
        if self.champion is not None:
            self.champion = self._generate_initial_champion()
            self.champion_trace = self._evaluate(self.champion)
            self.champion_fitness = self._compute_fitness(self.champion, self.champion_trace)

        # Log
        self._log(f"Curriculum advanced: L{old_level} -> L{self.champion_level}")

    # ------------------------------------------------------------------
    # Recurrent feedback
    # ------------------------------------------------------------------

    def _try_recurrent_feedback(self) -> bool:
        """Check if curriculum has stalled and save recurrent feedback data.

        When curriculum stalls >= 2 times, saves champion program and trace
        to milestone1/recurrent_data/recurrent_data.jsonl.

        Returns True if >= 50 examples accumulated (signal to retrain).
        """
        if self.generations_stalled < 2:
            return False

        # Guard: champion must exist before checking fitness
        if not self.champion or not self.champion_trace:
            return False

        # Fitness gate: don't save failing champion traces unless deeply stalled
        # Low-fitness champions produce training signal that teaches the model
        # to generate failing programs. Only bypass the gate when the system
        # has been stalled long enough that any champion data is better than none.
        champion_fitness = getattr(self, 'champion_fitness', None)
        if champion_fitness is not None and champion_fitness <= 0 and self.generations_stalled < self.config.circuit_breaker_stall_limit:
            self._log(f"Fitness gate blocked trace: fitness={champion_fitness:.2f}, stalled={self.generations_stalled} < limit={self.config.circuit_breaker_stall_limit}")
            return False

        # Raw task score gate: block traces where champion barely scored
        # (positive fitness may come from novelty/QD blending, not task performance).
        # Only apply when fitness > 0 (crashing programs with negative fitness
        # still provide useful training signal about what NOT to do).
        raw = getattr(self, '_raw_task_score', 0.0)
        if champion_fitness is not None and champion_fitness > 0 and raw < 10.0:
            self._log(f"Task score gate blocked trace: champion_fitness={champion_fitness:.2f}, raw_task_score={raw:.2f} < 10.0")
            return False

        # Determine recurrent data directory
        recur_dir = self.config.recurrent_feedback_dir
        if not recur_dir:
            recur_dir = os.path.join(MILESTONE1_DIR, "recurrent_data")
        os.makedirs(recur_dir, exist_ok=True)

        # Build example
        example = {
            "level": self.champion_level,
            "generation": self.current_generation,
            "fitness": self.champion_fitness,
            "champion_sha256": sha256_program(self.champion),
            "champion_words": len(self.champion),
            "champion_program": [hex(w) for w in self.champion],
            "trace_status": self.champion_trace.get("status", ""),
            "trace_instr_count": self.champion_trace.get("instructions_count", 0),
            "stalled_count": self.generations_stalled,
        }

        # Append to JSONL
        jsonl_path = os.path.join(recur_dir, "recurrent_data.jsonl")
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(example) + "\n")

        # Count accumulated examples
        self.recurrent_examples_count += 1

        # Log
        count = self.recurrent_examples_count
        self._log(f"Recurrent feedback saved ({count} total)")

        return count >= 50

    # ------------------------------------------------------------------
    # Strategy archive
    # ------------------------------------------------------------------

    def _record_strategy_outcomes(self, results: list[dict[str, Any]]) -> None:
        """Record strategy outcomes in the strategy archive.

        Matches each challenger's strategy (from _last_challenger_strategies)
        against its evaluation result. Top 10% by fitness are recorded as
        successes, the rest as failures. This gives the archive fine-grained
        signal even when no challenger beats the champion.
        """
        if self.strategy_archive is None:
            return
        strategies = self._last_challenger_strategies
        # Need the challenger programs to compute actual fitness
        # (results dicts from _evaluate_batch don't carry fitness values)
        if not strategies:
            return

        # Compute fitness from trace results and stored challenger programs
        fitnesses = []
        for i, r in enumerate(results):
            if hasattr(self, '_last_challenger_programs') and i < len(self._last_challenger_programs):
                prog = self._last_challenger_programs[i]
                fit = self._compute_fitness(prog, r)
                fitnesses.append(fit)
            else:
                fitnesses.append(0.0)

        if not fitnesses:
            return

        # Top 10% threshold
        sorted_f = sorted(fitnesses, reverse=True)
        threshold = sorted_f[max(0, len(sorted_f) // 10)]

        # Record each challenger's outcome
        gen = self.current_generation
        lvl = self.champion_level
        for strat, fit in zip(strategies, fitnesses):
            self.strategy_archive.record(
                strategy=strat,
                level=lvl,
                success=fit >= threshold,
                generation=gen,
            )

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
        """Save a full checkpoint of the evolution state."""
        if not self.config.checkpoint_interval:
            return

        cp_dir = os.path.join(SCRIPT_DIR, "..", "milestone1", "checkpoints")
        if self.config.log_dir:
            cp_dir = os.path.join(self.config.log_dir, "checkpoints")
        os.makedirs(cp_dir, exist_ok=True)

        # Save champion
        self._save_champion()

        # Save strategy archive
        if self.strategy_archive is not None:
            sa_path = self.config.strategy_archive_path
            if not sa_path:
                sa_path = os.path.join(cp_dir, f"strategy_archive_gen{self.current_generation}.json")
            self.strategy_archive.save(sa_path)

        # Save metadata
        meta = {
            "generation": self.current_generation,
            "level": self.champion_level,
            "fitness": self.champion_fitness,
            "program_words": len(self.champion) if self.champion else 0,
            "generations_stalled": self.generations_stalled,
        }
        meta_path = os.path.join(cp_dir, f"meta_gen{self.current_generation}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Log a message to stdout and optional log file."""
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] Gen {self.current_generation:4d} | {message}"
        print(line)
        if self.config.log_dir:
            log_path = os.path.join(self.config.log_dir, "evolution.log")
            with open(log_path, "a") as f:
                f.write(line + "\n")

    def _log_generation(self, gen: int, start_time: float) -> None:
        """Log per-generation stats."""
        elapsed = time.time() - start_time
        self._log(
            f"Level={self.champion_level}  "
            f"Fitness={self.champion_fitness:7.1f}  "
            f"W={len(self.champion) if self.champion else 0:3d}  "
            f"Stalled={self.generations_stalled:2d}  "
            f"T={elapsed:.2f}s"
        )

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

    print("\n" + "=" * 60)
    print("Evolution Run Complete")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
