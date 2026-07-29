"""Round 6 improvements — full loop reliability tests.

Tests for:
  1. _select_elite dedup no longer misaligns fitness values
  2. _record_strategy_outcomes length guard (no silent truncation)
  3. skip_evolution flag survives across cycles (RESET-AND-RELEARN works)
  4. _close_log_handle cleans up on exception in run()
  5. _try_recurrent_feedback works on gen 0 (no stall gate)
  6. EvolutionEngine._log_file initialized in __init__ (no hasattr needed)
  7. run() returns partial summary when exception occurs mid-loop
"""

import os
import random
import tempfile
from unittest.mock import MagicMock

# ── _select_elite dedup fitness alignment ────────────────────────────


class TestSelectEliteDedupAlignment:
    """Verify that dedup removes both program AND fitness together."""

    def test_tournament_dedup_preserves_fitness_alignment(self):
        """When a tournament winner duplicates a top-k elite, dedup
        must not leave a mismatched fitness value."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig(population_size=10, elite_ratio=0.2, tournament_size=3, eval_budget=1000)
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = config
        engine.rng = random.Random(42)
        engine.strategy_archive = None
        engine.champion_level = 0
        engine.champion_fitness = 0.0
        engine.champion_trace = None
        engine._raw_task_score = 0.0
        engine._current_diversity = 0.5
        engine._prefix_generator = None
        engine._external_evolution_params = {}
        engine._consecutive_same_strategy = 0
        engine._last_strategy_used = ""
        engine.circuit_breaker = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        engine.current_generation = 0
        engine._log_file = None

        # Create 10 challengers with known programs and fitnesses
        programs = [list(range(i, i + 5)) for i in range(10)]
        results = [{"status": "ok", "final_regs": {}, "instructions_count": 5} for _ in range(10)]

        # Patch _compute_fitness to return deterministic values
        fitness_map = {}
        for i, prog in enumerate(programs):
            fitness_map[tuple(prog)] = float(i * 10)  # 0, 10, 20, ..., 90

        def fake_fitness(prog, trace, eval_budget=None, raw_only=False):
            return fitness_map.get(tuple(prog), 0.0)

        engine._compute_fitness = fake_fitness

        # Patch _refresh_task_instance (not needed)
        engine.task_instance = {"inputs": {}, "expected_output": {"registers": {}}}

        elites, elite_fits, _idxs = engine._select_elite(programs, results)

        # Elites should be unique programs
        seen = set()
        for prog, fit in zip(elites, elite_fits):
            key = tuple(prog)
            assert key not in seen, f"Duplicate program in elites: {key}"
            seen.add(key)

        # Fitnesses should match their programs (not shuffled)
        for prog, fit in zip(elites, elite_fits):
            expected_fit = fitness_map.get(tuple(prog), -1)
            assert fit == expected_fit, (
                f"Fitness misalignment: program {prog} has fit {fit} " f"but expected {expected_fit}"
            )

    def test_elites_count_never_exceeds_population(self):
        """Elite selection should not produce more than population_size items."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig(population_size=5, elite_ratio=0.4, tournament_size=3, eval_budget=1000)
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = config
        engine.rng = random.Random(42)
        engine.strategy_archive = None
        engine.champion_level = 0
        engine.champion_fitness = 0.0
        engine.champion_trace = None
        engine._raw_task_score = 0.0
        engine._current_diversity = 0.5
        engine._prefix_generator = None
        engine._external_evolution_params = {}
        engine._consecutive_same_strategy = 0
        engine._last_strategy_used = ""
        engine.circuit_breaker = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        engine.current_generation = 0
        engine._log_file = None

        programs = [list(range(i, i + 3)) for i in range(5)]
        results = [{"status": "ok", "final_regs": {}, "instructions_count": 3} for _ in range(5)]

        engine._compute_fitness = lambda prog, trace, eval_budget=None, raw_only=False: float(sum(prog))
        engine.task_instance = {"inputs": {}, "expected_output": {"registers": {}}}

        elites, elite_fits, _idxs = engine._select_elite(programs, results)
        assert len(elites) <= len(programs), f"Elite count {len(elites)} exceeds population {len(programs)}"


# ── _record_strategy_outcomes length guard ──────────────────────────


class TestRecordStrategyOutcomesGuard:
    """Verify _record_strategy_outcomes handles mismatched lengths."""

    def test_mismatched_lengths_handled_safely(self):
        """When _last_challenger_strategies and _last_challenger_fitnesses
        have different lengths, the function should not crash."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = EvolutionConfig()
        engine._last_challenger_strategies = ["GROW", "HYBRID"]  # 2 strategies
        engine._last_challenger_fitnesses = [1.0, 2.0, 3.0]  # 3 fitnesses
        engine.current_generation = 5
        engine.champion_level = 3

        from strategy_archive import StrategyArchive

        engine.strategy_archive = StrategyArchive()

        # Should not raise
        engine._record_strategy_outcomes()

        # Should have recorded only 2 (the min length)
        assert engine.strategy_archive.total_records() == 2

    def test_empty_strategies_returns_early(self):
        """Empty strategies list should be a no-op."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = EvolutionConfig()
        engine._last_challenger_strategies = []
        engine._last_challenger_fitnesses = [1.0, 2.0]

        from strategy_archive import StrategyArchive

        engine.strategy_archive = StrategyArchive()

        engine._record_strategy_outcomes()
        assert engine.strategy_archive.total_records() == 0


# ── skip_evolution flag lifecycle ────────────────────────────────────


class TestSkipEvolutionFlag:
    """Verify skip_evolution is initialized before the loop and persists
    across cycles when set by RESET-AND-RELEARN."""

    def test_skip_evolution_initialized_outside_loop(self):
        """The skip_evolution variable is declared BEFORE the for-loop
        in main(), so it survives across iterations. Verify it does not
        reset at the top of each cycle."""
        # We test by reading the source file and checking the structure
        source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_multi_cycle.py")
        with open(source_path, encoding="utf-8") as f:
            source = f.read()

        # skip_evolution should be initialized BEFORE the for loop
        # Find the for loop line
        for_line = None
        skip_line = None
        for i, line in enumerate(source.splitlines()):
            stripped = line.strip()
            if "for cycle in range(1," in stripped and "args.cycles" in stripped:
                for_line = i
            if "skip_evolution = False" in stripped and "# R6" in stripped:
                skip_line = i

        assert for_line is not None, "for-loop not found in run_multi_cycle.py"
        assert skip_line is not None, "R6 skip_evolution init not found"
        assert skip_line < for_line, (
            f"skip_evolution (line {skip_line}) must be initialized " f"BEFORE the for-loop (line {for_line})"
        )


# ── _close_log_handle lifecycle ─────────────────────────────────────


class TestCloseLogHandle:
    """Verify _close_log_handle properly cleans up."""

    def test_close_log_handle_resets_to_none(self):
        """After calling _close_log_handle, the log file should be None."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = EvolutionConfig()
        engine._log_file = None

        # Should be safe even when None
        engine._close_log_handle()
        assert engine._log_file is None

    def test_close_log_handle_closes_file(self):
        """A real open file handle should be closed and reset to None."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = EvolutionConfig()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as f:
            log_path = f.name
        try:
            engine._log_file = open(log_path, "w")
            engine._close_log_handle()
            assert engine._log_file is None
        finally:
            os.unlink(log_path)

    def test_run_always_closes_log_handle_on_exception(self):
        """If an exception occurs during run(), the log handle is closed."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig(max_generations=3, population_size=5)
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = config
        engine.rng = random.Random(42)
        engine._log_file = None
        engine._log_path_cache = None
        engine._log_handle_closed = False
        engine.champion = None
        engine.champion_level = 0
        engine.champion_fitness = 0.0
        engine.champion_trace = None
        engine._raw_task_score = 0.0
        engine._current_diversity = 0.5
        engine._prefix_generator = None
        engine._external_evolution_params = {}
        engine._consecutive_same_strategy = 0
        engine._last_strategy_used = ""
        engine.circuit_breaker = config.CIRCUIT_BREAKER_SENTINEL
        engine.current_generation = 0
        engine.generations_stalled = 0
        engine._qd_champion_reset_tried = False
        engine.best_fitness_history = []
        engine._last_challenger_strategies = []
        engine._last_challenger_programs = []
        engine._last_challenger_fitnesses = []
        engine._last_challengers = []
        engine._last_results = []
        engine.recurrent_examples_count = 0
        engine.novelty_archive = None
        engine.qd_archive = None
        engine._seed_model_path = ""
        engine._saved_sha256s = {}

        # Create a real log file
        with tempfile.TemporaryDirectory() as tmp:
            engine.config.log_dir = tmp
            engine._log_file = open(os.path.join(tmp, "test.log"), "a")

            # Patch methods to raise on the first generation
            def bad_generate_challengers():
                raise RuntimeError("Simulated failure in challenger generation")

            engine._generate_challengers = bad_generate_challengers
            engine._refresh_task_instance = lambda: None
            engine._generate_initial_champion = lambda: [0x00100073]
            engine._evaluate = lambda p: {"status": "ok", "final_regs": {}, "instructions_count": 1}
            engine._compute_fitness = lambda p, t, **kw: 0.0

            # run() should catch the exception but still close the log
            try:
                engine.run()
            except Exception:
                pass

            assert engine._log_file is None, "Log handle not cleaned up after exception in run()"


# ── _try_recurrent_feedback gen 0 ───────────────────────────────────


class TestRecurrentFeedbackGen0:
    """Verify recurrent feedback works on generation 0."""

    def test_feedback_runs_without_stall(self):
        """_try_recurrent_feedback should run when generations_stalled=0
        (it only requires _last_challengers to be non-empty)."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = EvolutionConfig()
        engine.champion_level = 0
        engine.champion_fitness = 0.0
        engine.current_generation = 0
        engine.generations_stalled = 0  # not stalled at all
        # Program must have 3+ instructions (len(prog) < 3 is skipped)
        engine._last_challengers = [[0x00100073, 0x00100073, 0x00100073]]
        engine._last_results = [{"status": "ok", "final_regs": {}, "instructions_count": 3}]
        engine._last_challenger_fitnesses = [50.0]  # positive fitness
        engine.recurrent_examples_count = 0
        engine._saved_sha256s = {}
        engine.task_instance = {
            "inputs": {},
            "expected_output": {"registers": {}},
        }

        with tempfile.TemporaryDirectory() as tmp:
            engine.config.recurrent_feedback_dir = tmp
            result = engine._try_recurrent_feedback()

            # Should have saved at least one example
            assert engine.recurrent_examples_count >= 1, "Expected at least one recurrent example saved on gen 0"


# ── _log_file attribute in __init__ ─────────────────────────────────


class TestLogFileAttributeInit:
    """Verify _log_file is explicitly initialized in __init__."""

    def test_log_file_initialized(self):
        """EvolutionEngine.__init__ should set _log_file to None."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig()
        engine = EvolutionEngine(config=config)

        assert hasattr(engine, "_log_file"), "_log_file attribute not found on EvolutionEngine instance"
        assert engine._log_file is None, f"_log_file should be None after init, got {engine._log_file!r}"


# ── run() partial summary on exception ──────────────────────────────


class TestRunPartialSummary:
    """Verify run() returns a summary even when an exception occurs."""

    def test_run_returns_summary_on_exception(self):
        """When an exception occurs mid-loop, run() should still return
        a summary dict (not raise)."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig(max_generations=5, population_size=5, log_dir="", checkpoint_interval=0)
        engine = EvolutionEngine.__new__(EvolutionEngine)
        engine.config = config
        engine.rng = random.Random(42)
        engine._log_file = None
        engine.champion = [0x00100073]
        engine.champion_level = 0
        engine.champion_fitness = 10.0
        engine.champion_trace = {"status": "ok", "final_regs": {}, "instructions_count": 1}
        engine._raw_task_score = 10.0
        engine._current_diversity = 0.5
        engine._prefix_generator = None
        engine._external_evolution_params = {}
        engine._consecutive_same_strategy = 0
        engine._last_strategy_used = ""
        engine.circuit_breaker = config.CIRCUIT_BREAKER_SENTINEL
        engine.current_generation = 0
        engine.generations_stalled = 0
        engine._qd_champion_reset_tried = False
        engine.best_fitness_history = []
        engine._last_challenger_strategies = []
        engine._last_challenger_programs = []
        engine._last_challenger_fitnesses = []
        engine._last_challengers = []
        engine._last_results = []
        engine.recurrent_examples_count = 0
        engine.novelty_archive = None
        engine.qd_archive = None
        engine._seed_model_path = ""
        engine._saved_sha256s = {}
        engine.champion_anchor = MagicMock()
        engine.vm_farm = MagicMock()

        # Patch _refresh_task_instance and _evaluate to work normally
        engine.task_instance = {"inputs": {}, "expected_output": {"registers": {}}}
        engine._refresh_task_instance = lambda: None
        engine._evaluate = lambda p: {"status": "ok", "final_regs": {}, "instructions_count": 1}
        engine._compute_fitness = lambda p, t, **kw: 10.0

        # Patch _generate_challengers to always fail -- the exception
        # is caught inside the try/finally in run()
        def failing_challengers():
            raise RuntimeError("Simulated failure")

        engine._generate_challengers = failing_challengers
        engine._evaluate_batch = lambda p: [{"status": "ok", "final_regs": {}, "instructions_count": 1} for _ in p]
        engine._select_elite = lambda c, r: ([c[0]], [0.0], [0])
        engine._evaluate_batch_detailed = lambda p: [
            {"status": "ok", "final_regs": {}, "instructions_count": 1} for _ in p
        ]
        engine._update_champion = lambda e, r, stage1_fitnesses=None: False
        engine._log_generation = lambda g, t: None
        engine._try_recurrent_feedback = lambda: False
        engine._try_curriculum_advance = lambda: False
        engine._record_strategy_outcomes = lambda: None
        engine._update_qd_archive = lambda: None
        engine._try_qd_champion_reset = lambda: False
        engine._update_novelty_archive = lambda: None
        engine._checkpoint = lambda: None

        summary = engine.run()

        assert isinstance(summary, dict), "run() should return a dict even on exception"
        assert "final_fitness" in summary
        assert "final_level" in summary
        assert "generations_completed" in summary


# ── val_loss=None fallback does not undo rollback ────────────────────


class TestValLossNoneDoesNotUndoRollback:
    """When val_loss is None (unparsable) and a retry fails, the
    fallback checkpoint copy should NOT override a rollback that
    was just performed."""

    def test_fallback_skips_when_no_valid_checkpoint(self):
        """If new_ck is the same as the original retrain result but
        no retry succeeded, the code should not blindly copy."""
        # This is a structural test that the new code path exists
        source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_multi_cycle.py")
        with open(source_path, encoding="utf-8") as f:
            source = f.read()

        # The old unconditional copy (the bug) is no longer present
        assert (
            "Accept checkpoint unconditionally as fallback" not in source
        ), "The old unconditional fallback copy should be removed"
        # The new safe path should be present
        assert (
            "Fallback accept (val_loss unparsable)" in source
        ), "New safe fallback path not found in run_multi_cycle.py"
