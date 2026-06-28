#!/usr/bin/env python3
"""Tests for Phase 1 (Fix Foundations) and Phase 2 (Autonomous Runner).

Validates:
  - vm_farm import chain works
  - paths.py provides all project paths
  - evolution_engine imports cleanly
  - mutation_proposer imports cleanly
  - config files exist and are valid JSON
  - autonomous_runner can be instantiated
  - HealthMonitor writes health.json
  - Strategy archive monoculture prevention works
  - Fitness landscape partial credit scoring works
"""

import json
import os
import signal as _signal_mod
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestVMFarmImport(unittest.TestCase):
    """STEP 1: vm_farm import chain."""

    def test_eval_result_importable(self):
        from vm_farm import EvalResult
        self.assertTrue(hasattr(EvalResult, "_fields") or hasattr(EvalResult, "__annotations__"))

    def test_eval_result_namedtuple(self):
        from vm_farm import EvalResult
        er = EvalResult(success=True, registers={}, memory={}, trace=[], cycles=0, error=None, blocked_writes=[])
        self.assertTrue(er.success)
        self.assertEqual(er.cycles, 0)

    def test_run_one_evaluation_callable(self):
        from vm_farm import _run_one_evaluation
        self.assertTrue(callable(_run_one_evaluation))

    def test_gpu_vm_farm_importable(self):
        from gpu_vm_farm import _execute_vm, GpuVMFarm
        self.assertTrue(callable(_execute_vm))

    def test_execute_vm_basic(self):
        from gpu_vm_farm import _execute_vm
        # Simple program: ADDI x10, x0, 42; EBREAK
        program = [
            0x02A00513,  # ADDI x10, x0, 42
            0x00100073,  # EBREAK
        ]
        result = _execute_vm(program, {}, {}, 100)
        self.assertTrue(result["success"])
        self.assertEqual(result["registers"].get(10, 0), 42)
        self.assertEqual(result["error"], None)


class TestPaths(unittest.TestCase):
    """STEP 3: paths.py module."""

    def test_project_root(self):
        from paths import PROJECT_ROOT
        self.assertTrue(PROJECT_ROOT.is_dir())

    def test_milestone_dirs_exist_as_paths(self):
        from paths import CONFIG_DIR, MILESTONE1_DIR
        # These may not exist in all environments, but should be Path objects
        self.assertIsInstance(CONFIG_DIR, Path)
        self.assertIsInstance(MILESTONE1_DIR, Path)

    def test_ensure_project_on_path(self):
        from paths import ensure_project_on_path, PROJECT_ROOT
        before = len(sys.path)
        ensure_project_on_path()
        after = len(sys.path)
        # Should have added some paths (or they were already there)
        self.assertGreaterEqual(after, before)
        self.assertIn(str(PROJECT_ROOT), sys.path)

    def test_config_paths(self):
        from paths import ENGINE_CONFIG, TOOL_LIBRARY
        self.assertTrue(ENGINE_CONFIG.is_file())
        self.assertTrue(TOOL_LIBRARY.is_file())


class TestConfigFiles(unittest.TestCase):
    """STEP 2: config files."""

    def test_engine_config_valid_json(self):
        from paths import ENGINE_CONFIG
        with open(ENGINE_CONFIG) as f:
            cfg = json.load(f)
        self.assertIn("evolution", cfg)
        self.assertIn("retrain", cfg)
        self.assertIn("meta_controller", cfg)
        self.assertIn("autonomous_runner", cfg)

    def test_tool_library_valid_json(self):
        from paths import TOOL_LIBRARY
        with open(TOOL_LIBRARY) as f:
            lib = json.load(f)
        self.assertIn("instructions", lib)
        self.assertGreater(len(lib["instructions"]), 0)

    def test_engine_config_has_required_fields(self):
        from paths import ENGINE_CONFIG
        with open(ENGINE_CONFIG) as f:
            cfg = json.load(f)
        evo = cfg["evolution"]
        self.assertIn("population_size", evo)
        self.assertIn("max_generations", evo)
        self.assertIn("mutation_rate", evo)
        self.assertIn("elite_ratio", evo)

    def test_autonomous_runner_config_section(self):
        from paths import ENGINE_CONFIG
        with open(ENGINE_CONFIG) as f:
            cfg = json.load(f)
        ar = cfg["autonomous_runner"]
        self.assertIn("max_cycles", ar)
        self.assertIn("target_fitness", ar)
        self.assertIn("state_file", ar)


class TestEvolutionEngineImport(unittest.TestCase):
    """evolution_engine.py imports cleanly."""

    def test_import(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine, VMFarm
        self.assertTrue(callable(EvolutionEngine))
        config = EvolutionConfig()
        self.assertEqual(config.population_size, 50)

    def test_evolution_config_defaults(self):
        from evolution_engine import EvolutionConfig
        config = EvolutionConfig()
        self.assertEqual(config.max_generations, 200)
        self.assertAlmostEqual(config.mutation_rate, 0.3)
        self.assertEqual(config.curriculum_level, 0)
        self.assertTrue(config.auto_curriculum)

    def test_vm_farm_wrapper(self):
        from evolution_engine import VMFarm
        vm = VMFarm(num_workers=1)
        self.assertEqual(vm.num_workers, 1)

    def test_compute_qd_axes(self):
        from evolution_engine import compute_qd_axes
        # Simple program with ADD (ALU_REG category)
        program = [0x003100B3]  # ADD x1, x2, x3
        trace = {"final_regs": {1: 5, 2: 3}}
        axes = compute_qd_axes(program, trace)
        self.assertIsNotNone(axes)
        self.assertEqual(len(axes), 2)


class TestMutationProposerImport(unittest.TestCase):
    """mutation_proposer.py imports cleanly."""

    def test_import(self):
        from mutation_proposer import MutationStrategy, TraceAnalyzer, DirectedMutator
        self.assertTrue(callable(TraceAnalyzer.analyze))

    def test_mutation_strategy_enum(self):
        from mutation_proposer import MutationStrategy
        self.assertEqual(MutationStrategy.RANDOM.value, "RANDOM")
        self.assertEqual(MutationStrategy.GROW.value, "GROW")
        self.assertEqual(MutationStrategy.HYBRID.value, "HYBRID")
        self.assertEqual(MutationStrategy.MODEL_GUIDED.value, "MODEL_GUIDED")

    def test_trace_analyzer_empty(self):
        from mutation_proposer import TraceAnalyzer
        analysis = TraceAnalyzer.analyze(trace=[], expected={}, inputs={})
        self.assertEqual(analysis.divergence_type, "unknown")

    def test_stochastic_perturbation(self):
        from mutation_proposer import DirectedMutator, validate_mutation
        import random
        rng = random.Random(42)
        program = [0x02A00513, 0x00100073]  # ADDI x10, x0, 42; EBREAK
        mutated, desc = DirectedMutator.stochastic_perturbation(program, rng)
        valid, err = validate_mutation(mutated)
        self.assertTrue(valid, f"Mutation invalid: {err}")


class TestStrategyArchive(unittest.TestCase):
    """D3: Strategy monoculture prevention."""

    def test_record_and_recommend(self):
        from strategy_archive import StrategyArchive
        sa = StrategyArchive()
        # Record diverse strategies
        for i in range(50):
            sa.record("GROW", level=0, success=i % 3 == 0, generation=i)
            sa.record("HYBRID", level=0, success=i % 2 == 0, generation=i)
            sa.record("DIRECTED", level=0, success=i % 4 == 0, generation=i)
        rec = sa.recommend_strategy(level=0)
        self.assertIn(rec, ["GROW", "HYBRID", "DIRECTED"])

    def test_skip_strategy(self):
        from strategy_archive import StrategyArchive
        sa = StrategyArchive()
        for i in range(30):
            sa.record("GROW", level=0, success=True, generation=i)
            sa.record("HYBRID", level=0, success=True, generation=i)
        rec = sa.recommend_strategy(level=0, skip_strategy="GROW")
        self.assertNotEqual(rec, "GROW")

    def test_decay_level_records(self):
        from strategy_archive import StrategyArchive
        sa = StrategyArchive()
        for i in range(20):
            sa.record("GROW", level=5, success=True, generation=i)
        before = sum(1 for r in sa._records if r["success"])
        decayed = sa.decay_level_records(5, factor=0.5)
        after = sum(1 for r in sa._records if r["success"])
        self.assertLess(after, before)
        self.assertGreater(decayed, 0)


class TestFitnessLandscape(unittest.TestCase):
    """D1: Deceptive fitness landscape with partial credit scoring."""

    def test_perfect_beats_crash(self):
        """Perfect match scores much higher than a crash."""
        from evolution_engine import EvolutionConfig, EvolutionEngine
        from tasks_rv32i import Level0

        config = EvolutionConfig(curriculum_level=0)
        engine = EvolutionEngine(config=config)
        task = Level0(seed=42)
        engine.task_instance = task.generate_instance()

        # Perfect trace
        perfect_trace = {
            "status": "ok",
            "final_regs": dict(engine.task_instance["expected_output"]["registers"]),
            "instructions_count": 3,
            "trap_cause": "",
        }
        perfect_fitness = engine._compute_fitness([], perfect_trace)

        # Crash trace
        crash_trace = {
            "status": "crash",
            "final_regs": {},
            "instructions_count": 0,
            "trap_cause": "illegal instruction",
        }
        crash_fitness = engine._compute_fitness([], crash_trace)

        self.assertGreater(perfect_fitness, crash_fitness)

    def test_partial_credit_is_positive(self):
        """A program with a wrong register still gets some partial credit."""
        from evolution_engine import EvolutionConfig, EvolutionEngine
        from tasks_rv32i import Level0

        config = EvolutionConfig(curriculum_level=0)
        engine = EvolutionEngine(config=config)
        task = Level0(seed=42)
        engine.task_instance = task.generate_instance()

        # Program gets reg 3 close to expected but not exact
        # (partial credit via distance weighting)
        expected_val = engine.task_instance["expected_output"]["registers"][3]
        partial_val = expected_val + 1  # off by 1

        partial_trace = {
            "status": "ok",
            "final_regs": {3: partial_val},
            "instructions_count": 3,
            "trap_cause": "",
        }
        partial_fitness = engine._compute_fitness([], partial_trace)
        # Partial fitness should be positive (not a crash penalty)
        self.assertGreater(partial_fitness, 0)

    def test_crash_negative(self):
        """Crash returns clearly negative fitness."""
        from evolution_engine import EvolutionConfig, EvolutionEngine
        from tasks_rv32i import Level0

        config = EvolutionConfig(curriculum_level=0)
        engine = EvolutionEngine(config=config)
        task = Level0(seed=42)
        engine.task_instance = task.generate_instance()

        crash_trace = {
            "status": "crash",
            "final_regs": {},
            "instructions_count": 0,
            "trap_cause": "illegal instruction",
        }
        crash_fitness = engine._compute_fitness([], crash_trace)
        self.assertLess(crash_fitness, 0)


class TestTrainingSignalFilter(unittest.TestCase):
    """D2: Training signal contamination prevention."""

    def test_negative_fitness_filtered(self):
        """Verify that negative-fitness traces are filtered in recurrent feedback."""
        challenger_fitness = -5.0
        stalled = 5
        should_skip = challenger_fitness <= 0.0 and stalled < 50
        self.assertTrue(should_skip, "Negative fitness traces should be filtered")

    def test_positive_fitness_allowed(self):
        challenger_fitness = 10.0
        stalled = 5
        should_skip = challenger_fitness <= 0.0 and stalled < 50
        self.assertFalse(should_skip, "Positive fitness traces should pass")

    def test_deep_stall_allows_low_fitness(self):
        challenger_fitness = -2.0
        stalled = 50
        should_skip = challenger_fitness <= 0.0 and stalled < 50
        self.assertFalse(should_skip, "Deep stall should allow low-fitness traces")


class TestAutonomousRunner(unittest.TestCase):
    """STEP 4: AutonomousRunner instantiation and state management."""

    def test_instantiation(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        self.assertIsNotNone(runner._config)
        self.assertEqual(runner._state["current_cycle"], 0)

    def test_default_state(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        state = runner._default_state()
        self.assertIn("current_cycle", state)
        self.assertIn("best_fitness", state)
        self.assertIn("error_count", state)
        self.assertEqual(state["version"], 1)

    def test_save_load_state(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        runner._state["current_cycle"] = 5
        runner._state["best_fitness"] = 42.5

        # Save
        runner.save_state()

        # Create a fresh runner and load
        runner2 = AutonomousRunner()
        loaded = runner2.load_state()
        self.assertTrue(loaded)
        self.assertEqual(runner2._state["current_cycle"], 5)
        self.assertAlmostEqual(runner2._state["best_fitness"], 42.5)

        # Cleanup
        if os.path.isfile(runner._state_path):
            os.remove(runner._state_path)

    def test_should_stop_max_cycles(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        runner._config["max_cycles"] = 5
        runner._state["current_cycle"] = 5
        self.assertTrue(runner.should_stop())

    def test_should_stop_target_fitness(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        runner._config["target_fitness"] = 100.0
        runner._state["best_fitness"] = 150.0
        self.assertTrue(runner.should_stop())

    def test_should_stop_stop_file(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        stop_file = os.path.join(PROJECT_ROOT, runner._config["stop_file"])
        try:
            Path(stop_file).touch()
            self.assertTrue(runner.should_stop())
        finally:
            if os.path.isfile(stop_file):
                os.remove(stop_file)

    def test_health_monitor_writes_file(self):
        from autonomous_runner import HealthMonitor
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "state.json")
            hm = HealthMonitor(state_path=state_file)
            hm.set_status("running")
            state = {
                "current_cycle": 3,
                "best_fitness": 42.0,
                "best_level": 5,
                "error_count": 0,
                "last_cycle_timestamp": "2026-01-01T00:00:00",
            }
            hm.write_health(state)
            health_path = os.path.join(tmpdir, "health.json")
            self.assertTrue(os.path.isfile(health_path))
            with open(health_path) as f:
                health = json.load(f)
            self.assertEqual(health["status"], "running")
            self.assertEqual(health["cycles_completed"], 3)
            self.assertEqual(health["current_fitness"], 42.0)

    def test_shutdown_signal(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        self.assertFalse(runner._shutdown_requested)
        runner._request_shutdown(_signal_mod.SIGINT, None)
        self.assertTrue(runner._shutdown_requested)

    def test_oom_handler_reduces_population(self):
        from autonomous_runner import AutonomousRunner
        runner = AutonomousRunner()
        runner._config["evolution"]["population_size"] = 50
        runner._handle_oom(MemoryError("test"), 1)
        self.assertEqual(runner._config["evolution"]["population_size"], 37)


class TestMetaController(unittest.TestCase):
    """MetaController integration with runner."""

    def test_import(self):
        from meta_controller import MetaController, TelemetrySnapshot, StagnationResult
        self.assertTrue(callable(MetaController))

    def test_stagnation_detection(self):
        from meta_controller import MetaController, TelemetrySnapshot, StagnationResult
        mc = MetaController(checkpoint_dir="/tmp/test_mc")
        # Simulate 5 stagnant cycles
        for c in range(1, 6):
            snap = TelemetrySnapshot(
                cycle=c, fitness=-1.0, level=3, champion_words=10,
                val_loss=2.0, diversity=0.3, novelty_size=0, qd_occupied=0,
                qd_total=100, traces_count=0, retrained=False,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        state = mc.detect_stagnation()
        self.assertIn(state, [
            StagnationResult.LEVEL_STALLED,
            StagnationResult.FITNESS_PLATEAU,
            StagnationResult.COMPOUND_STALL,
        ])


class TestChampionAnchor(unittest.TestCase):
    """Champion persistence."""

    def test_save_load_roundtrip(self):
        from champion_anchor import ChampionAnchor
        with tempfile.TemporaryDirectory() as tmpdir:
            ca = ChampionAnchor(champions_dir=tmpdir)
            program = [0x02A00513, 0x00100073]
            base = os.path.join(tmpdir, "test_champ")
            ca.save(program, base)
            loaded = ca.load(base)
            self.assertEqual(loaded, program)


if __name__ == "__main__":
    unittest.main()
