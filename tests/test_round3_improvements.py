#!/usr/bin/env python3
"""Tests for Round 3 improvements to the Evolution Engine.

Validates:
  - EvolutionConfig comprehensive validation (prevents dangerous misconfig)
  - EvolutionEngine lazy log file handle (avoids fd exhaustion)
  - ChampionAnchor atomic saves (crash-safe persistence)
  - CycleState.load() non-mutation of input dict
  - Pre-initialized challenger state attributes
  - Strategy archive deterministic decay (seeded RNG)
  - Strategy archive seeded RNG passthrough
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestEvolutionConfigValidation(unittest.TestCase):
    """EvolutionConfig rejects dangerous values at construction time."""

    def test_population_size_must_be_at_least_2(self):
        """A population of 1 can't do tournament selection."""
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError) as ctx:
            EvolutionConfig(population_size=1)
        self.assertIn("population_size", str(ctx.exception))

    def test_max_generations_must_be_positive(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(max_generations=0)

    def test_mutation_rate_out_of_range(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(mutation_rate=1.5)

    def test_crossover_rate_negative(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(crossover_rate=-0.1)

    def test_elite_ratio_too_high(self):
        """elite_ratio > 0.5 means more elite than population -- nonsense."""
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(elite_ratio=0.6)

    def test_tournament_size_must_be_at_least_2(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(tournament_size=1)

    def test_tournament_size_exceeds_population(self):
        """Can't select 10 from a population of 5."""
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError) as ctx:
            EvolutionConfig(population_size=5, tournament_size=10)
        self.assertIn("tournament_size", str(ctx.exception))
        self.assertIn("population_size", str(ctx.exception))

    def test_instruction_budget_minimum(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(instruction_budget=10)

    def test_eval_budget_minimum(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(eval_budget=50)

    def test_stall_threshold_minimum(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(stall_threshold=0)

    def test_retrain_lr_must_be_positive(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(retrain_lr=0.0)

    def test_retrain_freeze_layers_negative(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(retrain_freeze_layers=-1)

    def test_retrain_patience_minimum(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(retrain_patience=0)

    def test_retrain_timeout_minimum(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(retrain_timeout=10)

    def test_max_consecutive_same_strategy_minimum(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(max_consecutive_same_strategy=1)

    def test_circuit_breaker_stall_limit_minimum(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(circuit_breaker_stall_limit=2)

    def test_novelty_weight_negative(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(novelty_weight=-0.5)

    def test_qd_bonus_weight_negative(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(qd_bonus_weight=-1.0)

    def test_valid_config_passes(self):
        """Default config should always be valid."""
        from evolution_engine import EvolutionConfig

        cfg = EvolutionConfig()  # should not raise
        self.assertEqual(cfg.population_size, 50)

    def test_valid_custom_config(self):
        """A non-default but valid config should pass."""
        from evolution_engine import EvolutionConfig

        cfg = EvolutionConfig(
            population_size=100,
            elite_ratio=0.15,
            tournament_size=5,
        )
        self.assertEqual(cfg.population_size, 100)
        self.assertEqual(cfg.elite_ratio, 0.15)


class TestEvolutionEngineLogFileHandle(unittest.TestCase):
    """EvolutionEngine._log uses lazy file handles, not per-call open()."""

    def test_log_does_not_crash_without_log_dir(self):
        """Logging without a log dir should just use structured logger."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EvolutionConfig(log_dir="", max_generations=1)
            engine = EvolutionEngine(config=cfg)
            engine.current_generation = 0
            # Should not raise
            engine._log("test message")

    def test_log_creates_file_with_content(self):
        """Logging with a log dir should create the file and write content."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            cfg = EvolutionConfig(log_dir=log_dir, max_generations=1)
            engine = EvolutionEngine(config=cfg)
            engine.current_generation = 42

            engine._log("hello world")

            log_path = os.path.join(log_dir, "evolution.log")
            self.assertTrue(os.path.isfile(log_path))
            with open(log_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("hello world", content)
            self.assertIn("Gen   42", content)

            # Release the lazily-opened file handle before the
            # TemporaryDirectory context manager tears down tmpdir --
            # otherwise Windows raises PermissionError on cleanup.
            engine._close_log_handle()

    def test_log_reuses_file_handle(self):
        """Multiple log calls should not create new file handles each time."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "logs")
            cfg = EvolutionConfig(log_dir=log_dir, max_generations=1)
            engine = EvolutionEngine(config=cfg)
            engine.current_generation = 0

            engine._log("msg1")
            engine._log("msg2")
            engine._log("msg3")

            # The same file handle should be reused
            self.assertTrue(hasattr(engine, "_log_file"))
            self.assertIsNotNone(engine._log_file)

            # Release the lazily-opened file handle before the
            # TemporaryDirectory context manager tears down tmpdir --
            # otherwise Windows raises PermissionError on cleanup.
            engine._close_log_handle()


class TestChampionAnchorAtomicSave(unittest.TestCase):
    """ChampionAnchor.save() uses atomic writes to prevent corruption."""

    def test_save_load_roundtrip(self):
        """Basic save/load with integrity check."""
        from champion_anchor import ChampionAnchor

        with tempfile.TemporaryDirectory() as tmpdir:
            ca = ChampionAnchor(champions_dir=tmpdir)
            program = [0x00100073, 0x00128093, 0x00100073]
            base_path = os.path.join(tmpdir, "champion_test")
            ca.save(program, base_path)
            loaded = ca.load(base_path)
            self.assertEqual(loaded, program)

    def test_save_creates_temp_files_cleanup(self):
        """After save, no .tmp files should remain."""
        from champion_anchor import ChampionAnchor

        with tempfile.TemporaryDirectory() as tmpdir:
            ca = ChampionAnchor(champions_dir=tmpdir)
            program = [0x00100073]
            base_path = os.path.join(tmpdir, "champion_test")
            ca.save(program, base_path)

            # No .tmp files should remain
            tmp_files = [f for f in os.listdir(tmpdir) if f.endswith(".tmp")]
            self.assertEqual(len(tmp_files), 0)

    def test_save_overwrites_existing(self):
        """Overwriting an existing champion should work cleanly."""
        from champion_anchor import ChampionAnchor

        with tempfile.TemporaryDirectory() as tmpdir:
            ca = ChampionAnchor(champions_dir=tmpdir)
            base_path = os.path.join(tmpdir, "champion_test")
            ca.save([0x00100073], base_path)
            ca.save([0x00100073, 0x00128093], base_path)
            loaded = ca.load(base_path)
            self.assertEqual(len(loaded), 2)


class TestCycleStateNonMutation(unittest.TestCase):
    """CycleState.load() does not mutate the raw dict passed to it."""

    def test_load_does_not_pop_checksum(self):
        """The 'checksum' key should remain in the raw data after load."""
        from cycle_state import CycleState

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = os.path.join(tmpdir, "cycle_state.json")
            cs = CycleState(state_path)
            cs.update(current_cycle=5, best_fitness=42.0)
            cs.save()

            # Load the raw JSON manually
            with open(state_path, encoding="utf-8") as f:
                raw = json.load(f)
            self.assertIn("checksum", raw)

            # Load via CycleState -- should NOT mutate raw
            cs2 = CycleState(state_path)
            loaded = cs2.load()
            self.assertTrue(loaded)

            # raw dict should still have checksum
            self.assertIn("checksum", raw)
            self.assertEqual(raw["current_cycle"], 5)


class TestPreInitializedChallengerState(unittest.TestCase):
    """Challenger state attributes are initialized before run()."""

    def test_last_challengers_initialized(self):
        """_last_challengers should exist before _try_recurrent_feedback."""
        from evolution_engine import EvolutionConfig, EvolutionEngine

        cfg = EvolutionConfig(max_generations=1)
        engine = EvolutionEngine(config=cfg)

        # These should exist before any generation runs
        self.assertIsInstance(engine._last_challengers, list)
        self.assertIsInstance(engine._last_results, list)
        self.assertEqual(len(engine._last_challengers), 0)
        self.assertEqual(len(engine._last_results), 0)


class TestStrategyArchiveDeterministicDecay(unittest.TestCase):
    """Strategy archive decay uses seeded RNG for reproducibility."""

    def test_deterministic_decay_with_seeded_rng(self):
        """Same seed should produce same decay pattern."""
        import random

        from strategy_archive import StrategyArchive

        sa1 = StrategyArchive()
        sa2 = StrategyArchive()

        for i in range(20):
            sa1.record("GROW", level=5, success=True, generation=i)
            sa2.record("GROW", level=5, success=True, generation=i)

        rng1 = random.Random(42)
        rng2 = random.Random(42)

        modified1 = sa1.decay_level_records(5, factor=0.5, rng=rng1)
        modified2 = sa2.decay_level_records(5, factor=0.5, rng=rng2)

        self.assertEqual(modified1, modified2)
        # Same records should be modified
        for r1, r2 in zip(sa1._records, sa2._records):
            self.assertEqual(r1["success"], r2["success"])

    def test_decay_returns_zero_for_no_matching_level(self):
        """Decaying a level with no records should return 0."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        sa.record("GROW", level=3, success=True)
        modified = sa.decay_level_records(99, factor=0.5)
        self.assertEqual(modified, 0)

    def test_decay_preserves_non_matching_records(self):
        """Records at other levels should not be affected."""
        import random

        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        sa.record("GROW", level=5, success=True)
        sa.record("GROW", level=7, success=True)

        rng = random.Random(42)
        sa.decay_level_records(5, factor=0.0, rng=rng)  # factor=0 means flip ALL

        # Level 7 record should be untouched
        l7_records = [r for r in sa._records if r["level"] == 7]
        self.assertTrue(all(r["success"] for r in l7_records))


class TestEvolutionConfigSentinelSingleton(unittest.TestCase):
    """Circuit breaker sentinel is a proper singleton."""

    def test_sentinel_is_always_open(self):
        """The sentinel never reports breached."""
        from evolution_engine import EvolutionConfig

        sentinel = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        self.assertFalse(sentinel.is_breached())

    def test_sentinel_repr(self):
        from evolution_engine import EvolutionConfig

        sentinel = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        r = repr(sentinel)
        self.assertIn("not configured", r)

    def test_sentinel_singleton_identity(self):
        """Two references should be the same object."""
        from evolution_engine import EvolutionConfig

        s1 = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        s2 = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        self.assertIs(s1, s2)


class TestQDAxesComputation(unittest.TestCase):
    """compute_qd_axes handles edge cases."""

    def test_none_program(self):
        """None program should return default axes."""
        from evolution_engine import compute_qd_axes

        axes = compute_qd_axes(None, None)
        self.assertEqual(axes, [-1.0, -1.0])

    def test_empty_program(self):
        from evolution_engine import compute_qd_axes

        axes = compute_qd_axes([], None)
        self.assertEqual(axes, [-1.0, -1.0])

    def test_full_diversity(self):
        """A program using all 11 opcode categories should score 1.0 on inst_div."""
        from evolution_engine import compute_qd_axes
        from mutation_proposer import rv32i_encode

        # Generate one instruction from each opcode category
        opcodes = [
            ("LUI", {}),
            ("AUIPC", {}),
            ("JAL", {}),
            ("JALR", {}),
            ("BEQ", {}),
            ("LB", {}),
            ("SB", {}),
            ("ADDI", {}),
            ("ADD", {}),
            ("FENCE", {}),
            ("EBREAK", {}),
        ]
        program = []
        for name, _ in opcodes:
            try:
                if name == "FENCE":
                    word = 0x0000000F  # FENCE encoding
                elif name == "EBREAK":
                    word = 0x00100073
                else:
                    word = rv32i_encode(name=name, rd=0, rs1=0, rs2=0, imm=0)
                program.append(word)
            except Exception:
                pass

        trace = {"final_regs": {i: i for i in range(1, 32)}}
        axes = compute_qd_axes(program, trace)
        # With all 11 categories, inst_div should be close to 1.0
        self.assertGreater(axes[0], 0.8)
        # With all 31 registers active, reg_press should be close to 1.0
        self.assertGreater(axes[1], 0.8)

    def test_minimal_program(self):
        """A single EBREAK should give minimum diversity."""
        from evolution_engine import compute_qd_axes

        axes = compute_qd_axes([0x00100073], {"final_regs": {}})
        # Only SYSTEM category, 0 active registers
        self.assertLess(axes[0], 0.0)
        self.assertLess(axes[1], 0.0)


class TestConfigValidationRanges(unittest.TestCase):
    """Config validation catches edge cases in float ranges."""

    def test_retrain_val_split_boundary(self):
        from evolution_engine import EvolutionConfig

        # Exactly 0.0 and 1.0 are valid
        cfg1 = EvolutionConfig(retrain_val_split=0.0)
        self.assertEqual(cfg1.retrain_val_split, 0.0)
        cfg2 = EvolutionConfig(retrain_val_split=1.0)
        self.assertEqual(cfg2.retrain_val_split, 1.0)
        # Negative and > 1.0 are invalid
        with self.assertRaises(ValueError):
            EvolutionConfig(retrain_val_split=-0.1)
        with self.assertRaises(ValueError):
            EvolutionConfig(retrain_val_split=1.1)

    def test_retrain_weight_decay_boundary(self):
        from evolution_engine import EvolutionConfig

        # 0.0 is valid, 1.0 is not (would zero all weights)
        cfg = EvolutionConfig(retrain_weight_decay=0.0)
        self.assertEqual(cfg.retrain_weight_decay, 0.0)
        with self.assertRaises(ValueError):
            EvolutionConfig(retrain_weight_decay=1.0)

    def test_max_level_positive(self):
        from evolution_engine import EvolutionConfig

        with self.assertRaises(ValueError):
            EvolutionConfig(max_level=0)


class TestMetaEvolutionSeedValidation(unittest.TestCase):
    """MetaEvolutionEngine validates seed configs."""

    def test_valid_seeds_accepted(self):
        """Default seeds should all pass validation."""
        from meta_evolution import MetaEvolutionConfig, MetaEvolutionEngine

        config = MetaEvolutionConfig(
            meta_population_size=4,
            meta_generations=1,
            base_generations=5,
        )
        # Should not raise
        engine = MetaEvolutionEngine(meta_config=config)
        self.assertEqual(len(engine.population), 4)


if __name__ == "__main__":
    unittest.main()
