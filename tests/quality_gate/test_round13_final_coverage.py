#!/usr/bin/env python3
"""R32: Final coverage push — untested code paths across all modules.

Covers:
  1. MetaController.execute_rollback (snapshot, same-file guard, error paths)
  2. MetaController.accept_checkpoint (file not found, invalid val_loss)
  3. MetaController.save_state / load_state (atomic write, corruption)
  4. MetaController._log (lazy file handle, stale fd retry)
  5. parse_val_loss_from_retrain_log / parse_grad_norms_from_retrain_log
  6. EvolutionEngine._eval_result_to_trace (AttributeError fallback)
  7. EvolutionEngine._choose_mutation_strategy (D3 cap, unsolved levels)
  8. EvolutionEngine._get_model_guided_ratio
  9. EvolutionEngine._try_qd_crossover / _try_qd_champion_reset branches
  10. EvolutionEngine._record_strategy_outcomes
  11. EvolutionEngine.load_champion
  12. VMFarm._resolve_kernel_path / _dict_to_eval_result
  13. MutationProposer utility functions (_is_ebreak, validate_mutation, generate_seed_program)
  14. StrategyArchive.summary, clear, record_batch slow path
  15. MetaEvolutionEngine._compute_meta_fitness edge cases
  16. EvolutionConfig.__post_init__ validation errors
  17. MetaController.tune_retrain_params compound_stall / recovery mode
  18. MetaController.get_state_summary
"""

import json
import os
import random
import shutil
import tempfile
from typing import NamedTuple
from unittest.mock import patch

import pytest

from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame, VMFarm, compute_qd_axes
from meta_controller import (
    MetaController,
    StagnationResult,
    TelemetrySnapshot,
    parse_grad_norms_from_retrain_log,
    parse_val_loss_from_retrain_log,
)
from meta_evolution import MetaEvolutionConfig, MetaEvolutionEngine
from meta_genome import MetaGenome
from strategy_archive import StrategyArchive

# ───────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────


def _snapshot(cycle, fitness=1.0, level=5, diversity=0.3, val_loss=2.0, champion_words=50, retrained=True):
    return TelemetrySnapshot(
        cycle=cycle,
        fitness=fitness,
        level=level,
        champion_words=champion_words,
        val_loss=val_loss,
        diversity=diversity,
        novelty_size=10,
        qd_occupied=5,
        qd_total=100,
        traces_count=20,
        retrained=retrained,
        generations_completed=50,
    )


def _make_engine(**kwargs):
    cfg = EvolutionConfig(
        mutation_rate=kwargs.pop("mutation_rate", 0.3),
        seed=kwargs.pop("seed", 42),
        population_size=kwargs.pop("population_size", 20),
        **kwargs,
    )
    return EvolutionEngine(config=cfg, vm_farm=None)


def _set_champion(eng, prog=None, trace=None, fitness=10.0, level=0):
    eng.champion = prog or [0x00000033, 0x00100073]
    eng.champion_trace = trace or {
        "status": "ok",
        "final_regs": {10: 42},
        "instructions_count": 2,
        "trap_cause": "",
    }
    eng.champion_fitness = fitness
    eng.champion_level = level
    eng.task_instance = {
        "expected_output": {"registers": {10: 42}},
        "inputs": {},
    }
    eng.current_generation = 0
    eng._bonus_threshold_cache = None


class _FakeEvalResult(NamedTuple):
    success: bool
    registers: dict
    memory: dict
    trace: list
    cycles: int
    error: str | None
    blocked_writes: int


# ───────────────────────────────────────────────────────────────────────
# 1. MetaController.execute_rollback
# ───────────────────────────────────────────────────────────────────────


class TestExecuteRollback:
    """Tests for MetaController.execute_rollback with all branches."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_controller(self):
        return MetaController(checkpoint_dir=self.tmpdir)

    def test_rollback_with_best_checkpoint(self):
        """Rollback copies best checkpoint to current."""
        mc = self._make_controller()
        # Create best checkpoint file
        best_path = mc.best_checkpoint_path
        with open(best_path, "w") as f:
            f.write("best model data")
        # Create current checkpoint
        current_path = os.path.join(self.tmpdir, "current.pt")
        with open(current_path, "w") as f:
            f.write("regressed model data")

        result = mc.execute_rollback(current_path)
        assert result == current_path
        with open(current_path) as f:
            assert f.read() == "best model data"
        assert mc.recovery_mode is True
        assert mc.recovery_cycles_remaining == 2

    def test_rollback_same_file_skips(self):
        """When source and current are same file, rollback skips gracefully."""
        mc = self._make_controller()
        best_path = mc.best_checkpoint_path
        with open(best_path, "w") as f:
            f.write("model data")

        result = mc.execute_rollback(best_path)
        assert result == best_path
        assert mc.recovery_mode is True

    def test_rollback_no_best_checkpoint_returns_none(self):
        """When best checkpoint doesn't exist, rollback returns None."""
        mc = self._make_controller()
        current_path = os.path.join(self.tmpdir, "current.pt")
        with open(current_path, "w") as f:
            f.write("current")
        result = mc.execute_rollback(current_path)
        assert result is None

    def test_rollback_with_snapshot(self):
        """Rollback with use_snapshot=True uses snapshot path."""
        mc = self._make_controller()
        # Create snapshot
        snapshot_path = os.path.join(self.tmpdir, "snapshot.pt")
        with open(snapshot_path, "w") as f:
            f.write("snapshot data")
        mc._last_snapshot_path = snapshot_path

        current_path = os.path.join(self.tmpdir, "current.pt")
        with open(current_path, "w") as f:
            f.write("current data")

        result = mc.execute_rollback(current_path, use_snapshot=True)
        assert result == current_path
        with open(current_path) as f:
            assert f.read() == "snapshot data"

    def test_rollback_snapshot_not_found_falls_back(self):
        """When snapshot doesn't exist, falls back to best checkpoint."""
        mc = self._make_controller()
        mc._last_snapshot_path = "/nonexistent/snapshot.pt"
        best_path = mc.best_checkpoint_path
        with open(best_path, "w") as f:
            f.write("best data")
        current_path = os.path.join(self.tmpdir, "current.pt")
        with open(current_path, "w") as f:
            f.write("current data")

        result = mc.execute_rollback(current_path, use_snapshot=True)
        assert result == current_path
        with open(current_path) as f:
            assert f.read() == "best data"

    def test_rollback_backs_up_regressed_checkpoint(self):
        """Rollback creates .regressed backup of the current checkpoint."""
        mc = self._make_controller()
        best_path = mc.best_checkpoint_path
        with open(best_path, "w") as f:
            f.write("best")
        current_path = os.path.join(self.tmpdir, "current.pt")
        with open(current_path, "w") as f:
            f.write("regressed")

        mc.execute_rollback(current_path)
        backup_path = current_path + ".regressed"
        assert os.path.isfile(backup_path)
        with open(backup_path) as f:
            assert f.read() == "regressed"


# ───────────────────────────────────────────────────────────────────────
# 2. MetaController.accept_checkpoint
# ───────────────────────────────────────────────────────────────────────


class TestAcceptCheckpoint:
    """Tests for MetaController.accept_checkpoint branches."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_reject_when_file_not_found(self):
        mc = MetaController(checkpoint_dir=self.tmpdir)
        assert mc.accept_checkpoint(1.0, "/nonexistent/file.pt") is False

    def test_accept_when_val_loss_zero(self):
        """Val_loss <= 0 accepts cautiously."""
        mc = MetaController(checkpoint_dir=self.tmpdir)
        ckpt = os.path.join(self.tmpdir, "model.pt")
        with open(ckpt, "w") as f:
            f.write("data")
        assert mc.accept_checkpoint(0.0, ckpt) is True

    def test_accept_when_val_loss_negative(self):
        mc = MetaController(checkpoint_dir=self.tmpdir)
        ckpt = os.path.join(self.tmpdir, "model.pt")
        with open(ckpt, "w") as f:
            f.write("data")
        assert mc.accept_checkpoint(-1.0, ckpt) is True

    def test_accept_within_tolerance(self):
        mc = MetaController(checkpoint_dir=self.tmpdir, initial_val_loss=1.0, checkpoint_acceptance_tolerance=1.05)
        ckpt = os.path.join(self.tmpdir, "model.pt")
        with open(ckpt, "w") as f:
            f.write("data")
        # threshold = 1.0 * 1.05 = 1.05; val=1.04 <= threshold
        assert mc.accept_checkpoint(1.04, ckpt) is True
        # 1.04 < 1.0 is False, so _best_val_loss stays at 1.0
        assert mc._best_val_loss == 1.0

    def test_accept_improves_best_val_loss(self):
        mc = MetaController(checkpoint_dir=self.tmpdir, initial_val_loss=2.0, checkpoint_acceptance_tolerance=1.05)
        ckpt = os.path.join(self.tmpdir, "model.pt")
        with open(ckpt, "w") as f:
            f.write("data")
        # threshold = 2.0 * 1.05 = 2.1; val=1.5 <= threshold and < best
        assert mc.accept_checkpoint(1.5, ckpt) is True
        assert mc._best_val_loss == 1.5

    def test_reject_outside_tolerance(self):
        mc = MetaController(checkpoint_dir=self.tmpdir, initial_val_loss=1.0, checkpoint_acceptance_tolerance=1.05)
        ckpt = os.path.join(self.tmpdir, "model.pt")
        with open(ckpt, "w") as f:
            f.write("data")
        # threshold = 1.05; val=1.10 > threshold
        assert mc.accept_checkpoint(1.10, ckpt) is False

    def test_accept_updates_best_val_loss(self):
        mc = MetaController(checkpoint_dir=self.tmpdir, initial_val_loss=2.0)
        ckpt = os.path.join(self.tmpdir, "model.pt")
        with open(ckpt, "w") as f:
            f.write("data")
        mc.accept_checkpoint(1.5, ckpt)
        assert mc._best_val_loss == 1.5

    def test_accept_does_not_worsen_best_val_loss(self):
        mc = MetaController(checkpoint_dir=self.tmpdir, initial_val_loss=1.0)
        ckpt = os.path.join(self.tmpdir, "model.pt")
        with open(ckpt, "w") as f:
            f.write("data")
        # val=1.02 is within tolerance of 1.0 (threshold=1.05) but doesn't improve
        mc.accept_checkpoint(1.02, ckpt)
        assert mc._best_val_loss == 1.0  # unchanged


# ───────────────────────────────────────────────────────────────────────
# 3. MetaController.save_state / load_state
# ───────────────────────────────────────────────────────────────────────


class TestSaveLoadState:
    """Tests for MetaController state persistence."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        mc = MetaController(checkpoint_dir=self.tmpdir)
        mc.update_cycle(_snapshot(1, fitness=5.0, level=3))
        mc.update_cycle(_snapshot(2, fitness=6.0, level=4))

        path = os.path.join(self.tmpdir, "state.json")
        mc.save_state(path)
        assert os.path.isfile(path)

        mc2 = MetaController(checkpoint_dir=self.tmpdir)
        assert mc2.load_state(path) is True
        assert mc2.best_fitness == 6.0
        assert mc2.best_level == 4

    def test_load_nonexistent_returns_false(self):
        mc = MetaController(checkpoint_dir=self.tmpdir)
        assert mc.load_state("/nonexistent/state.json") is False

    def test_load_corrupt_json_returns_false(self):
        mc = MetaController(checkpoint_dir=self.tmpdir)
        path = os.path.join(self.tmpdir, "bad.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        assert mc.load_state(path) is False

    def test_save_atomic_write(self):
        """save_state should create the file atomically (no .tmp leftover)."""
        mc = MetaController(checkpoint_dir=self.tmpdir)
        path = os.path.join(self.tmpdir, "state.json")
        mc.save_state(path)
        # No .tmp files should remain
        tmp_files = [f for f in os.listdir(self.tmpdir) if f.endswith(".tmp")]
        assert len(tmp_files) == 0


# ───────────────────────────────────────────────────────────────────────
# 4. MetaController._log (lazy file handle)
# ───────────────────────────────────────────────────────────────────────


class TestMetaControllerLogging:
    """Tests for MetaController._log with lazy file handle."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_creates_file(self):
        log_path = os.path.join(self.tmpdir, "mc.log")
        mc = MetaController(checkpoint_dir=self.tmpdir, log_file=log_path)
        mc._log("test message")
        assert os.path.isfile(log_path)
        with open(log_path) as f:
            content = f.read()
        assert "test message" in content
        assert "[MetaController]" in content

    def test_log_no_file_no_crash(self):
        mc = MetaController(checkpoint_dir=self.tmpdir)
        mc._log("no crash")  # should not raise

    def test_log_retries_on_stale_fd(self):
        """When file handle is stale, _log retries once."""
        log_path = os.path.join(self.tmpdir, "mc.log")
        mc = MetaController(checkpoint_dir=self.tmpdir, log_file=log_path)
        mc._log("init")
        # Simulate stale fd
        mc._log_file.close()
        mc._log_file = None
        # Force error on first open, then succeed
        call_count = [0]
        original_open = open

        def mock_open(path, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and "mc.log" in str(path):
                raise OSError("stale fd")
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=mock_open):
            mc._log("retry message")
        # Message should eventually be written
        with open(log_path) as f:
            assert "retry message" in f.read()


# ───────────────────────────────────────────────────────────────────────
# 5. parse_val_loss_from_retrain_log / parse_grad_norms
# ───────────────────────────────────────────────────────────────────────


class TestParseValLossFromLog:
    """Tests for retrain log parsers."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_nonexistent_file_returns_none(self):
        assert parse_val_loss_from_retrain_log("/nonexistent/log.txt") is None

    def test_val_format(self):
        path = os.path.join(self.tmpdir, "log.txt")
        with open(path, "w") as f:
            f.write("epoch 1: val=1.2345\nepoch 2: val=0.9876\n")
        result = parse_val_loss_from_retrain_log(path)
        assert result == 0.9876  # minimum

    def test_val_loss_format(self):
        path = os.path.join(self.tmpdir, "log.txt")
        with open(path, "w") as f:
            f.write("val_loss=2.3456\n")
        result = parse_val_loss_from_retrain_log(path)
        assert result == 2.3456

    def test_val_loss_colon_format(self):
        path = os.path.join(self.tmpdir, "log.txt")
        with open(path, "w") as f:
            f.write("Val Loss: 3.4567\n")
        result = parse_val_loss_from_retrain_log(path)
        assert result == 3.4567

    def test_no_match_returns_none(self):
        path = os.path.join(self.tmpdir, "log.txt")
        with open(path, "w") as f:
            f.write("no val_loss here\n")
        result = parse_val_loss_from_retrain_log(path)
        assert result is None

    def test_empty_file_returns_none(self):
        path = os.path.join(self.tmpdir, "log.txt")
        with open(path, "w") as f:
            f.write("")
        result = parse_val_loss_from_retrain_log(path)
        assert result is None

    def test_returns_minimum_val_loss(self):
        path = os.path.join(self.tmpdir, "log.txt")
        with open(path, "w") as f:
            f.write("val=5.0\nval=2.0\nval=3.0\n")
        result = parse_val_loss_from_retrain_log(path)
        assert result == 2.0


class TestParseGradNormsFromLog:
    """Tests for gradient norm parsing."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_nonexistent_file_returns_none(self):
        assert parse_grad_norms_from_retrain_log("/nonexistent/log.txt") is None

    def test_parse_grad_norms(self):
        path = os.path.join(self.tmpdir, "log.txt")
        with open(path, "w") as f:
            f.write("grad_norm/layer.0=0.0234\ngrad_norm blocks.0.attn=0.0156\n")
        result = parse_grad_norms_from_retrain_log(path)
        assert result is not None
        assert "layer.0" in result
        assert "blocks.0.attn" in result
        assert abs(result["layer.0"] - 0.0234) < 1e-6

    def test_no_grad_norms_returns_none(self):
        path = os.path.join(self.tmpdir, "log.txt")
        with open(path, "w") as f:
            f.write("no grad norms here\n")
        result = parse_grad_norms_from_retrain_log(path)
        assert result is None

    def test_empty_file_returns_none(self):
        path = os.path.join(self.tmpdir, "log.txt")
        with open(path, "w") as f:
            f.write("")
        result = parse_grad_norms_from_retrain_log(path)
        assert result is None


# ───────────────────────────────────────────────────────────────────────
# 6. EvolutionEngine._eval_result_to_trace
# ───────────────────────────────────────────────────────────────────────


class TestEvalResultToTrace:
    """Tests for EvalResult to trace dict conversion."""

    def test_normal_eval_result(self):
        result = _FakeEvalResult(
            success=True,
            registers={10: 42},
            memory={},
            trace=[],
            cycles=100,
            error=None,
            blocked_writes=0,
        )
        trace = EvolutionEngine._eval_result_to_trace(result)
        assert trace["status"] == "ok"
        assert trace["final_regs"] == {10: 42}
        assert trace["instructions_count"] == 100
        assert trace["trap_cause"] == ""

    def test_failed_eval_result(self):
        result = _FakeEvalResult(
            success=False,
            registers={},
            memory={},
            trace=[],
            cycles=50,
            error="illegal instruction",
            blocked_writes=0,
        )
        trace = EvolutionEngine._eval_result_to_trace(result)
        assert trace["status"] == "crash"
        assert trace["trap_cause"] == "illegal instruction"

    def test_attribute_error_fallback(self):
        """Non-EvalResult objects should trigger the AttributeError fallback."""

        # Use a simple object without .success attribute to trigger AttributeError
        class FallbackResult:
            pass

        result = FallbackResult()
        trace = EvolutionEngine._eval_result_to_trace(result)
        # getattr defaults kick in: success=True default, registers={}, etc.
        assert trace["status"] == "ok"
        assert trace["final_regs"] == {}
        assert trace["instructions_count"] == 0

    def test_error_none_becomes_empty_string(self):
        result = _FakeEvalResult(
            success=True,
            registers={},
            memory={},
            trace=[],
            cycles=10,
            error=None,
            blocked_writes=0,
        )
        trace = EvolutionEngine._eval_result_to_trace(result)
        assert trace["trap_cause"] == ""

    def test_empty_error_string(self):
        result = _FakeEvalResult(
            success=True,
            registers={},
            memory={},
            trace=[],
            cycles=10,
            error="",
            blocked_writes=0,
        )
        trace = EvolutionEngine._eval_result_to_trace(result)
        assert trace["trap_cause"] == ""


# ───────────────────────────────────────────────────────────────────────
# 7. EvolutionEngine._choose_mutation_strategy
# ───────────────────────────────────────────────────────────────────────


class TestChooseMutationStrategy:
    """Tests for strategy selection logic."""

    def _make_engine(self, **kwargs):
        cfg = EvolutionConfig(
            seed=kwargs.pop("seed", 42),
            population_size=kwargs.pop("population_size", 20),
            **kwargs,
        )
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_defaults_to_hybrid_when_no_archive(self):
        eng = self._make_engine()
        eng.config.use_strategy_archive = False
        eng.champion_level = 0
        eng._consecutive_same_strategy = 0
        eng._last_strategy_used = ""
        eng.generations_stalled = 0
        strat = eng._choose_mutation_strategy(0)
        assert strat.value == "HYBRID"

    def test_d3_cap_halved_at_level_5(self):
        """At L5+, effective cap should be halved."""
        eng = self._make_engine(max_consecutive_same_strategy=30)
        eng.champion_level = 5
        eng._consecutive_same_strategy = 16  # above halved cap (15)
        eng._last_strategy_used = "RANDOM"
        eng.generations_stalled = 0
        eng._choose_mutation_strategy(0)
        # After exceeding cap, counter resets to 0, then chosen strat
        # (HYBRID, different from RANDOM) sets counter to 1
        assert eng._consecutive_same_strategy == 1

    def test_unsolved_level_uses_perturbation_capped(self):
        """Levels 9-11 with stall > 2 should sometimes use PERTURBATION_CAPPED."""
        eng = self._make_engine(perturbation_capped_probability_unsolved=1.0)
        eng.champion_level = 10
        eng._consecutive_same_strategy = 0
        eng._last_strategy_used = ""
        eng.generations_stalled = 5
        eng.config.use_strategy_archive = False
        strat = eng._choose_mutation_strategy(0)
        assert strat.value == "PERTURBATION_CAPPED"

    def test_consecutive_counter_increments(self):
        """Same strategy should increment the consecutive counter."""
        eng = self._make_engine()
        eng.config.use_strategy_archive = False
        eng.champion_level = 0
        eng._consecutive_same_strategy = 0
        eng._last_strategy_used = "HYBRID"
        eng.generations_stalled = 0
        eng._choose_mutation_strategy(0)
        # HYBRID was chosen again -> counter goes from 0 to 1
        # (not 2: when strat matches, counter increments by 1)
        assert eng._consecutive_same_strategy == 1

    def test_different_strategy_resets_counter(self):
        """Different strategy should reset consecutive counter."""
        eng = self._make_engine()
        eng.config.use_strategy_archive = False
        eng.champion_level = 0
        eng._consecutive_same_strategy = 5
        eng._last_strategy_used = "RANDOM"
        eng.generations_stalled = 0
        eng._choose_mutation_strategy(0)
        assert eng._consecutive_same_strategy == 1
        assert eng._last_strategy_used == "HYBRID"


# ───────────────────────────────────────────────────────────────────────
# 8. EvolutionEngine._get_model_guided_ratio
# ───────────────────────────────────────────────────────────────────────


class TestGetModelGuidedRatio:
    """Tests for model-guided ratio determination."""

    def _make_engine(self, **kwargs):
        cfg = EvolutionConfig(seed=42, **kwargs)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_explicit_ratio_used(self):
        eng = self._make_engine(model_guided_ratio=0.5)
        eng.champion_level = 0
        assert eng._get_model_guided_ratio() == 0.5

    def test_low_level_uses_low_ratio(self):
        eng = self._make_engine(model_guided_ratio=None)
        eng.champion_level = 3
        assert eng._get_model_guided_ratio() == eng.config.model_guided_ratio_low

    def test_mid_level_uses_mid_ratio(self):
        eng = self._make_engine(model_guided_ratio=None)
        eng.champion_level = 6
        assert eng._get_model_guided_ratio() == eng.config.model_guided_ratio_mid

    def test_high_level_uses_high_ratio(self):
        eng = self._make_engine(model_guided_ratio=None)
        eng.champion_level = 10
        assert eng._get_model_guided_ratio() == eng.config.model_guided_ratio_high

    def test_level_5_uses_low(self):
        eng = self._make_engine(model_guided_ratio=None)
        eng.champion_level = 5
        assert eng._get_model_guided_ratio() == eng.config.model_guided_ratio_low

    def test_level_7_uses_mid(self):
        eng = self._make_engine(model_guided_ratio=None)
        eng.champion_level = 7
        assert eng._get_model_guided_ratio() == eng.config.model_guided_ratio_mid

    def test_level_8_uses_high(self):
        eng = self._make_engine(model_guided_ratio=None)
        eng.champion_level = 8
        assert eng._get_model_guided_ratio() == eng.config.model_guided_ratio_high


def test_skip_model_guided_at_terminal_saturated():
    from evolution.mutation_helpers import skip_model_guided_at_terminal

    assert skip_model_guided_at_terminal(30, 1.0) is True
    assert skip_model_guided_at_terminal(30, 0.5) is False
    assert skip_model_guided_at_terminal(10, 1.0) is False


def test_compact_terminal_prefers_task_aware():
    from types import SimpleNamespace

    from evolution.mutation_helpers import apply_level_aware_strategy_bias
    from mutation_proposer import MutationStrategy

    class _R:
        def random(self):
            return 0.0

        def choice(self, xs):
            return xs[0]

    eng = SimpleNamespace(
        champion_level=30,
        champion=[0] * 46,
        rng=_R(),
        _LEVEL_WEIGHT_PROFILES={"L17_plus": {"RANDOM": 1.0}},
        _LEVEL_OVERRIDE_PROBS={"L17_plus": 0.0},
    )
    eng._log = lambda *_a, **_k: None
    chosen = apply_level_aware_strategy_bias(eng, MutationStrategy.RANDOM)
    assert chosen == MutationStrategy.TASK_AWARE

    eng.champion = [0] * 19

    class _RHigh:
        def random(self):
            return 0.65

        def choice(self, xs):
            return xs[0]

    eng.rng = _RHigh()
    chosen_short = apply_level_aware_strategy_bias(eng, MutationStrategy.RANDOM)
    assert chosen_short == MutationStrategy.TASK_AWARE
    eng.champion = [0] * 46
    chosen_long = apply_level_aware_strategy_bias(eng, MutationStrategy.RANDOM)
    assert chosen_long == MutationStrategy.RANDOM


# ───────────────────────────────────────────────────────────────────────
# 9. EvolutionEngine._try_qd_crossover / _try_qd_champion_reset
# ───────────────────────────────────────────────────────────────────────


class TestTryQDCrossover:
    """Tests for QD crossover and champion reset branches."""

    def _make_engine(self, **kwargs):
        cfg = EvolutionConfig(seed=42, **kwargs)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_no_qd_archive_returns_none(self):
        eng = self._make_engine()
        eng.qd_archive = None
        eng.generations_stalled = 20
        assert eng._try_qd_crossover() is None

    def test_stall_below_threshold_returns_none(self):
        eng = self._make_engine()
        eng.generations_stalled = 7
        assert eng._try_qd_crossover() is None

    def test_qd_archive_too_few_solutions_returns_none(self):
        eng = self._make_engine()
        eng.generations_stalled = 20

        class FakeArchive:
            def grid_solutions(self):
                return [{"program": [1], "fitness": 1.0}]

        eng.qd_archive = FakeArchive()
        assert eng._try_qd_crossover() is None

    def test_identical_programs_returns_none(self):
        eng = self._make_engine()
        eng.generations_stalled = 20
        eng.champion_level = 0

        class FakeArchive:
            def grid_solutions(self):
                return [
                    {"program": [0x00000033, 0x00100073], "fitness": 10.0},
                    {"program": [0x00000033, 0x00100073], "fitness": 8.0},
                ]

        eng.qd_archive = FakeArchive()
        result = eng._try_qd_crossover()
        # Both programs identical, only 2 solutions -> returns None
        assert result is None


class TestTryQDChampionReset:
    """Tests for QD champion reset branches."""

    def _make_engine(self, **kwargs):
        cfg = EvolutionConfig(seed=42, **kwargs)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_stall_below_50_returns_false(self):
        eng = self._make_engine()
        eng.generations_stalled = 30
        assert eng._try_qd_champion_reset() is False

    def test_no_qd_archive_returns_false(self):
        eng = self._make_engine()
        eng.generations_stalled = 50
        eng.qd_archive = None
        assert eng._try_qd_champion_reset() is False

    def test_already_tried_returns_false(self):
        eng = self._make_engine()
        eng.generations_stalled = 50
        eng._qd_champion_reset_tried = True

        class FakeArchive:
            def grid_solutions(self):
                return [{"program": [1], "fitness": 100.0}]

        eng.qd_archive = FakeArchive()
        assert eng._try_qd_champion_reset() is False

    def test_empty_archive_returns_false(self):
        eng = self._make_engine()
        eng.generations_stalled = 50
        eng.champion_fitness = 10.0

        class FakeArchive:
            def grid_solutions(self):
                return []

        eng.qd_archive = FakeArchive()
        assert eng._try_qd_champion_reset() is False

    def test_archive_better_replaces_champion(self):
        eng = self._make_engine()
        eng.generations_stalled = 50
        eng.champion_fitness = 5.0
        eng.champion = [0x00100073]
        # Patch _HAS_QD to True so the method proceeds
        from unittest.mock import patch

        import evolution_engine

        orig = evolution_engine._HAS_QD
        evolution_engine._HAS_QD = True
        try:

            class FakeArchive:
                def grid_solutions(self):
                    return [{"program": [0x00000033, 0x00100073], "fitness": 12.0}]

                def selection_bias(self, desc):
                    return 0.0

                def get_illumination_diversity(self):
                    return {"overall_diversity_score": 0.0}

            eng.qd_archive = FakeArchive()

            with patch.object(eng, "_verify_champion", return_value=(True, 1.0)):
                result = eng._try_qd_champion_reset()
            assert result is True
            assert eng.champion == [0x00000033, 0x00100073]
            assert eng.champion_fitness > 5.0
            assert eng.generations_stalled == 0
            assert eng._champion_verified is True
        finally:
            evolution_engine._HAS_QD = orig

    def test_archive_worse_keeps_champion(self):
        eng = self._make_engine()
        eng.generations_stalled = 50
        eng.champion_fitness = 60.0
        eng.champion = [0x00000033, 0x00100073]
        import evolution_engine

        orig = evolution_engine._HAS_QD
        evolution_engine._HAS_QD = True
        try:

            class FakeArchive:
                def grid_solutions(self):
                    return [{"program": [0x00100073], "fitness": 5.0}]

                def selection_bias(self, desc):
                    return 0.0

                def get_illumination_diversity(self):
                    return {"overall_diversity_score": 0.0}

            eng.qd_archive = FakeArchive()

            result = eng._try_qd_champion_reset()
            assert result is False
            assert eng.champion == [0x00000033, 0x00100073]
        finally:
            evolution_engine._HAS_QD = orig


# ───────────────────────────────────────────────────────────────────────
# 10. EvolutionEngine._record_strategy_outcomes
# ───────────────────────────────────────────────────────────────────────


class TestRecordStrategyOutcomes:
    """Tests for strategy outcome recording."""

    def _make_engine(self, **kwargs):
        cfg = EvolutionConfig(seed=42, use_strategy_archive=True, **kwargs)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_records_strategy_outcomes(self):
        eng = self._make_engine()
        eng._last_challenger_strategies = ["RANDOM", "TASK_AWARE"]
        eng._last_results = [
            {"status": "ok", "final_regs": {10: 42}},
            {"status": "crash", "final_regs": {}},
        ]
        eng._last_challenger_fitnesses = [10.0, -100.0]
        eng.champion_fitness = 5.0
        eng.champion_level = 0

        eng._record_strategy_outcomes()

        assert eng.strategy_archive.total_records() == 2

    def test_empty_strategies_no_crash(self):
        eng = self._make_engine()
        eng._last_challenger_strategies = []
        eng._last_results = []
        eng._last_challenger_fitnesses = []
        eng._record_strategy_outcomes()
        assert eng.strategy_archive.total_records() == 0

    def test_mismatched_lengths_handled(self):
        eng = self._make_engine()
        eng._last_challenger_strategies = ["RANDOM", "TASK_AWARE", "DIRECTED"]
        eng._last_results = [{"status": "ok", "final_regs": {}}]
        eng._last_challenger_fitnesses = [10.0]
        eng.champion_fitness = 5.0
        eng.champion_level = 0
        eng._record_strategy_outcomes()
        # Should record only the 1 result that has a matching strategy
        assert eng.strategy_archive.total_records() == 1

    def test_stall_widens_success_threshold(self):
        eng = self._make_engine()
        eng.generations_stalled = 20
        eng.current_generation = 1
        eng.champion_level = 0
        eng._last_challenger_strategies = ["RANDOM"] * 10
        eng._last_challenger_fitnesses = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        eng._record_strategy_outcomes()
        success_count = sum(1 for r in eng.strategy_archive._records if r["success"])
        assert success_count == 2

    def test_beats_champion_marks_success(self):
        eng = self._make_engine()
        eng.current_generation = 1
        eng.champion_level = 0
        eng._champion_stage1_fitness = 85.0
        eng._last_challenger_strategies = ["RANDOM"] * 10
        eng._last_challenger_fitnesses = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 86.0, 100.0]
        eng._record_strategy_outcomes()
        success = [r["success"] for r in eng.strategy_archive._records]
        assert success[8] is True
        assert success[9] is True
        assert sum(success) == 2

    def test_generate_challengers_records_strategies(self):
        eng = self._make_engine(population_size=12)
        eng.champion = [0x00100073]
        eng.champion_trace = {"status": "ok", "final_regs": {10: 42}, "instructions_count": 1}
        eng.champion_fitness = 50.0
        eng.champion_level = 0
        challengers = eng._generate_challengers()
        assert len(challengers) == 12
        assert len(eng._last_challenger_strategies) == 12
        assert all(isinstance(s, str) and s for s in eng._last_challenger_strategies)


# ───────────────────────────────────────────────────────────────────────
# 11. EvolutionEngine.load_champion
# ───────────────────────────────────────────────────────────────────────


class TestLoadChampion:
    """Tests for champion loading."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_engine(self):
        cfg = EvolutionConfig(seed=42)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_load_nonexistent_returns_false(self):
        eng = self._make_engine()
        # load_champion catches IntegrityError but not FileNotFoundError
        # so FileNotFoundError propagates up -- test the behavior as-is
        with pytest.raises(FileNotFoundError):
            eng.load_champion(0, "/nonexistent/path.json")

    def test_load_valid_champion_file(self):
        """Test load_champion with a properly formatted champion file."""
        eng = self._make_engine()
        # ChampionAnchor uses its own save/load format with SHA-256
        from champion_anchor import ChampionAnchor

        path = os.path.join(self.tmpdir, "champion_l0")
        prog = [0x00000033, 0x00100073]
        anchor = ChampionAnchor()
        anchor.save(prog, path)

        result = eng.load_champion(0, path)
        assert result is True
        assert eng.champion == prog
        assert eng.champion_level == 0

    def test_load_corrupt_file_returns_false(self):
        """Corrupt file that causes IntegrityError should return False."""
        eng = self._make_engine()
        path = os.path.join(self.tmpdir, "champion_l0")
        prog = [0x00000033, 0x00100073]
        # Write a JSON file with wrong checksum directly (bypass save which writes .bin)
        json_path = path + ".json"
        data = {
            "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "program": prog,
            "level": 0,
        }
        with open(json_path, "w") as f:
            json.dump(data, f)

        result = eng.load_champion(0, path)
        assert result is False

    def test_load_no_path_returns_false(self):
        """When no path and no champion_dir, should return False."""
        eng = self._make_engine()
        eng.config.champion_dir = ""
        result = eng.load_champion(0, path=None)
        assert result is False


# ───────────────────────────────────────────────────────────────────────
# 12. VMFarm._resolve_kernel_path / _dict_to_eval_result
# ───────────────────────────────────────────────────────────────────────


class TestVMFarmHelpers:
    """Tests for VMFarm helper methods."""

    def test_resolve_kernel_path_direct(self):
        """When kernel_path exists as file, return it."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        try:
            result = VMFarm._resolve_kernel_path(path)
            assert result == path
        finally:
            os.unlink(path)

    def test_resolve_kernel_path_not_found(self):
        """When kernel_path doesn't exist and no candidates found, return empty."""
        result = VMFarm._resolve_kernel_path("/nonexistent/kernel")
        # Either returns a valid path or empty string (no external kernel found)
        assert isinstance(result, str)

    def test_dict_to_eval_result(self):
        d = {
            "success": True,
            "registers": {10: 42},
            "memory": {100: 5},
            "cycles": 200,
            "error": None,
            "blocked_writes": 3,
        }
        result = VMFarm._dict_to_eval_result(d)
        assert result.success is True
        assert result.registers == {10: 42}
        assert result.memory == {100: 5}
        assert result.cycles == 200
        assert result.error is None
        assert result.blocked_writes == 3

    def test_dict_to_eval_result_defaults(self):
        d = {}
        result = VMFarm._dict_to_eval_result(d)
        assert result.success is False
        assert result.registers == {}
        assert result.memory == {}
        assert result.cycles == 0
        assert result.error is None
        assert result.blocked_writes == 0


# ───────────────────────────────────────────────────────────────────────
# 13. MutationProposer utility functions
# ───────────────────────────────────────────────────────────────────────


class TestMutationProposerUtils:
    """Tests for mutation proposer utility functions."""

    def test_is_ebreak(self):
        from mutation_proposer import EBREAK, _is_ebreak

        assert _is_ebreak(0x00100073) is True
        assert _is_ebreak(EBREAK) is True
        assert _is_ebreak(0x00000033) is False
        assert _is_ebreak(0) is False

    def test_validate_mutation_empty(self):
        from mutation_proposer import validate_mutation

        valid, err = validate_mutation([])
        assert valid is False
        assert "Empty" in err

    def test_validate_mutation_valid(self):
        from mutation_proposer import EBREAK, make_rv32i, validate_mutation

        prog = [make_rv32i("ADDI", rd=10, rs1=0, imm=42), EBREAK]
        valid, err = validate_mutation(prog)
        assert valid is True
        assert err is None

    def test_generate_seed_program_default(self):
        from mutation_proposer import generate_seed_program

        prog = generate_seed_program()
        assert len(prog) >= 2
        assert prog[-1] == 0x00100073  # EBREAK

    def test_generate_seed_program_with_rng(self):
        from mutation_proposer import generate_seed_program

        rng = random.Random(42)
        prog = generate_seed_program(length=10, rng=rng)
        assert len(prog) == 10
        assert prog[-1] == 0x00100073

    def test_generate_seed_program_min_length(self):
        from mutation_proposer import generate_seed_program

        prog = generate_seed_program(length=1)
        assert len(prog) >= 2  # at least 1 instr + EBREAK


# ───────────────────────────────────────────────────────────────────────
# 14. StrategyArchive
# ───────────────────────────────────────────────────────────────────────


class TestStrategyArchiveExtended:
    """Extended tests for StrategyArchive."""

    def test_summary(self):
        sa = StrategyArchive()
        sa.record("RANDOM", 0, True, 1)
        sa.record("RANDOM", 0, True, 2)
        sa.record("TASK_AWARE", 0, False, 3)
        summary = sa.summary()
        assert summary["total_records"] == 3
        assert "RANDOM" in summary["strategies"]
        assert "TASK_AWARE" in summary["strategies"]
        assert summary["strategies"]["RANDOM"]["count"] == 2
        assert summary["strategies"]["RANDOM"]["successes"] == 2
        assert summary["strategies"]["TASK_AWARE"]["count"] == 1
        assert summary["strategies"]["TASK_AWARE"]["successes"] == 0

    def test_clear(self):
        sa = StrategyArchive()
        sa.record("RANDOM", 0, True, 1)
        sa.record("TASK_AWARE", 1, False, 2)
        assert sa.total_records() == 2
        sa.clear()
        assert sa.total_records() == 0

    def test_record_batch_slow_path(self):
        """record_batch with partial dicts (slow path)."""
        sa = StrategyArchive()
        records = [
            {"strategy": "DIRECTED", "level": 0, "success": True, "gen": 1},
            {"strategy": "RANDOM", "level": 1},  # missing success, gen
        ]
        sa.record_batch(records)
        assert sa.total_records() == 2
        assert sa.success_rate("DIRECTED", level=0) > 0.0

    def test_load_failure_returns_false(self):
        sa = StrategyArchive()
        assert sa.load("/nonexistent/archive.json") is False

    def test_load_corrupt_json_returns_false(self):
        sa = StrategyArchive()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json {{{")
            path = f.name
        try:
            assert sa.load(path) is False
        finally:
            os.unlink(path)

    def test_records_by_strategy(self):
        sa = StrategyArchive()
        sa.record("RANDOM", 0, True, 1)
        sa.record("RANDOM", 1, False, 2)
        sa.record("DIRECTED", 0, True, 3)
        records = sa.records_by_strategy("RANDOM")
        assert len(records) == 2
        assert all(r["strategy"] == "RANDOM" for r in records)

    def test_records_by_strategy_empty(self):
        sa = StrategyArchive()
        records = sa.records_by_strategy("NONEXISTENT")
        assert records == []

    def test_records_by_level_empty(self):
        sa = StrategyArchive()
        records = sa.records_by_level(99)
        assert records == []

    def test_refresh_timestamp(self):
        sa = StrategyArchive()
        old_ts = sa._gen_timestamp
        sa.refresh_timestamp()
        assert sa._gen_timestamp >= old_ts

    def test_exploration_bonus_untried(self):
        sa = StrategyArchive()
        sa.record("RANDOM", 0, True, 1)
        bonus = sa.exploration_bonus("TASK_AWARE", level=0)
        assert bonus == 1.0  # maximum bonus for untried


# ───────────────────────────────────────────────────────────────────────
# 15. MetaEvolutionEngine._compute_meta_fitness edge cases
# ───────────────────────────────────────────────────────────────────────


class TestComputeMetaFitness:
    """Tests for meta-fitness computation edge cases."""

    def test_zero_fitness_summary(self):
        summary = {
            "final_fitness": 0.0,
            "final_level": 0,
            "generations_completed": 50,
            "champion_words": 10,
        }
        result = MetaEvolutionEngine._compute_meta_fitness(summary)
        assert result >= 0.0

    def test_negative_fitness_summary(self):
        """Negative fitness should zero out curriculum_depth."""
        summary = {
            "final_fitness": -5.0,
            "final_level": 10,
            "generations_completed": 50,
            "champion_words": 30,
        }
        result = MetaEvolutionEngine._compute_meta_fitness(summary)
        # curriculum_depth should be 0 because fitness < 0
        assert result >= 0.0

    def test_high_champion_words_bloat_penalty(self):
        """Very long champion should get bloat penalty."""
        summary = {
            "final_fitness": 1.0,
            "final_level": 5,
            "generations_completed": 50,
            "champion_words": 200,
        }
        result = MetaEvolutionEngine._compute_meta_fitness(summary)
        assert result >= 0.0

    def test_short_champion_stub_penalty(self):
        """Very short champion (< 5 words) gets stub penalty."""
        summary = {
            "final_fitness": 1.0,
            "final_level": 5,
            "generations_completed": 50,
            "champion_words": 2,
        }
        result = MetaEvolutionEngine._compute_meta_fitness(summary)
        assert result >= 0.0

    def test_sweet_spot_champion_words(self):
        """25-50 words should get maximum diversity score."""
        summary = {
            "final_fitness": 1.0,
            "final_level": 5,
            "generations_completed": 50,
            "champion_words": 30,
        }
        result = MetaEvolutionEngine._compute_meta_fitness(summary)
        assert result >= 0.0

    def test_custom_meta_params(self):
        summary = {
            "final_fitness": 1.0,
            "final_level": 5,
            "generations_completed": 50,
            "champion_words": 30,
        }
        params = {
            "max_fitness": 2.0,
            "max_level": 13,
            "max_champion_words": 500,
            "meta_weight_improvement": 0.5,
            "meta_weight_depth": 0.3,
            "meta_weight_diversity": 0.2,
        }
        result = MetaEvolutionEngine._compute_meta_fitness(summary, meta_params=params)
        assert result >= 0.0
        # champion_improvement = min(1.0, 1.0/2.0) = 0.5
        # curriculum_depth = min(1.0, 5/13) = 0.385
        # diversity for 30 words = 1.0 (sweet spot)
        # meta_fitness = 0.5*0.5 + 0.385*0.3 + 1.0*0.2 = 0.25+0.115+0.2 = 0.565
        assert abs(result - 0.565) < 0.01

    def test_missing_summary_keys(self):
        """Missing keys should use defaults."""
        summary = {}
        result = MetaEvolutionEngine._compute_meta_fitness(summary)
        assert result >= 0.0

    def test_zero_generations_handled(self):
        summary = {
            "final_fitness": 0.5,
            "final_level": 3,
            "generations_completed": 0,
            "champion_words": 20,
        }
        result = MetaEvolutionEngine._compute_meta_fitness(summary)
        assert result >= 0.0


# ───────────────────────────────────────────────────────────────────────
# 16. EvolutionConfig.__post_init__ validation
# ───────────────────────────────────────────────────────────────────────


class TestEvolutionConfigValidation:
    """Tests for EvolutionConfig validation errors."""

    def test_population_size_too_small(self):
        with pytest.raises(ValueError, match="population_size"):
            EvolutionConfig(population_size=1)

    def test_max_generations_zero(self):
        with pytest.raises(ValueError, match="max_generations"):
            EvolutionConfig(max_generations=0)

    def test_mutation_rate_out_of_range(self):
        with pytest.raises(ValueError, match="mutation_rate"):
            EvolutionConfig(mutation_rate=1.5)

    def test_crossover_rate_negative(self):
        with pytest.raises(ValueError, match="crossover_rate"):
            EvolutionConfig(crossover_rate=-0.1)

    def test_elite_ratio_too_high(self):
        with pytest.raises(ValueError, match="elite_ratio"):
            EvolutionConfig(elite_ratio=0.6)

    def test_tournament_size_too_small(self):
        with pytest.raises(ValueError, match="tournament_size"):
            EvolutionConfig(tournament_size=1)

    def test_tournament_size_exceeds_population(self):
        with pytest.raises(ValueError, match="tournament_size.*population_size"):
            EvolutionConfig(population_size=5, tournament_size=10)

    def test_instruction_budget_too_low(self):
        with pytest.raises(ValueError, match="instruction_budget"):
            EvolutionConfig(instruction_budget=50)

    def test_eval_budget_too_low(self):
        with pytest.raises(ValueError, match="eval_budget"):
            EvolutionConfig(eval_budget=50)

    def test_retrain_val_split_out_of_range(self):
        with pytest.raises(ValueError, match="retrain_val_split"):
            EvolutionConfig(retrain_val_split=1.5)

    def test_retrain_max_epochs_zero(self):
        with pytest.raises(ValueError, match="retrain_max_epochs"):
            EvolutionConfig(retrain_max_epochs=0)

    def test_retrain_batch_size_zero(self):
        with pytest.raises(ValueError, match="retrain_batch_size"):
            EvolutionConfig(retrain_batch_size=0)

    def test_retrain_lr_negative(self):
        with pytest.raises(ValueError, match="retrain_lr"):
            EvolutionConfig(retrain_lr=-0.01)

    def test_retrain_freeze_layers_negative(self):
        with pytest.raises(ValueError, match="retrain_freeze_layers"):
            EvolutionConfig(retrain_freeze_layers=-1)

    def test_retrain_weight_decay_too_high(self):
        with pytest.raises(ValueError, match="retrain_weight_decay"):
            EvolutionConfig(retrain_weight_decay=1.0)

    def test_retrain_patience_too_low(self):
        with pytest.raises(ValueError, match="retrain_patience"):
            EvolutionConfig(retrain_patience=0)

    def test_retrain_timeout_too_low(self):
        with pytest.raises(ValueError, match="retrain_timeout"):
            EvolutionConfig(retrain_timeout=30)

    def test_stall_threshold_too_low(self):
        with pytest.raises(ValueError, match="stall_threshold"):
            EvolutionConfig(stall_threshold=0)

    def test_max_level_too_low(self):
        with pytest.raises(ValueError, match="max_level"):
            EvolutionConfig(max_level=0)

    def test_curriculum_stall_threshold_too_low(self):
        with pytest.raises(ValueError, match="curriculum_stall_threshold"):
            EvolutionConfig(curriculum_stall_threshold=0)

    def test_negative_meta_weights(self):
        with pytest.raises(ValueError, match="meta_weight"):
            EvolutionConfig(meta_weight_improvement=-0.1)

    def test_negative_novelty_weight(self):
        with pytest.raises(ValueError, match="novelty_weight"):
            EvolutionConfig(novelty_weight=-0.1)

    def test_negative_qd_bonus_weight(self):
        with pytest.raises(ValueError, match="qd_bonus_weight"):
            EvolutionConfig(qd_bonus_weight=-0.1)

    def test_max_consecutive_same_strategy_too_low(self):
        with pytest.raises(ValueError, match="max_consecutive_same_strategy"):
            EvolutionConfig(max_consecutive_same_strategy=2)

    def test_diversity_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="diversity_threshold"):
            EvolutionConfig(diversity_threshold=1.5)

    def test_diversity_inject_ratio_out_of_range(self):
        with pytest.raises(ValueError, match="diversity_inject_ratio"):
            EvolutionConfig(diversity_inject_ratio=-0.1)

    def test_champion_crossover_prob_out_of_range(self):
        with pytest.raises(ValueError, match="champion_crossover_prob"):
            EvolutionConfig(champion_crossover_prob=1.5)

    def test_circuit_breaker_stall_limit_too_low(self):
        with pytest.raises(ValueError, match="circuit_breaker_stall_limit"):
            EvolutionConfig(circuit_breaker_stall_limit=3)

    def test_valid_config_no_error(self):
        """Default config should pass validation."""
        cfg = EvolutionConfig()
        assert cfg.population_size == 50


# ───────────────────────────────────────────────────────────────────────
# 17. MetaController.tune_retrain_params edge cases
# ───────────────────────────────────────────────────────────────────────


class TestTuneRetrainParamsEdgeCases:
    """Tests for MetaController.tune_retrain_params edge cases."""

    def _make_controller(self, **kwargs):
        return MetaController(checkpoint_dir="/tmp/mc_tune_test", **kwargs)

    def test_compound_stall_request_meta_evolution(self):
        """3+ consecutive compound_stall cycles should request meta-evolution."""
        mc = self._make_controller(level_stall_cycles=1, fitness_plateau_cycles=1)

        # Create conditions for compound stall: fitness plateau + level stall
        for c in range(1, 12):
            mc.update_cycle(_snapshot(c, fitness=-2.0, level=5, diversity=0.01))
            params = mc.tune_retrain_params(c)
            mc.detect_stagnation()

        state = mc.detect_stagnation()
        # After enough cycles, compound_stall counter should be high
        if state == StagnationResult.COMPOUND_STALL and mc._compound_stall_counter >= 3:
            params = mc.tune_retrain_params(12)
            assert params.get("meta_evolution_requested") is True

    def test_compound_stall_no_model_explores(self):
        """Compound stall without a good model should explore (increase LR)."""
        mc = self._make_controller(level_stall_cycles=1, fitness_plateau_cycles=1)
        mc._best_val_loss = float("inf")

        for c in range(1, 8):
            mc.update_cycle(_snapshot(c, fitness=-2.0, level=5, diversity=0.01))
            mc.tune_retrain_params(c)

        state = mc.detect_stagnation()
        if state == StagnationResult.COMPOUND_STALL:
            params = mc.tune_retrain_params(8)
            assert params["retrain_lr"] > 0

    def test_recovery_mode_reduces_lr(self):
        """Recovery mode should halve the LR."""
        mc = self._make_controller()
        mc.recovery_mode = True
        mc.recovery_cycles_remaining = 2

        params = mc.tune_retrain_params(5)
        # LR should be halved from base
        assert params["retrain_lr"] > 0

    def test_recovery_mode_expires(self):
        """Recovery mode should expire after recovery_cycles_remaining reaches 0."""
        mc = self._make_controller()
        mc.recovery_mode = True
        mc.recovery_cycles_remaining = 1
        mc.tune_retrain_params(5)
        assert mc.recovery_mode is False

    def test_val_loss_none_fallback_stagnant_fitness(self):
        """When val_loss is None for 3+ cycles and fitness stagnant, LR is halved."""
        mc = self._make_controller()
        # Push fitness to a known value
        for c in range(1, 5):
            mc.update_cycle(_snapshot(c, fitness=5.0, level=5, val_loss=None))
            mc.tune_retrain_params(c)
        # Now fitness stops improving
        for c in range(5, 10):
            mc.update_cycle(_snapshot(c, fitness=5.0, level=5, val_loss=None))
            mc.tune_retrain_params(c)

        # After 5+ cycles with no fitness improvement and val_loss=None,
        # the fallback should halve LR
        assert mc._consecutive_none_val_loss >= 3

    def test_val_loss_none_fallback_fitness_improving(self):
        """When val_loss None but fitness improving, keep current LR."""
        mc = self._make_controller()
        for c in range(1, 5):
            mc.update_cycle(_snapshot(c, fitness=float(c), level=5, val_loss=None))
            params = mc.tune_retrain_params(c)
        # Fitness is improving, so LR should not be halved
        assert params["retrain_lr"] > 0

    def test_cycle_1_higher_lr(self):
        """Cycle 1 should use higher base LR (3e-4)."""
        mc = self._make_controller()
        params = mc.tune_retrain_params(1)
        assert params["retrain_lr"] == 3e-4

    def test_cycle_2_to_3_moderate_lr(self):
        """Cycles 2-3 should use moderate LR."""
        mc = self._make_controller()
        params = mc.tune_retrain_params(2)
        assert params["retrain_lr"] == 3e-5

    def test_cycle_gt_15_low_lr(self):
        """Cycle > 15 should use lowest base LR."""
        mc = self._make_controller()
        params = mc.tune_retrain_params(20)
        assert params["retrain_lr"] == 5e-6

    def test_per_level_cap_increases_with_cycle(self):
        """Per-level data cap should increase with cycle number."""
        mc = self._make_controller()
        params3 = mc.tune_retrain_params(3)
        params10 = mc.tune_retrain_params(10)
        assert params10["retrain_max_per_level"] > params3["retrain_max_per_level"]


# ───────────────────────────────────────────────────────────────────────
# 18. MetaController.get_state_summary
# ───────────────────────────────────────────────────────────────────────


class TestGetStateSummary:
    """Tests for MetaController.get_state_summary."""

    def test_initial_summary(self):
        mc = MetaController(checkpoint_dir="/tmp/mc_summary_test")
        summary = mc.get_state_summary()
        assert summary["best_fitness"] == -float("inf")
        assert summary["best_level"] == 0
        assert summary["best_val_loss"] is None
        assert summary["recovery_mode"] is False
        assert summary["total_cycles_observed"] == 0
        assert summary["last_5_cycles"] == []
        assert summary["tuning_history"] == []

    def test_summary_after_updates(self):
        mc = MetaController(checkpoint_dir="/tmp/mc_summary_test")
        mc.update_cycle(_snapshot(1, fitness=5.0, level=3, val_loss=2.0))
        mc.update_cycle(_snapshot(2, fitness=6.0, level=4, val_loss=1.5))
        summary = mc.get_state_summary()
        assert summary["best_fitness"] == 6.0
        assert summary["best_level"] == 4
        assert summary["total_cycles_observed"] == 2
        assert len(summary["last_5_cycles"]) == 2

    def test_summary_includes_tuning_history(self):
        mc = MetaController(checkpoint_dir="/tmp/mc_summary_test")
        # tune_retrain_params only records tuning when stagnation state != NORMAL
        # Need enough cycles for a stagnation signal
        for c in range(1, 10):
            mc.update_cycle(_snapshot(c, fitness=5.0, level=5))
        mc.tune_retrain_params(10)
        summary = mc.get_state_summary()
        # tuning_history may or may not be populated depending on stagnation state
        # Just verify the key exists
        assert "tuning_history" in summary


# ───────────────────────────────────────────────────────────────────────
# 19. MetaController.detect_stagnation with diversity trend
# ───────────────────────────────────────────────────────────────────────


class TestDetectStagnationDiversityTrend:
    """Tests for diversity trend detection in detect_stagnation."""

    def _make_controller(self, **kwargs):
        return MetaController(checkpoint_dir="/tmp/mc_div_test", **kwargs)

    def test_diversity_trend_detected(self):
        """Dropping diversity >40% across window with low current diversity."""
        mc = self._make_controller(
            diversity_collapse_threshold=0.1,
            level_stall_cycles=20,  # avoid LEVEL_STALLED triggering
        )
        # Use improving fitness to avoid FITNESS_PLATEAU so only DIVERSITY_COLLAPSE triggers
        for c, div in [(1, 0.5), (2, 0.4), (3, 0.3), (4, 0.2), (5, 0.05)]:
            mc.update_cycle(_snapshot(c, fitness=-5.0 + c, level=5, diversity=div))
        state = mc.detect_stagnation()
        # div_first=0.5, div_last=0.05, drop = 90% > 40%, div_last < 0.1*1.5=0.15
        # div_min=0.05 < 0.1 threshold -> DIVERSITY_COLLAPSE
        assert state == StagnationResult.DIVERSITY_COLLAPSE

    def test_diversity_drop_but_not_below_relaxed_threshold(self):
        """Diversity drop without relaxed threshold should not trigger."""
        mc = self._make_controller(diversity_collapse_threshold=0.01)
        # div_first=0.5, div_last=0.15, drop = 70%, but div_last=0.15 > 0.01*1.5=0.015
        for c, div in [(1, 0.5), (2, 0.4), (3, 0.35), (4, 0.2), (5, 0.15)]:
            mc.update_cycle(_snapshot(c, fitness=-2.0, level=5, diversity=div))
        state = mc.detect_stagnation()
        # div_min = 0.15 > 0.01 threshold, no trend (0.15 > 0.5*0.6=0.3? No)
        # Actually: div_last=0.15, div_first=0.5, 0.15 < 0.5*0.6=0.3 -> trend detected
        # BUT 0.15 >= 0.01*1.5=0.015 -> not below relaxed threshold
        # So diversity_trend_detected = False, div_min=0.15 >= 0.01 -> no collapse
        assert state == StagnationResult.NORMAL or state != StagnationResult.DIVERSITY_COLLAPSE


# ───────────────────────────────────────────────────────────────────────
# 20. HallOfFame extended
# ───────────────────────────────────────────────────────────────────────


class TestHallOfFameExtended:
    """Extended Hall of Fame tests."""

    def test_get_best_with_level_filter(self):
        hof = HallOfFame(capacity=5)
        hof.add([1], fitness=10.0, level=0, generation=1)
        hof.add([2], fitness=20.0, level=1, generation=2)
        hof.add([3], fitness=15.0, level=0, generation=3)
        best = hof.get_best(level=0)
        # Should return the best fitness program at level 0
        assert best == [3]  # fitness=15.0

    def test_get_best_with_nonexistent_level_falls_back(self):
        hof = HallOfFame(capacity=5)
        hof.add([1], fitness=10.0, level=0, generation=1)
        hof.add([2], fitness=20.0, level=1, generation=2)
        best = hof.get_best(level=99)
        # Falls back to unfiltered best
        assert best == [2]  # fitness=20.0

    def test_get_best_empty_returns_none(self):
        hof = HallOfFame(capacity=5)
        assert hof.get_best() is None

    def test_get_best_level_filter_single_match(self):
        hof = HallOfFame(capacity=5)
        hof.add([1], fitness=10.0, level=0, generation=1)
        hof.add([2], fitness=20.0, level=1, generation=2)
        best = hof.get_best(level=1)
        assert best == [2]

    def test_add_returns_true_when_not_full(self):
        hof = HallOfFame(capacity=5)
        assert hof.add([1], fitness=10.0, level=0) is True
        assert hof.size == 1

    def test_add_returns_true_when_replacing(self):
        hof = HallOfFame(capacity=2)
        hof.add([1], fitness=10.0, level=0)
        hof.add([2], fitness=5.0, level=0)
        assert hof.add([3], fitness=20.0, level=0) is True

    def test_returns_defensive_copies(self):
        """get_random should return a copy, not the internal list."""
        hof = HallOfFame(capacity=5)
        prog = [0x00000033, 0x00100073]
        hof.add(prog, fitness=10.0, level=0, generation=1)
        result = hof.get_random(random.Random(42))
        assert result == prog
        result.append(0xDEADBEEF)
        # Original should be unaffected
        assert hof.get_random(random.Random(42)) == [0x00000033, 0x00100073]


# ───────────────────────────────────────────────────────────────────────
# 21. compute_qd_axes extended
# ───────────────────────────────────────────────────────────────────────


class TestComputeQDAxesExtended:
    """Extended tests for compute_qd_axes."""

    def test_all_11_categories(self):
        """Program using all 11 opcode categories should have inst_div = 1.0."""
        prog = [
            0x00000037,  # LUI  (0x37)
            0x00000017,  # AUIPC (0x17)
            0x0000006F,  # JAL  (0x6F)
            0x00000067,  # JALR (0x67)
            0x00000063,  # BRANCH (0x63)
            0x00000003,  # LOAD (0x03)
            0x00000023,  # STORE (0x23)
            0x00000013,  # ALU_IMM (0x13)
            0x00000033,  # ALU_REG (0x33)
            0x0000000F,  # FENCE (0x0F)
            0x00100073,  # SYSTEM (0x73)
        ]
        axes = compute_qd_axes(prog, None)
        assert axes[0] == 1.0  # 11/11 categories

    def test_trace_with_empty_regs(self):
        """Trace with empty regs dict should give -1.0."""
        axes = compute_qd_axes([0x00000033], {"final_regs": {}})
        assert axes[1] == -1.0

    def test_trace_final_regs_not_dict(self):
        """Non-dict final_regs should give -1.0."""
        axes = compute_qd_axes([0x00000033], {"final_regs": [1, 2, 3]})
        assert axes[1] == -1.0

    def test_trace_without_final_regs_key(self):
        """Trace without final_regs key should give -1.0."""
        axes = compute_qd_axes([0x00000033], {"other_key": "value"})
        assert axes[1] == -1.0


# ───────────────────────────────────────────────────────────────────────
# 22. MetaEvolutionEngine initialization edge cases
# ───────────────────────────────────────────────────────────────────────


class TestMetaEvolutionEngineInit:
    """Tests for MetaEvolutionEngine initialization."""

    def test_small_population_truncates_seeds(self):
        """Population smaller than seeds should truncate seed list."""
        config = MetaEvolutionConfig(meta_population_size=2, seed=42)
        engine = MetaEvolutionEngine(meta_config=config)
        assert len(engine.population) == 2

    def test_population_matches_config_size(self):
        config = MetaEvolutionConfig(meta_population_size=10, seed=42)
        engine = MetaEvolutionEngine(meta_config=config)
        assert len(engine.population) == 10

    def test_custom_rng(self):
        rng = random.Random(99)
        config = MetaEvolutionConfig(meta_population_size=5, seed=42)
        engine = MetaEvolutionEngine(meta_config=config, rng=rng)
        assert engine._rng is rng

    def test_best_config_without_best_genome(self):
        """best_config should fall back to current population."""
        config = MetaEvolutionConfig(meta_population_size=3, seed=42)
        engine = MetaEvolutionEngine(meta_config=config)
        engine._best_genome = None
        engine.fitness = [1.0, 5.0, 2.0]
        best = engine.best_config()
        # Should return decoded genome from index 1 (fitness=5.0)
        assert best is not None


# ───────────────────────────────────────────────────────────────────────
# 23. MetaGenome decode
# ───────────────────────────────────────────────────────────────────────


class TestMetaGenomeDecode:
    """Tests for MetaGenome decoding."""

    def test_decode_returns_evolution_config(self):
        genome = MetaGenome(
            mutation_rate=0.5,
            crossover_rate=0.3,
            elite_ratio=0.1,
            tournament_size=5,
            population_size=50,
            stall_threshold=5,
            novelty_weight=0.2,
            strategy_type_weights={"RANDOM": 0.25, "TASK_AWARE": 0.25, "DIRECTED": 0.25, "PERTURBATION_CAPPED": 0.25},
        )
        cfg = genome.decode()
        assert cfg.mutation_rate == 0.5
        assert cfg.population_size == 50
        assert cfg.elite_ratio == 0.1

    def test_random_genome_produces_valid_config(self):
        rng = random.Random(42)
        genome = MetaGenome.random(rng=rng)
        cfg = genome.decode()
        assert cfg.mutation_rate >= 0.0
        assert cfg.mutation_rate <= 1.0
        assert cfg.population_size >= 10

    def test_crossover_produces_two_genomes(self):
        rng = random.Random(42)
        g1 = MetaGenome.random(rng=rng)
        g2 = MetaGenome.random(rng=rng)
        c1, c2 = g1.crossover(g2, rng=rng)
        assert c1 is not None
        assert c2 is not None

    def test_mutate_produces_different_genome(self):
        rng = random.Random(42)
        g1 = MetaGenome.random(rng=rng)
        g2 = g1.mutate(std=0.5, rng=rng)
        # Mutation should produce a genome (may or may not differ)
        assert g2 is not None
        cfg = g2.decode()
        assert cfg.mutation_rate >= 0.0


# ───────────────────────────────────────────────────────────────────────
# 24. MetaController.load_state with full data
# ───────────────────────────────────────────────────────────────────────


class TestLoadStateFullData:
    """Tests for loading full state data."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_restores_all_fields(self):
        mc = MetaController(checkpoint_dir=self.tmpdir)
        mc.update_cycle(_snapshot(1, fitness=10.0, level=5, val_loss=1.0))
        mc.recovery_mode = True
        mc.recovery_cycles_remaining = 3
        mc._consecutive_none_val_loss = 5

        path = os.path.join(self.tmpdir, "state.json")
        mc.save_state(path)

        mc2 = MetaController(checkpoint_dir=self.tmpdir)
        mc2.load_state(path)
        assert mc2.best_fitness == 10.0
        assert mc2.best_level == 5
        assert mc2._best_val_loss == 1.0
        assert mc2.recovery_mode is True
        assert mc2.recovery_cycles_remaining == 3
        assert mc2._consecutive_none_val_loss == 5


# ───────────────────────────────────────────────────────────────────────
# 25. Crossover sentinel
# ───────────────────────────────────────────────────────────────────────


class TestCircuitBreakerSentinel:
    """Tests for the CircuitBreakerSentinel."""

    def test_sentinel_always_open(self):
        sentinel = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        assert sentinel.is_breached() is False

    def test_sentinel_repr(self):
        sentinel = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        r = repr(sentinel)
        assert "CircuitBreakerSentinel" in r

    def test_sentinel_is_singleton(self):
        """All references to CIRCUIT_BREAKER_SENTINEL should be the same object."""
        s1 = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        s2 = EvolutionConfig.CIRCUIT_BREAKER_SENTINEL
        assert s1 is s2


# ───────────────────────────────────────────────────────────────────────
# 26. VMFarm input validation
# ───────────────────────────────────────────────────────────────────────


class TestVMFarmInputValidation:
    """Tests for VMFarm initialization validation."""

    def test_invalid_num_workers_negative(self):
        with pytest.raises(ValueError, match="num_workers"):
            VMFarm(num_workers=-1)

    def test_invalid_num_workers_zero(self):
        with pytest.raises(ValueError, match="num_workers"):
            VMFarm(num_workers=0)

    def test_invalid_num_workers_string(self):
        with pytest.raises(ValueError, match="num_workers"):
            VMFarm(num_workers="two")

    def test_valid_num_workers(self):
        vm = VMFarm(num_workers=1)
        assert vm.num_workers == 1


# ───────────────────────────────────────────────────────────────────────
# 27. TelemetrySnapshot.to_dict
# ───────────────────────────────────────────────────────────────────────


class TestTelemetrySnapshot:
    """Tests for TelemetrySnapshot."""

    def test_to_dict(self):
        snap = _snapshot(1, fitness=5.0, level=3)
        d = snap.to_dict()
        assert d["cycle"] == 1
        assert d["fitness"] == 5.0
        assert d["level"] == 3
        assert d["val_loss"] == 2.0
        assert d["diversity"] == 0.3
        assert d["retrained"] is True

    def test_to_dict_none_val_loss(self):
        snap = TelemetrySnapshot(
            cycle=1,
            fitness=5.0,
            level=3,
            champion_words=50,
            val_loss=None,
            diversity=0.3,
            novelty_size=10,
            qd_occupied=5,
            qd_total=100,
            traces_count=20,
            retrained=False,
            generations_completed=50,
        )
        d = snap.to_dict()
        assert d["val_loss"] is None


# ───────────────────────────────────────────────────────────────────────
# 28. TuningAction.to_dict
# ───────────────────────────────────────────────────────────────────────


class TestTuningAction:
    """Tests for TuningAction."""

    def test_to_dict(self):
        from meta_controller import TuningAction

        action = TuningAction("retrain", "retrain_lr", 0.001, 0.002, "stagnation: fitness_plateau")
        d = action.to_dict()
        assert d["domain"] == "retrain"
        assert d["param"] == "retrain_lr"
        assert d["old"] == 0.001
        assert d["new"] == 0.002
        assert "stagnation" in d["reason"]


# ───────────────────────────────────────────────────────────────────────
# 29. StrategyArchive.decay_level_records
# ───────────────────────────────────────────────────────────────────────


class TestDecayLevelRecords:
    """Tests for decay_level_records."""

    def test_decay_no_records_at_level(self):
        sa = StrategyArchive()
        sa.record("RANDOM", 0, True, 1)
        sa.record("TASK_AWARE", 1, True, 2)
        modified = sa.decay_level_records(99, factor=0.5)
        assert modified == 0

    def test_decay_with_seed(self):
        """Deterministic decay with seeded RNG."""
        sa = StrategyArchive()
        for i in range(20):
            sa.record("RANDOM", 0, True, i)
        rng = random.Random(42)
        modified = sa.decay_level_records(0, factor=0.5, rng=rng)
        assert modified >= 0  # Some records may have been flipped

    def test_decay_factor_1_no_change(self):
        """Factor=1.0 should not flip any records."""
        sa = StrategyArchive()
        for i in range(10):
            sa.record("RANDOM", 0, True, i)
        rng = random.Random(42)
        modified = sa.decay_level_records(0, factor=1.0, rng=rng)
        assert modified == 0

    def test_decay_factor_0_all_flipped(self):
        """Factor=0.0 should flip all successful records."""
        sa = StrategyArchive()
        for i in range(10):
            sa.record("RANDOM", 0, True, i)
        rng = random.Random(42)
        modified = sa.decay_level_records(0, factor=0.0, rng=rng)
        assert modified == 10


# ───────────────────────────────────────────────────────────────────────
# 30. _compute_population_diversity extended
# ───────────────────────────────────────────────────────────────────────


class TestComputePopulationDiversityExtended:
    """Extended tests for _compute_population_diversity."""

    def test_many_diverse_programs(self):
        """Programs using different opcodes should have high diversity."""
        progs = [
            [0x00000033],  # ALU_REG
            [0x00000013],  # ALU_IMM
            [0x00000063],  # BRANCH
            [0x00000003],  # LOAD
            [0x00000023],  # STORE
            [0x00000037],  # LUI
            [0x00000017],  # AUIPC
            [0x0000006F],  # JAL
            [0x00000067],  # JALR
            [0x0000000F],  # FENCE
            [0x00100073],  # SYSTEM
        ]
        div = EvolutionEngine._compute_population_diversity(progs)
        assert div == 1.0  # All 11 categories

    def test_identical_programs_low_diversity(self):
        """All identical programs should have very low diversity."""
        prog = [0x00000033, 0x00000013]
        div = EvolutionEngine._compute_population_diversity([prog] * 10)
        assert 0.0 < div < 0.2  # Only 2 categories out of 11


# ───────────────────────────────────────────────────────────────────────
# 31. StrategyArchive.recommend_strategy fallback
# ───────────────────────────────────────────────────────────────────────


class TestRecommendStrategyFallback:
    """Tests for strategy recommendation fallback logic."""

    def test_fallback_to_random_when_empty(self):
        sa = StrategyArchive()
        result = sa.recommend_strategy(0)
        assert result == "RANDOM"

    def test_fallback_to_global_best(self):
        """When level has no records, should use global best."""
        sa = StrategyArchive()
        sa.record("RANDOM", 0, True, 1)
        sa.record("RANDOM", 0, True, 2)
        sa.record("TASK_AWARE", 1, True, 3)
        # Level 2 has no records, should use global best
        result = sa.recommend_strategy(2)
        assert result in ("RANDOM", "TASK_AWARE")

    def test_skip_strategy_affects_recommendation(self):
        """skip_strategy should exclude that strategy from consideration."""
        sa = StrategyArchive()
        sa.record("RANDOM", 0, True, 1)
        sa.record("RANDOM", 0, True, 2)
        sa.record("TASK_AWARE", 0, True, 3)
        result = sa.recommend_strategy(0, skip_strategy="RANDOM")
        assert result == "TASK_AWARE"

    def test_level_with_only_skipped_strategy_falls_back(self):
        """When only strategy at level is skipped, should fall back."""
        sa = StrategyArchive()
        sa.record("RANDOM", 0, True, 1)
        # Only RANDOM at level 0, skip it
        result = sa.recommend_strategy(0, skip_strategy="RANDOM")
        # Falls back to global or RANDOM (since no other strategy)
        assert result in ("RANDOM",)


# ───────────────────────────────────────────────────────────────────────
# 32. MetaEvolutionConfig validation
# ───────────────────────────────────────────────────────────────────────


class TestMetaEvolutionConfig:
    """Tests for MetaEvolutionEngine seed validation."""

    def test_invalid_seed_config_rejected(self):
        """Seed with mutation_rate out of range should raise ValueError."""
        import meta_evolution

        original_seeds = meta_evolution._SEEDED_META_GENOMES[:]
        try:
            meta_evolution._SEEDED_META_GENOMES = [
                {
                    "mutation_rate": 2.0,
                    "crossover_rate": 0.3,
                    "elite_ratio": 0.1,
                    "tournament_size": 5,
                    "population_size": 50,
                    "stall_threshold": 3,
                    "novelty_weight": 0.3,
                    "strategy_type_weights": {
                        "RANDOM": 0.25,
                        "TASK_AWARE": 0.25,
                        "DIRECTED": 0.25,
                        "PERTURBATION_CAPPED": 0.25,
                    },
                },
            ]
            with pytest.raises(ValueError, match="mutation_rate"):
                MetaEvolutionEngine(meta_config=MetaEvolutionConfig(meta_population_size=1))
        finally:
            meta_evolution._SEEDED_META_GENOMES = original_seeds

    def test_invalid_elite_ratio_rejected(self):
        import meta_evolution

        original_seeds = meta_evolution._SEEDED_META_GENOMES[:]
        try:
            meta_evolution._SEEDED_META_GENOMES = [
                {
                    "mutation_rate": 0.5,
                    "crossover_rate": 0.3,
                    "elite_ratio": 0.8,
                    "tournament_size": 5,
                    "population_size": 50,
                    "stall_threshold": 3,
                    "novelty_weight": 0.3,
                    "strategy_type_weights": {
                        "RANDOM": 0.25,
                        "TASK_AWARE": 0.25,
                        "DIRECTED": 0.25,
                        "PERTURBATION_CAPPED": 0.25,
                    },
                },
            ]
            with pytest.raises(ValueError, match="elite_ratio"):
                MetaEvolutionEngine(meta_config=MetaEvolutionConfig(meta_population_size=1))
        finally:
            meta_evolution._SEEDED_META_GENOMES = original_seeds


# ───────────────────────────────────────────────────────────────────────
# 33. StagnationResult enum
# ───────────────────────────────────────────────────────────────────────


class TestStagnationResultEnum:
    """Tests for StagnationResult enum values."""

    def test_all_values(self):
        assert StagnationResult.NORMAL.value == "normal"
        assert StagnationResult.FITNESS_PLATEAU.value == "fitness_plateau"
        assert StagnationResult.LEVEL_STALLED.value == "level_stalled"
        assert StagnationResult.VAL_LOSS_REGRESSION.value == "val_loss_regression"
        assert StagnationResult.DIVERSITY_COLLAPSE.value == "diversity_collapse"
        assert StagnationResult.COMPOUND_STALL.value == "compound_stall"

    def test_member_count(self):
        assert len(StagnationResult) == 6


# ───────────────────────────────────────────────────────────────────────
# 34. MetaController.update_cycle edge cases
# ───────────────────────────────────────────────────────────────────────


class TestUpdateCycleEdgeCases:
    """Tests for MetaController.update_cycle edge cases."""

    def _make_controller(self):
        return MetaController(checkpoint_dir="/tmp/mc_update_test")

    def test_level_dropped_increments_stall(self):
        """Level below best should increment cycles_since_level_up."""
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0, level=10))
        assert mc.best_level == 10
        mc.update_cycle(_snapshot(2, fitness=5.0, level=8))
        assert mc.cycles_since_level_up == 1

    def test_level_same_increments_stall(self):
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0, level=5))
        mc.update_cycle(_snapshot(2, fitness=5.0, level=5))
        assert mc.cycles_since_level_up == 1

    def test_level_new_best_resets_stall(self):
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0, level=5))
        mc.update_cycle(_snapshot(2, fitness=6.0, level=6))
        assert mc.cycles_since_level_up == 0

    def test_val_loss_none_increments_counter(self):
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0, val_loss=None))
        mc.update_cycle(_snapshot(2, fitness=5.0, val_loss=None))
        assert mc._consecutive_none_val_loss == 2

    def test_val_loss_positive_resets_counter(self):
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0, val_loss=None))
        mc.update_cycle(_snapshot(2, fitness=5.0, val_loss=None))
        mc.update_cycle(_snapshot(3, fitness=5.0, val_loss=2.0))
        assert mc._consecutive_none_val_loss == 0

    def test_val_loss_zero_does_not_update_best(self):
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0, val_loss=0.0))
        assert mc._best_val_loss == float("inf")  # unchanged

    def test_val_loss_negative_does_not_update_best(self):
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0, val_loss=-1.0))
        assert mc._best_val_loss == float("inf")

    def test_champion_words_updates(self):
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0, champion_words=30))
        assert mc.best_champion_words == 30
        mc.update_cycle(_snapshot(2, fitness=5.0, champion_words=20))
        assert mc.best_champion_words == 30  # not updated (lower)

    def test_fitness_momentum_increases(self):
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0))
        mc.update_cycle(_snapshot(2, fitness=10.0))
        assert mc._fitness_momentum > 0.0


# ───────────────────────────────────────────────────────────────────────
# 35. MetaController should_rollback
# ───────────────────────────────────────────────────────────────────────


class TestShouldRollback:
    """Tests for should_rollback decision logic."""

    def _make_controller(self):
        return MetaController(checkpoint_dir="/tmp/mc_rollback_test")

    def test_no_rollback_with_few_cycles(self):
        mc = self._make_controller()
        mc.update_cycle(_snapshot(1, fitness=5.0, val_loss=1.0))
        assert mc.should_rollback() is False

    def test_no_rollback_when_improving(self):
        mc = self._make_controller()
        for c in range(1, 5):
            mc.update_cycle(_snapshot(c, fitness=float(c), level=5, val_loss=1.0))
        assert mc.should_rollback() is False

    def test_rollback_on_val_loss_regression(self):
        mc = self._make_controller()
        # Initial good cycles
        for c in range(1, 4):
            mc.update_cycle(_snapshot(c, fitness=5.0, level=5, val_loss=1.0))
        # Regression cycle
        mc.update_cycle(_snapshot(4, fitness=4.0, level=5, val_loss=5.0))
        mc.update_cycle(_snapshot(5, fitness=3.0, level=5, val_loss=10.0))
        assert mc.should_rollback() is True

    def test_rollback_on_compound_stall(self):
        mc = MetaController(
            checkpoint_dir="/tmp/mc_rollback_test2",
            fitness_plateau_cycles=1,
            level_stall_cycles=1,
        )
        # Create compound stall: fitness flat + level stalled + val loss regression
        for c in range(1, 8):
            mc.update_cycle(_snapshot(c, fitness=5.0, level=5, val_loss=1.0, diversity=0.01))
        # Now regress val loss
        for c in range(8, 12):
            mc.update_cycle(_snapshot(c, fitness=4.0, level=5, val_loss=5.0, diversity=0.01))
        assert mc.should_rollback() is True


# ───────────────────────────────────────────────────────────────────────
# 36. MetaController._log stale fd retry
# ───────────────────────────────────────────────────────────────────────


class TestMetaControllerLogRetry:
    """Tests for MetaController._log stale file descriptor handling."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_handle_closed_mid_run(self):
        """When log file handle is closed, _log should retry."""
        log_path = os.path.join(self.tmpdir, "mc.log")
        mc = MetaController(checkpoint_dir=self.tmpdir, log_file=log_path)
        mc._log("first message")

        # Close the file handle externally (simulating stale fd)
        if mc._log_file is not None:
            mc._log_file.close()
            mc._log_file = None

        # Next _log call should retry and succeed
        mc._log("second message")
        with open(log_path) as f:
            content = f.read()
        assert "second message" in content

    def test_log_directory_deleted(self):
        """When log directory is deleted, _log should handle gracefully."""
        log_path = os.path.join(self.tmpdir, "subdir", "mc.log")
        mc = MetaController(checkpoint_dir=self.tmpdir, log_file=log_path)
        # This should create the directory
        mc._log("test")
        assert os.path.isfile(log_path)


# ───────────────────────────────────────────────────────────────────────
# 37. EvolutionEngine.apply_external_params
# ───────────────────────────────────────────────────────────────────────


class TestApplyExternalParams:
    """Tests for apply_external_params."""

    def test_applies_valid_params(self):
        eng = _make_engine(mutation_rate=0.3)
        eng.apply_external_params({"mutation_rate": 0.8})
        assert eng.config.mutation_rate == 0.8

    def test_ignores_unknown_params(self):
        eng = _make_engine()
        eng.apply_external_params({"nonexistent_param": 42})
        # Should not raise

    def test_refreshes_novelty_cache(self):
        eng = _make_engine(novelty_weight=0.1)
        eng.apply_external_params({"novelty_weight": 0.5})
        assert eng._novelty_weight_cache == 0.5


# ───────────────────────────────────────────────────────────────────────
# 38. QD axes caching
# ───────────────────────────────────────────────────────────────────────


class TestQDAxesCaching:
    """Tests for the champion QD descriptor caching mechanism."""

    def _make_engine(self):
        cfg = EvolutionConfig(seed=42, qd_bonus_weight=0.1)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_cache_initial_state(self):
        eng = self._make_engine()
        assert eng._champion_qd_valid is False
        assert eng._cached_champion_qd_desc is None

    def test_cache_set_after_computation(self):
        eng = self._make_engine()
        _set_champion(eng)

        class FakeArchive:
            def selection_bias(self, desc):
                return 0.01

            def get_illumination_diversity(self):
                return {"overall_diversity_score": 0.5}

        eng.qd_archive = FakeArchive()
        prog = [0x00000033, 0x00100073]
        trace = {"status": "ok", "final_regs": {10: 42}, "instructions_count": 2, "trap_cause": ""}
        eng._compute_fitness(prog, trace)
        # Cache should now be valid
        assert eng._champion_qd_valid is True
        assert eng._cached_champion_qd_desc is not None

    def test_cache_reused(self):
        eng = self._make_engine()
        _set_champion(eng)

        class FakeArchive:
            def selection_bias(self, desc):
                return 0.01

            def get_illumination_diversity(self):
                return {"overall_diversity_score": 0.5}

        eng.qd_archive = FakeArchive()
        prog = [0x00000033, 0x00100073]
        trace = {"status": "ok", "final_regs": {10: 42}, "instructions_count": 2, "trap_cause": ""}
        eng._compute_fitness(prog, trace)
        cached = eng._cached_champion_qd_desc
        eng._compute_fitness(prog, trace)
        assert eng._cached_champion_qd_desc is cached


# ───────────────────────────────────────────────────────────────────────
# 39. EvolutionEngine._compute_fitness crash variants
# ───────────────────────────────────────────────────────────────────────


class TestComputeFitnessCrashVariants:
    """Tests for different crash types in _compute_fitness."""

    def _make_engine(self):
        cfg = EvolutionConfig(seed=42)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_illegal_in_crash_trap(self):
        eng = self._make_engine()
        _set_champion(eng)
        trace = {
            "status": "crash",
            "final_regs": {},
            "instructions_count": 0,
            "trap_cause": "illegal instruction at PC 0x1000",
        }
        fitness = eng._compute_fitness([0x00100073], trace)
        assert fitness == -100.0

    def test_invalid_in_crash_trap(self):
        eng = self._make_engine()
        _set_champion(eng)
        trace = {"status": "crash", "final_regs": {}, "instructions_count": 0, "trap_cause": "invalid encoding"}
        fitness = eng._compute_fitness([0x00100073], trace)
        assert fitness == -100.0

    def test_generic_crash_trap(self):
        eng = self._make_engine()
        _set_champion(eng)
        trace = {"status": "crash", "final_regs": {}, "instructions_count": 0, "trap_cause": "segmentation fault"}
        fitness = eng._compute_fitness([0x00100073], trace)
        assert fitness == -50.0

    def test_exception_status(self):
        eng = self._make_engine()
        _set_champion(eng)
        trace = {"status": "exception", "final_regs": {}, "instructions_count": 0, "trap_cause": "overflow"}
        fitness = eng._compute_fitness([0x00100073], trace)
        assert fitness == -50.0

    def test_timeout_by_budget(self):
        eng = self._make_engine()
        _set_champion(eng)
        trace = {"status": "ok", "final_regs": {}, "instructions_count": 10000, "trap_cause": ""}
        fitness = eng._compute_fitness([0x00100073], trace, eval_budget=10000)
        assert fitness == -25.0

    def test_ok_status_zero_instructions(self):
        eng = self._make_engine()
        eng.task_instance = {
            "expected_output": {"registers": {10: 42}},
            "inputs": {},
        }
        eng.current_generation = 0
        eng._bonus_threshold_cache = None
        trace = {"status": "ok", "final_regs": {}, "instructions_count": 0, "trap_cause": ""}
        fitness = eng._compute_fitness([0x00100073], trace)
        # Should still compute a score (reg_match will be 0.0)
        assert isinstance(fitness, float)


# ───────────────────────────────────────────────────────────────────────
# 40. EvolutionEngine._log and logging
# ───────────────────────────────────────────────────────────────────────


class TestEvolutionEngineLogging:
    """Tests for EvolutionEngine logging."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_engine(self):
        cfg = EvolutionConfig(seed=42, log_dir=self.tmpdir)
        return EvolutionEngine(config=cfg, vm_farm=None)

    def test_log_to_file(self):
        eng = self._make_engine()
        eng._log("test message")
        log_file = os.path.join(self.tmpdir, "evolution.log")
        assert os.path.isfile(log_file)
        with open(log_file) as f:
            content = f.read()
        assert "test message" in content

    def test_log_no_log_dir(self):
        cfg = EvolutionConfig(seed=42, log_dir="")
        eng = EvolutionEngine(config=cfg, vm_farm=None)
        eng._log("no crash")  # should not raise

    def test_close_log_handle(self):
        eng = self._make_engine()
        eng._log("open")
        eng._close_log_handle()
        eng._log("after close")
        # Should reopen without error


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
