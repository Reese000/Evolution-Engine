#!/usr/bin/env python3
"""tests/test_wave10_comprehensive.py — Comprehensive edge-case and robustness tests.

Eight test classes covering config defaults, DirectedMutator static methods,
data rotation, logging, paths, checkpoint manager, cycle state, and metrics.
All tests are focused and fast; file-based tests use tempfile.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path for imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ═══════════════════════════════════════════════════════════════════════
# 1. TestEvolutionConfigDefaults
# ═══════════════════════════════════════════════════════════════════════

class TestEvolutionConfigDefaults:
    """Verify EvolutionSection default values and Config creation."""

    def test_default_config_values_exist(self):
        from config import EvolutionSection
        evo = EvolutionSection()
        # Spot-check key defaults
        assert evo.population_size == 50
        assert evo.max_generations == 200
        assert evo.mutation_rate == 0.3
        assert evo.seed == 42
        assert evo.instruction_budget == 10000
        assert evo.eval_budget == 5000
        assert evo.max_level == 30
        assert evo.use_gpu is False

    def test_config_creation_no_args(self):
        from config import Config
        cfg = Config(config_path=None)
        # Should have all four sections
        assert hasattr(cfg, "evolution")
        assert hasattr(cfg, "retrain")
        assert hasattr(cfg, "meta_controller")
        assert hasattr(cfg, "autonomous_runner")
        # Defaults should be intact (max_cycles may be overridden by config file)
        assert cfg.evolution.population_size == 50
        assert cfg.retrain.retrain_lr == 3e-5
        assert cfg.autonomous_runner.target_fitness == 200.0

    def test_config_with_custom_seed(self):
        from config import Config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"evolution": {"seed": 999, "population_size": 10}}, f)
            f.flush()
            cfg = Config(config_path=f.name)
        try:
            assert cfg.evolution.seed == 999
            assert cfg.evolution.population_size == 10
            # Non-overridden defaults should remain
            assert cfg.evolution.max_generations == 200
        finally:
            os.unlink(f.name)

    def test_config_to_dict_roundtrip(self):
        from config import Config
        cfg = Config(config_path=None)
        d = cfg.to_dict()
        assert "evolution" in d
        assert "retrain" in d
        assert isinstance(d["evolution"], dict)
        assert d["evolution"]["seed"] == 42

    def test_config_validate_catches_bad_values(self):
        from config import Config
        cfg = Config(config_path=None)
        cfg.evolution.population_size = -1
        with pytest.raises(ValueError, match="population_size"):
            cfg._validate()


# ═══════════════════════════════════════════════════════════════════════
# 2. TestDirectedMutatorStaticMethods
# ═══════════════════════════════════════════════════════════════════════

class TestDirectedMutatorStaticMethods:
    """Test find_ebreak_index, ensure_ebreak, trim_dead_code, simplify_program."""

    EBREAK = 0x00100073

    def test_find_ebreak_index_no_ebreak(self):
        from mutation_proposer import find_ebreak_index
        program = [0x00000013, 0x00000013, 0x00000013]  # 3x NOP
        assert find_ebreak_index(program) is None

    def test_find_ebreak_index_with_ebreak(self):
        from mutation_proposer import find_ebreak_index
        nop = 0x00000013
        program = [nop, nop, self.EBREAK, nop]
        assert find_ebreak_index(program) == 2

    def test_find_ebreak_index_ebreak_at_end(self):
        from mutation_proposer import find_ebreak_index
        nop = 0x00000013
        program = [nop, nop, self.EBREAK]
        assert find_ebreak_index(program) == 2

    def test_find_ebreak_index_empty(self):
        from mutation_proposer import find_ebreak_index
        assert find_ebreak_index([]) is None

    def test_ensure_ebreak_adds_ebreak(self):
        from mutation_proposer import DirectedMutator
        nop = 0x00000013
        program = [nop, nop]
        result = DirectedMutator.ensure_ebreak(program)
        assert result[-1] == self.EBREAK
        assert len(result) == 3

    def test_ensure_ebreak_already_has_ebreak(self):
        from mutation_proposer import DirectedMutator
        nop = 0x00000013
        program = [nop, self.EBREAK]
        result = DirectedMutator.ensure_ebreak(program)
        assert result[-1] == self.EBREAK
        assert result == [nop, self.EBREAK]

    def test_ensure_ebreak_moves_ebreak_to_end(self):
        from mutation_proposer import DirectedMutator
        nop = 0x00000013
        program = [self.EBREAK, nop, nop]
        result = DirectedMutator.ensure_ebreak(program)
        assert result[-1] == self.EBREAK
        assert result[0] == nop

    def test_ensure_ebreak_empty_program(self):
        from mutation_proposer import DirectedMutator
        result = DirectedMutator.ensure_ebreak([])
        assert result == [self.EBREAK]

    def test_trim_dead_code_removes_after_ebreak(self):
        from mutation_proposer import trim_dead_code
        nop = 0x00000013
        program = [nop, self.EBREAK, nop, nop]
        result = trim_dead_code(program)
        assert result == [nop, self.EBREAK]

    def test_trim_dead_code_no_ebreak(self):
        from mutation_proposer import trim_dead_code
        nop = 0x00000013
        program = [nop, nop]
        result = trim_dead_code(program)
        assert result == [nop, nop]

    def test_trim_dead_code_empty(self):
        from mutation_proposer import trim_dead_code
        assert trim_dead_code([]) == []

    def test_simplify_program_empty(self):
        from mutation_proposer import simplify_program
        result = simplify_program([])
        # R38: empty program returns [NOP, EBREAK]
        assert self.EBREAK in result

    def test_simplify_program_single_nop(self):
        from mutation_proposer import simplify_program
        nop = 0x00000013
        result = simplify_program([nop])
        # Single non-EBREAK should get EBREAK appended
        assert result[-1] == self.EBREAK

    def test_simplify_program_consecutive_nops(self):
        from mutation_proposer import simplify_program
        nop = 0x00000013
        program = [nop, nop, nop, self.EBREAK]
        result = simplify_program(program)
        # Should reduce consecutive NOPs to at most 1
        nop_count = sum(1 for x in result if x == nop)
        assert nop_count <= 1

    def test_simplify_program_only_ebreak(self):
        from mutation_proposer import simplify_program
        result = simplify_program([self.EBREAK])
        assert result == [self.EBREAK]


# ═══════════════════════════════════════════════════════════════════════
# 3. TestDataRotationFunctions
# ═══════════════════════════════════════════════════════════════════════

class TestDataRotationFunctions:
    """Test run_all_rotations, rotate_checkpoints, rotate_reports."""

    def test_run_all_rotations_returns_dict(self):
        from data_rotation import run_all_rotations
        result = run_all_rotations()
        assert isinstance(result, dict)
        assert "checkpoints" in result
        assert "recurrent" in result
        assert "archives" in result
        assert "reports" in result
        assert all(isinstance(v, int) for v in result.values())

    def test_rotate_checkpoints_with_keep_n_zero(self):
        """rotate_checkpoints with keep_n=0 should attempt to delete
        non-sacred checkpoint files (but preserve sacred ones)."""
        from data_rotation import rotate_checkpoints
        with tempfile.TemporaryDirectory() as tmpdir:
            # Monkey-patch M1_CHECKPOINTS to our temp dir
            import data_rotation
            original = data_rotation.M1_CHECKPOINTS
            data_rotation.M1_CHECKPOINTS = Path(tmpdir)
            try:
                # Create some fake checkpoint files
                for i in range(3):
                    (Path(tmpdir) / f"checkpoint_cycle_{i}.pt").touch()
                (Path(tmpdir) / "checkpoint_best.pt").touch()  # sacred
                (Path(tmpdir) / "checkpoint_current.pt").touch()  # sacred
                deleted = rotate_checkpoints(keep_n=0)
                # Non-sacred files should be deleted
                assert len(deleted) == 3
                # Sacred files should still exist
                assert (Path(tmpdir) / "checkpoint_best.pt").exists()
                assert (Path(tmpdir) / "checkpoint_current.pt").exists()
            finally:
                data_rotation.M1_CHECKPOINTS = original

    def test_rotate_reports_with_custom_keep_n(self):
        """rotate_reports with small keep_n should delete old reports."""
        import data_rotation
        from data_rotation import rotate_reports
        with tempfile.TemporaryDirectory() as tmpdir:
            original = data_rotation.PROJECT_ROOT
            data_rotation.PROJECT_ROOT = Path(tmpdir)
            try:
                # Create report files
                for i in range(5):
                    fpath = Path(tmpdir) / f"progress_report_{i}.json"
                    fpath.write_text("{}")
                # Keep only 2
                deleted = rotate_reports(keep_n=2)
                assert len(deleted) == 3
                # Verify remaining files
                remaining = list(Path(tmpdir).glob("progress_report_*.json"))
                assert len(remaining) == 2
            finally:
                data_rotation.PROJECT_ROOT = original


# ═══════════════════════════════════════════════════════════════════════
# 4. TestLoggingConfig
# ═══════════════════════════════════════════════════════════════════════

class TestLoggingConfig:
    """Test setup_logging creates logger and configures handlers."""

    def test_setup_logging_creates_logger(self):
        import logging_config
        from logging_config import get_logger, setup_logging
        logging_config._initialized = False
        tmpdir = tempfile.mkdtemp()
        try:
            setup_logging(log_dir=tmpdir)
            logger = get_logger("TestComponent")
            assert isinstance(logger, logging.Logger)
            # Should have at least console + file handlers
            root = logging.getLogger()
            handler_types = {type(h).__name__ for h in root.handlers}
            assert "StreamHandler" in handler_types or "RotatingFileHandler" in handler_types
        finally:
            # Close all handlers so Windows can release file locks before cleanup
            root = logging.getLogger()
            for h in root.handlers[:]:
                h.close()
                root.removeHandler(h)
            logging_config._initialized = False
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_log_rotation_settings(self):
        """Verify RotatingFileHandler is configured with proper settings."""
        import logging_config
        logging_config._initialized = False
        tmpdir = tempfile.mkdtemp()
        try:
            logging_config.setup_logging(
                log_dir=tmpdir,
                max_bytes=1024 * 1024,
                backup_count=3,
            )
            root = logging.getLogger()
            file_handlers = [
                h for h in root.handlers
                if isinstance(h, logging.handlers.RotatingFileHandler)
            ]
            assert len(file_handlers) >= 1
            handler = file_handlers[0]
            assert handler.maxBytes == 1024 * 1024
            assert handler.backupCount == 3
        finally:
            root = logging.getLogger()
            for h in root.handlers[:]:
                h.close()
                root.removeHandler(h)
            logging_config._initialized = False
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_logger_returns_named_logger(self):
        from logging_config import get_logger
        logger = get_logger("MySpecificName")
        assert logger.name == "MySpecificName"


# ═══════════════════════════════════════════════════════════════════════
# 5. TestPaths
# ═══════════════════════════════════════════════════════════════════════

class TestPaths:
    """Test centralized path constants."""

    def test_m1_checkpoints_is_path(self):
        from paths import M1_CHECKPOINTS
        assert isinstance(M1_CHECKPOINTS, Path)

    def test_m1_recurrent_is_path(self):
        from paths import M1_RECURRENT
        assert isinstance(M1_RECURRENT, Path)

    def test_project_root_exists(self):
        from paths import PROJECT_ROOT
        assert isinstance(PROJECT_ROOT, Path)
        assert PROJECT_ROOT.exists()

    def test_project_root_contains_config_dir(self):
        from paths import PROJECT_ROOT
        assert (PROJECT_ROOT / "config").is_dir()

    def test_project_root_contains_tests_dir(self):
        from paths import PROJECT_ROOT
        assert (PROJECT_ROOT / "tests").is_dir()


# ═══════════════════════════════════════════════════════════════════════
# 6. TestCheckpointManager
# ═══════════════════════════════════════════════════════════════════════

class TestCheckpointManager:
    """Test CheckpointManager save/load and missing file handling."""

    def test_checkpoint_exists_returns_false_for_missing(self):
        from checkpoint_manager import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            assert mgr.exists("nonexistent") is False

    def test_checkpoint_exists_returns_true_after_save(self):
        """Save a raw checkpoint dict (bypassing torch) and verify exists()."""
        import torch

        from checkpoint_manager import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            # Manually create a checkpoint file
            ckpt = {
                "config_name": "small",
                "config": {},
                "model_state_dict": {"weight": torch.randn(3, 3)},
                "meta": {"val_loss": 0.5, "cycle": 1},
            }
            path = os.path.join(tmpdir, "checkpoint_best.pt")
            torch.save(ckpt, path)
            assert mgr.exists("checkpoint_best") is True

    def test_checkpoint_load_missing_raises(self):
        from checkpoint_manager import CheckpointManager
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(base_dir=tmpdir)
            with pytest.raises(FileNotFoundError):
                mgr.load("nonexistent")

    def test_checkpoint_meta_dataclass(self):
        from checkpoint_manager import CheckpointMeta
        meta = CheckpointMeta(
            config_name="small",
            config={"layers": 4},
            val_loss=0.123,
            cycle=10,
            timestamp=1000.0,
        )
        assert meta.config_name == "small"
        assert meta.val_loss == 0.123
        assert meta.cycle == 10


# ═══════════════════════════════════════════════════════════════════════
# 7. TestCycleState
# ═══════════════════════════════════════════════════════════════════════

class TestCycleState:
    """Test CycleState creation, save/load, and transitions."""

    def test_cycle_state_creation_defaults(self):
        from cycle_state import CycleState
        cs = CycleState.__new__(CycleState)
        from cycle_state import default_cycle_state
        cs.data = default_cycle_state()
        assert cs.data["version"] == 2
        assert cs.data["current_cycle"] == 0
        assert cs.data["best_fitness"] == 0.0
        assert cs.data["cycles_completed"] == 0

    def test_cycle_state_save_and_load(self):
        from cycle_state import CycleState
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cycle_state.json")
            cs = CycleState(path)
            cs.data["current_cycle"] = 5
            cs.data["best_fitness"] = 0.75
            cs.save()

            # Load into a fresh CycleState
            cs2 = CycleState(path)
            success = cs2.load()
            assert success is True
            assert cs2.data["current_cycle"] == 5
            assert cs2.data["best_fitness"] == 0.75

    def test_cycle_state_load_missing_file(self):
        from cycle_state import CycleState
        with tempfile.TemporaryDirectory() as tmpdir:
            cs = CycleState(os.path.join(tmpdir, "nonexistent.json"))
            success = cs.load()
            assert success is False
            # Data should be reset to defaults
            assert cs.data["current_cycle"] == 0

    def test_cycle_state_transitions(self):
        from cycle_state import CycleState
        with tempfile.TemporaryDirectory() as tmpdir:
            cs = CycleState(os.path.join(tmpdir, "cs.json"))
            # Simulate cycle transitions
            cs.update(current_cycle=1, cycles_completed=1, best_fitness=0.3)
            assert cs["current_cycle"] == 1
            cs.update(current_cycle=2, cycles_completed=2, best_fitness=0.5)
            assert cs["current_cycle"] == 2
            assert cs.data["best_fitness"] == 0.5
            # Save and reload
            cs.save()
            cs2 = CycleState(os.path.join(tmpdir, "cs.json"))
            cs2.load()
            assert cs2["current_cycle"] == 2

    def test_cycle_state_checksum_integrity(self):
        """Corrupted file should fail to load."""
        from cycle_state import CycleState
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cs.json")
            cs = CycleState(path)
            cs.save()
            # Corrupt the file
            with open(path, "r") as f:
                data = json.load(f)
            data["best_fitness"] = 999.0
            with open(path, "w") as f:
                json.dump(data, f)
            cs2 = CycleState(path)
            success = cs2.load()
            assert success is False

    def test_cycle_state_reset(self):
        from cycle_state import CycleState
        cs = CycleState.__new__(CycleState)
        from cycle_state import default_cycle_state
        cs.path = "dummy"
        cs.data = default_cycle_state()
        cs.data["current_cycle"] = 10
        cs.data["start_timestamp"] = "2025-01-01T00:00:00"
        cs.reset()
        assert cs.data["current_cycle"] == 0
        # start_timestamp should be preserved
        assert cs.data["start_timestamp"] == "2025-01-01T00:00:00"


# ═══════════════════════════════════════════════════════════════════════
# 8. TestMetricsEdgeCases
# ═══════════════════════════════════════════════════════════════════════

class TestMetricsEdgeCases:
    """Test MetricsCollector edge cases: empty files, insufficient data."""

    def test_record_cycle_adds_timestamp(self):
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            mc.record_cycle({"cycle": 1, "fitness": 0.5})
            history = mc.get_history()
            assert len(history) == 1
            assert "timestamp" in history[0]
            assert history[0]["cycle"] == 1

    def test_record_cycle_preserves_existing_timestamp(self):
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            mc.record_cycle({"cycle": 1, "fitness": 0.5, "timestamp": "2025-01-01"})
            history = mc.get_history()
            assert history[0]["timestamp"] == "2025-01-01"

    def test_get_history_no_file(self):
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.jsonl")
            mc = MetricsCollector(path)
            history = mc.get_history()
            assert history == []

    def test_get_history_last_n(self):
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            for i in range(10):
                mc.record_cycle({"cycle": i, "fitness": float(i)})
            history = mc.get_history(last_n=3)
            assert len(history) == 3
            assert history[0]["cycle"] == 7

    def test_get_trend_with_insufficient_data(self):
        """With 0 records, trend should return 'unknown'."""
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            trend = mc.get_trend("fitness")
            assert trend["trend"] == "unknown"
            assert trend["window_size"] == 0
            assert trend["current"] is None

    def test_get_trend_single_record(self):
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            mc.record_cycle({"cycle": 1, "fitness": 0.5})
            trend = mc.get_trend("fitness")
            assert trend["trend"] == "stable"
            assert trend["current"] == 0.5
            assert trend["window_size"] == 1

    def test_get_trend_improving(self):
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            for i in range(10):
                mc.record_cycle({"cycle": i, "fitness": float(i) * 0.1})
            trend = mc.get_trend("fitness")
            assert trend["trend"] == "improving"
            assert trend["current"] > trend["avg"] or trend["current"] >= 0.0

    def test_count_records_empty_file(self):
        """count_records on empty/missing file should return 0."""
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.jsonl")
            mc = MetricsCollector(path)
            assert mc.count_records() == 0

    def test_count_records_after_writes(self):
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            for i in range(5):
                mc.record_cycle({"cycle": i})
            assert mc.count_records() == 5

    def test_clear_removes_file(self):
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            mc.record_cycle({"cycle": 1})
            assert mc.count_records() == 1
            mc.clear()
            assert mc.count_records() == 0
            assert not os.path.exists(path)

    def test_correlate_metrics_empty(self):
        from metrics import MetricsCollector
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metrics.jsonl")
            mc = MetricsCollector(path)
            result = mc.correlate_metrics("fitness", "level")
            assert result["sample_size"] == 0
            assert result["strength"] == "none"
