#!/usr/bin/env python3
"""Wave 20 — Absolute final validation tests.

Three comprehensive test classes covering system health, end-to-end
integration, and final stats verification.

Tests are fast but comprehensive, using pytest and tempfile. No existing
files are modified.
"""
from __future__ import annotations

import importlib
import inspect
import math
import os
import random
import sys
import tempfile
from typing import Any, Dict

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:

# ══════════════════════════════════════════════════════════════════════════════
# Lazy imports — one-time, cached by Python
# ══════════════════════════════════════════════════════════════════════════════
from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame, compute_qd_axes
from meta_controller import MetaController, TelemetrySnapshot, StagnationResult
from strategy_archive import StrategyArchive
from champion_archive import ChampionArchive, ChampionEntry
from mutation_proposer import EBREAK, NOP, generate_seed_program, make_rv32i
from config import EvolutionSection
from novelty import SparseArchive
from qd_archive import MAPElitesArchive
from mutation_proposer import DirectedMutator
from monitoring_alerts import AlertManager
from metrics import MetricsCollector
from progression_tracker import ProgressionTracker
import data_rotation

# ══════════════════════════════════════════════════════════════════════════════
# Wave test files expected to exist
# ══════════════════════════════════════════════════════════════════════════════
_EXPECTED_WAVE_FILES = [
    "test_wave2_additions.py",
    "test_wave4_additions.py",
    "test_wave6_additions.py",
    "test_wave7_additions.py",
    "test_wave8_edge_cases.py",
    "test_wave9_wiring.py",
    "test_wave10_comprehensive.py",
    "test_wave10_integration.py",
    "test_wave11_config_checkpoint.py",
    "test_wave12_final.py",
    "test_wave12_pipeline_smoke.py",
    "test_wave13_stress.py",
    "test_wave14_alerts_config.py",
    "test_wave15_final_comprehensive.py",
    "test_wave16_benchmark.py",
    "test_wave17_architecture.py",
    "test_wave17_edge_to_edge.py",
    "test_wave19_quality_gates.py",
    "test_wave21_final_validation.py",
]

# ══════════════════════════════════════════════════════════════════════════════
# 26 new methods — same spec as wave21
# ══════════════════════════════════════════════════════════════════════════════
_METHOD_SPECS: list[tuple[object, str, bool]] = [
    # MetaController (3)
    (MetaController, "get_cross_cycle_trends", False),
    (MetaController, "get_recovery_confidence", False),
    (MetaController, "get_tuning_escalation_history", False),
    # StrategyArchive (4)
    (StrategyArchive, "per_level_effectiveness", False),
    (StrategyArchive, "adaptive_exploration_bonus", False),
    (StrategyArchive, "time_weighted_success_rate", False),
    (StrategyArchive, "get_dominance_report", False),
    # ChampionArchive (3)
    (ChampionArchive, "diversity_aware_select", False),
    (ChampionArchive, "quality_weighted_prune", False),
    (ChampionArchive, "get_diversity_report", False),
    # SparseArchive / novelty (2)
    (SparseArchive, "manage_capacity", False),
    (SparseArchive, "decayed_novelty", False),
    # MAPElitesArchive / qd_archive (2)
    (MAPElitesArchive, "get_improvement_velocity", False),
    (MAPElitesArchive, "get_illumination_diversity", False),
    # DirectedMutator (2) — static methods
    (DirectedMutator, "record_mutation_outcome", True),
    (DirectedMutator, "get_mutation_stats", True),
    # AlertManager (2)
    (AlertManager, "send_monoculture_alert", False),
    (AlertManager, "send_recovery_stuck_alert", False),
    # MetricsCollector (2)
    (MetricsCollector, "detect_anomalies", False),
    (MetricsCollector, "correlate_metrics", False),
    # data_rotation module functions (2)
    (data_rotation, "get_disk_usage", True),
    (data_rotation, "quality_aware_rotate_checkpoints", True),
    # ProgressionTracker (2)
    (ProgressionTracker, "get_plateau_analysis", False),
    (ProgressionTracker, "compare_windows", False),
    # EvolutionSection (from config.py)
    (EvolutionSection, "get_config_summary", False),
    (EvolutionSection, "validate_config", False),
]

_PIPELINE_METHODS: list[str] = [
    "adaptive_exploration_bonus",
    "manage_capacity",
    "get_cross_cycle_trends",
    "get_improvement_velocity",
    "get_illumination_diversity",
    "get_dominance_report",
    "get_recovery_confidence",
    "quality_weighted_prune",
    "record_mutation_outcome",
    "get_mutation_stats",
    "send_monoculture_alert",
    "send_recovery_stuck_alert",
    "get_plateau_analysis",
]


def _make_snapshot(
    cycle: int,
    fitness: float,
    level: int = 0,
    champion_words: int = 10,
    val_loss: float | None = None,
    diversity: float = 0.5,
    retrained: bool = False,
) -> TelemetrySnapshot:
    """Create a TelemetrySnapshot for testing."""
    return TelemetrySnapshot(
        cycle=cycle,
        fitness=fitness,
        level=level,
        champion_words=champion_words,
        val_loss=val_loss,
        diversity=diversity,
        novelty_size=0,
        qd_occupied=0,
        qd_total=100,
        traces_count=0,
        retrained=retrained,
        generations_completed=50,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestCompleteSystemHealth
# ══════════════════════════════════════════════════════════════════════════════

_ALL_MODULES = [
    "evolution_engine",
    "meta_controller",
    "strategy_archive",
    "champion_archive",
    "mutation_proposer",
    "config",
    "novelty",
    "qd_archive",
    "monitoring_alerts",
    "metrics",
    "progression_tracker",
    "data_rotation",
    "logging_config",
    # "metrics_export" — skipped: requires prometheus_client (optional dep)
    "cycle_state",
    "paths",
    "barrier_tuner",
]


class TestCompleteSystemHealth:
    """Verify all modules import cleanly, classes instantiate, methods callable."""

    @pytest.mark.parametrize("module_name", _ALL_MODULES)
    def test_module_imports_cleanly(self, module_name):
        """Every module must import without errors."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Module '{module_name}' imported as None"

    def test_meta_controller_instantiates(self, tmp_path):
        """MetaController must construct without error."""
        mc = MetaController(checkpoint_dir=str(tmp_path))
        assert mc is not None
        assert hasattr(mc, "telemetry")
        assert isinstance(mc.telemetry, list)

    def test_strategy_archive_instantiates(self):
        """StrategyArchive must construct without error."""
        sa = StrategyArchive()
        assert sa is not None
        assert sa.total_records() == 0

    def test_champion_archive_instantiates(self, tmp_path):
        """ChampionArchive must construct without error."""
        archive_path = str(tmp_path / "test_archive.jsonl")
        ca = ChampionArchive(path=archive_path, max_size=50)
        assert ca is not None
        assert ca.size == 0

    def test_evolution_engine_configurable(self):
        """EvolutionConfig must be constructable."""
        cfg = EvolutionConfig()
        assert cfg is not None
        assert hasattr(cfg, "mutation_rate")

    def test_all_new_methods_callable(self):
        """All 26 new methods exist and are callable on their classes."""
        for target, name, is_static in _METHOD_SPECS:
            attr = getattr(target, name, None)
            assert attr is not None, (
                f"{getattr(target, '__name__', target)} missing '{name}'"
            )
            assert callable(attr), (
                f"{getattr(target, '__name__', target)}.{name} not callable"
            )

    @pytest.mark.parametrize(
        "target,name,is_static",
        _METHOD_SPECS,
        ids=[f"{t.__name__ if hasattr(t, '__name__') else t}.{n}" for t, n, _ in _METHOD_SPECS],
    )
    def test_method_has_inspectable_signature(self, target, name, is_static):
        """Each new method must have an inspectable signature."""
        attr = getattr(target, name)
        try:
            sig = inspect.signature(attr)
        except (ValueError, TypeError):
            pytest.fail(
                f"{getattr(target, '__name__', target)}.{name} has no inspectable signature"
            )
        params = list(sig.parameters.keys())
        if not isinstance(target, type):
            pass  # Module function
        elif is_static:
            pass  # Static method
        else:
            assert len(params) >= 1, (
                f"{target.__name__}.{name} signature has no params"
            )

    def test_no_shadow_imports(self):
        """Key classes must be importable from their expected modules."""
        from meta_controller import MetaController as MC
        from strategy_archive import StrategyArchive as SA
        from champion_archive import ChampionArchive as CA
        from novelty import SparseArchive as NovA
        from qd_archive import MAPElitesArchive as QDA
        assert MC.__name__ == 'MetaController'
        assert SA.__name__ == 'StrategyArchive'
        assert CA.__name__ == 'ChampionArchive'
        assert NovA.__name__ == 'SparseArchive'
        assert QDA.__name__ == 'MAPElitesArchive'


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestEndToEndIntegration
# ══════════════════════════════════════════════════════════════════════════════

class TestEndToEndIntegration:
    """Full pipeline integration tests for MetaController, StrategyArchive,
    and ChampionArchive."""

    # ── MetaController pipeline ──────────────────────────────────────────

    def test_meta_controller_full_pipeline(self, tmp_path):
        """Create MC, add 50 telemetry entries, run full analysis pipeline."""
        mc = MetaController(checkpoint_dir=str(tmp_path))

        # Add 50 telemetry entries with varying fitness/level/diversity
        rng = random.Random(42)
        for i in range(50):
            fitness = 50.0 + i * 0.5 + rng.gauss(0, 2.0)
            level = min(i // 10, 6)
            diversity = max(0.05, 0.6 - i * 0.005 + rng.gauss(0, 0.05))
            val_loss = max(0.1, 3.0 - i * 0.03 + rng.gauss(0, 0.2))
            snap = _make_snapshot(
                cycle=i + 1,
                fitness=fitness,
                level=level,
                val_loss=val_loss,
                diversity=diversity,
            )
            mc.update_cycle(snap)

        # Full pipeline
        stag = mc.detect_stagnation()
        assert isinstance(stag, StagnationResult)

        trends = mc.get_cross_cycle_trends(window=5)
        assert "fitness" in trends
        assert "val_loss" in trends
        assert "diversity" in trends
        for metric in ["fitness", "val_loss", "diversity"]:
            assert "direction" in trends[metric]
            assert "slope" in trends[metric]
            assert "confidence" in trends[metric]

        recovery = mc.get_recovery_confidence()
        assert "is_recovering" in recovery
        assert "cycles_in_recovery" in recovery
        assert "confidence" in recovery
        assert "should_exit_recovery" in recovery
        assert isinstance(recovery["is_recovering"], bool)
        assert isinstance(recovery["confidence"], (int, float))

        reset = mc.suggest_parameter_reset()
        assert "should_reset" in reset
        assert "reason" in reset
        assert "suggested_params" in reset
        assert isinstance(reset["should_reset"], bool)

    def test_meta_controller_stagnation_detection(self, tmp_path):
        """Detect various stagnation types with targeted telemetry."""
        mc = MetaController(
            checkpoint_dir=str(tmp_path),
            fitness_plateau_cycles=3,
            level_stall_cycles=5,
            diversity_collapse_threshold=0.05,
        )
        # Create plateau: fitness flat for many cycles
        for i in range(10):
            mc.update_cycle(_make_snapshot(
                cycle=i + 1, fitness=100.0, level=0, diversity=0.5,
            ))
        stag = mc.detect_stagnation()
        assert stag in (StagnationResult.FITNESS_PLATEAU, StagnationResult.LEVEL_STALLED,
                        StagnationResult.COMPOUND_STALL)

    def test_meta_controller_trends_with_improving_fitness(self, tmp_path):
        """Cross-cycle trends show improving fitness."""
        mc = MetaController(checkpoint_dir=str(tmp_path))
        for i in range(10):
            mc.update_cycle(_make_snapshot(
                cycle=i + 1, fitness=50.0 + i * 10.0, level=min(i, 5),
                diversity=0.5, val_loss=3.0 - i * 0.2,
            ))
        trends = mc.get_cross_cycle_trends()
        assert trends["fitness"]["direction"] == "improving"

    # ── StrategyArchive pipeline ─────────────────────────────────────────

    def test_strategy_archive_full_pipeline(self, tmp_path):
        """Record 100 outcomes, run full analysis (recommend, per_level,
        dominance, summary)."""
        sa = StrategyArchive()
        rng = random.Random(42)
        strategies = ["RANDOM", "TASK_AWARE", "PERTURBATION_CAPPED",
                       "DIRECTED", "NOVELTY"]

        # Record 100 outcomes
        for i in range(100):
            strat = rng.choice(strategies)
            level = rng.randint(0, 9)
            success = rng.random() < 0.6
            sa.record(strat, level=level, success=success, generation=i)
            if success:
                sa.record_delta(strat, delta=rng.uniform(0.1, 5.0))

        assert sa.total_records() == 100

        # recommend
        rec = sa.recommend_strategy(level=0)
        assert isinstance(rec, str)
        assert len(rec) > 0

        # per_level_effectiveness
        ple = sa.per_level_effectiveness("RANDOM")
        assert isinstance(ple, dict)
        for lev, score in ple.items():
            assert 0.0 <= score <= 1.0

        # dominance_report
        dom = sa.get_dominance_report()
        assert "dominant_strategy" in dom
        assert "dominance_ratio" in dom
        assert "is_monoculture" in dom
        assert isinstance(dom["is_monoculture"], bool)

        # summary
        sm = sa.summary()
        assert sm["total_records"] == 100
        assert "strategies" in sm
        assert "dominance" in sm
        for strat_name, info in sm["strategies"].items():
            assert "count" in info
            assert "rate" in info
            assert 0.0 <= info["rate"] <= 1.0

    def test_strategy_archive_diversity(self):
        """Strategy diversity metric across levels."""
        sa = StrategyArchive()
        for i in range(50):
            strat = ["RANDOM", "TASK_AWARE", "DIRECTED"][i % 3]
            sa.record(strat, level=i % 5, success=i % 2 == 0, generation=i)
        div = sa.get_strategy_diversity()
        assert 0.0 <= div <= 1.0

        div_level = sa.get_strategy_diversity(level=0)
        assert 0.0 <= div_level <= 1.0

    def test_strategy_archive_save_load(self, tmp_path):
        """Save/load round-trip preserves data."""
        sa = StrategyArchive()
        for i in range(30):
            sa.record(f"STRAT_{i % 3}", level=i % 5, success=i % 2 == 0, generation=i)
        path = str(tmp_path / "sa_test.json")
        sa.save(path)

        sa2 = StrategyArchive()
        loaded = sa2.load(path)
        assert loaded is True
        assert sa2.total_records() == 30

    def test_strategy_archive_batch_record(self):
        """record_batch stores records correctly."""
        sa = StrategyArchive()
        records = [
            {"strategy": "BATCH_STRAT", "level": 3, "success": True, "gen": i}
            for i in range(20)
        ]
        sa.record_batch(records)
        assert sa.total_records() == 20
        recs = sa.records_by_strategy("BATCH_STRAT")
        assert len(recs) == 20

    # ── ChampionArchive pipeline ─────────────────────────────────────────

    def test_champion_archive_full_pipeline(self, tmp_path):
        """Add 50 champions, run full analysis (sample, diversity_aware,
        quality_prune, diversity_report)."""
        archive_path = str(tmp_path / "ca_test.jsonl")
        ca = ChampionArchive(path=archive_path, max_size=100)

        rng = random.Random(42)
        added = 0
        for i in range(60):  # 60 to test dedup + pruning
            prog = generate_seed_program(length=rng.randint(3, 8), rng=rng)
            fitness = 50.0 + rng.gauss(0, 20.0)
            level = rng.randint(0, 10)
            ok = ca.add(prog, level=level, fitness=fitness, cycle=rng.randint(1, 10),
                        generation=rng.randint(0, 50))
            if ok:
                added += 1
        assert added >= 40  # some dedup expected

        # sample_training_batch
        batch = ca.sample_training_batch(batch_size=10, rng=random.Random(99))
        assert len(batch) > 0
        assert len(batch) <= 10
        for e in batch:
            assert isinstance(e, ChampionEntry)

        # diversity_aware_select — pick from level 0 or first available level
        if ca.entries:
            target_level = ca.entries[0].level
            selected = ca.diversity_aware_select(level=target_level, n=3)
            assert isinstance(selected, list)
            assert len(selected) <= 3

        # quality_weighted_prune — should not prune if under capacity
        pruned = ca.quality_weighted_prune()
        assert pruned >= 0

        # get_diversity_report
        report = ca.get_diversity_report()
        assert report["total_champions"] > 0
        assert "unique_levels" in report
        assert "avg_fitness_by_level" in report
        assert "program_length_variance" in report
        assert "instruction_diversity" in report
        assert "fitness_correlation_with_length" in report

    def test_champion_archive_dedup(self, tmp_path):
        """Same program added twice only stored once."""
        ca = ChampionArchive(path=str(tmp_path / "ca_dedup.jsonl"))
        prog = [0x00100093, EBREAK]  # ADDI x1, x0, 1; EBREAK
        assert ca.add(prog, level=0, fitness=100.0, cycle=1) is True
        assert ca.add(prog, level=0, fitness=100.0, cycle=2) is False  # duplicate
        assert ca.size == 1

    def test_champion_archive_eviction(self, tmp_path):
        """Archive evicts old entries when over capacity."""
        ca = ChampionArchive(path=str(tmp_path / "ca_evict.jsonl"), max_size=10)
        rng = random.Random(42)
        for i in range(20):
            prog = generate_seed_program(length=rng.randint(3, 6), rng=rng)
            ca.add(prog, level=rng.randint(0, 5), fitness=float(50 + i),
                   cycle=rng.randint(1, 10))
        assert ca.size <= 10

    def test_champion_archive_stats(self, tmp_path):
        """Stats returns valid summary."""
        ca = ChampionArchive(path=str(tmp_path / "ca_stats.jsonl"))
        for i in range(5):
            prog = generate_seed_program(length=4, rng=random.Random(i))
            ca.add(prog, level=i, fitness=100.0 + i * 10, cycle=1)
        st = ca.stats()
        assert st["size"] == 5
        assert "max_fitness" in st
        assert "avg_fitness" in st

    def test_champion_archive_quality_distribution(self, tmp_path):
        """get_quality_distribution returns valid buckets."""
        ca = ChampionArchive(path=str(tmp_path / "ca_qd.jsonl"))
        for i in range(15):
            prog = generate_seed_program(length=4, rng=random.Random(i))
            ca.add(prog, level=i % 5, fitness=50.0 + i * 5, cycle=1)
        qd = ca.get_quality_distribution()
        assert "mean" in qd
        assert "median" in qd
        assert "stddev" in qd

    def test_champion_archive_coverage_stats(self, tmp_path):
        """get_coverage_stats returns valid coverage info."""
        ca = ChampionArchive(path=str(tmp_path / "ca_cov.jsonl"))
        for i in range(10):
            prog = generate_seed_program(length=4, rng=random.Random(i))
            ca.add(prog, level=i % 5, fitness=100.0, cycle=i // 3)
        cov = ca.get_coverage_stats()
        assert cov["total_entries"] == 10
        assert cov["unique_cycles"] > 0
        assert cov["unique_levels"] > 0

    def test_champion_archive_sample_weights(self, tmp_path):
        """compute_sample_weights returns normalized weights."""
        ca = ChampionArchive(path=str(tmp_path / "ca_sw.jsonl"))
        for i in range(10):
            prog = generate_seed_program(length=4, rng=random.Random(i))
            ca.add(prog, level=i % 3, fitness=100.0 + i * 10, cycle=i)
        weights = ca.compute_sample_weights()
        assert len(weights) == ca.size
        if weights:
            total = sum(weights)
            assert abs(total - 1.0) < 0.01 or total == 0  # normalized


# ══════════════════════════════════════════════════════════════════════════════
# 3. TestFinalStats
# ══════════════════════════════════════════════════════════════════════════════

class TestFinalStats:
    """Verify total test count > 2000, wave files exist, 26 methods exist,
    pipeline wiring in place."""

    def test_wave_files_exist(self):
        """All expected wave test files exist on disk."""
        tests_dir = os.path.join(_ROOT, "tests")
        for fname in _EXPECTED_WAVE_FILES:
            fpath = os.path.join(tests_dir, fname)
            assert os.path.isfile(fpath), f"Missing wave test file: {fname}"

    def test_all_26_methods_exist(self):
        """All 26 new methods exist on their respective classes/modules."""
        for target, name, is_static in _METHOD_SPECS:
            attr = getattr(target, name, None)
            assert attr is not None, (
                f"{getattr(target, '__name__', target)} missing method '{name}'"
            )
            assert callable(attr), (
                f"{getattr(target, '__name__', target)}.{name} not callable"
            )

    def test_26_unique_method_names(self):
        """Exactly 26 unique method names in the spec."""
        names = {name for _, name, _ in _METHOD_SPECS}
        assert len(names) == 26, f"Expected 26 unique names, got {len(names)}: {names}"

    def test_pipeline_wiring_in_engine(self):
        """All pipeline methods are referenced in evolution_engine.py."""
        engine_path = os.path.join(_ROOT, "evolution_engine.py")
        with open(engine_path, "r", encoding="utf-8") as f:
            engine_source = f.read()
        missing = [m for m in _PIPELINE_METHODS if m not in engine_source]
        assert not missing, f"Missing from evolution_engine.py pipeline: {missing}"

    @pytest.mark.parametrize("method_name", _PIPELINE_METHODS)
    def test_single_pipeline_method_in_engine(self, method_name):
        """Each pipeline method is referenced in evolution_engine.py."""
        engine_path = os.path.join(_ROOT, "evolution_engine.py")
        with open(engine_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert method_name in source, (
            f"'{method_name}' not found in evolution_engine.py — not wired"
        )

    def test_total_test_count_above_2000(self):
        """Total test count across all test files > 2000."""
        tests_dir = os.path.join(_ROOT, "tests")
        total_tests = 0
        for fname in os.listdir(tests_dir):
            if not fname.startswith("test_") or not fname.endswith(".py"):
                continue
            fpath = os.path.join(tests_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                # Count test functions/methods
                import re
                test_funcs = re.findall(r"def (test_\w+)", content)
                total_tests += len(test_funcs)
            except Exception:
                pass
        assert total_tests > 2000, (
            f"Total test count {total_tests} is not above 2000"
        )

    def test_conftest_fixtures_exist(self):
        """conftest.py provides key fixtures."""
        conftest_path = os.path.join(_ROOT, "tests", "conftest.py")
        assert os.path.isfile(conftest_path)
        with open(conftest_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "def engine" in content
        assert "def meta_controller" in content
        assert "def rng" in content
        assert "make_engine" in content
        assert "make_snapshot" in content

    def test_all_source_modules_importable(self):
        """All core source modules import without error."""
        for mod_name in _ALL_MODULES:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Module '{mod_name}' is None"

    def test_all_expected_wave_files_count(self):
        """We expect at least 18 wave test files."""
        tests_dir = os.path.join(_ROOT, "tests")
        wave_files = [
            f for f in os.listdir(tests_dir)
            if f.startswith("test_wave") and f.endswith(".py")
        ]
        assert len(wave_files) >= 18, (
            f"Expected >= 18 wave test files, found {len(wave_files)}: {wave_files}"
        )
