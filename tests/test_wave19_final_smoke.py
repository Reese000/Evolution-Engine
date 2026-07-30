"""Wave 19 Final Smoke Tests — Method inventory, pipeline integrity, suite health.

Fast smoke tests that verify:
  1. Every new method exists and is callable
  2. Core pipeline classes can be instantiated
  3. Test suite is healthy (count, files, syntax)
"""
from __future__ import annotations

import ast
import importlib
import os
import sys
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Path setup — mirrors conftest.py
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestCompleteMethodInventory
# ═══════════════════════════════════════════════════════════════════════════

class TestCompleteMethodInventory:
    """Verify every new method exists and is callable on its owning class/object."""

    # --- MetaController methods ---

    def test_get_cross_cycle_trends(self):
        from meta_controller import MetaController
        assert callable(getattr(MetaController, "get_cross_cycle_trends", None))

    def test_get_recovery_confidence(self):
        from meta_controller import MetaController
        assert callable(getattr(MetaController, "get_recovery_confidence", None))

    def test_get_tuning_escalation_history(self):
        from meta_controller import MetaController
        assert callable(getattr(MetaController, "get_tuning_escalation_history", None))

    # --- StrategyArchive methods ---

    def test_per_level_effectiveness(self):
        from strategy_archive import StrategyArchive
        assert callable(getattr(StrategyArchive, "per_level_effectiveness", None))

    def test_adaptive_exploration_bonus(self):
        from strategy_archive import StrategyArchive
        assert callable(getattr(StrategyArchive, "adaptive_exploration_bonus", None))

    def test_time_weighted_success_rate(self):
        from strategy_archive import StrategyArchive
        assert callable(getattr(StrategyArchive, "time_weighted_success_rate", None))

    def test_get_dominance_report(self):
        from strategy_archive import StrategyArchive
        assert callable(getattr(StrategyArchive, "get_dominance_report", None))

    # --- ChampionArchive methods ---

    def test_diversity_aware_select(self):
        from champion_archive import ChampionArchive
        assert callable(getattr(ChampionArchive, "diversity_aware_select", None))

    def test_quality_weighted_prune(self):
        from champion_archive import ChampionArchive
        assert callable(getattr(ChampionArchive, "quality_weighted_prune", None))

    def test_get_diversity_report(self):
        from champion_archive import ChampionArchive
        assert callable(getattr(ChampionArchive, "get_diversity_report", None))

    # --- SparseArchive (novelty) methods ---

    def test_manage_capacity(self):
        from novelty import SparseArchive
        assert callable(getattr(SparseArchive, "manage_capacity", None))

    def test_decayed_novelty(self):
        from novelty import SparseArchive
        assert callable(getattr(SparseArchive, "decayed_novelty", None))

    # --- MAPElitesArchive (qd_archive) methods ---

    def test_get_improvement_velocity(self):
        from qd_archive import MAPElitesArchive
        assert callable(getattr(MAPElitesArchive, "get_improvement_velocity", None))

    def test_get_illumination_diversity(self):
        from qd_archive import MAPElitesArchive
        assert callable(getattr(MAPElitesArchive, "get_illumination_diversity", None))

    # --- DirectedMutator static methods ---

    def test_record_mutation_outcome(self):
        from mutation_proposer import DirectedMutator
        assert callable(getattr(DirectedMutator, "record_mutation_outcome", None))

    def test_get_mutation_stats(self):
        from mutation_proposer import DirectedMutator
        assert callable(getattr(DirectedMutator, "get_mutation_stats", None))

    # --- AlertManager methods ---

    def test_send_monoculture_alert(self):
        from monitoring_alerts import AlertManager
        assert callable(getattr(AlertManager, "send_monoculture_alert", None))

    def test_send_recovery_stuck_alert(self):
        from monitoring_alerts import AlertManager
        assert callable(getattr(AlertManager, "send_recovery_stuck_alert", None))

    # --- MetricsCollector methods ---

    def test_detect_anomalies(self):
        from metrics import MetricsCollector
        assert callable(getattr(MetricsCollector, "detect_anomalies", None))

    def test_correlate_metrics(self):
        from metrics import MetricsCollector
        assert callable(getattr(MetricsCollector, "correlate_metrics", None))

    # --- data_rotation module-level functions ---

    def test_get_disk_usage(self):
        import data_rotation
        assert callable(getattr(data_rotation, "get_disk_usage", None))

    def test_quality_aware_rotate_checkpoints(self):
        import data_rotation
        assert callable(getattr(data_rotation, "quality_aware_rotate_checkpoints", None))

    # --- ProgressionTracker methods ---

    def test_get_plateau_analysis(self):
        from progression_tracker import ProgressionTracker
        assert callable(getattr(ProgressionTracker, "get_plateau_analysis", None))

    def test_compare_windows(self):
        from progression_tracker import ProgressionTracker
        assert callable(getattr(ProgressionTracker, "compare_windows", None))

    # --- EvolutionSection (config.py) methods ---

    def test_get_config_summary(self):
        from config import EvolutionSection
        assert callable(getattr(EvolutionSection, "get_config_summary", None))

    def test_validate_config(self):
        from config import EvolutionSection
        assert callable(getattr(EvolutionSection, "validate_config", None))


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestPipelineIntegrity
# ═══════════════════════════════════════════════════════════════════════════

class TestPipelineIntegrity:
    """Verify imports, class instantiation, and module-level health."""

    def test_evolution_engine_imports(self):
        """evolution_engine.py imports without error."""
        mod = importlib.import_module("evolution_engine")
        assert hasattr(mod, "EvolutionEngine")
        assert hasattr(mod, "EvolutionConfig")

    def test_evolution_config_instantiation(self):
        from evolution_engine import EvolutionConfig
        cfg = EvolutionConfig()
        assert cfg is not None
        assert isinstance(cfg.mutation_rate, float)

    def test_meta_controller_instantiation(self):
        import tempfile
        from meta_controller import MetaController
        with tempfile.TemporaryDirectory() as td:
            mc = MetaController(checkpoint_dir=td)
            assert mc is not None

    def test_strategy_archive_instantiation(self):
        from strategy_archive import StrategyArchive
        sa = StrategyArchive()
        assert sa is not None

    def test_champion_archive_instantiation(self):
        from champion_archive import ChampionArchive
        ca = ChampionArchive()
        assert ca is not None

    def test_novelty_archive_instantiation(self):
        from novelty import SparseArchive
        sa = SparseArchive()
        assert sa is not None

    def test_qd_archive_instantiation(self):
        from qd_archive import MAPElitesArchive
        qa = MAPElitesArchive()
        assert qa is not None

    def test_progression_tracker_instantiation(self):
        from progression_tracker import ProgressionTracker
        pt = ProgressionTracker()
        assert pt is not None

    def test_monitoring_alerts_instantiation(self):
        from monitoring_alerts import AlertManager
        am = AlertManager()
        assert am is not None

    def test_metrics_collector_instantiation(self):
        from metrics import MetricsCollector
        mc = MetricsCollector()
        assert mc is not None

    def test_config_instantiation(self):
        from config import Config
        cfg = Config(config_path=None)
        assert cfg is not None
        assert hasattr(cfg, "evolution")

    def test_no_import_errors_core_modules(self):
        """All core modules import without error."""
        core_modules = [
            "evolution_engine",
            "meta_controller",
            "mutation_proposer",
            "strategy_archive",
            "champion_archive",
            "novelty",
            "qd_archive",
            "progression_tracker",
            "monitoring_alerts",
            "metrics",
            "config",
            "data_rotation",
            "logging_config",
            "paths",
            "checkpoint_manager",
            "cycle_state",
            "champion_anchor",
        ]
        for mod_name in core_modules:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Module '{mod_name}' failed to import"


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestTestSuiteHealth
# ═══════════════════════════════════════════════════════════════════════════

# All wave test files that should exist
WAVE_TEST_FILES = [
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
]


class TestTestSuiteHealth:
    """Verify the overall test suite is healthy."""

    @pytest.mark.skipif(
        os.environ.get("CI_SMOKE_ONLY") == "1",
        reason="Slow — collection may time out in CI smoke mode",
    )
    def test_test_count_above_2000(self):
        """Total test count should be above 2000."""
        tests_dir = os.path.join(_ROOT, "tests")
        count = 0
        for fname in os.listdir(tests_dir):
            if fname.startswith("test_") and fname.endswith(".py"):
                fpath = os.path.join(tests_dir, fname)
                try:
                    with open(fpath, encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=fpath)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if node.name.startswith("test_"):
                                count += 1
                except SyntaxError:
                    pass  # syntax errors caught separately
        assert count > 2000, f"Expected >2000 test functions, found {count}"

    def test_wave_files_exist(self):
        """All expected wave test files exist on disk."""
        tests_dir = os.path.join(_ROOT, "tests")
        missing = [f for f in WAVE_TEST_FILES if not os.path.isfile(os.path.join(tests_dir, f))]
        assert not missing, f"Missing wave test files: {missing}"

    def test_wave_files_parseable(self):
        """All wave test files have valid Python syntax."""
        tests_dir = os.path.join(_ROOT, "tests")
        errors = []
        for fname in WAVE_TEST_FILES:
            fpath = os.path.join(tests_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, encoding="utf-8") as f:
                    ast.parse(f.read(), filename=fpath)
            except SyntaxError as e:
                errors.append(f"{fname}: {e}")
        assert not errors, f"Syntax errors in wave test files:\n" + "\n".join(errors)

    def test_all_test_files_parseable(self):
        """Every test_*.py file in tests/ has valid Python syntax."""
        tests_dir = os.path.join(_ROOT, "tests")
        errors = []
        for fname in sorted(os.listdir(tests_dir)):
            if not fname.startswith("test_") or not fname.endswith(".py"):
                continue
            fpath = os.path.join(tests_dir, fname)
            try:
                with open(fpath, encoding="utf-8") as f:
                    ast.parse(f.read(), filename=fpath)
            except SyntaxError as e:
                errors.append(f"{fname}: {e}")
        assert not errors, f"Syntax errors in test files:\n" + "\n".join(errors)
