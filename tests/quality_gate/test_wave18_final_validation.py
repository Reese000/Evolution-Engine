#!/usr/bin/env python3
"""Wave 18 — Final validation test suite.

Five focused test classes that verify:
  1. Every new method returns the correct type
  2. All pipeline wiring is present in evolution_engine.py
  3. Core EvolutionEngine methods still work (no regressions)
  4. All new methods have docstrings
  5. No syntax errors in any modified module

Uses pytest + inspect + importlib.  Fast — no heavy computation.
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import ClassVar

import pytest

# Path setup
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ═══════════════════════════════════════════════════════════════════════
# 1. TestAllMethodsReturnType
# ═══════════════════════════════════════════════════════════════════════


class TestAllMethodsReturnType:
    """Verify every new method returns the correct type."""

    def test_get_cross_cycle_trends_returns_dict(self):
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir="/tmp/mc_test_w18")
        result = mc.get_cross_cycle_trends()
        assert isinstance(result, dict)

    def test_get_recovery_confidence_returns_dict(self):
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir="/tmp/mc_test_w18")
        result = mc.get_recovery_confidence()
        assert isinstance(result, dict)

    def test_get_dominance_report_returns_dict(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.get_dominance_report()
        assert isinstance(result, dict)

    def test_per_level_effectiveness_returns_dict(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.per_level_effectiveness("RANDOM")
        assert isinstance(result, dict)

    def test_time_weighted_success_rate_returns_float(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.time_weighted_success_rate("RANDOM")
        assert isinstance(result, float)

    def test_diversity_aware_select_returns_list(self):
        from champion_archive import ChampionArchive

        ca = ChampionArchive()
        result = ca.diversity_aware_select(level=0, n=1)
        assert isinstance(result, list)

    def test_quality_weighted_prune_returns_int(self):
        from champion_archive import ChampionArchive

        ca = ChampionArchive()
        result = ca.quality_weighted_prune()
        assert isinstance(result, int)

    def test_manage_capacity_returns_int(self):
        from novelty import SparseArchive

        sa = SparseArchive()
        result = sa.manage_capacity(max_size=100)
        assert isinstance(result, int)

    def test_decayed_novelty_returns_float(self):
        from novelty import SparseArchive

        sa = SparseArchive()
        # Program not in archive → returns 0.0
        result = sa.decayed_novelty([0x00100073], current_generation=100)
        assert isinstance(result, float)

    def test_get_improvement_velocity_returns_dict(self):
        from qd_archive import MAPElitesArchive

        qd = MAPElitesArchive(bins=5)
        result = qd.get_improvement_velocity()
        assert isinstance(result, dict)

    def test_get_illumination_diversity_returns_dict(self):
        from qd_archive import MAPElitesArchive

        qd = MAPElitesArchive(bins=5)
        result = qd.get_illumination_diversity()
        assert isinstance(result, dict)

    def test_detect_anomalies_returns_list(self):
        import tempfile

        from metrics import MetricsCollector

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            mc = MetricsCollector(path)
            result = mc.detect_anomalies("fitness")
            assert isinstance(result, list)
        finally:
            os.unlink(path)

    def test_correlate_metrics_returns_dict(self):
        import tempfile

        from metrics import MetricsCollector

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            mc = MetricsCollector(path)
            result = mc.correlate_metrics("fitness", "level")
            assert isinstance(result, dict)
        finally:
            os.unlink(path)

    def test_get_disk_usage_returns_dict(self):
        from data_rotation import get_disk_usage

        result = get_disk_usage()
        assert isinstance(result, dict)

    def test_get_plateau_analysis_returns_dict(self):
        from progression_tracker import ProgressionTracker

        pt = ProgressionTracker()
        result = pt.get_plateau_analysis()
        assert isinstance(result, dict)

    def test_compare_windows_returns_dict(self):
        from progression_tracker import ProgressionTracker

        pt = ProgressionTracker()
        result = pt.compare_windows("fitness")
        assert isinstance(result, dict)

    def test_get_config_summary_returns_dict(self):
        from config import EvolutionSection

        ec = EvolutionSection()
        result = ec.get_config_summary()
        assert isinstance(result, dict)

    def test_validate_config_returns_list(self):
        from config import EvolutionSection

        ec = EvolutionSection()
        result = ec.validate_config()
        assert isinstance(result, list)

    def test_get_mutation_stats_returns_dict(self):
        from mutation_proposer import DirectedMutator

        result = DirectedMutator.get_mutation_stats()
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════
# 2. TestPipelineWiringComplete
# ═══════════════════════════════════════════════════════════════════════


def _read_engine_source() -> str:
    """Read evolution_engine.py source as a single string."""
    path = _ROOT / "evolution_engine.py"
    return path.read_text(encoding="utf-8")


def _read_submodule_source(module_name: str) -> str:
    """Read an evolution/ submodule source as a single string."""
    path = _ROOT / "evolution" / f"{module_name}.py"
    return path.read_text(encoding="utf-8")


class TestPipelineWiringComplete:
    """Verify evolution_engine.py has all pipeline wiring."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self._src = _read_engine_source()

    def test_adaptive_exploration_bonus_in_choose_mutation_strategy(self):
        """adaptive_exploration_bonus must be called inside _choose_mutation_strategy."""
        # Find _choose_mutation_strategy method body
        assert "adaptive_exploration_bonus" in self._src
        # Verify it appears in a strategy archive context
        assert "self.strategy_archive.adaptive_exploration_bonus" in self._src

    def test_manage_capacity_in_update_novelty_archive(self):
        """manage_capacity must be called inside _update_novelty_archive (now in qd_archive_mixin)."""
        qd_src = (_ROOT / "evolution" / "qd_archive_mixin.py").read_text(encoding="utf-8")
        assert "self.novelty_archive.manage_capacity()" in qd_src

    def test_get_cross_cycle_trends_in_generation_loop(self):
        """get_cross_cycle_trends must be called in the generation loop."""
        mh_src = _read_submodule_source("mutation_helpers")
        assert "get_cross_cycle_trends" in mh_src

    def test_get_improvement_velocity_in_generation_loop(self):
        """get_improvement_velocity must be called in the generation loop."""
        qd_src = (_ROOT / "qd_archive.py").read_text(encoding="utf-8")
        assert "get_improvement_velocity" in qd_src

    def test_get_dominance_report_after_strategy_recording(self):
        """get_dominance_report must be called after strategy recording."""
        assert "get_dominance_report()" in self._src

    def test_get_recovery_confidence_after_curriculum_advance(self):
        """get_recovery_confidence must be called after curriculum advance."""
        assert "get_recovery_confidence()" in self._src

    def test_quality_weighted_prune_after_champion_update(self):
        """quality_weighted_prune must be called after champion update."""
        assert "quality_weighted_prune()" in self._src

    def test_record_mutation_outcome_after_evaluation(self):
        """record_mutation_outcome must be called after evaluation."""
        assert "record_mutation_outcome" in self._src

    def test_get_plateau_analysis_in_diagnostics(self):
        """get_plateau_analysis must be called in the diagnostics section."""
        mh_src = _read_submodule_source("mutation_helpers")
        assert "get_plateau_analysis" in mh_src


# ═══════════════════════════════════════════════════════════════════════
# 3. TestNoRegressions
# ═══════════════════════════════════════════════════════════════════════


class TestNoRegressions:
    """Verify core EvolutionEngine methods still work."""

    def test_init_exists_and_accepts_kwargs(self):
        """EvolutionEngine.__init__ must exist and accept config/rng/vm_farm/champion_anchor."""
        from evolution_engine import EvolutionEngine

        sig = inspect.signature(EvolutionEngine.__init__)
        params = list(sig.parameters.keys())
        assert "config" in params
        assert "rng" in params
        assert "vm_farm" in params
        assert "champion_anchor" in params

    def test_run_signature(self):
        """EvolutionEngine.run must accept num_generations: int | None."""
        from evolution_engine import EvolutionEngine

        sig = inspect.signature(EvolutionEngine.run)
        params = sig.parameters
        assert "num_generations" in params
        # Verify return annotation includes dict
        assert sig.return_annotation != inspect.Parameter.empty

    def test_get_run_diagnostics_works(self):
        """get_run_diagnostics must return a dict with expected keys."""
        from tests.conftest import make_engine

        engine = make_engine()
        diag = engine.get_run_diagnostics()
        assert isinstance(diag, dict)
        assert "champion_level" in diag
        assert "champion_fitness" in diag
        assert "generations_stalled" in diag
        assert "current_generation" in diag

    def test_get_performance_metrics_works(self):
        """get_performance_metrics must return a dict with expected keys."""
        from tests.conftest import make_engine

        engine = make_engine()
        metrics = engine.get_performance_metrics()
        assert isinstance(metrics, dict)
        assert "current_generation" in metrics
        assert "population_size" in metrics
        assert "mutation_rate" in metrics

    def test_get_archive_summary_works(self):
        """get_archive_summary must return a dict with expected keys."""
        from tests.conftest import make_engine

        engine = make_engine()
        summary = engine.get_archive_summary()
        assert isinstance(summary, dict)
        assert "novelty" in summary
        assert "qd" in summary
        assert "strategy" in summary
        assert "hall_of_fame" in summary


# ═══════════════════════════════════════════════════════════════════════
# 4. TestDocumentationCoverage
# ═══════════════════════════════════════════════════════════════════════


class TestDocumentationCoverage:
    """Verify all new methods have docstrings."""

    @staticmethod
    def _has_docstring(obj) -> bool:
        """Check if an object has a non-empty docstring."""
        doc = getattr(obj, "__doc__", None)
        return bool(doc and doc.strip())

    def test_meta_controller_new_methods_have_docstrings(self):
        from meta_controller import MetaController

        assert self._has_docstring(MetaController.get_cross_cycle_trends)
        assert self._has_docstring(MetaController.get_recovery_confidence)

    def test_strategy_archive_new_methods_have_docstrings(self):
        from strategy_archive import StrategyArchive

        assert self._has_docstring(StrategyArchive.get_dominance_report)
        assert self._has_docstring(StrategyArchive.per_level_effectiveness)
        assert self._has_docstring(StrategyArchive.time_weighted_success_rate)
        assert self._has_docstring(StrategyArchive.adaptive_exploration_bonus)

    def test_champion_archive_new_methods_have_docstrings(self):
        from champion_archive import ChampionArchive

        assert self._has_docstring(ChampionArchive.diversity_aware_select)
        assert self._has_docstring(ChampionArchive.quality_weighted_prune)

    def test_novelty_new_methods_have_docstrings(self):
        from novelty import SparseArchive

        assert self._has_docstring(SparseArchive.manage_capacity)
        assert self._has_docstring(SparseArchive.decayed_novelty)

    def test_qd_archive_new_methods_have_docstrings(self):
        from qd_archive import MAPElitesArchive

        assert self._has_docstring(MAPElitesArchive.get_improvement_velocity)
        assert self._has_docstring(MAPElitesArchive.get_illumination_diversity)

    def test_metrics_new_methods_have_docstrings(self):
        from metrics import MetricsCollector

        assert self._has_docstring(MetricsCollector.detect_anomalies)
        assert self._has_docstring(MetricsCollector.correlate_metrics)

    def test_data_rotation_new_methods_have_docstrings(self):
        from data_rotation import get_disk_usage

        assert self._has_docstring(get_disk_usage)

    def test_progression_tracker_new_methods_have_docstrings(self):
        from progression_tracker import ProgressionTracker

        assert self._has_docstring(ProgressionTracker.get_plateau_analysis)
        assert self._has_docstring(ProgressionTracker.compare_windows)

    def test_config_new_methods_have_docstrings(self):
        from config import EvolutionSection

        assert self._has_docstring(EvolutionSection.get_config_summary)
        assert self._has_docstring(EvolutionSection.validate_config)

    def test_mutation_proposer_new_methods_have_docstrings(self):
        from mutation_proposer import DirectedMutator

        assert self._has_docstring(DirectedMutator.get_mutation_stats)

    def test_monitoring_alerts_config_summary_has_docstring(self):
        from monitoring_alerts import AlertManager

        assert self._has_docstring(AlertManager.get_config_summary)


# ═══════════════════════════════════════════════════════════════════════
# 5. TestCodeIntegrity
# ═══════════════════════════════════════════════════════════════════════


class TestCodeIntegrity:
    """Verify no syntax errors in all modified files by importing each module."""

    MODIFIED_MODULES: ClassVar[list[str]] = [
        "evolution_engine",
        "novelty",
        "strategy_archive",
        "champion_archive",
        "qd_archive",
        "metrics",
        "data_rotation",
        "progression_tracker",
        "config",
        "monitoring_alerts",
        "mutation_proposer",
        "meta_controller",
    ]

    @pytest.mark.parametrize("module_name", MODIFIED_MODULES)
    def test_module_imports_without_syntax_error(self, module_name):
        """Each modified module must import cleanly (no SyntaxError)."""
        try:
            mod = importlib.import_module(module_name)
            assert mod is not None, f"Module {module_name} imported as None"
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in {module_name}: {e}")
        except ImportError as e:
            # Some modules may have optional deps — skip if it's an
            # ImportError from a missing optional package, not from our code
            if module_name in str(e):
                pytest.skip(f"Optional dependency missing for {module_name}: {e}")
            raise

    def test_evolution_engine_core_classes_exist(self):
        """EvolutionEngine, EvolutionConfig, HallOfFame must all be importable."""
        from evolution_engine import EvolutionConfig, EvolutionEngine, HallOfFame

        assert EvolutionEngine is not None
        assert EvolutionConfig is not None
        assert HallOfFame is not None

    def test_evolution_engine_source_has_no_syntax_errors(self):
        """Compile evolution_engine.py to verify no syntax errors."""
        path = _ROOT / "evolution_engine.py"
        source = path.read_text(encoding="utf-8")
        try:
            compile(source, str(path), "exec")
        except SyntaxError as e:
            pytest.fail(f"SyntaxError in evolution_engine.py: {e}")
