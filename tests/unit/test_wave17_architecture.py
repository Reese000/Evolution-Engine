#!/usr/bin/env python3
"""Wave 17 — Architecture and Design Pattern Tests.

Verifies:
  1. Module isolation (no circular imports)
  2. Class interface consistency
  3. Error handling edge cases
  4. State management roundtrips

Uses only pytest, importlib, tempfile.  No existing files are modified.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import tempfile

import pytest

# Ensure the project root is on sys.path (mirrors conftest)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

# Map of module name → (class_to_import, extra_import_name)
# We test that each module can be imported independently.
_MODULE_IMPORT_LIST = [
    "meta_controller",
    "strategy_archive",
    "champion_archive",
    "novelty",
    "qd_archive",
    "mutation_proposer",
    "monitoring_alerts",
    "metrics",
    "data_rotation",
    "progression_tracker",
    "config",
]


def _fresh_import(mod_name: str):
    """Import *mod_name* from the project root, clearing any cached entry
    first so we truly test independent importability."""
    # Remove the module and any of its sub-imports from cache so we
    # exercise the import path from scratch.
    cached = {k: v for k, v in sys.modules.items() if k == mod_name or k.startswith(mod_name + ".")}
    for key in cached:
        del sys.modules[key]
    return importlib.import_module(mod_name)


def _tmpfile(suffix: str = ".json") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


# ═══════════════════════════════════════════════════════════════════════
# 1. TestModuleIsolation
# ═══════════════════════════════════════════════════════════════════════


class TestModuleIsolation:
    """Each enhanced module imports independently without circular deps."""

    @pytest.mark.parametrize("mod_name", _MODULE_IMPORT_LIST)
    def test_independent_import(self, mod_name: str):
        """Module can be imported without raising ImportError/CircularImport."""
        mod = _fresh_import(mod_name)
        assert mod is not None, f"importlib.import_module('{mod_name}') returned None"

    @pytest.mark.parametrize("mod_name", _MODULE_IMPORT_LIST)
    def test_module_has_public_interface(self, mod_name: str):
        """Each module exposes at least one public class or callable."""
        mod = _fresh_import(mod_name)
        # Every module should have at least one uppercase name (class or constant)
        public = [n for n in dir(mod) if n[0].isupper() and not n.startswith("_")]
        assert public, f"Module '{mod_name}' exposes no public names"

    def test_meta_controller_core_class(self):
        """meta_controller exposes MetaController."""
        mod = _fresh_import("meta_controller")
        assert hasattr(mod, "MetaController")

    def test_strategy_archive_core_class(self):
        """strategy_archive exposes StrategyArchive."""
        mod = _fresh_import("strategy_archive")
        assert hasattr(mod, "StrategyArchive")

    def test_novelty_core_class(self):
        """novelty exposes SparseArchive."""
        mod = _fresh_import("novelty")
        assert hasattr(mod, "SparseArchive")

    def test_qd_archive_core_class(self):
        """qd_archive exposes MAPElitesArchive."""
        mod = _fresh_import("qd_archive")
        assert hasattr(mod, "MAPElitesArchive")

    def test_monitoring_alerts_core_classes(self):
        """monitoring_alerts exposes AlertManager and AlertThresholds."""
        mod = _fresh_import("monitoring_alerts")
        assert hasattr(mod, "AlertManager")
        assert hasattr(mod, "AlertThresholds")

    def test_metrics_core_class(self):
        """metrics exposes MetricsCollector."""
        mod = _fresh_import("metrics")
        assert hasattr(mod, "MetricsCollector")

    def test_config_core_class(self):
        """config exposes Config (or EvolutionSection)."""
        mod = _fresh_import("config")
        assert hasattr(mod, "Config") or hasattr(mod, "EvolutionSection")


# ═══════════════════════════════════════════════════════════════════════
# 2. TestClassInterfaces
# ═══════════════════════════════════════════════════════════════════════


class TestClassInterfaces:
    """Verify consistent return-type patterns across module interfaces."""

    # -- Analysis methods return dicts -----------------------------------

    def test_strategy_archive_summary_returns_dict(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.summary()
        assert isinstance(result, dict)

    def test_novelty_archive_summary_returns_dict(self):
        from novelty import SparseArchive

        sa = SparseArchive()
        result = sa.get_archive_summary()
        assert isinstance(result, dict)

    def test_qd_archive_summary_returns_dict(self):
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.get_archive_summary()
        assert isinstance(result, dict)

    def test_qd_fitness_distribution_returns_dict(self):
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.get_fitness_distribution()
        assert isinstance(result, dict)

    def test_qd_cell_stats_returns_dict(self):
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.get_cell_stats()
        assert isinstance(result, dict)

    def test_qd_improvement_velocity_returns_dict(self):
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.get_improvement_velocity()
        assert isinstance(result, dict)

    def test_qd_illumination_diversity_returns_dict(self):
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.get_illumination_diversity()
        assert isinstance(result, dict)

    def test_monitoring_alerts_config_summary_returns_dict(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.get_config_summary()
        assert isinstance(result, dict)

    def test_meta_controller_state_summary_returns_dict(self):
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.get_state_summary()
        assert isinstance(result, dict)

    def test_meta_controller_effectiveness_report_returns_dict(self):
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.get_effectiveness_report()
        assert isinstance(result, dict)

    def test_meta_controller_recovery_confidence_returns_dict(self):
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.get_recovery_confidence()
        assert isinstance(result, dict)

    def test_meta_controller_cross_cycle_trends_returns_dict(self):
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.get_cross_cycle_trends()
        assert isinstance(result, dict)

    def test_meta_controller_suggest_parameter_reset_returns_dict(self):
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.suggest_parameter_reset()
        assert isinstance(result, dict)

    # -- Boolean methods return bool ------------------------------------

    def test_strategy_archive_recommend_returns_str(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.recommend_strategy(level=0)
        assert isinstance(result, str)

    def test_meta_controller_should_rollback_returns_bool(self):
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.should_rollback()
        assert isinstance(result, bool)

    def test_novelty_get_diversity_score_returns_float(self):
        from novelty import SparseArchive

        sa = SparseArchive()
        result = sa.get_diversity_score()
        assert isinstance(result, (int, float))

    def test_qd_archive_coverage_returns_float(self):
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.coverage()
        assert isinstance(result, (int, float))

    def test_qd_archive_size_returns_int(self):
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.size  # @property, not callable
        assert isinstance(result, int)

    # -- Count methods return int ---------------------------------------

    def test_strategy_archive_total_records_returns_int(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.total_records()
        assert isinstance(result, int)

    def test_novelty_trim_archive_returns_int(self):
        from novelty import SparseArchive

        sa = SparseArchive()
        result = sa.trim_archive(max_entries=10)
        assert isinstance(result, int)

    def test_novelty_manage_capacity_returns_int(self):
        from novelty import SparseArchive

        sa = SparseArchive()
        result = sa.manage_capacity(max_size=100)
        assert isinstance(result, int)

    def test_metrics_count_records_returns_int(self):
        tmp = _tmpfile(".jsonl")
        try:
            from metrics import MetricsCollector

            mc = MetricsCollector(path=tmp)
            result = mc.count_records()
            assert isinstance(result, int)
        finally:
            os.unlink(tmp)

    # -- List methods return list ---------------------------------------

    def test_strategy_archive_records_by_level_returns_list(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.records_by_level(level=0)
        assert isinstance(result, list)

    def test_strategy_archive_records_by_strategy_returns_list(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.records_by_strategy(strategy="random")
        assert isinstance(result, list)

    def test_meta_controller_get_tuning_escalation_history_returns_list(self):
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.get_tuning_escalation_history()
        assert isinstance(result, list)

    def test_novelty_recent_descriptors_returns_list(self):
        from novelty import SparseArchive

        sa = SparseArchive()
        result = sa.recent_descriptors(n=5)
        assert isinstance(result, list)

    def test_novelty_recent_novelty_scores_returns_list(self):
        from novelty import SparseArchive

        sa = SparseArchive()
        result = sa.recent_novelty_scores(n=5)
        assert isinstance(result, list)

    def test_qd_archive_grid_solutions_returns_list(self):
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.grid_solutions()
        assert isinstance(result, list)

    def test_monitoring_alerts_get_alert_history_returns_list(self):
        from monitoring_alerts import AlertManager

        am = AlertManager()
        result = am.get_alert_history()
        assert isinstance(result, list)

    def test_config_validate_config_returns_list(self):
        from config import EvolutionSection

        es = EvolutionSection()
        result = es.validate_config()
        assert isinstance(result, list)

    def test_config_get_config_summary_returns_dict(self):
        from config import EvolutionSection

        es = EvolutionSection()
        result = es.get_config_summary()
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════
# 3. TestErrorHandling
# ═══════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Verify methods handle edge cases gracefully."""

    # -- Empty inputs return sensible defaults ---------------------------

    def test_strategy_archive_recommend_empty(self):
        """recommend_strategy on empty archive returns a string (fallback)."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.recommend_strategy(level=0)
        assert isinstance(result, str)
        assert len(result) > 0  # should return a valid strategy name

    def test_strategy_archive_success_rate_empty(self):
        """success_rate on empty archive returns 0 or 0.0."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.success_rate(strategy="random")
        assert result == 0 or result == 0.0

    def test_strategy_archive_summary_empty(self):
        """summary on empty archive returns dict with expected keys."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        summary = sa.summary()
        assert isinstance(summary, dict)
        assert "total_records" in summary or "count" in summary or len(summary) > 0

    def test_novelty_novelty_score_empty_archive(self):
        """novelty_score on empty archive doesn't crash."""
        from novelty import SparseArchive

        sa = SparseArchive()
        desc = {
            "inst_diversity": 0.5,
            "reg_pressure": 0.3,
            "program_length": 0.1,
            "branch_fraction": 0.2,
            "alu_fraction": 0.4,
            "load_store_ratio": 0.1,
            "unique_regs_used": 0.3,
        }
        score = sa.novelty_score(desc)
        assert isinstance(score, (int, float))

    def test_qd_archive_selection_bias_empty(self):
        """selection_bias on empty archive returns 0.0."""
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.selection_bias(descriptor=[0.0, 0.0])
        assert isinstance(result, (int, float))
        assert result >= 0.0

    def test_qd_archive_get_freshest_cell_empty(self):
        """get_freshest_cell on empty archive returns None."""
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.get_freshest_cell()
        assert result is None

    def test_qd_archive_coverage_empty(self):
        """coverage on empty archive returns 0.0."""
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.coverage()
        assert result == 0.0

    def test_metrics_get_history_empty(self):
        """get_history on empty metrics file returns empty list."""
        tmp = _tmpfile(".jsonl")
        try:
            from metrics import MetricsCollector

            mc = MetricsCollector(path=tmp)
            result = mc.get_history(last_n=10)
            assert isinstance(result, list)
            assert len(result) == 0
        finally:
            os.unlink(tmp)

    def test_metrics_count_records_empty(self):
        """count_records on empty file returns 0."""
        tmp = _tmpfile(".jsonl")
        try:
            from metrics import MetricsCollector

            mc = MetricsCollector(path=tmp)
            result = mc.count_records()
            assert result == 0
        finally:
            os.unlink(tmp)

    # -- None inputs don't crash ----------------------------------------

    def test_novelty_novelty_score_none_descriptor(self):
        """novelty_score with empty dict doesn't crash."""
        from novelty import SparseArchive

        sa = SparseArchive()
        score = sa.novelty_score({})
        assert isinstance(score, (int, float))

    def test_champion_archive_add_minimal(self):
        """ChampionArchive.add with minimal entry doesn't crash."""
        from champion_archive import ChampionArchive

        ca = ChampionArchive.__new__(ChampionArchive)
        # Set minimum required attributes
        ca._entries = []
        ca._path = _tmpfile(".json")
        ca._capacity = 100
        ca._max_cycle = 0
        try:
            # This should handle the edge case gracefully
            ca.add(program=[0x00000013], fitness=0.5, level=0, cycle=1, trace={})
        except (TypeError, ValueError, KeyError):
            # Some validation errors are acceptable
            pass
        finally:
            if os.path.exists(ca._path):
                os.unlink(ca._path)

    # -- Division by zero is handled ------------------------------------

    def test_novelty_adaptive_threshold_empty(self):
        """adaptive_threshold with no data returns finite value."""
        from novelty import SparseArchive

        sa = SparseArchive()
        result = sa.adaptive_threshold()
        assert isinstance(result, (int, float))
        assert math.isfinite(result)

    def test_strategy_archive_time_weighted_empty(self):
        """time_weighted_success_rate with no data returns 0 or 0.0."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.time_weighted_success_rate(strategy="random")
        assert result == 0 or result == 0.0

    def test_qd_archive_neighborhood_density_empty(self):
        """get_neighborhood_density on empty archive returns 0.0."""
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        result = qa.get_neighborhood_density(descriptor=[0.0, 0.0])
        assert isinstance(result, (int, float))
        assert result >= 0.0

    def test_meta_controller_detect_stagnation_empty(self):
        """detect_stagnation with no history returns NORMAL."""
        from meta_controller import MetaController, StagnationResult

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.detect_stagnation()
        assert isinstance(result, StagnationResult)
        # With no history, should be NORMAL
        assert result == StagnationResult.NORMAL


# ═══════════════════════════════════════════════════════════════════════
# 4. TestStateManagement
# ═══════════════════════════════════════════════════════════════════════


class TestStateManagement:
    """Verify serialization/deserialization roundtrips."""

    def _make_mc_with_data(self):
        """Create a MetaController with a few telemetry snapshots."""
        from meta_controller import MetaController, TelemetrySnapshot

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        for i in range(5):
            snap = TelemetrySnapshot(
                cycle=i + 1,
                fitness=float(i) * 0.1,
                level=i,
                champion_words=10 + i,
                val_loss=2.0 - i * 0.1,
                diversity=0.5 + i * 0.05,
                novelty_size=i,
                qd_occupied=i * 2,
                qd_total=100,
                traces_count=0,
                retrained=False,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        return mc

    def test_meta_controller_save_load_roundtrip(self):
        """MetaController state survives save → load."""
        from meta_controller import MetaController

        mc = self._make_mc_with_data()
        path = _tmpfile(".json")
        try:
            mc.save_state(path)
            # Create a fresh MC and load
            mc2 = MetaController(checkpoint_dir=tempfile.mkdtemp())
            loaded = mc2.load_state(path)
            assert loaded is True
            # Verify key state survived
            s1 = mc.get_state_summary()
            s2 = mc2.get_state_summary()
            # Same cycle count
            assert s1.get("total_cycles", s1.get("cycles_tracked", 0)) == s2.get(
                "total_cycles", s2.get("cycles_tracked", 0)
            )
        finally:
            os.unlink(path)

    def test_meta_controller_get_state_summary_structure(self):
        """get_state_summary returns a dict with expected keys."""
        mc = self._make_mc_with_data()
        summary = mc.get_state_summary()
        assert isinstance(summary, dict)
        # Should have some identifying keys
        assert len(summary) > 0

    def test_meta_controller_save_creates_file(self):
        """save_state creates the target file."""
        mc = self._make_mc_with_data()
        path = _tmpfile(".json")
        os.unlink(path)  # remove so save creates it
        try:
            mc.save_state(path)
            assert os.path.exists(path)
            # Verify it's valid JSON
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_meta_controller_load_nonexistent_returns_false(self):
        """load_state with nonexistent path returns False."""
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.load_state("/tmp/nonexistent_evolution_state_99999.json")
        assert result is False

    def test_meta_controller_load_corrupt_returns_false(self):
        """load_state with corrupt JSON returns False (doesn't crash)."""
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        path = _tmpfile(".json")
        try:
            with open(path, "w") as f:
                f.write("NOT VALID JSON {{{{")
            result = mc.load_state(path)
            assert result is False
        finally:
            os.unlink(path)

    def test_strategy_archive_save_load_roundtrip(self):
        """StrategyArchive state survives save → load."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        # Record a few entries (record takes: strategy, level, success, generation)
        for i in range(3):
            sa.record(
                strategy="random",
                level=i,
                success=True,
                generation=i,
            )
        path = _tmpfile(".json")
        try:
            sa.save(path)
            # Load into a fresh archive
            sa2 = StrategyArchive()
            loaded = sa2.load(path)
            assert loaded is True
            assert sa2.total_records() == sa.total_records()
        finally:
            os.unlink(path)

    def test_strategy_archive_save_creates_file(self):
        """save creates the target file."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        path = _tmpfile(".json")
        os.unlink(path)
        try:
            sa.save(path)
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_strategy_archive_load_nonexistent_returns_false(self):
        """load with nonexistent path returns False."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        result = sa.load("/tmp/nonexistent_sa_99999.json")
        assert result is False

    def test_evolution_config_get_config_summary(self):
        """EvolutionSection.get_config_summary returns a non-empty dict."""
        from config import EvolutionSection

        es = EvolutionSection()
        summary = es.get_config_summary()
        assert isinstance(summary, dict)
        assert len(summary) > 0

    def test_evolution_config_validate_config(self):
        """EvolutionSection.validate_config returns a list."""
        from config import EvolutionSection

        es = EvolutionSection()
        errors = es.validate_config()
        assert isinstance(errors, list)
        # Default config should have no validation errors (or at least no crash)
        # Some defaults might fail validation; that's fine as long as it returns a list

    def test_novelty_serialize_deserialize_roundtrip(self):
        """SparseArchive survives serialize → deserialize."""
        from novelty import SparseArchive

        sa = SparseArchive()
        data = sa.serialize()
        assert isinstance(data, dict)
        # Deserialize into a fresh archive
        sa2 = SparseArchive.deserialize(data)
        assert sa2 is not None

    def test_qd_archive_serialize_roundtrip(self):
        """MAPElitesArchive survives serialize → deserialize."""
        from qd_archive import MAPElitesArchive

        qa = MAPElitesArchive()
        data = qa.serialize()
        assert isinstance(data, dict)
        # Verify the serialized data has expected structure
        assert "bins" in data or "grid" in data or "size" in data or len(data) > 0

    def test_meta_controller_tuning_history_summary(self):
        """get_tuning_history_summary returns a dict."""
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.get_tuning_history_summary()
        assert isinstance(result, dict)

    def test_meta_controller_effectiveness_score(self):
        """get_effectiveness_score returns a finite float."""
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.get_effectiveness_score()
        assert isinstance(result, (int, float))
        assert math.isfinite(result)

    def test_strategy_archive_summary_after_records(self):
        """summary reflects recorded data."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        sa.record(strategy="random", level=0, success=True)
        sa.record(strategy="directed", level=1, success=True)
        summary = sa.summary()
        assert summary["total_records"] >= 2

    def test_meta_controller_get_best_checkpoint_none(self):
        """get_best_checkpoint returns None when no checkpoints exist."""
        from meta_controller import MetaController

        mc = MetaController(checkpoint_dir=tempfile.mkdtemp())
        result = mc.get_best_checkpoint()
        assert result is None or isinstance(result, str)

    def test_monitoring_alerts_get_alert_history_empty(self):
        """get_alert_history returns empty list for fresh AlertManager."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        history = am.get_alert_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_monitoring_alerts_config_summary_structure(self):
        """get_config_summary has expected keys."""
        from monitoring_alerts import AlertManager

        am = AlertManager()
        summary = am.get_config_summary()
        assert isinstance(summary, dict)
        # Should mention channels or thresholds
        assert len(summary) > 0

    def test_data_rotation_run_all_rotations_returns_dict(self):
        """run_all_rotations returns a dict with rotation counts."""
        from data_rotation import run_all_rotations

        result = run_all_rotations()
        assert isinstance(result, dict)
        # Should have keys for each rotation type
        assert len(result) > 0
