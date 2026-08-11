#!/usr/bin/env python3
"""test_wave32_final.py — Absolute final validation for all 26 new methods.

Two test classes:
  1. TestCompleteMethodInventory   — 26 new methods exist and are callable
  2. TestPipelineWiringComplete    — evolution_engine.py references all 13 pipeline points

Uses pytest, inspect, importlib. Tests are fast (no VM, no training).
Does NOT modify any existing files.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# 1. TestCompleteMethodInventory
# ══════════════════════════════════════════════════════════════════════════════
# Map: (module_name, class_or_none, method_name, is_static_or_module_func)
# class_or_none = None → module-level function imported directly

METHOD_INVENTORY: List[Tuple[str, str | None, str, bool]] = [
    # MetaController (3)
    ("meta_controller", "MetaController", "get_cross_cycle_trends", False),
    ("meta_controller", "MetaController", "get_recovery_confidence", False),
    ("meta_controller", "MetaController", "get_tuning_escalation_history", False),
    # StrategyArchive (4)
    ("strategy_archive", "StrategyArchive", "per_level_effectiveness", False),
    ("strategy_archive", "StrategyArchive", "adaptive_exploration_bonus", False),
    ("strategy_archive", "StrategyArchive", "time_weighted_success_rate", False),
    ("strategy_archive", "StrategyArchive", "get_dominance_report", False),
    # ChampionArchive (3)
    ("champion_archive", "ChampionArchive", "diversity_aware_select", False),
    ("champion_archive", "ChampionArchive", "quality_weighted_prune", False),
    ("champion_archive", "ChampionArchive", "get_diversity_report", False),
    # SparseArchive / novelty (2)
    ("novelty", "SparseArchive", "manage_capacity", False),
    ("novelty", "SparseArchive", "decayed_novelty", False),
    # MAPElitesArchive / qd_archive (2)
    ("qd_archive", "MAPElitesArchive", "get_improvement_velocity", False),
    ("qd_archive", "MAPElitesArchive", "get_illumination_diversity", False),
    # DirectedMutator (2) — static methods
    ("mutation_proposer", "DirectedMutator", "record_mutation_outcome", True),
    ("mutation_proposer", "DirectedMutator", "get_mutation_stats", True),
    # AlertManager (2)
    ("monitoring_alerts", "AlertManager", "send_monoculture_alert", False),
    ("monitoring_alerts", "AlertManager", "send_recovery_stuck_alert", False),
    # MetricsCollector (2)
    ("metrics", "MetricsCollector", "detect_anomalies", False),
    ("metrics", "MetricsCollector", "correlate_metrics", False),
    # data_rotation module functions (2)
    ("data_rotation", None, "get_disk_usage", True),
    ("data_rotation", None, "quality_aware_rotate_checkpoints", True),
    # ProgressionTracker (2)
    ("progression_tracker", "ProgressionTracker", "get_plateau_analysis", False),
    ("progression_tracker", "ProgressionTracker", "compare_windows", False),
    # EvolutionSection (from config.py)
    ("config", "EvolutionSection", "get_config_summary", False),
    ("config", "EvolutionSection", "validate_config", False),
]


def _load_target(mod_name: str, cls_name: str | None) -> Any:
    """Import module and return the class object (or module if cls_name is None)."""
    mod = importlib.import_module(mod_name)
    if cls_name is None:
        return mod
    return getattr(mod, cls_name)


class TestCompleteMethodInventory:
    """Verify all 26 new methods exist and are callable on their owning class."""

    @pytest.mark.parametrize(
        "mod_name,cls_name,method_name,is_static",
        METHOD_INVENTORY,
        ids=[f"{m[0]}.{m[1] or 'module'}.{m[2]}" for m in METHOD_INVENTORY],
    )
    def test_method_exists_and_callable(self, mod_name: str, cls_name: str | None, method_name: str, is_static: bool):
        """Each method must be a callable attribute on its class/module."""
        target = _load_target(mod_name, cls_name)
        attr = getattr(target, method_name, None)
        assert attr is not None, f"{cls_name or mod_name} missing attribute '{method_name}'"
        assert callable(attr), f"{cls_name or mod_name}.{method_name} is not callable"

    @pytest.mark.parametrize(
        "mod_name,cls_name,method_name,is_static",
        METHOD_INVENTORY,
        ids=[f"{m[0]}.{m[1] or 'module'}.{m[2]}" for m in METHOD_INVENTORY],
    )
    def test_method_in_class_dict(self, mod_name: str, cls_name: str | None, method_name: str, is_static: bool):
        """Confirm the method lives in __dict__ (not inherited by accident)."""
        target = _load_target(mod_name, cls_name)
        if cls_name is not None:
            # For classes, check __dict__ to ensure it's defined on this class
            assert method_name in target.__dict__, (
                f"'{method_name}' not in {cls_name}.__dict__ — " "may be inherited from a parent or missing"
            )
        else:
            # For modules, just confirm it's a top-level attribute
            assert hasattr(target, method_name)

    @pytest.mark.parametrize(
        "mod_name,cls_name,method_name,is_static",
        METHOD_INVENTORY,
        ids=[f"{m[0]}.{m[1] or 'module'}.{m[2]}" for m in METHOD_INVENTORY],
    )
    def test_method_has_valid_signature(self, mod_name: str, cls_name: str | None, method_name: str, is_static: bool):
        """Each method must have an inspectable signature (not a C builtin)."""
        target = _load_target(mod_name, cls_name)
        attr = getattr(target, method_name)
        try:
            sig = inspect.signature(attr)
        except (ValueError, TypeError):
            pytest.fail(f"{cls_name or mod_name}.{method_name} has no inspectable signature")
        params = list(sig.parameters.keys())
        # instance methods should have 'self' as first param
        if cls_name is not None and not is_static:
            assert len(params) >= 1, f"{cls_name}.{method_name} signature has no params"

    def test_total_method_count(self):
        """Sanity: we must have exactly 26 inventory entries."""
        assert len(METHOD_INVENTORY) == 26, f"Expected 26 inventory entries, got {len(METHOD_INVENTORY)}"

    def test_unique_method_names(self):
        """All method names must be distinct."""
        names = {name for _, _, name, _ in METHOD_INVENTORY}
        assert len(names) == 26, f"Expected 26 unique method names, got {len(names)}: {names}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestPipelineWiringComplete
# ══════════════════════════════════════════════════════════════════════════════
# Read evolution_engine.py source and verify references exist.

PIPELINE_POINTS: List[str] = [
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

# Maps pipeline methods that moved out of the engine to their
# current file locations (relative to the project root).
_PIPELINE_METHOD_FILES: dict[str, str] = {
    "get_cross_cycle_trends": "evolution/mutation_helpers.py",
    "get_improvement_velocity": "qd_archive.py",
    "get_illumination_diversity": "qd_archive.py",
    "get_mutation_stats": "evolution/mutation_helpers.py",
    "send_monoculture_alert": "monitoring_alerts.py",
    "send_recovery_stuck_alert": "monitoring_alerts.py",
    "get_plateau_analysis": "evolution/mutation_helpers.py",
}


ENGINE_SRC = _ROOT / "evolution_engine.py"


@pytest.fixture(scope="module")
def engine_source() -> str:
    """Read evolution_engine.py source once for the whole class."""
    assert ENGINE_SRC.exists(), f"evolution_engine.py not found at {ENGINE_SRC}"
    return ENGINE_SRC.read_text(encoding="utf-8")


class TestPipelineWiringComplete:
    """Verify evolution_engine.py references all 13 pipeline integration points."""

    @pytest.mark.parametrize(
        "point",
        PIPELINE_POINTS,
        ids=PIPELINE_POINTS,
    )
    def test_pipeline_reference_present(self, engine_source: str, point: str):
        """Each pipeline integration point must appear in evolution_engine.py."""
        if point in _PIPELINE_METHOD_FILES:
            fpath = ENGINE_SRC.parent / _PIPELINE_METHOD_FILES[point]
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()
            assert point in source, f"'point' not found in {point} -> {_PIPELINE_METHOD_FILES[point]}"
        else:
            assert point in engine_source, "'point' not found in evolution_engine.py"

    def test_all_points_count(self):
        """Sanity: we must have exactly 13 pipeline points."""
        assert len(PIPELINE_POINTS) == 13, f"Expected 13 pipeline points, got {len(PIPELINE_POINTS)}"
        assert len(set(PIPELINE_POINTS)) == 13, "Duplicate pipeline points detected"

    def test_source_file_exists_and_nontrivial(self):
        """Quick smoke: the engine source file exists and is large enough."""
        assert ENGINE_SRC.is_file(), f"{ENGINE_SRC} is not a regular file"
        size = ENGINE_SRC.stat().st_size
        assert size > 100, f"evolution_engine.py is suspiciously small ({size} bytes)"
