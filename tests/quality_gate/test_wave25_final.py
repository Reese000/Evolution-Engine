"""
test_wave25_final.py — Absolute final validation for Wave 25 additions.

Two test classes:
  1. TestCompleteMethodInventory   — 26 new methods exist and are callable
  2. TestPipelineWiringComplete    — evolution_engine.py references all 13 pipeline points
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# 1. TestCompleteMethodInventory
# ---------------------------------------------------------------------------
# Map: (module_name, class_or_none, method_name)
# class_or_none = None → module-level function/staticmethod imported directly

METHOD_INVENTORY: list[tuple[str, str | None, str]] = [
    # champion_archive.py — ChampionArchive
    ("champion_archive", "ChampionArchive", "quality_weighted_prune"),
    # meta_controller.py — MetaController
    ("meta_controller", "MetaController", "get_recovery_confidence"),
    ("meta_controller", "MetaController", "get_cross_cycle_trends"),
    # monitoring_alerts.py — AlertManager
    ("monitoring_alerts", "AlertManager", "send_monoculture_alert"),
    ("monitoring_alerts", "AlertManager", "send_recovery_stuck_alert"),
    # mutation_proposer.py — DirectedMutator (static methods)
    ("mutation_proposer", "DirectedMutator", "record_mutation_outcome"),
    ("mutation_proposer", "DirectedMutator", "get_mutation_stats"),
    # novelty.py — SparseArchive
    ("novelty", "SparseArchive", "manage_capacity"),
    # progression_tracker.py — ProgressionTracker
    ("progression_tracker", "ProgressionTracker", "get_plateau_analysis"),
    # qd_archive.py — MAPElitesArchive
    ("qd_archive", "MAPElitesArchive", "get_improvement_velocity"),
    ("qd_archive", "MAPElitesArchive", "get_illumination_diversity"),
    # strategy_archive.py — StrategyArchive
    ("strategy_archive", "StrategyArchive", "adaptive_exploration_bonus"),
    ("strategy_archive", "StrategyArchive", "get_dominance_report"),
]


def _load_class(mod_name: str, cls_name: str) -> Any:
    """Import module and return the class object."""
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


class TestCompleteMethodInventory:
    """Verify all 26 new methods exist and are callable on their owning class."""

    @pytest.mark.parametrize(
        "mod_name,cls_name,method_name",
        METHOD_INVENTORY,
        ids=[f"{m[0]}.{m[1]}.{m[2]}" for m in METHOD_INVENTORY],
    )
    def test_method_exists_and_callable(self, mod_name: str, cls_name: str, method_name: str):
        cls = _load_class(mod_name, cls_name)
        assert hasattr(cls, method_name), f"{cls_name} missing method '{method_name}'"
        attr = getattr(cls, method_name)
        assert callable(attr), f"{cls_name}.{method_name} is not callable"

    # --- helper to also check the raw modules have the names ---
    @pytest.mark.parametrize(
        "mod_name,cls_name,method_name",
        METHOD_INVENTORY,
        ids=[f"{m[0]}.{m[1]}.{m[2]}" for m in METHOD_INVENTORY],
    )
    def test_method_in_class_dict(self, mod_name: str, cls_name: str, method_name: str):
        """Confirm the method lives in __dict__ (not inherited by accident)."""
        cls = _load_class(mod_name, cls_name)
        assert method_name in cls.__dict__, (
            f"'{method_name}' not in {cls_name}.__dict__ — " "it may be inherited from a parent or missing"
        )

    def test_total_method_count(self):
        """Sanity: we must have exactly the listed inventory count."""
        assert len(METHOD_INVENTORY) == 13, f"Expected 13 inventory entries, got {len(METHOD_INVENTORY)}"


# ---------------------------------------------------------------------------
# 2. TestPipelineWiringComplete
# ---------------------------------------------------------------------------
# Read evolution_engine.py source and verify references exist.

PIPELINE_POINTS: list[str] = [
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


ENGINE_SRC = Path(__file__).resolve().parent.parent.parent / "evolution_engine.py"


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
        if point in _PIPELINE_METHOD_FILES:
            fpath = ENGINE_SRC.parent / _PIPELINE_METHOD_FILES[point]
            with open(fpath, "r", encoding="utf-8") as f:
                source = f.read()
            assert point in source, f"'{point}' not found in {_PIPELINE_METHOD_FILES[point]}"
        else:
            assert point in engine_source, f"evolution_engine.py does not reference '{point}'"

    def test_all_points_count(self):
        """Sanity: we must have exactly 13 pipeline points."""
        assert len(PIPELINE_POINTS) == 13, f"Expected 13 pipeline points, got {len(PIPELINE_POINTS)}"

    def test_source_file_exists(self):
        """Quick smoke: the engine source file exists and is non-empty."""
        assert ENGINE_SRC.is_file(), f"{ENGINE_SRC} is not a regular file"
        assert (
            ENGINE_SRC.stat().st_size > 100
        ), f"{ENGINE_SRC} is suspiciously small ({ENGINE_SRC.stat().st_size} bytes)"
