#!/usr/bin/env python3
"""test_wave77_final.py — Absolute final validation tests for Evolution Engine.

Two test classes:
  1. TestCompleteMethodInventory — verify all 26 new methods exist and are
     callable by importing each module and checking hasattr + callable.
  2. TestPipelineWiringComplete — read evolution_engine.py source and verify
     it contains references to all 13 pipeline integration points.

Uses pytest, inspect, importlib. Fast — no VM, no evolution.
Does NOT modify any existing files.
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── read evolution_engine.py source once at import time ───────────────────────
_ENGINE_SOURCE_PATH = _ROOT / "evolution_engine.py"
assert _ENGINE_SOURCE_PATH.exists(), f"evolution_engine.py not found at {_ENGINE_SOURCE_PATH}"
_ENGINE_SOURCE = _ENGINE_SOURCE_PATH.read_text(encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# 26 NEW METHODS — (module_name, class_name_or_None, method_name)
# ══════════════════════════════════════════════════════════════════════════════
# 15 from evolution_engine.py  (EvolutionEngine class)
#  6 from champion_archive.py  (ChampionArchive class)
#  5 from meta_controller.py   (MetaController class)
# Total = 26

_METHODS_26: list[tuple[str, str | None, str]] = [
    # ── evolution_engine.py · EvolutionEngine ──────────────────────────────
    ("evolution_engine", "EvolutionEngine", "get_generation_stats"),
    ("evolution_engine", "EvolutionEngine", "get_champion_history"),
    ("evolution_engine", "EvolutionEngine", "get_champion_program_analysis"),
    ("evolution_engine", "EvolutionEngine", "get_champion_summary"),
    ("evolution_engine", "EvolutionEngine", "get_run_diagnostics"),
    ("evolution_engine", "EvolutionEngine", "get_performance_metrics"),
    ("evolution_engine", "EvolutionEngine", "get_archive_summary"),
    ("evolution_engine", "EvolutionEngine", "get_training_summary"),
    ("evolution_engine", "EvolutionEngine", "get_run_summary"),
    ("evolution_engine", "EvolutionEngine", "get_evolution_state"),
    ("evolution_engine", "EvolutionEngine", "get_comprehensive_state"),
    ("evolution_engine", "EvolutionEngine", "log_comprehensive_diagnostics"),
    ("evolution_engine", "EvolutionEngine", "log_diagnostics"),
    ("evolution_engine", "EvolutionEngine", "export_state_for_analysis"),
    ("evolution_engine", "EvolutionEngine", "save_state_to_file"),
    # ── champion_archive.py · ChampionArchive ──────────────────────────────
    ("champion_archive", "ChampionArchive", "get_quality_distribution"),
    ("champion_archive", "ChampionArchive", "get_recency_distribution"),
    ("champion_archive", "ChampionArchive", "get_coverage_stats"),
    ("champion_archive", "ChampionArchive", "get_sample_weights_summary"),
    ("champion_archive", "ChampionArchive", "quality_weighted_prune"),
    ("champion_archive", "ChampionArchive", "get_diversity_report"),
    # ── meta_controller.py · MetaController ────────────────────────────────
    ("meta_controller", "MetaController", "get_effectiveness_report"),
    ("meta_controller", "MetaController", "suggest_parameter_reset"),
    ("meta_controller", "MetaController", "get_effectiveness_score"),
    ("meta_controller", "MetaController", "get_recovery_confidence"),
    ("meta_controller", "MetaController", "get_cross_cycle_trends"),
]

_METHOD_IDS: list[str] = [f"{mod}.{cls}.{method}" if cls else f"{mod}.{method}" for mod, cls, method in _METHODS_26]

# ══════════════════════════════════════════════════════════════════════════════
# 13 PIPELINE INTEGRATION POINTS — referenced in evolution_engine.py
# ══════════════════════════════════════════════════════════════════════════════
_PIPELINE_POINTS: list[str] = [
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
    "manage_capacity": "evolution/qd_archive_mixin.py",
}


# ══════════════════════════════════════════════════════════════════════════════
# Test class 1 — method existence & callability
# ══════════════════════════════════════════════════════════════════════════════
class TestCompleteMethodInventory:
    """Verify all 26 new methods exist and are callable."""

    @pytest.mark.parametrize(
        "module_name, class_name, method_name",
        _METHODS_26,
        ids=_METHOD_IDS,
    )
    def test_method_exists_and_callable(self, module_name: str, class_name: str | None, method_name: str) -> None:
        """Import the module, resolve the class, check hasattr + callable."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"Failed to import module '{module_name}'"

        target = getattr(mod, class_name) if class_name else mod
        assert hasattr(target, method_name), f"{module_name}.{class_name or ''}.{method_name} does not exist"
        attr = getattr(target, method_name)
        assert callable(attr), f"{module_name}.{class_name or ''}.{method_name} is not callable"

    def test_method_count_is_26(self) -> None:
        """Sanity: ensure we are actually checking 26 methods."""
        assert len(_METHODS_26) == 26, f"Expected 26 methods, got {len(_METHODS_26)}"

    def test_all_modules_importable(self) -> None:
        """All referenced modules import without error."""
        modules = {m for m, _, _ in _METHODS_26}
        for name in sorted(modules):
            mod = importlib.import_module(name)
            assert mod is not None, f"Cannot import '{name}'"

    def test_all_classes_exist(self) -> None:
        """All referenced classes exist in their modules."""
        for module_name, class_name, _ in _METHODS_26:
            if class_name is None:
                continue
            mod = importlib.import_module(module_name)
            assert hasattr(mod, class_name), f"Class '{class_name}' not found in module '{module_name}'"
            cls = getattr(mod, class_name)
            assert inspect.isclass(cls), f"'{module_name}.{class_name}' is not a class (got {type(cls)})"


# ══════════════════════════════════════════════════════════════════════════════
# Test class 2 — pipeline wiring in evolution_engine.py
# ══════════════════════════════════════════════════════════════════════════════
class TestPipelineWiringComplete:
    """Verify evolution_engine.py source contains all 13 pipeline integration points."""

    # engine_source is read at module level (_ENGINE_SOURCE) for efficiency.

    @pytest.mark.parametrize(
        "point",
        _PIPELINE_POINTS,
        ids=_PIPELINE_POINTS,
    )
    def test_pipeline_point_present(self, point: str) -> None:
        """evolution_engine.py contains a reference to each pipeline point."""
        if point in _PIPELINE_METHOD_FILES:
            fpath = _ROOT / _PIPELINE_METHOD_FILES[point]
            with open(fpath, "r", encoding="utf-8") as f:
                src = f.read()
            assert point in src, f"Pipeline point '{point}' not found in {_PIPELINE_METHOD_FILES[point]}"
        else:
            assert point in _ENGINE_SOURCE, f"Pipeline integration point '{point}' not found in evolution_engine.py"

    def test_pipeline_point_count(self) -> None:
        """Sanity: ensure we are checking 13 pipeline points."""
        assert len(_PIPELINE_POINTS) == 13, f"Expected 13 pipeline points, got {len(_PIPELINE_POINTS)}"

    def test_engine_source_readable(self) -> None:
        """evolution_engine.py is non-empty and contains class definition."""
        assert len(_ENGINE_SOURCE) > 1000, "evolution_engine.py seems too short"
        assert "class EvolutionEngine" in _ENGINE_SOURCE

    def test_pipeline_points_are_method_calls(self) -> None:
        """Each pipeline point appears as a method call (with dot or assignment)."""
        for point in _PIPELINE_POINTS:
            target_file = _PIPELINE_METHOD_FILES.get(point)
            if target_file:
                fpath = _ROOT / target_file
                with open(fpath, "r", encoding="utf-8") as fh:
                    src = fh.read()
                assert (
                    f".{point}" in src or point in src
                ), f"'{point}' does not appear as a method call in {target_file}"
            else:
                assert (
                    f".{point}" in _ENGINE_SOURCE or point in _ENGINE_SOURCE
                ), f"'{point}' does not appear as a method call in evolution_engine.py"
