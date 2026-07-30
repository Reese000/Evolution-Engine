#!/usr/bin/env python3
"""Wave 23 — Absolute final validation tests.

Two fast, read-only test classes:

1. TestCompleteMethodInventory — verifies all 26 new methods exist and are
   callable on their respective classes/modules.

2. TestPipelineWiringComplete — verifies evolution_engine.py contains
   references to all 13 pipeline integration points.

Uses pytest parametrize for speed. No existing files are modified.
"""

from __future__ import annotations

import inspect
import os
import sys

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:

# ── imports ───────────────────────────────────────────────────────────────────
from config import EvolutionSection
from meta_controller import MetaController
from strategy_archive import StrategyArchive
from champion_archive import ChampionArchive
from novelty import SparseArchive
from qd_archive import MAPElitesArchive
from mutation_proposer import DirectedMutator
from monitoring_alerts import AlertManager
from metrics import MetricsCollector
from progression_tracker import ProgressionTracker
import data_rotation


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestCompleteMethodInventory — all 26 new methods exist and are callable
# ══════════════════════════════════════════════════════════════════════════════

# (target_class_or_module, method_name, is_static_or_module_func)
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
    # EvolutionSection (from config.py — EvolutionConfig's evolution params)
    (EvolutionSection, "get_config_summary", False),
    (EvolutionSection, "validate_config", False),
]

# Sanity check: exactly 26 specs
assert len(_METHOD_SPECS) == 26, f"Expected 26 method specs, got {len(_METHOD_SPECS)}"


class TestCompleteMethodInventory:
    """Verify all 26 wave-23 methods exist and are callable."""

    @pytest.mark.parametrize(
        "target,name,is_static",
        _METHOD_SPECS,
        ids=[f"{t.__name__ if hasattr(t, '__name__') else t}.{n}" for t, n, _ in _METHOD_SPECS],
    )
    def test_method_exists_and_is_callable(self, target, name, is_static):
        """Each method must be a callable attribute on its class/module."""
        attr = getattr(target, name, None)
        assert attr is not None, (
            f"{getattr(target, '__name__', target)} has no attribute '{name}'"
        )
        assert callable(attr), (
            f"{getattr(target, '__name__', target)}.{name} is not callable"
        )

    @pytest.mark.parametrize(
        "target,name,is_static",
        _METHOD_SPECS,
        ids=[f"{t.__name__ if hasattr(t, '__name__') else t}.{n}" for t, n, _ in _METHOD_SPECS],
    )
    def test_method_has_valid_signature(self, target, name, is_static):
        """Each method must have an inspectable signature (not a C builtin)."""
        attr = getattr(target, name)
        try:
            sig = inspect.signature(attr)
        except (ValueError, TypeError):
            pytest.fail(
                f"{getattr(target, '__name__', target)}.{name} has no inspectable signature"
            )
        params = list(sig.parameters.keys())
        if is_static and isinstance(target, type):
            pass  # static/class methods — just needs to be inspectable
        elif not isinstance(target, type):
            pass  # module function — just needs to exist
        else:
            # instance method — should have 'self' as first param
            assert len(params) >= 1, (
                f"{target.__name__}.{name} signature has no params"
            )

    def test_total_method_count(self):
        """Verify we test exactly 26 distinct method names."""
        names = {name for _, name, _ in _METHOD_SPECS}
        assert len(names) == 26, f"Expected 26 unique names, got {len(names)}: {names}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestPipelineWiringComplete — evolution_engine.py references all 13 points
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


class TestPipelineWiringComplete:
    """Verify evolution_engine.py contains references to all 13 integration points."""

    @pytest.fixture
    def engine_source(self) -> str:
        """Read evolution_engine.py once per test class."""
        path = os.path.join(_ROOT, "evolution_engine.py")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @pytest.mark.parametrize("method_name", _PIPELINE_POINTS)
    def test_method_referenced_in_engine(self, method_name, engine_source):
        """Each pipeline integration point must appear in evolution_engine.py."""
        assert method_name in engine_source, (
            f"'{method_name}' not found in evolution_engine.py — pipeline not wired"
        )

    def test_all_thirteen_points_present(self, engine_source):
        """Aggregate check: all 13 pipeline integration points referenced."""
        missing = [m for m in _PIPELINE_POINTS if m not in engine_source]
        assert not missing, (
            f"Missing from evolution_engine.py pipeline: {missing}"
        )

    def test_pipeline_point_count(self):
        """Verify we test exactly 13 unique pipeline integration points."""
        assert len(_PIPELINE_POINTS) == 13, (
            f"Expected 13 pipeline points, got {len(_PIPELINE_POINTS)}"
        )
        assert len(set(_PIPELINE_POINTS)) == 13, (
            "Duplicate pipeline points detected"
        )
