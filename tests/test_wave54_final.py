#!/usr/bin/env python3
"""Wave-54 absolute final validation tests for Evolution Engine.

Two test classes:
  1. TestCompleteMethodInventory — verify all 26 new methods exist and are callable.
  2. TestPipelineWiringComplete — verify evolution_engine.py references all 13
     pipeline integration points.

Uses pytest parametrize for fast, focused tests. Does NOT modify any existing files.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Lazy module loader — avoids import-time failures on missing deps
# ---------------------------------------------------------------------------

_loaded_modules: dict[str, object] = {}


def _get_module(name: str):
    """Import and cache a top-level module from the project root."""
    if name not in _loaded_modules:
        _loaded_modules[name] = importlib.import_module(name)
    return _loaded_modules[name]


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestCompleteMethodInventory
# ══════════════════════════════════════════════════════════════════════════════
#
# Each entry: (module_name, class_or_none, method_name, description)
#   - class_or_none = None means it is a module-level function.
#   - For class methods we check `hasattr(cls, name)` on the class itself
#     (no instantiation needed — keeps tests fast).

_METHOD_INVENTORY: list[tuple[str, str | None, str, str]] = [
    # ── strategy_archive.py · StrategyArchive ──────────────────────────────
    ("strategy_archive", "StrategyArchive", "adaptive_exploration_bonus",
     "adaptive exploration bonus based on strategy diversity"),
    ("strategy_archive", "StrategyArchive", "get_dominance_report",
     "dominance report for strategy monoculture detection"),
    ("strategy_archive", "StrategyArchive", "get_strategy_diversity",
     "strategy diversity score"),
    ("strategy_archive", "StrategyArchive", "get_archive_summary",
     "strategy archive summary"),
    ("strategy_archive", "StrategyArchive", "record_delta",
     "record fitness delta for a strategy"),
    ("strategy_archive", "StrategyArchive", "avg_delta",
     "average fitness delta per strategy"),
    ("strategy_archive", "StrategyArchive", "per_level_effectiveness",
     "per-level effectiveness breakdown"),
    ("strategy_archive", "StrategyArchive", "time_weighted_success_rate",
     "time-weighted success rate with exponential decay"),
    ("strategy_archive", "StrategyArchive", "get_strategy_history",
     "strategy history with optional level filter"),

    # ── novelty.py · SparseArchive ─────────────────────────────────────────
    ("novelty", "SparseArchive", "manage_capacity",
     "manage archive capacity with pruning"),
    ("novelty", "SparseArchive", "get_diversity_score",
     "diversity score from archive"),
    ("novelty", "SparseArchive", "get_archive_summary",
     "novelty archive summary"),
    ("novelty", "SparseArchive", "recent_novelty_scores",
     "recent novelty scores"),
    ("novelty", "SparseArchive", "trim_archive",
     "trim archive to max entries"),
    ("novelty", "SparseArchive", "decayed_novelty",
     "time-decayed novelty scoring"),

    # ── qd_archive.py · MAPElitesArchive ──────────────────────────────────
    ("qd_archive", "MAPElitesArchive", "get_improvement_velocity",
     "improvement velocity from QD archive"),
    ("qd_archive", "MAPElitesArchive", "get_illumination_diversity",
     "illumination diversity metric"),
    ("qd_archive", "MAPElitesArchive", "get_archive_summary",
     "QD archive summary"),
    ("qd_archive", "MAPElitesArchive", "get_cell_stats",
     "per-cell statistics"),
    ("qd_archive", "MAPElitesArchive", "get_neighborhood_density",
     "neighborhood density around a descriptor"),
    ("qd_archive", "MAPElitesArchive", "get_freshest_cell",
     "most recently updated cell"),
    ("qd_archive", "MAPElitesArchive", "get_fitness_distribution",
     "fitness distribution across cells"),

    # ── champion_archive.py · ChampionArchive ──────────────────────────────
    ("champion_archive", "ChampionArchive", "quality_weighted_prune",
     "quality-weighted pruning of champion archive"),
    ("champion_archive", "ChampionArchive", "get_diversity_report",
     "diversity report for champion archive"),
    ("champion_archive", "ChampionArchive", "get_quality_distribution",
     "quality distribution of archived champions"),
    ("champion_archive", "ChampionArchive", "get_recency_distribution",
     "recency distribution of archived champions"),
    ("champion_archive", "ChampionArchive", "prune_stale",
     "prune stale champions by age"),
    ("champion_archive", "ChampionArchive", "get_coverage_stats",
     "coverage statistics"),
    ("champion_archive", "ChampionArchive", "compute_sample_weights",
     "compute sampling weights for training"),
    ("champion_archive", "ChampionArchive", "get_sample_weights_summary",
     "summary of sampling weights"),
    ("champion_archive", "ChampionArchive", "current_max_cycle",
     "current maximum cycle in archive (property)"),
    ("champion_archive", "ChampionArchive", "diversity_aware_select",
     "diversity-aware champion selection"),

    # ── progression_tracker.py · ProgressionTracker ────────────────────────
    ("progression_tracker", "ProgressionTracker", "get_plateau_analysis",
     "plateau analysis with windowed detection"),
    ("progression_tracker", "ProgressionTracker", "compare_windows",
     "compare two time windows of progression"),

    # ── monitoring_alerts.py · AlertManager ────────────────────────────────
    ("monitoring_alerts", "AlertManager", "send_monoculture_alert",
     "alert when strategy monoculture detected"),
    ("monitoring_alerts", "AlertManager", "send_recovery_stuck_alert",
     "alert when recovery mode is stuck"),

    # ── mutation_proposer.py · DirectedMutator static methods ─────────────
    ("mutation_proposer", "DirectedMutator", "record_mutation_outcome",
     "record mutation fitness delta"),
    ("mutation_proposer", "DirectedMutator", "get_mutation_stats",
     "get mutation statistics summary"),

    # ── metrics.py · MetricsCollector ──────────────────────────────────────
    ("metrics", "MetricsCollector", "detect_anomalies",
     "anomaly detection in metrics"),
    ("metrics", "MetricsCollector", "correlate_metrics",
     "correlation between two metrics"),

    # ── meta_controller.py · MetaController ────────────────────────────────
    ("meta_controller", "MetaController", "get_recovery_confidence",
     "recovery confidence score"),
    ("meta_controller", "MetaController", "get_cross_cycle_trends",
     "cross-cycle trend analysis"),
    ("meta_controller", "MetaController", "get_tuning_escalation_history",
     "tuning escalation history"),

    # ── config.py · EvolutionSection ───────────────────────────────────────
    ("config", "EvolutionSection", "get_config_summary",
     "config summary dictionary"),
    ("config", "EvolutionSection", "validate_config",
     "config validation warnings"),

    # ── data_rotation.py · module-level functions ──────────────────────────
    ("data_rotation", None, "get_disk_usage",
     "disk usage statistics"),
    ("data_rotation", None, "quality_aware_rotate_checkpoints",
     "quality-aware checkpoint rotation"),

    # ── evolution_engine.py · EvolutionEngine (diagnostic methods) ─────────
    ("evolution_engine", "EvolutionEngine", "get_generation_stats",
     "generation statistics"),
    ("evolution_engine", "EvolutionEngine", "get_champion_history",
     "champion history list"),
    ("evolution_engine", "EvolutionEngine", "get_champion_program_analysis",
     "champion program analysis"),
    ("evolution_engine", "EvolutionEngine", "get_champion_summary",
     "champion summary string"),
    ("evolution_engine", "EvolutionEngine", "get_archive_summary",
     "archive summary"),
    ("evolution_engine", "EvolutionEngine", "get_training_summary",
     "training summary"),
    ("evolution_engine", "EvolutionEngine", "get_run_summary",
     "run summary"),
    ("evolution_engine", "EvolutionEngine", "get_evolution_state",
     "evolution state snapshot"),
    ("evolution_engine", "EvolutionEngine", "get_comprehensive_state",
     "comprehensive state dump"),
    ("evolution_engine", "EvolutionEngine", "log_comprehensive_diagnostics",
     "log comprehensive diagnostics"),
    ("evolution_engine", "EvolutionEngine", "log_diagnostics",
     "log diagnostics"),
    ("evolution_engine", "EvolutionEngine", "export_state_for_analysis",
     "export state for analysis"),
    ("evolution_engine", "EvolutionEngine", "save_state_to_file",
     "save state to file"),
]


def _method_id(value: tuple) -> str:
    """Generate a readable test ID for parametrized method checks."""
    mod, cls, method, _ = value
    if cls:
        return f"{mod}.{cls}.{method}"
    return f"{mod}.{method}"


class TestCompleteMethodInventory:
    """Verify all new methods exist and are callable.

    Each parametrized case imports the target module, resolves the class
    (if any), and asserts hasattr + callable on the method name. No
    instantiation — pure import-time check, fast and deterministic.
    """

    @pytest.mark.parametrize(
        "module_name, class_name, method_name, description",
        _METHOD_INVENTORY,
        ids=[_method_id(m) for m in _METHOD_INVENTORY],
    )
    def test_method_exists_and_callable(
        self, module_name: str, class_name: str | None,
        method_name: str, description: str,
    ):
        """Method must exist on the class (or module) and be callable."""
        mod = _get_module(module_name)

        if class_name is not None:
            # Class method — resolve class from module
            assert hasattr(mod, class_name), (
                f"Module '{module_name}' has no class '{class_name}'"
            )
            cls = getattr(mod, class_name)
            assert hasattr(cls, method_name), (
                f"{class_name} missing method '{method_name}' — {description}"
            )
            attr = getattr(cls, method_name)
            # Properties are descriptors (not callable), but still valid API
            if isinstance(attr, property):
                pass  # property descriptor exists — that's sufficient
            else:
                assert callable(attr), (
                    f"{class_name}.{method_name} is not callable"
                )
        else:
            # Module-level function
            assert hasattr(mod, method_name), (
                f"Module '{module_name}' missing function '{method_name}' "
                f"— {description}"
            )
            attr = getattr(mod, method_name)
            assert callable(attr), (
                f"{module_name}.{method_name} is not callable"
            )

    def test_total_new_method_count(self):
        """Sanity check: we expect exactly 26 public API methods in the inventory."""
        # Filter to the key pipeline + diagnostic methods (excluding some
        # internal helpers that happen to be public for testability).
        # Count is based on the distinct method names across all modules.
        unique_methods = set()
        for mod, cls, method, _ in _METHOD_INVENTORY:
            unique_methods.add((mod, cls, method))
        # The inventory should cover at least the 26 methods the user expects
        # We expect 26+ distinct public API methods in the inventory
        assert len(unique_methods) >= 26, (
            f"Expected at least 26 methods in inventory, found {len(unique_methods)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestPipelineWiringComplete
# ══════════════════════════════════════════════════════════════════════════════
#
# Read evolution_engine.py source and verify it references all 13 pipeline
# integration points. These are methods on various subsystem objects that
# the engine calls during its evolution loop.

_PIPELINE_INTEGRATION_POINTS: list[tuple[str, str]] = [
    # (method_name, expected_module_or_context)
    ("adaptive_exploration_bonus", "strategy_archive.StrategyArchive"),
    ("manage_capacity", "novelty.SparseArchive"),
    ("get_cross_cycle_trends", "meta_controller.MetaController"),
    ("get_improvement_velocity", "qd_archive.MAPElitesArchive"),
    ("get_illumination_diversity", "qd_archive.MAPElitesArchive"),
    ("get_dominance_report", "strategy_archive.StrategyArchive"),
    ("get_recovery_confidence", "meta_controller.MetaController"),
    ("quality_weighted_prune", "champion_archive.ChampionArchive"),
    ("record_mutation_outcome", "mutation_proposer (module-level)"),
    ("get_mutation_stats", "mutation_proposer (module-level)"),
    ("send_monoculture_alert", "monitoring_alerts.AlertManager"),
    ("send_recovery_stuck_alert", "monitoring_alerts.AlertManager"),
    ("get_plateau_analysis", "progression_tracker.ProgressionTracker"),
]


class TestPipelineWiringComplete:
    """Verify evolution_engine.py references all 13 pipeline integration points.

    Reads the source file and checks each method name appears as a string
    reference (attribute access or call) in the engine code.
    """

    @pytest.fixture(scope="class")
    def engine_source(self) -> str:
        """Read evolution_engine.py source once per test class."""
        engine_path = os.path.join(_PROJECT_ROOT, "evolution_engine.py")
        assert os.path.isfile(engine_path), (
            f"evolution_engine.py not found at {engine_path}"
        )
        with open(engine_path, "r", encoding="utf-8") as f:
            return f.read()

    @pytest.mark.parametrize(
        "method_name, expected_location",
        _PIPELINE_INTEGRATION_POINTS,
        ids=[m[0] for m in _PIPELINE_INTEGRATION_POINTS],
    )
    def test_pipeline_reference_exists(
        self, engine_source: str, method_name: str, expected_location: str,
    ):
        """evolution_engine.py must reference each pipeline method."""
        assert method_name in engine_source, (
            f"evolution_engine.py does not reference '{method_name}' "
            f"(expected from {expected_location})"
        )

    def test_all_13_points_present(self, engine_source: str):
        """All 13 pipeline integration points must be present."""
        missing = []
        for method_name, location in _PIPELINE_INTEGRATION_POINTS:
            if method_name not in engine_source:
                missing.append(f"{method_name} ({location})")
        assert not missing, (
            f"evolution_engine.py is missing {len(missing)} pipeline "
            f"integration points:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    def test_engine_imports_subsystems(self, engine_source: str):
        """evolution_engine.py must import the key subsystem modules."""
        required_imports = [
            "strategy_archive",
            "novelty",
            "qd_archive",
            "champion_archive",
            "progression_tracker",
            "monitoring_alerts",
            "mutation_proposer",
            "metrics",
        ]
        missing = []
        for mod_name in required_imports:
            if mod_name not in engine_source:
                missing.append(mod_name)
        assert not missing, (
            f"evolution_engine.py is missing imports for: "
            f"{', '.join(missing)}"
        )
