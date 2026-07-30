"""Wave 22 Final Comprehensive Validation Tests.

Three test classes verifying:
1. All 12 enhanced modules are importable without errors
2. All 26 new methods exist and are callable
3. evolution_engine.py source contains all 13 wiring references
"""
from __future__ import annotations

import importlib
import inspect
import os
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# Class 1 — All 12 Enhanced Modules Importable
# ===========================================================================


class TestAllModulesImportable:
    """Verify all 12 enhanced modules import without errors."""

    MODULES = [
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
        "evolution_engine",
    ]

    @pytest.mark.parametrize("module_name", MODULES, ids=MODULES)
    def test_module_importable(self, module_name: str) -> None:
        """Each enhanced module must import cleanly."""
        mod = importlib.import_module(module_name)
        assert mod is not None, f"importlib.import_module returned None for {module_name}"
        assert hasattr(mod, "__name__"), f"Module {module_name} missing __name__"

    def test_all_twelve_imported(self) -> None:
        """Sanity check: import all modules in one shot and verify count."""
        imported = []
        for name in self.MODULES:
            mod = importlib.import_module(name)
            assert mod is not None
            imported.append(name)
        assert len(imported) == 12, f"Expected 12 modules, got {len(imported)}"


# ===========================================================================
# Class 2 — All 26 New Methods Exist and Are Callable
# ===========================================================================


class TestAllNewMethodsCallable:
    """Verify all 26 new methods exist and are callable.

    For each method we check:
    - The owning class/module has the attribute (hasattr)
    - The attribute is callable (callable)
    """

    # -- strategy_archive.py : StrategyArchive class — 6 methods -----------
    SA_METHODS = [
        "adaptive_exploration_bonus",
        "get_strategy_diversity",
        "get_dominance_report",
        "get_archive_summary",
        "time_weighted_success_rate",
        "per_level_effectiveness",
    ]

    # -- champion_archive.py : ChampionArchive class — 6 methods ------------
    CA_METHODS = [
        "quality_weighted_prune",
        "get_diversity_report",
        "get_quality_distribution",
        "get_recency_distribution",
        "prune_stale",
        "get_coverage_stats",
    ]

    # -- novelty.py : SparseArchive class — 5 methods -----------------------
    NOV_METHODS = [
        "manage_capacity",
        "decayed_novelty",
        "adaptive_threshold",
        "get_diversity_score",
        "get_archive_summary",
    ]

    # -- qd_archive.py : MAPElitesArchive class — 4 methods ----------------
    QD_METHODS = [
        "get_improvement_velocity",
        "get_illumination_diversity",
        "selection_bias",
        "get_freshest_cell",
    ]

    # -- mutation_proposer.py : DirectedMutator class — 2 static methods ----
    MP_METHODS = [
        "record_mutation_outcome",
        "get_mutation_stats",
    ]

    # -- monitoring_alerts.py : AlertManager class — 2 methods --------------
    MON_METHODS = [
        "send_monoculture_alert",
        "send_recovery_stuck_alert",
    ]

    # -- progression_tracker.py : ProgressionTracker class — 1 method -------
    PT_METHODS = [
        "get_plateau_analysis",
    ]

    # Aggregate map: class_or_module_path -> (import_path, class_name, methods)
    METHOD_MAP = [
        ("strategy_archive", "StrategyArchive", SA_METHODS),
        ("champion_archive", "ChampionArchive", CA_METHODS),
        ("novelty", "SparseArchive", NOV_METHODS),
        ("qd_archive", "MAPElitesArchive", QD_METHODS),
        ("mutation_proposer", "DirectedMutator", MP_METHODS),
        ("monitoring_alerts", "AlertManager", MON_METHODS),
        ("progression_tracker", "ProgressionTracker", PT_METHODS),
    ]

    def test_total_method_count(self) -> None:
        """Verify we are checking exactly 26 methods."""
        total = sum(len(methods) for _, _, methods in self.METHOD_MAP)
        assert total == 26, f"Expected 26 methods, got {total}"

    @pytest.mark.parametrize(
        "module_name,class_name,method_name",
        [
            (mod, cls, m)
            for mod, cls, methods in [
                ("strategy_archive", "StrategyArchive", SA_METHODS),
                ("champion_archive", "ChampionArchive", CA_METHODS),
                ("novelty", "SparseArchive", NOV_METHODS),
                ("qd_archive", "MAPElitesArchive", QD_METHODS),
                ("mutation_proposer", "DirectedMutator", MP_METHODS),
                ("monitoring_alerts", "AlertManager", MON_METHODS),
                ("progression_tracker", "ProgressionTracker", PT_METHODS),
            ]
            for m in methods
        ],
        ids=[
            f"{cls}.{m}"
            for _, cls, methods in [
                ("strategy_archive", "StrategyArchive", SA_METHODS),
                ("champion_archive", "ChampionArchive", CA_METHODS),
                ("novelty", "SparseArchive", NOV_METHODS),
                ("qd_archive", "MAPElitesArchive", QD_METHODS),
                ("mutation_proposer", "DirectedMutator", MP_METHODS),
                ("monitoring_alerts", "AlertManager", MON_METHODS),
                ("progression_tracker", "ProgressionTracker", PT_METHODS),
            ]
            for m in methods
        ],
    )
    def test_method_exists_and_callable(
        self, module_name: str, class_name: str, method_name: str
    ) -> None:
        """Each new method must exist on its owning class and be callable."""
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        assert hasattr(cls, method_name), (
            f"{class_name}.{method_name} not found on {module_name}.{class_name}"
        )
        attr = getattr(cls, method_name)
        assert callable(attr), (
            f"{class_name}.{method_name} is not callable (type={type(attr).__name__})"
        )


# ===========================================================================
# Class 3 — Pipeline Wiring Verified in Source
# ===========================================================================


class TestPipelineWiringVerified:
    """Verify evolution_engine.py source contains all 13 wiring references."""

    WIRING_REFS = [
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

    @pytest.fixture(scope="class")
    def evolution_source(self) -> str:
        """Read evolution_engine.py source once per class."""
        src_path = os.path.join(PROJECT_ROOT, "evolution_engine.py")
        assert os.path.isfile(src_path), f"Missing {src_path}"
        with open(src_path, encoding="utf-8") as f:
            return f.read()

    @pytest.mark.parametrize("ref", WIRING_REFS, ids=WIRING_REFS)
    def test_wiring_reference_present(
        self, ref: str, evolution_source: str
    ) -> None:
        """Each wiring reference must appear in evolution_engine.py source."""
        assert ref in evolution_source, (
            f"'{ref}' not found in evolution_engine.py source"
        )

    def test_total_wiring_count(self) -> None:
        """Verify exactly 13 wiring references are tested."""
        assert len(self.WIRING_REFS) == 13, (
            f"Expected 13 wiring refs, got {len(self.WIRING_REFS)}"
        )

    def test_all_wiring_found(self, evolution_source: str) -> None:
        """All 13 wiring references must be found in one check."""
        missing = [ref for ref in self.WIRING_REFS if ref not in evolution_source]
        assert not missing, (
            f"Missing wiring references in evolution_engine.py: {missing}"
        )
