#!/usr/bin/env python3
"""Wave 21 — Final documentation validation tests.

Two fast, read-only test classes:

1. TestAllNewMethodsDocumented — uses inspect to verify every new method has
   a non-empty docstring with at least 10 characters. Checks all 26 new methods
   across all modules.

2. TestCodeQualityFinal — verifies no TODO/FIXME/HACK comments in new methods,
   verifies all new methods follow snake_case naming, verifies all dict-returning
   methods have return type annotations.

Uses pytest parametrize for speed. No existing files are modified.
"""
from __future__ import annotations

import inspect
import os
import re
import sys
import typing

import pytest

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:

# ── lazy imports (one-time, cached by Python) ─────────────────────────────────
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
# Shared method specs — all 26 new methods across all modules
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
    # EvolutionSection (from config.py)
    (EvolutionSection, "get_config_summary", False),
    (EvolutionSection, "validate_config", False),
]

# Sanity: exactly 26 specs
assert len(_METHOD_SPECS) == 26, f"Expected 26 method specs, got {len(_METHOD_SPECS)}"

_METHOD_IDS = [
    f"{t.__name__ if hasattr(t, '__name__') else 'module'}.{n}"
    for t, n, _ in _METHOD_SPECS
]

# ── Dict-returning methods (must have return type annotations) ────────────────
_DICT_RETURN_METHODS: set[str] = {
    "get_cross_cycle_trends",
    "get_recovery_confidence",
    "get_tuning_escalation_history",
    "per_level_effectiveness",
    "get_dominance_report",
    "get_diversity_report",
    "get_improvement_velocity",
    "get_illumination_diversity",
    "get_mutation_stats",
    "detect_anomalies",
    "correlate_metrics",
    "get_disk_usage",
    "get_plateau_analysis",
    "compare_windows",
    "get_config_summary",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestAllNewMethodsDocumented — docstring presence & quality
# ══════════════════════════════════════════════════════════════════════════════

class TestAllNewMethodsDocumented:
    """Verify every new method has a non-empty docstring ≥ 10 characters."""

    @pytest.mark.parametrize(
        "target,name,is_static",
        _METHOD_SPECS,
        ids=_METHOD_IDS,
    )
    def test_docstring_exists(self, target, name, is_static):
        """Each method must have a docstring with at least 10 characters."""
        attr = getattr(target, name)
        docstring = inspect.getdoc(attr)
        assert docstring is not None, (
            f"{getattr(target, '__name__', 'module')}.{name} has no docstring"
        )
        stripped = docstring.strip()
        assert len(stripped) >= 10, (
            f"{getattr(target, '__name__', 'module')}.{name} docstring too short: "
            f"{len(stripped)} chars (need ≥ 10): {stripped!r}"
        )

    def test_total_documented_count(self):
        """Verify we check exactly 26 methods and all are documented."""
        documented = 0
        for target, name, _ in _METHOD_SPECS:
            attr = getattr(target, name)
            docstring = inspect.getdoc(attr)
            if docstring and len(docstring.strip()) >= 10:
                documented += 1
        assert documented == 26, (
            f"Expected 26 fully documented methods, got {documented}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestCodeQualityFinal — naming, comments, type annotations
# ══════════════════════════════════════════════════════════════════════════════

# Regex for snake_case: lowercase letters/digits/underscores, no leading/trailing _
_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

# Patterns that indicate code-smell comments
_TODO_PATTERNS = re.compile(r"\bTODO\b|\bFIXME\b|\bHACK\b", re.IGNORECASE)


class TestCodeQualityFinal:
    """Verify no TODO/FIXME/HACK in new methods, snake_case naming,
    and return type annotations on dict-returning methods."""

    @pytest.mark.parametrize(
        "target,name,is_static",
        _METHOD_SPECS,
        ids=_METHOD_IDS,
    )
    def test_no_todo_fixme_hack(self, target, name, is_static):
        """New methods must not contain TODO/FIXME/HACK comments."""
        attr = getattr(target, name)
        source, _ = inspect.getsourcelines(attr)
        for lineno, line in enumerate(source, 1):
            # Skip comments that are in docstrings — only check code lines
            # Simple heuristic: skip lines starting with # (pure comment lines)
            # but catch inline comments on code lines
            stripped = line.strip()
            # Allow # noqa-style comments but flag TODO/FIXME/HACK
            if _TODO_PATTERNS.search(stripped):
                # Check if it's inside a docstring (between triple quotes)
                # Simple: check if the match is in a docstring context
                in_docstring = False
                for prev_line in source[:lineno - 1]:
                    if '"""' in prev_line or "'''" in prev_line:
                        in_docstring = not in_docstring
                if not in_docstring:
                    pytest.fail(
                        f"{getattr(target, '__name__', 'module')}.{name} "
                        f"line {lineno} contains TODO/FIXME/HACK: {stripped!r}"
                    )

    @pytest.mark.parametrize(
        "target,name,is_static",
        _METHOD_SPECS,
        ids=_METHOD_IDS,
    )
    def test_snake_case_naming(self, target, name, is_static):
        """All new method names must follow snake_case convention."""
        assert _SNAKE_CASE_RE.match(name), (
            f"{getattr(target, '__name__', 'module')}.{name} "
            f"does not follow snake_case naming"
        )

    @pytest.mark.parametrize(
        "target,name,is_static",
        _METHOD_SPECS,
        ids=_METHOD_IDS,
    )
    def test_return_type_annotation(self, target, name, is_static):
        """All new methods must have a return type annotation."""
        attr = getattr(target, name)
        sig = inspect.signature(attr)
        ret = sig.return_annotation
        assert ret is not inspect.Signature.empty, (
            f"{getattr(target, '__name__', 'module')}.{name} "
            f"missing return type annotation"
        )

    @pytest.mark.parametrize(
        "target,name,is_static",
        [(t, n, s) for t, n, s in _METHOD_SPECS if n in _DICT_RETURN_METHODS],
        ids=[
            f"{t.__name__ if hasattr(t, '__name__') else 'module'}.{n}"
            for t, n, s in _METHOD_SPECS
            if n in _DICT_RETURN_METHODS
        ],
    )
    def test_dict_returning_method_has_annotation(self, target, name, is_static):
        """Methods returning dicts must have explicit Dict/dict return annotations."""
        attr = getattr(target, name)
        sig = inspect.signature(attr)
        ret = sig.return_annotation
        ret_str = str(ret)
        assert "dict" in ret_str.lower() or "Dict" in ret_str, (
            f"{getattr(target, '__name__', 'module')}.{name} "
            f"returns dict but annotation is {ret_str!r} (expected Dict/dict)"
        )

    def test_all_methods_snake_case(self):
        """Aggregate check: all 26 method names are snake_case."""
        bad = []
        for _, name, _ in _METHOD_SPECS:
            if not _SNAKE_CASE_RE.match(name):
                bad.append(name)
        assert not bad, f"Non-snake_case names: {bad}"

    def test_all_methods_have_return_annotation(self):
        """Aggregate check: all 26 methods have return type annotations."""
        missing = []
        for target, name, _ in _METHOD_SPECS:
            attr = getattr(target, name)
            sig = inspect.signature(attr)
            if sig.return_annotation is inspect.Signature.empty:
                missing.append(
                    f"{getattr(target, '__name__', 'module')}.{name}"
                )
        assert not missing, f"Missing return annotations: {missing}"
