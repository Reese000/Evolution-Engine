#!/usr/bin/env python3
"""Wave 15: Documentation and validation tests for enhanced modules.

Verifies that all enhanced modules and methods have proper docstrings,
type annotations, and code quality standards.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import re
import sys
from pathlib import Path

import pytest

# ── Path setup ──────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:

# ── Module imports ──────────────────────────────────────────────────────
# Lazy imports to keep collection fast; actual import happens in tests.
_MOD_PATHS = {
    "meta_controller": _PROJECT_ROOT / "meta_controller.py",
    "strategy_archive": _PROJECT_ROOT / "strategy_archive.py",
    "champion_archive": _PROJECT_ROOT / "champion_archive.py",
    "novelty": _PROJECT_ROOT / "novelty.py",
    "qd_archive": _PROJECT_ROOT / "qd_archive.py",
    "mutation_proposer": _PROJECT_ROOT / "mutation_proposer.py",
    "monitoring_alerts": _PROJECT_ROOT / "monitoring_alerts.py",
    "metrics": _PROJECT_ROOT / "metrics.py",
    "data_rotation": _PROJECT_ROOT / "data_rotation.py",
    "progression_tracker": _PROJECT_ROOT / "progression_tracker.py",
    "config": _PROJECT_ROOT / "config.py",
}

# Cache for module objects (imported once, reused across tests)
_mod_cache: dict[str, object] = {}


def _get_module(name: str):
    """Import and cache a module from the project root."""
    if name not in _mod_cache:
        _mod_cache[name] = importlib.import_module(name)
    return _mod_cache[name]


def _get_source(name: str) -> str:
    """Read the source text of a module file."""
    path = _MOD_PATHS[name]
    return path.read_text(encoding="utf-8")


def _get_source_lines(name: str) -> list[str]:
    """Read source lines of a module file."""
    path = _MOD_PATHS[name]
    return path.read_text(encoding="utf-8").splitlines()


# ═════════════════════════════════════════════════════════════════════════
# Test 1: Module Docstrings
# ═════════════════════════════════════════════════════════════════════════

class TestModuleDocstrings:
    """Verify all enhanced modules have docstrings."""

    @pytest.mark.parametrize("module_name", [
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
    ])
    def test_module_has_docstring(self, module_name):
        """Each enhanced module must have a non-empty module-level docstring."""
        source = _get_source(module_name)
        tree = ast.parse(source, filename=f"{module_name}.py")
        docstring = ast.get_docstring(tree)
        assert docstring, (
            f"Module '{module_name}' is missing a module-level docstring"
        )

    @pytest.mark.parametrize("module_name", [
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
    ])
    def test_module_file_exists(self, module_name):
        """Verify the module source file exists on disk."""
        path = _MOD_PATHS[module_name]
        assert path.is_file(), f"Source file for '{module_name}' not found at {path}"


# ═════════════════════════════════════════════════════════════════════════
# Test 2: Method Docstrings
# ═════════════════════════════════════════════════════════════════════════

# Mapping: method_name -> (module_name, class_or_func_name_or_None)
# class_or_func_name is the class name if the method is on a class, or None for module-level functions.
_METHOD_MAP: dict[str, tuple[str, str | None]] = {
    # MetaController methods
    "get_cross_cycle_trends": ("meta_controller", "MetaController"),
    "get_recovery_confidence": ("meta_controller", "MetaController"),
    # StrategyArchive methods
    "get_dominance_report": ("strategy_archive", "StrategyArchive"),
    # ChampionArchive methods
    "quality_weighted_prune": ("champion_archive", "ChampionArchive"),
    # Novelty (SparseArchive) methods
    "manage_capacity": ("novelty", "SparseArchive"),
    "decayed_novelty": ("novelty", "SparseArchive"),
    # QDArchive (MAPElitesArchive) methods
    "get_improvement_velocity": ("qd_archive", "MAPElitesArchive"),
    "get_illumination_diversity": ("qd_archive", "MAPElitesArchive"),
    # MetricsCollector methods
    "detect_anomalies": ("metrics", "MetricsCollector"),
    "correlate_metrics": ("metrics", "MetricsCollector"),
    # data_rotation module-level functions
    "get_disk_usage": ("data_rotation", None),
    "quality_aware_rotate_checkpoints": ("data_rotation", None),
    # ProgressionTracker methods
    "get_plateau_analysis": ("progression_tracker", "ProgressionTracker"),
    "compare_windows": ("progression_tracker", "ProgressionTracker"),
    # EvolutionSection methods (nested inside config.py)
    "get_config_summary": ("config", "EvolutionSection"),
    "validate_config": ("config", "EvolutionSection"),
    # DirectedMutator static methods (mutation_proposer.py)
    "record_mutation_outcome": ("mutation_proposer", "DirectedMutator"),
    "get_mutation_stats": ("mutation_proposer", "DirectedMutator"),
    # AlertManager methods
    "send_monoculture_alert": ("monitoring_alerts", "AlertManager"),
    "send_recovery_stuck_alert": ("monitoring_alerts", "AlertManager"),
}


def _get_method_obj(method_name: str):
    """Resolve a method name to its actual Python object."""
    mod_name, class_name = _METHOD_MAP[method_name]
    mod = _get_module(mod_name)
    if class_name is None:
        return getattr(mod, method_name)
    cls = getattr(mod, class_name)
    return getattr(cls, method_name)


class TestMethodDocstrings:
    """Verify all new/enhanced methods have docstrings."""

    @pytest.mark.parametrize("method_name", list(_METHOD_MAP.keys()))
    def test_method_has_docstring(self, method_name):
        """Each enhanced method must have a non-empty docstring."""
        mod_name, class_name = _METHOD_MAP[method_name]
        source = _get_source(mod_name)
        tree = ast.parse(source, filename=f"{mod_name}.py")

        found = False
        docstring_found = False

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    docstring = ast.get_docstring(node)
                    if docstring:
                        docstring_found = True
                    found = True
                    # Don't break — check all matches (e.g., static methods)

        assert found, (
            f"Method '{method_name}' not found in module '{mod_name}'"
        )
        assert docstring_found, (
            f"Method '{method_name}' in module '{mod_name}' "
            f"(class={class_name}) is missing a docstring"
        )


# ═════════════════════════════════════════════════════════════════════════
# Test 3: Type Annotations
# ═════════════════════════════════════════════════════════════════════════

# Key methods that should have return type annotations.
# Format: method_name -> (module_name, class_name_or_None)
_ANNOTATED_METHODS: dict[str, tuple[str, str | None]] = {
    "get_cross_cycle_trends": ("meta_controller", "MetaController"),
    "get_recovery_confidence": ("meta_controller", "MetaController"),
    "get_dominance_report": ("strategy_archive", "StrategyArchive"),
    "quality_weighted_prune": ("champion_archive", "ChampionArchive"),
    "manage_capacity": ("novelty", "SparseArchive"),
    "decayed_novelty": ("novelty", "SparseArchive"),
    "get_improvement_velocity": ("qd_archive", "MAPElitesArchive"),
    "get_illumination_diversity": ("qd_archive", "MAPElitesArchive"),
    "detect_anomalies": ("metrics", "MetricsCollector"),
    "correlate_metrics": ("metrics", "MetricsCollector"),
    "get_disk_usage": ("data_rotation", None),
    "quality_aware_rotate_checkpoints": ("data_rotation", None),
    "get_plateau_analysis": ("progression_tracker", "ProgressionTracker"),
    "compare_windows": ("progression_tracker", "ProgressionTracker"),
    "get_config_summary": ("config", "EvolutionSection"),
    "validate_config": ("config", "EvolutionSection"),
    "send_monoculture_alert": ("monitoring_alerts", "AlertManager"),
    "send_recovery_stuck_alert": ("monitoring_alerts", "AlertManager"),
}


class TestTypeAnnotations:
    """Verify key methods have return type annotations."""

    @pytest.mark.parametrize("method_name", list(_ANNOTATED_METHODS.keys()))
    def test_return_type_annotation(self, method_name):
        """Each key method must have a return type annotation."""
        mod_name, class_name = _ANNOTATED_METHODS[method_name]
        method = _get_method_obj(method_name)

        sig = inspect.signature(method)
        assert sig.return_annotation is not inspect.Parameter.empty, (
            f"Method '{method_name}' (module='{mod_name}', "
            f"class='{class_name}') is missing a return type annotation"
        )


# ═════════════════════════════════════════════════════════════════════════
# Test 4: Code Quality
# ═════════════════════════════════════════════════════════════════════════

# Methods to scan for TODO/FIXME/HACK markers in their source bodies.
_QC_METHOD_LOCATIONS: list[tuple[str, str, str | None]] = [
    ("meta_controller", "get_cross_cycle_trends", "MetaController"),
    ("meta_controller", "get_recovery_confidence", "MetaController"),
    ("strategy_archive", "get_dominance_report", "StrategyArchive"),
    ("champion_archive", "quality_weighted_prune", "ChampionArchive"),
    ("novelty", "manage_capacity", "SparseArchive"),
    ("novelty", "decayed_novelty", "SparseArchive"),
    ("qd_archive", "get_improvement_velocity", "MAPElitesArchive"),
    ("qd_archive", "get_illumination_diversity", "MAPElitesArchive"),
    ("metrics", "detect_anomalies", "MetricsCollector"),
    ("metrics", "correlate_metrics", "MetricsCollector"),
    ("data_rotation", "get_disk_usage", None),
    ("data_rotation", "quality_aware_rotate_checkpoints", None),
    ("progression_tracker", "get_plateau_analysis", "ProgressionTracker"),
    ("progression_tracker", "compare_windows", "ProgressionTracker"),
    ("config", "get_config_summary", "EvolutionSection"),
    ("config", "validate_config", "EvolutionSection"),
    ("mutation_proposer", "record_mutation_outcome", "DirectedMutator"),
    ("mutation_proposer", "get_mutation_stats", "DirectedMutator"),
    ("monitoring_alerts", "send_monoculture_alert", "AlertManager"),
    ("monitoring_alerts", "send_recovery_stuck_alert", "AlertManager"),
]


def _get_method_ast_node(
    mod_name: str, method_name: str, class_name: str | None
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find the AST node for a method/function in the given module."""
    source = _get_source(mod_name)
    tree = ast.parse(source, filename=f"{mod_name}.py")

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and class_name and node.name == class_name:
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == method_name:
                        return item
        elif class_name is None:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    return node
    return None


class TestCodeQuality:
    """Verify code quality standards for enhanced methods."""

    @pytest.mark.parametrize(
        "mod_name,method_name,class_name",
        _QC_METHOD_LOCATIONS,
        ids=[f"{m[1]}" for m in _QC_METHOD_LOCATIONS],
    )
    def test_no_todo_fixme_hack(self, mod_name, method_name, class_name):
        """New methods must not contain TODO, FIXME, or HACK markers."""
        node = _get_method_ast_node(mod_name, method_name, class_name)
        assert node is not None, (
            f"Could not find '{method_name}' in '{mod_name}' "
            f"(class={class_name})"
        )

        # Get the source lines of the method body
        source = _get_source(mod_name)
        lines = source.splitlines()
        start = node.lineno - 1  # 0-indexed
        end = node.end_lineno if hasattr(node, "end_lineno") else start + 50
        method_body = "\n".join(lines[start:end])

        # Strip comments to avoid false positives in docstrings
        # Check each line for TODO/FIXME/HACK (case-insensitive)
        violations = []
        for i, line in enumerate(lines[start:end], start=start + 1):
            stripped = line.strip()
            # Skip lines that are part of a docstring (triple-quoted)
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                # Only flag comments containing these markers
                if re.search(r"\b(TODO|FIXME|HACK)\b", stripped, re.IGNORECASE):
                    violations.append(f"  Line {i}: {stripped}")

        assert not violations, (
            f"Method '{method_name}' in '{mod_name}' contains "
            f"TODO/FIXME/HACK markers:\n" + "\n".join(violations)
        )

    @pytest.mark.parametrize(
        "mod_name,method_name,class_name",
        _QC_METHOD_LOCATIONS,
        ids=[f"{m[1]}" for m in _QC_METHOD_LOCATIONS],
    )
    def test_snake_case_naming(self, mod_name, method_name, class_name):
        """Method names must follow snake_case convention."""
        assert re.match(
            r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$", method_name
        ), (
            f"Method name '{method_name}' in '{mod_name}' "
            f"does not follow snake_case convention"
        )

    def test_all_enhanced_modules_exist(self):
        """Verify all enhanced module files exist."""
        for mod_name, path in _MOD_PATHS.items():
            assert path.is_file(), (
                f"Enhanced module '{mod_name}' not found at {path}"
            )

    def test_module_docstrings_are_meaningful(self):
        """Module docstrings should have at least 10 characters."""
        for mod_name in _MOD_PATHS:
            source = _get_source(mod_name)
            tree = ast.parse(source, filename=f"{mod_name}.py")
            docstring = ast.get_docstring(tree)
            assert docstring and len(docstring.strip()) >= 10, (
                f"Module '{mod_name}' docstring is too short or empty "
                f"(got {len(docstring.strip()) if docstring else 0} chars)"
            )

    @pytest.mark.parametrize(
        "mod_name,method_name,class_name",
        _QC_METHOD_LOCATIONS,
        ids=[f"{m[1]}" for m in _QC_METHOD_LOCATIONS],
    )
    def test_method_docstrings_are_meaningful(self, mod_name, method_name, class_name):
        """Method docstrings should have at least 20 characters."""
        node = _get_method_ast_node(mod_name, method_name, class_name)
        assert node is not None
        docstring = ast.get_docstring(node)
        assert docstring and len(docstring.strip()) >= 20, (
            f"Method '{method_name}' in '{mod_name}' docstring is too short "
            f"or empty (got {len(docstring.strip()) if docstring else 0} chars)"
        )

    def test_new_methods_have_parameter_docs(self):
        """Methods with parameters should document them (at least Args/Parameters section)."""
        # Check a subset of methods that take non-self parameters
        methods_with_params = [
            ("meta_controller", "get_cross_cycle_trends", "MetaController"),
            ("metrics", "detect_anomalies", "MetricsCollector"),
            ("metrics", "correlate_metrics", "MetricsCollector"),
            ("progression_tracker", "compare_windows", "ProgressionTracker"),
        ]
        for mod_name, method_name, class_name in methods_with_params:
            node = _get_method_ast_node(mod_name, method_name, class_name)
            assert node is not None
            docstring = ast.get_docstring(node)
            assert docstring, (
                f"Method '{method_name}' in '{mod_name}' has no docstring"
            )
            # Should mention Args or Parameters or have parameter descriptions
            has_params_section = any(
                kw in docstring.lower()
                for kw in ("args", "parameters", "params", "attributes")
            )
            # Also accept if all params are mentioned by name in the docstring
            args_node = node.args
            param_names = [
                a.arg for a in args_node.args if a.arg != "self"
            ]
            has_param_mentions = any(
                pname in docstring for pname in param_names
            )
            assert has_params_section or has_param_mentions, (
                f"Method '{method_name}' in '{mod_name}' has parameters "
                f"{param_names} but docstring doesn't document them"
            )
