#!/usr/bin/env python3
"""Wave 16 — Documentation & Code Quality Coverage Tests.

Verifies that all new methods have docstrings, all enhanced modules have
module-level docstrings, and that code quality invariants (no hardcoded
secrets, snake_case naming, no bare except clauses) hold.

Uses inspect for docstring checks — no network or filesystem side-effects.
Tests are fast (pure import + introspection).
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent  # Evolution Engine root

# ---------------------------------------------------------------------------
# Method → (module_name, class_name_or_None)
# Every new method added in the Wave 16 enhancement.
# ---------------------------------------------------------------------------
_METHOD_MAP: dict[str, tuple[str, str | None]] = {
    # meta_controller.py
    "get_cross_cycle_trends":        ("meta_controller", "MetaController"),
    "get_recovery_confidence":       ("meta_controller", "MetaController"),
    "get_tuning_escalation_history": ("meta_controller", "MetaController"),
    # strategy_archive.py
    "per_level_effectiveness":       ("strategy_archive", "StrategyArchive"),
    "adaptive_exploration_bonus":    ("strategy_archive", "StrategyArchive"),
    "time_weighted_success_rate":    ("strategy_archive", "StrategyArchive"),
    "get_dominance_report":          ("strategy_archive", "StrategyArchive"),
    # champion_archive.py
    "diversity_aware_select":        ("champion_archive", "ChampionArchive"),
    "quality_weighted_prune":        ("champion_archive", "ChampionArchive"),
    "get_diversity_report":          ("champion_archive", "ChampionArchive"),
    # novelty.py
    "manage_capacity":               ("novelty", "NoveltyArchive"),
    "decayed_novelty":               ("novelty", "NoveltyArchive"),
    # qd_archive.py
    "get_improvement_velocity":      ("qd_archive", "MAPElitesArchive"),
    "get_illumination_diversity":    ("qd_archive", "MAPElitesArchive"),
    # mutation_proposer.py
    "record_mutation_outcome":       ("mutation_proposer", "DirectedMutator"),
    "get_mutation_stats":            ("mutation_proposer", "DirectedMutator"),
    # monitoring_alerts.py
    "send_monoculture_alert":        ("monitoring_alerts", "AlertManager"),
    "send_recovery_stuck_alert":     ("monitoring_alerts", "AlertManager"),
    # metrics.py
    "detect_anomalies":              ("metrics", "MetricsCollector"),
    "correlate_metrics":             ("metrics", "MetricsCollector"),
    # data_rotation.py  (module-level)
    "get_disk_usage":                ("data_rotation", None),
    "quality_aware_rotate_checkpoints": ("data_rotation", None),
    # progression_tracker.py
    "get_plateau_analysis":          ("progression_tracker", "ProgressionTracker"),
    "compare_windows":               ("progression_tracker", "ProgressionTracker"),
    # config.py
    "get_config_summary":            ("config", "EvolutionSection"),
    "validate_config":               ("config", "EvolutionSection"),
}

# ---------------------------------------------------------------------------
# Modules with module-level docstrings
# ---------------------------------------------------------------------------
_ENHANCED_MODULES: list[str] = [
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

# Source files to scan for code quality
_SOURCE_FILES: list[str] = [
    "evolution_engine.py",
    "meta_controller.py",
    "strategy_archive.py",
    "champion_archive.py",
    "novelty.py",
    "qd_archive.py",
    "mutation_proposer.py",
    "monitoring_alerts.py",
    "metrics.py",
    "data_rotation.py",
    "progression_tracker.py",
    "config.py",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. TestNewMethodsHaveDocstrings
# ═══════════════════════════════════════════════════════════════════════════
class TestNewMethodsHaveDocstrings:
    """Every new method must have a non-empty docstring."""

    @pytest.fixture(scope="class")
    def _method_obj(self, request):
        """Resolve a (method_name, module_name, class_name) triple to the
        actual callable and stash it on the request for parametrized tests."""
        # We use a parametrize helper instead — see each test below.

    @staticmethod
    def _resolve(method_name: str):
        """Import module, locate class/func, return the callable."""
        mod_name, cls_name = _METHOD_MAP[method_name]
        mod = importlib.import_module(mod_name)
        if cls_name:
            return getattr(mod, cls_name)
        return mod

    @pytest.mark.parametrize(
        "method_name",
        sorted(_METHOD_MAP.keys()),
        ids=lambda mn: mn,
    )
    def test_method_exists(self, method_name: str):
        """Method must be importable from its module (class or module level)."""
        mod_name, cls_name = _METHOD_MAP[method_name]
        mod = importlib.import_module(mod_name)
        if cls_name:
            cls = getattr(mod, cls_name)
            assert hasattr(cls, method_name), (
                f"{cls_name} in {mod_name} missing method {method_name}"
            )
        else:
            assert hasattr(mod, method_name), (
                f"{mod_name} missing module-level function {method_name}"
            )

    @pytest.mark.parametrize(
        "method_name",
        sorted(_METHOD_MAP.keys()),
        ids=lambda mn: mn,
    )
    def test_method_has_docstring(self, method_name: str):
        """Method docstring must exist and be non-empty."""
        mod_name, cls_name = _METHOD_MAP[method_name]
        mod = importlib.import_module(mod_name)
        if cls_name:
            obj = getattr(mod, cls_name)
        else:
            obj = mod
        method = getattr(obj, method_name)
        doc = getattr(method, "__doc__", None)
        assert doc and doc.strip(), (
            f"{method_name} ({mod_name}.{cls_name or 'module'}) "
            f"has no docstring or docstring is empty"
        )

    def test_all_expected_methods_present(self):
        """Sanity: every name in _METHOD_MAP was found at import time."""
        missing = []
        for mn, (mod_name, cls_name) in _METHOD_MAP.items():
            mod = importlib.import_module(mod_name)
            if cls_name:
                obj = getattr(mod, cls_name, None)
                if not obj or not hasattr(obj, mn):
                    missing.append(f"{mod_name}.{cls_name}.{mn}")
            else:
                if not hasattr(mod, mn):
                    missing.append(f"{mod_name}.{mn}")
        assert not missing, f"Missing methods: {missing}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. TestNewModulesHaveDocstrings
# ═══════════════════════════════════════════════════════════════════════════
class TestNewModulesHaveDocstrings:
    """Every enhanced module must have a module-level docstring."""

    @pytest.mark.parametrize(
        "mod_name",
        _ENHANCED_MODULES,
        ids=lambda m: m,
    )
    def test_module_importable(self, mod_name: str):
        """Module must be importable without error."""
        mod = importlib.import_module(mod_name)
        assert mod is not None, f"{mod_name} import returned None"

    @pytest.mark.parametrize(
        "mod_name",
        _ENHANCED_MODULES,
        ids=lambda m: m,
    )
    def test_module_has_docstring(self, mod_name: str):
        """Module __doc__ must exist and be non-empty."""
        mod = importlib.import_module(mod_name)
        doc = getattr(mod, "__doc__", None)
        assert doc and doc.strip(), (
            f"Module {mod_name} has no docstring or docstring is empty"
        )

    def test_all_enhanced_modules_count(self):
        """Verify we're checking the expected number of modules."""
        assert len(_ENHANCED_MODULES) == 11, (
            f"Expected 11 enhanced modules, got {len(_ENHANCED_MODULES)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. TestCodeQualityChecks
# ═══════════════════════════════════════════════════════════════════════════
class TestCodeQualityChecks:
    """Static code quality checks on the source files."""

    # ---- helpers ----

    @staticmethod
    def _read_source(filename: str) -> str:
        path = _REPO / filename
        if not path.exists():
            pytest.skip(f"{filename} not found in repo root")
        return path.read_text(encoding="utf-8")

    # ---- secrets ----

    @pytest.mark.parametrize(
        "filename",
        _SOURCE_FILES,
        ids=lambda f: f,
    )
    def test_no_hardcoded_secrets(self, filename: str):
        """No hardcoded API keys, passwords, tokens, or AWS secrets."""
        source = self._read_source(filename)
        patterns = [
            r'(?i)api[_-]?key\s*=\s*["\'][^"\']{8,}["\']',
            r'(?i)secret[_-]?key\s*=\s*["\'][^"\']{8,}["\']',
            r'(?i)password\s*=\s*["\'][^"\']{3,}["\']',
            r'(?i)token\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']',
            r'(?i)aws[_-]?(?:access|secret)\s*=\s*["\'][^"\']{10,}["\']',
        ]
        for pat in patterns:
            match = re.search(pat, source)
            assert not match, (
                f"{filename} appears to contain a hardcoded secret: "
                f"{match.group().strip()[:60]}"
            )

    def test_no_hardcoded_secrets_across_all_files(self):
        """Aggregate secret scan across all source files."""
        violations: list[str] = []
        patterns = [
            r'(?i)api[_-]?key\s*=\s*["\'][^"\']{8,}["\']',
            r'(?i)secret[_-]?key\s*=\s*["\'][^"\']{8,}["\']',
            r'(?i)password\s*=\s*["\'][^"\']{3,}["\']',
            r'(?i)token\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']',
            r'(?i)aws[_-]?(?:access|secret)\s*=\s*["\'][^"\']{10,}["\']',
        ]
        for filename in _SOURCE_FILES:
            source = self._read_source(filename)
            for pat in patterns:
                for m in re.finditer(pat, source):
                    violations.append(f"{filename}: {m.group().strip()[:60]}")
        assert not violations, f"Hardcoded secrets found: {violations}"

    # ---- naming ----

    @pytest.mark.parametrize(
        "filename",
        _SOURCE_FILES,
        ids=lambda f: f,
    )
    def test_functions_use_snake_case(self, filename: str):
        """All top-level and method function names must be snake_case."""
        source = self._read_source(filename)
        tree = ast.parse(source)
        snake_re = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                # Skip dunder methods and private names starting with _
                if name.startswith("_"):
                    continue
                if not snake_re.match(name):
                    violations.append(f"line {node.lineno}: {name}")
        assert not violations, (
            f"{filename} has non-snake_case function names: {violations}"
        )

    # ---- bare except ----

    @pytest.mark.parametrize(
        "filename",
        _SOURCE_FILES,
        ids=lambda f: f,
    )
    def test_no_bare_except(self, filename: str):
        """No bare ``except:`` clauses — always catch ``Exception`` or a
        specific subclass."""
        source = self._read_source(filename)
        tree = ast.parse(source)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    violations.append(f"line {node.lineno}: bare except")
        assert not violations, (
            f"{filename} has bare except clauses: {violations}"
        )

    def test_no_bare_except_across_all_files(self):
        """Aggregate bare-except scan across all source files."""
        violations: list[str] = []
        for filename in _SOURCE_FILES:
            source = self._read_source(filename)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    violations.append(
                        f"{filename}: line {node.lineno}: bare except"
                    )
        assert not violations, f"Bare except clauses found: {violations}"

    # ---- docstring completeness (cross-check) ----

    def test_method_count_matches_expectation(self):
        """Verify we're testing the expected number of new methods."""
        assert len(_METHOD_MAP) == 26, (
            f"Expected 26 new methods, got {len(_METHOD_MAP)}"
        )
