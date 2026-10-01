#!/usr/bin/env python3
"""Wave 19 quality gate tests.

Scans source files for coding-quality violations:
  1. Bare except clauses
  2. Hardcoded secrets
  3. Consistent error handling
  4. Method complexity (≤50 lines excluding docstrings)

All tests are read-only and finish quickly.
"""
from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent


# ── Source files to scan ──────────────────────────────────────────────────────
def _discover_source_files() -> list[Path]:
    """Return top-level .py source files (non-test, non-archive)."""
    skip_dirs = {
        "tests",
        "archive",
        "champions",
        "cycle_data",
        "milestone0",
        "milestone1",
        "milestone2",
        "milestone3",
        "milestone4",
        "milestone5",
        "milestone6",
        "milestone7",
        "simulation",
        "benchmarks",
        "metrics",
        "logs",
        "__pycache__",
        ".git",
        "config",
    }
    files: list[Path] = []
    for child in sorted(ROOT.iterdir()):
        if child.is_dir() and (child.name in skip_dirs or child.name.startswith("cycle_")):
            continue
        if child.is_file() and child.suffix == ".py":
            files.append(child)
    return files


SOURCE_FILES = _discover_source_files()


def _get_recently_modified_files() -> list[Path]:
    """Get files modified in the last 5 git commits."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~5", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            modified = set()
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if line.endswith(".py"):
                    fp = ROOT / line
                    if fp.exists() and fp in SOURCE_FILES:
                        modified.add(fp)
            return sorted(modified)
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return []


RECENTLY_MODIFIED = _get_recently_modified_files()

# Skip pre-existing files that have known violations
_BARE_EXCEPT_SKIP = {"v9_monitor.py"}
_SECRET_SKIP_FILES = {"paths.py", "logging_config.py", "config.py"}
_DICT_RETURN_SKIP = {"meta_controller.py", "prefix_generator.py", "v11_dashboard.py"}
_COMPLEXITY_SKIP = {"run_multi_cycle.py"}


# ══════════════════════════════════════════════════════════════════════════════
# 1. TestNoBareExceptClauses
# ══════════════════════════════════════════════════════════════════════════════


class TestNoBareExceptClauses:
    """Verify no bare ``except:`` clauses in source files."""

    # ``except`` at start of line content, followed by ``:``
    # but NOT followed by an exception class (e.g. ``except ValueError:``).
    _BARE_EXCEPT_RE = re.compile(
        r"^[ \t]*except\s*:\s*(?:#.*)?$",
        re.MULTILINE,
    )

    @pytest.mark.parametrize(
        "src_file",
        SOURCE_FILES,
        ids=lambda p: p.name,
    )
    def test_no_bare_except(self, src_file: Path):
        """Each source file must contain zero bare except clauses."""
        if src_file.name in _BARE_EXCEPT_SKIP:
            pytest.skip(f"{src_file.name}: pre-existing bare except (known)")

        content = src_file.read_text(encoding="utf-8", errors="replace")
        matches = self._BARE_EXCEPT_RE.findall(content)
        assert not matches, f"{src_file.name}: found {len(matches)} bare except clause(s).\n" + "\n".join(
            f"  • {m.strip()}" for m in matches
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. TestNoHardcodedSecrets
# ══════════════════════════════════════════════════════════════════════════════

_SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"""(?:api_key|api_secret|password|passwd|secret|token|access_key|aws_secret_access_key)"""
        r"""\s*=\s*(?:(?!os\.environ|os\.getenv|None|""|'').)*"""
        r"""(?:"[^"]{3,}"|'[^']{3,}')""",
        re.IGNORECASE,
    ),
]


class TestNoHardcodedSecrets:
    """Detect hardcoded API keys, passwords, tokens, and secrets."""

    @pytest.mark.parametrize(
        "src_file",
        [f for f in SOURCE_FILES if f.name not in _SECRET_SKIP_FILES],
        ids=lambda p: p.name,
    )
    def test_no_hardcoded_secrets(self, src_file: Path):
        """No source file should contain hardcoded secret patterns."""
        content = src_file.read_text(encoding="utf-8", errors="replace")
        violations: list[str] = []
        for pat in _SECRET_PATTERNS:
            for m in pat.finditer(content):
                line_no = content[: m.start()].count("\n") + 1
                violations.append(f"  line {line_no}: {m.group().strip()}")
        assert not violations, f"{src_file.name}: possible hardcoded secrets found:\n" + "\n".join(violations)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TestConsistentErrorHandling
# ══════════════════════════════════════════════════════════════════════════════


class TestConsistentErrorHandling:
    """Verify specific exceptions are caught and dict-returning methods are consistent."""

    @pytest.mark.parametrize(
        "src_file",
        SOURCE_FILES,
        ids=lambda p: p.name,
    )
    def test_no_bare_except_in_methods(self, src_file: Path):
        """All try/except blocks must catch a specific exception."""
        if src_file.name in _BARE_EXCEPT_SKIP:
            pytest.skip(f"{src_file.name}: pre-existing bare except (known)")

        try:
            tree = ast.parse(src_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            pytest.skip(f"{src_file.name}: syntax error – cannot parse")

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append(f"  line {node.lineno}: bare except in function/method body")

        assert not violations, f"{src_file.name}: bare except found in {len(violations)} location(s):\n" + "\n".join(
            violations
        )

    @pytest.mark.parametrize(
        "src_file",
        SOURCE_FILES,
        ids=lambda p: p.name,
    )
    def test_dict_typed_methods_never_return_none(self, src_file: Path):
        """Methods with explicit ``-> dict`` annotation must not return None.

        Only checks methods with an explicit ``-> dict`` (or ``-> Dict``)
        type annotation, avoiding false positives on methods that happen
        to return dicts sometimes.
        """
        if src_file.name in _DICT_RETURN_SKIP:
            pytest.skip(f"{src_file.name}: pre-existing dict-return with None (known)")

        try:
            source = src_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except SyntaxError:
            pytest.skip(f"{src_file.name}: syntax error – cannot parse")

        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Only check methods with explicit dict return annotation
            ret_ann = node.returns
            if ret_ann is None:
                continue
            ann_str = ast.dump(ret_ann)
            if "dict" not in ann_str.lower():
                continue

            # Check for bare return or return None
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return):
                    if sub.value is None:
                        violations.append(
                            f"  line {sub.lineno}: {node.name}() has explicit -> dict "
                            f"annotation but bare return (returns None)"
                        )
                    elif isinstance(sub.value, ast.Constant) and sub.value.value is None:
                        violations.append(
                            f"  line {sub.lineno}: {node.name}() has explicit -> dict "
                            f"annotation but returns None explicitly"
                        )

        assert not violations, f"{src_file.name}: dict-annotated methods with None return:\n" + "\n".join(violations)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TestMethodComplexity
# ══════════════════════════════════════════════════════════════════════════════

_MAX_METHOD_LINES = 50


def _method_body_lines(
    source_lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    """Count non-blank, non-comment, non-docstring lines in a method body.

    Uses ``node.lineno`` / ``node.end_lineno`` from the AST (available in
    Python 3.8+) to slice the source lines, so it works without importing
    the module under test.
    """
    if not node.end_lineno:
        return 0

    # node.lineno is the ``def`` line; body starts one line later.
    start = node.lineno  # 1-indexed def line
    end = node.end_lineno  # 1-indexed last line of method (inclusive)

    # Extract body lines (skip the def line itself)
    body_lines = source_lines[start:end]  # 0-indexed slice

    # Detect docstring: first non-blank line that starts with triple quotes
    docstring_end = 0
    in_docstring = False
    for i, line in enumerate(body_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if not in_docstring:
            if stripped.startswith(('"""', "'''")):
                quote = stripped[:3]
                # Single-line docstring: opening and closing quotes on same line
                if stripped.endswith(quote) and len(stripped) > 6:
                    docstring_end = i + 1
                    break
                # Multi-line docstring
                in_docstring = True
                docstring_end = i + 1
                continue
            break  # First non-blank line is not a docstring
        else:
            docstring_end = i + 1
            if stripped.endswith(quote):
                break
    else:
        if in_docstring:
            docstring_end = len(body_lines)

    # Count non-blank, non-comment lines after docstring
    count = 0
    for line in body_lines[docstring_end:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


class TestMethodComplexity:
    """Verify no method exceeds 50 non-blank, non-comment, non-docstring lines.

    Scans only recently modified files (last 5 commits) to catch new
    violations without flagging pre-existing long methods.
    """

    @pytest.mark.parametrize(
        "src_file",
        RECENTLY_MODIFIED,
        ids=lambda p: p.name,
    )
    def test_methods_within_line_limit(self, src_file: Path):
        """Methods in recently modified files must be ≤50 lines."""
        if src_file.name in _COMPLEXITY_SKIP:
            pytest.skip(f"{src_file.name}: pre-existing long methods (known)")

        try:
            source = src_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except SyntaxError:
            pytest.skip(f"{src_file.name}: syntax error – cannot parse")

        source_lines = source.splitlines()

        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            line_count = _method_body_lines(source_lines, node)
            if line_count > _MAX_METHOD_LINES:
                violations.append(f"  {node.name}(): {line_count} lines (limit {_MAX_METHOD_LINES})")

        assert (
            not violations
        ), f"{src_file.name}: {len(violations)} method(s) exceed {_MAX_METHOD_LINES}-line limit:\n" + "\n".join(
            violations
        )

    @pytest.mark.parametrize(
        "src_file",
        SOURCE_FILES,
        ids=lambda p: p.name,
    )
    def test_no_method_over_100_lines(self, src_file: Path):
        """No method anywhere should exceed 100 lines (absolute ceiling).

        Scoped to recently modified files to avoid pre-existing violations.
        """
        if src_file.name in _COMPLEXITY_SKIP:
            pytest.skip(f"{src_file.name}: pre-existing long methods (known)")
        if src_file not in RECENTLY_MODIFIED:
            pytest.skip(f"{src_file.name}: not recently modified")

        try:
            source = src_file.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except SyntaxError:
            pytest.skip(f"{src_file.name}: syntax error – cannot parse")

        source_lines = source.splitlines()

        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            line_count = _method_body_lines(source_lines, node)
            if line_count > 100:
                violations.append(f"  {node.name}(): {line_count} lines (absolute ceiling: 100)")

        assert not violations, f"{src_file.name}: {len(violations)} method(s) exceed 100-line ceiling:\n" + "\n".join(
            violations
        )
