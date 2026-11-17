# ADR-003: Test Parallelization with pytest-xdist

**Status:** Proposed

**Date:** 2026-09-06

## Context

The Evolution Engine test suite has grown to 8,887+ tests. Running them sequentially takes ~3-5 minutes on a typical development machine. This slows down the feedback loop for developers and CI pipelines. The tests are largely independent — they create isolated engine instances, mock external dependencies, and don't share mutable state. This makes them good candidates for parallel execution.

## Decision

Adopt `pytest-xdist` for test parallelization. The configuration will:

- Use `pytest-xdist` with `-n auto` to auto-detect CPU count
- Set `tests/property/` as a separate test directory for property-based tests (which are slower per-test due to Hypothesis generation)
- Configure `pytest.ini_options` in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
```

Run with: `python -m pytest tests/ -x -q --tb=line -n auto`

## Consequences

### Positive
- Test suite runtime reduced from ~3-5 min to ~1-2 min on 4-core machines
- Faster CI feedback loop
- No code changes required — xdist is a pytest plugin

### Negative
- Adds `pytest-xdist` as a dev dependency
- Tests that use global state or temp directories may need `--forked` flag
- Debugging parallel test failures can be harder (less deterministic output)
- Hypothesis tests with `@given` may interact poorly with xdist worker processes (each worker gets different random state)

### Neutral
- `pytest-xdist` is the standard solution for pytest parallelization
- The `-n auto` flag gracefully degrades to sequential on single-core machines

## Alternatives Considered

1. **pytest-parallel**: Less mature than xdist, fewer features.
2. **Run tests in separate CI jobs**: More complex CI configuration, no local speedup.
3. **Cache test results**: Would skip re-running tests that haven't changed, but requires infrastructure (pytest-cache is insufficient for this).

## References

- pytest-xdist: https://pytest-xdist.readthedocs.io/
- Task t_89d73b6e: CI setup
- Current test count: 8,887+ passing
