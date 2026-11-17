# ADR-002: Pin numpy<2.0 for Compatibility

**Status:** Accepted

**Date:** 2026-09-05

## Context

The Evolution Engine depends on numpy for array operations in the VM farm, QD archive, and fitness computation. numpy 2.0 (released June 2024) introduced breaking API changes including removed deprecated functions, changed default integer types, and modified array behavior that caused runtime failures in several modules. Specifically:

- `numpy.bool` and `numpy.int` aliases removed (breaking legacy code)
- Default integer type changed from `int64` to platform-native `int` (breaking type-sensitive comparisons)
- Several deprecated functions removed without fallback paths

The project was blocked: upgrading to numpy 2.x caused import errors and runtime crashes that would have required extensive codebase-wide refactoring to resolve.

## Decision

Pin numpy to `>=1.24.0,<2.0` in `pyproject.toml` dependencies. This ensures:

- Minimum numpy 1.24.0 for Python 3.10+ compatibility and modern features
- Upper bound <2.0 to avoid the breaking API changes in numpy 2.x
- scipy 1.11.x (which requires numpy<2.0) remains compatible

```toml
dependencies = [
    "torch==2.4.1",
    "numpy>=1.24.0,<2.0",
]
```

## Consequences

### Positive
- Immediate unblock: test suite goes from 0 passing to 9,018+ passing
- No code changes required in existing modules
- scipy 1.11.4 remains compatible without additional pins

### Negative
- Cannot use numpy 2.0+ features (string operations, new array API)
- Must re-evaluate pin when numpy 2.x ecosystem matures
- Locks to numpy 1.x line which will eventually reach end-of-life

### Neutral
- numpy 1.26.4 (the version resolved by this pin) is widely used and stable
- The pin is documented in pyproject.toml with a comment explaining the rationale

## Alternatives Considered

1. **Fix all numpy 2.x incompatibilities**: Would require touching ~15 files with `numpy.bool` → `bool`, `numpy.int` → `int` replacements, plus testing every module. Estimated effort: 2-3 days. Not worth it for a research/prototype codebase.
2. **Use numpy 2.x with compatibility shims**: Possible but adds fragile conditional imports throughout the codebase.
3. **Wait for upstream fixes**: Depends on torch/scipy supporting numpy 2.x, which is out of our control.

## References

- Task t_0d9a2567: numpy/scipy fix implementation
- pyproject.toml: dependency pin
- scipy 1.11.4: compatible with numpy<2.0
