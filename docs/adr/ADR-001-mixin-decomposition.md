# ADR-001: Mixin Decomposition of evolution_engine.py

**Status:** Accepted

**Date:** 2026-09-06

## Context

`evolution_engine.py` had grown to 8,795 lines, combining the core evolutionary loop with selection logic, QD archive management, recurrent feedback, and diagnostics. This monolithic structure made the codebase difficult to navigate, test in isolation, and maintain. The single-file design also created merge conflicts when multiple agents worked on different subsystems simultaneously.

## Decision

Decompose `evolution_engine.py` into a mixin-based architecture:

- **SelectionMixin** (`evolution/selection_mixin.py`, ~736 lines): Elite selection, champion verification, Hall of Fame management
- **QDArchiveMixin** (`evolution/qd_archive_mixin.py`, ~616 lines): MAP-Elites QD archive, novelty archive management
- **RecurrentMixin** (`evolution/recurrent_mixin.py`, ~733 lines): Recurrent feedback, strategy outcome tracking
- **DiagnosticsMixin** (`evolution/diagnostics_mixin.py`, ~751 lines): Logging, diagnostics, summaries, state export
- **Shared constants** (`evolution/shared.py`, ~10 lines): `_CRASH_STATUSES`, `_TIMEOUT_STATUSES`, `MIN_FITNESS_FOR_TRACE`, etc.

`EvolutionEngine` inherits from all four mixins via Python MRO, preserving the existing API surface. No method signatures changed; no circular imports introduced (lazy imports for `compute_qd_axes` and `_HAS_QD` where needed).

## Consequences

### Positive
- `evolution_engine.py` reduced from 8,795 to 6,151 lines (30% reduction)
- Each mixin can be understood and tested in relative isolation
- Reduced merge conflict surface: agents working on selection vs. diagnostics no longer touch the same file
- Shared constants extracted to a dedicated module with clear ownership

### Negative
- Debugging across mixins requires understanding the MRO chain
- Lazy imports for QD/novelty modules add minor indirection

### Neutral
- The mixin pattern is idiomatic Python and familiar to contributors
- No API changes — existing callers see no difference

## Alternatives Considered

1. **Plugin architecture**: Would allow runtime composition but adds complexity disproportionate to the benefit. Rejected for a codebase with 4 clear subsystems.
2. **Full refactoring to separate modules with explicit composition**: More Pythonic OOP but would require API changes across all callers. The mixin approach preserves backward compatibility.

## References

- Task t_64cabb9a: Decomposition implementation
- Original evolution_engine.py: 8,795 lines
- Post-decomposition: 6,151 lines
- Test suite: 8,887 pass, 0 new failures
