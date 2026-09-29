"""Per-generation phase timing for the evolution run loop.

Splits each generation's wall time into additive phases so the
``phase_evolution_seconds`` band (measured 638-4577s) can be attributed
either to VM-farm fitness evaluation or to Python orchestration overhead:

* ``vm_eval``              -- VM-farm fitness evaluation (GPU/CPU batches)
* ``mutation_propagation`` -- challenger generation / mutation / curriculum init
* ``selection_archives``   -- elite selection, champion update, QD/novelty archives
* ``diagnostics_logging``  -- diversity metrics, per-gen logging, meta monitoring
* ``unattributed``         -- residual (block-boundary overhead)

The split is an exact partition: the four named phases plus the VM time
sum to the generation wall time (vm time is measured once at the source
and subtracted from the enclosing orchestration phase, so nothing is
double counted).  One JSON line per generation is appended to a rolling
JSONL log.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

#: Phase names emitted per generation, in report order.
PHASES: tuple[str, ...] = (
    "vm_eval",
    "mutation_propagation",
    "selection_archives",
    "diagnostics_logging",
    "unattributed",
)

#: The four phases the card asks to be reported (excludes residual).
NAMED_PHASES: tuple[str, ...] = PHASES[:-1]

DEFAULT_LOG_NAME = "generation_phase_timing.jsonl"
DEFAULT_MAX_BYTES = 8 * 1024 * 1024

_vm_lock = threading.Lock()
_vm_eval_seconds = 0.0
_vm_eval_calls = 0
_vm_depth = 0


def record_vm_eval(seconds: float, calls: int = 1) -> None:
    """Accumulate wall seconds spent inside the VM farm (source-level)."""
    global _vm_eval_seconds, _vm_eval_calls
    if seconds <= 0.0:
        return
    with _vm_lock:
        _vm_eval_seconds += seconds
        _vm_eval_calls += calls


def vm_eval_totals() -> tuple[float, int]:
    """Return ``(cumulative_seconds, cumulative_calls)`` for the VM farm."""
    with _vm_lock:
        return _vm_eval_seconds, _vm_eval_calls


def reset_vm_eval() -> None:
    """Reset the process-wide VM accumulator (tests only)."""
    global _vm_eval_seconds, _vm_eval_calls, _vm_depth
    with _vm_lock:
        _vm_eval_seconds = 0.0
        _vm_eval_calls = 0
        _vm_depth = 0


@contextmanager
def timed_vm_eval() -> Iterator[None]:
    """Record the wall time of the OUTERMOST enclosed VM-farm call.

    Re-entrancy guard: ``VMFarm.evaluate_batch`` fans out to
    ``VMFarm.evaluate`` (single-worker CPU path), so only the outermost
    call is timed -- otherwise the same wall seconds would be counted
    twice and the per-generation partition would not sum to the
    generation wall time.  Depth is process-wide on purpose: a batch
    call timed on the main thread already covers the worker threads it
    spawns.  (Concurrent *independent* VM calls from two threads would
    sum to more than elapsed wall time; this engine runs evaluations
    sequentially or as one batch fan-out, so that does not occur.)
    """
    global _vm_depth
    with _vm_lock:
        _vm_depth += 1
        outermost = _vm_depth == 1
    start = time.perf_counter()
    try:
        yield
    finally:
        with _vm_lock:
            _vm_depth -= 1
        if outermost:
            record_vm_eval(time.perf_counter() - start)


class PhaseTimer:
    """Accumulate per-phase wall time for one evolution run.

    ``phase(name)`` measures the enclosed block's wall time minus any
    VM-farm time that elapsed inside it (VM time is accounted once, as
    ``vm_eval``).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._net: dict[str, float] = {p: 0.0 for p in PHASES}
        self._gen_start: float | None = None
        self._gen_vm_start: float = 0.0
        self._gen_start_calls: int = 0
        self._cycle: int = 0
        self._generation: int = 0
        self.records: list[dict[str, Any]] = []
        self.run_init_seconds = 0.0
        self.run_finalize_seconds = 0.0
        self._run_start = time.perf_counter()
        self._log_path: str | None = None

    # -- configuration -------------------------------------------------
    def set_cycle(self, cycle: int) -> None:
        self._cycle = int(cycle)

    def set_log_path(self, path: str) -> None:
        self._log_path = path

    @property
    def log_path(self) -> str | None:
        return self._log_path

    @property
    def active(self) -> bool:
        return self.enabled

    # -- measurement ---------------------------------------------------
    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Attribute the enclosed block's non-VM wall time to ``name``."""
        if not self.enabled or name not in self._net:
            yield
            return
        vm_before, _ = vm_eval_totals()
        start = time.perf_counter()
        try:
            yield
        finally:
            wall = time.perf_counter() - start
            vm_inside = vm_eval_totals()[0] - vm_before
            self._net[name] += max(0.0, wall - vm_inside)

    def begin_generation(self, generation: int) -> None:
        if not self.enabled:
            return
        self._generation = int(generation)
        self._gen_start = time.perf_counter()
        self._gen_vm_start, self._gen_start_calls = vm_eval_totals()
        # Reset residual phase time: if the previous generation aborted
        # before end_generation, its `_net` would otherwise leak into this
        # record and drive `unattributed` negative (EE-W5 review).
        for p in self._net:
            self._net[p] = 0.0

    def end_generation(self, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Close the current generation and emit its JSON record."""
        if not self.enabled or self._gen_start is None:
            return None
        wall = time.perf_counter() - self._gen_start
        vm_now, calls_now = vm_eval_totals()
        vm_seconds = max(0.0, vm_now - self._gen_vm_start)
        vm_calls = max(0, calls_now - self._gen_start_calls)

        phases = {p: round(self._net[p], 4) for p in NAMED_PHASES}
        # The evaluation phase is reported as a whole: pure VM-farm time
        # plus the Python orchestration measured around the eval calls.
        vm_eval_total = self._net["vm_eval"] + vm_seconds
        named_totals = {**self._net, "vm_eval": vm_eval_total}
        # Sum the UNROUNDED totals: rounding each phase first lets accounted
        # exceed wall by up to 4*5e-5, which reported coverage_ratio > 1.0
        # and unattributed = -0.0 (EE-W5 review, observed on a real run).
        accounted = sum(named_totals[p] for p in NAMED_PHASES)
        phases["vm_eval"] = round(vm_eval_total, 4)
        unattributed = wall - accounted
        phases["unattributed"] = round(unattributed, 4)

        record: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "cycle": self._cycle,
            "generation": self._generation,
            "generation_seconds": round(wall, 4),
            "vm_eval_seconds": round(vm_seconds, 4),
            "vm_eval_calls": vm_calls,
            "phase_seconds": phases,
            "accounted_seconds": round(accounted, 4),
            "coverage_ratio": round(accounted / wall, 4) if wall > 0 else 1.0,
        }
        if extra:
            record.update(extra)

        self.records.append(record)
        self._reset_gen()
        self._emit(record)
        return record

    def _reset_gen(self) -> None:
        self._gen_start = None
        for p in self._net:
            self._net[p] = 0.0

    # -- persistence ---------------------------------------------------
    def _emit(self, record: dict[str, Any]) -> None:
        if not self._log_path:
            return
        try:
            parent = os.path.dirname(os.path.abspath(self._log_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            roll_log(self._log_path)
        except (OSError, TypeError, ValueError) as exc:
            # Instrumentation must never break evolution. OSError covers IO;
            # TypeError/ValueError cover json.dumps on a non-serializable
            # `extra` payload -- without them the error escapes end_generation
            # and aborts the generation loop (EE-W5 review).
            sys.stderr.write(f"[phase_timing] log disabled after emit failure: {exc!r}\n")
            self._log_path = None

    # -- run-level summary ---------------------------------------------
    def summary(self, run_wall: float | None = None) -> dict[str, Any]:
        """Additive per-phase summary for the cycle result dict."""
        if not self.enabled:
            return {}
        totals = {p: round(sum(r["phase_seconds"][p] for r in self.records), 3) for p in PHASES}
        gen_seconds = round(sum(r["generation_seconds"] for r in self.records), 3)
        vm_calls = sum(int(r.get("vm_eval_calls", 0)) for r in self.records)
        wall = run_wall if run_wall is not None else (time.perf_counter() - self._run_start)
        named = sum(totals[p] for p in NAMED_PHASES)
        vm_farm = round(sum(float(r.get("vm_eval_seconds", 0.0)) for r in self.records), 3)
        return {
            "gen_phase_generations": len(self.records),
            "gen_phase_seconds": gen_seconds,
            "gen_phase_vm_eval_seconds": totals["vm_eval"],
            "gen_phase_vm_farm_seconds": vm_farm,
            "gen_phase_mutation_propagation_seconds": totals["mutation_propagation"],
            "gen_phase_selection_archives_seconds": totals["selection_archives"],
            "gen_phase_diagnostics_logging_seconds": totals["diagnostics_logging"],
            "gen_phase_unattributed_seconds": totals["unattributed"],
            "gen_phase_named_seconds": round(named, 3),
            "gen_phase_vm_eval_calls": vm_calls,
            "gen_phase_run_init_seconds": round(self.run_init_seconds, 3),
            "gen_phase_run_finalize_seconds": round(self.run_finalize_seconds, 3),
            "gen_phase_run_seconds": round(wall, 3),
            "gen_phase_log_path": self._log_path,
        }


def roll_log(path: str, max_bytes: int | None = None) -> None:
    """Keep the JSONL log rolling: retain only the newest half when full."""
    if max_bytes is None:
        max_bytes = int(os.environ.get("EE_GEN_TIMING_LOG_MAX_BYTES", str(DEFAULT_MAX_BYTES)))
    if max_bytes <= 0:
        return
    try:
        if os.path.getsize(path) <= max_bytes:
            return
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        keep = len(lines) // 2 or 1
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines[-keep:])
    except OSError:
        return


def default_log_path(project_root: str) -> str:
    override = os.environ.get("EE_GEN_TIMING_LOG")
    if override:
        return override
    return os.path.join(project_root, DEFAULT_LOG_NAME)


def timers_enabled() -> bool:
    return os.environ.get("EE_GEN_TIMING", "1").strip().lower() not in {"0", "false", "off", "no"}
