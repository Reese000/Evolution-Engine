"""Shared constants for evolution engine mixins."""

from __future__ import annotations

_EMPTY_TUPLE: tuple = ()
_EBREAK_INST: int = 0x00100073
_CRASH_STATUSES: frozenset[str] = frozenset(("crash", "trap", "exception"))
_TIMEOUT_STATUSES: frozenset[str] = frozenset(("timeout",))
MIN_FITNESS_FOR_TRACE: float = 0.1
MAX_RECURRENT_ENTRIES: int = 500
MAX_TERMINAL_RECURRENT_PER_CYCLE: int = 128
TERMINAL_RECURRENT_GENERATION_INTERVAL: int = 5
RECURRENT_DECAY_RATE: float = 0.01
