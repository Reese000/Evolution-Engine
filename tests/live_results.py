"""Tolerant readers for the LIVE production ``multi_cycle_results.jsonl``.

The running evolution daemon appends to the production results file while the
test suite streams it, so a read can land mid-append: the trailing line then
holds a truncated JSON document (possibly with a multi-byte character cut in
half).  ``scripts.dedupe_cycle_results.dedupe()`` additionally rewrites the
file out from under us (``os.replace`` on the live path, reached from
``ensure_metrics_integrity``), so an ``isfile`` check can go stale before
``open`` runs.

This module is the shared, imported implementation of the tolerant read
contract.  ``tests/unit/test_terminal_probe_scan_floor.py`` predates it (card
t_0598757b); it originally carried its OWN earlier copy that streamed while
reading and lacked the non-object guard -- same intent, not byte-equivalent --
and now imports from here (card t_9107d993), so no test module reimplements
this reader any more.  A few modules still inline the same absent-file
``pytest.skip`` clause as their own guard, with byte-identical wording.
``evolution.loop_integrity.count_result_
rows`` is production's third copy of the same contract: undecodable /
unparsable lines are ignored, complete lines are trusted.  Test modules that
read the live file import from here instead of re-implementing the skip /
torn-line / retry handling N times.

Fixture-local copies of the results file (``tmp_path / "multi_cycle_results
.jsonl"``) are NOT this module's business -- those are written by the test
itself and are never torn.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from typing import Any

import pytest

#: Relative path used by every live-file reader; resolves against the pytest
#: cwd (the repo root for a normal run).
RESULTS_JSONL = "multi_cycle_results.jsonl"


def live_result_rows(path: str = RESULTS_JSONL) -> Iterator[dict[str, Any]]:
    """Yield parsed rows from the LIVE results jsonl, skipping torn lines.

    Behaviour contract (identical for every caller):

    * file absent -> ``pytest.skip("needs live results jsonl")``;
    * file vanishing between ``isfile`` and ``open`` (the dedupe
      ``os.replace`` window) -> retried 3x with a 50ms backoff, then skipped
      with the usual missing-file reason carrying the last ``OSError``;
    * a line that does not decode (``errors="replace"`` keeps a split
      multi-byte character from raising) or does not parse
      (``json.JSONDecodeError``) -> skipped as a torn mid-append line;
    * a line that parses to a non-object -> skipped (callers use ``.get``);
    * blank lines -> skipped.

    The whole file is parsed into a list BEFORE the first row is yielded, so
    a retried read (attempt 2 after an ``OSError``) can never emit a row
    twice.  That matters for callers that ACCUMULATE rows -- e.g.
    ``tests/unit/test_archive_scan_floor_probes.py::test_archive_intercross_
    rotates_offsets`` collects up to 4 parents -- where a re-yield would hand
    them duplicates and let a ``len(parents) < 2`` skip guard pass vacuously.
    """
    if not os.path.isfile(path):
        pytest.skip("needs live results jsonl")
    last_error: OSError | None = None
    rows: list[dict[str, Any]] = []
    for _attempt in range(3):
        try:
            rows = []
            with open(path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # Torn mid-append line: the daemon is still writing it.
                        continue
                    if isinstance(row, dict):
                        rows.append(row)
            break
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    else:
        pytest.skip(f"needs live results jsonl ({last_error})")
    yield from rows


def live_champion_row(words: int, path: str = RESULTS_JSONL) -> dict[str, Any] | None:
    """Last complete L30 row with ``champion_words == words``, else None.

    Returns None (never skips) when the file exists but holds no matching
    row, so callers keep their own ``assert row is not None`` / dedicated
    ``pytest.skip`` text.
    """
    row: dict[str, Any] | None = None
    for candidate in live_result_rows(path):
        if candidate.get("champion_words") == words and candidate.get("final_level") == 30:
            row = candidate
    return row


def live_champion_program(words: int, path: str = RESULTS_JSONL) -> list[int]:
    """``champion_program`` of the last ``words``-word L30 row, as ints.

    The one shared copy of the probe-side lookup several test modules used to
    duplicate verbatim as ``_live_60w_champion()`` (card t_eda54001).  Same
    raises / skips as that wrapper:

    * file absent (or vanishing under the ``os.replace`` window) -> the
      ``pytest.skip`` reasons inherited from :func:`live_result_rows`;
    * file present but no matching row -> ``assert row is not None`` fires, so
      the probe FAILS loudly instead of running with no program;
    * otherwise a fresh ``list[int]`` of the row's hex words -- safe for the
      call sites to slice (``[:48]``, ``[:27]``) or mutate in place.
    """
    row = live_champion_row(words, path)
    assert row is not None
    return [int(x, 16) for x in row["champion_program"]]
