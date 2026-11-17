#!/usr/bin/env python3
r"""EE-W1b-FU5 (t_3af5cf11): ``save()`` must not delete the rows ``_load`` skipped.

``ChampionArchive._load`` is corruption-tolerant: a row it cannot use is skipped
on its own and counted in ``_load_skipped``. ``save()`` used to rewrite
``champions/archive.jsonl`` from ``self.entries`` alone, so every row the
tolerant loader refused was permanently deleted on the next save -- the
tolerance turned a recoverable problem into an unrecoverable one -- and the
count itself was reachable only as a private attribute, so nothing outside the
test suite ever learned that rows had vanished.

Two halves of the fix are pinned here:

1. every rejected row keeps its raw text and its file position, and ``save()``
   re-emits both, so a skipped row survives a save round trip and stays where it
   was (the reviewer's repro: a valid row carrying ``"generation": null``),
2. the count is surfaced: ``ChampionArchive.load_skipped`` is public and
   ``_load`` logs a WARNING -- once per (path, count) per process, because
   ``merge_archive_to_recurrent`` rebuilds an archive every cycle and this
   project treats read-path log spam as a defect.

Deliberately out of scope, per the card's decision note: ``_entry_loadable``'s
accept/reject policy is untouched, so the parent card's
``test_champion_archive_load_*`` tests keep asserting the same skip counts.
"""

from __future__ import annotations

import builtins
import json
import logging
from pathlib import Path

import champion_archive
from champion_archive import ChampionArchive, _hash_program


def _row(sha: str) -> str:
    """One JSONL row carrying every ChampionEntry field, in dataclass field order.

    That order plus ``json.dumps`` defaults is byte-identical to what ``save()``
    writes for a healthy entry, which is what makes the row-for-row assertions
    below exact rather than approximate.
    """
    return json.dumps(
        {
            "program": [0, 1],
            "program_hex": ["0x00000000", "0x00000001"],
            "level": 1,
            "fitness": 0.5,
            "cycle": 1,
            "generation": 1,
            "instructions_count": 2,
            "sha256": sha,
            "trace_status": "ok",
            "task_score": 1.0,
            "timestamp": 0.0,
        }
    )


GOOD_1 = _row("sha_good_1")
GOOD_2 = _row("sha_good_2")
NULL_GENERATION = _row("sha_null_gen").replace('"generation": 1', '"generation": null')
TRUNCATED = '{"program": [0, }'
UNHASHABLE_SHA = (
    '{"program":[0,1],"program_hex":["0x00000000"],"level":1,"fitness":0.5,'
    '"cycle":1,"generation":1,"instructions_count":2,"sha256":[]}'
)

# good, unusable, good, corrupt, unhashable -- three rejected rows among two usable ones
DAMAGED = [GOOD_1, NULL_GENERATION, GOOD_2, TRUNCATED, UNHASHABLE_SHA]


def _write(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_save_keeps_skipped_rows_verbatim_and_in_place(tmp_path: Path) -> None:
    """Reviewer repro: a skipped row must survive save() instead of vanishing.

    Before the fix the file went 5 rows -> 2 rows on save() and a reload then
    reported ``_load_skipped == 0``, i.e. the engine had no way to notice the
    three rows were gone.
    """
    path = tmp_path / "archive.jsonl"
    _write(path, DAMAGED)

    archive = ChampionArchive(path=str(path))
    assert [e.sha256 for e in archive.entries] == ["sha_good_1", "sha_good_2"]
    assert archive.load_skipped == 3

    before = _lines(path)
    archive.save()
    after = _lines(path)

    # byte-for-byte identical rows, rejected ones still at indexes 1, 3 and 4
    assert after == before, "save() must re-emit rejected rows at their original position"

    reloaded = ChampionArchive(path=str(path))
    assert [e.sha256 for e in reloaded.entries] == ["sha_good_1", "sha_good_2"]
    assert reloaded.load_skipped == 3


def test_skipped_rows_survive_a_save_that_also_writes_new_entries(tmp_path: Path) -> None:
    """Mutation path: adding a champion must not evict the preserved raw rows."""
    path = tmp_path / "archive.jsonl"
    _write(path, DAMAGED)

    archive = ChampionArchive(path=str(path))
    assert archive.add([0x00100073], level=2, fitness=9.0, cycle=7)
    archive.save()

    lines = _lines(path)
    assert NULL_GENERATION in lines, "unusable row deleted by a save that had new data"
    assert TRUNCATED in lines
    assert UNHASHABLE_SHA in lines
    assert len(lines) == len(DAMAGED) + 1
    # the new entry is still appended after everything that came from disk
    assert lines[-1] == json.dumps(archive.entries[-1].to_dict())

    reloaded = ChampionArchive(path=str(path))
    assert reloaded.load_skipped == 3
    assert [e.sha256 for e in reloaded.entries][-1] == archive.entries[-1].sha256


def test_skipped_rows_survive_repeated_saves(tmp_path: Path) -> None:
    """save() is called more than once over a run; nothing may drift per call."""
    path = tmp_path / "archive.jsonl"
    _write(path, DAMAGED)

    archive = ChampionArchive(path=str(path))
    for _ in range(3):
        archive.save()
        assert _lines(path) == DAMAGED

    reloaded = ChampionArchive(path=str(path))
    assert reloaded.load_skipped == 3


def test_save_still_dedupes_duplicate_sha_rows(tmp_path: Path) -> None:
    """The one row save() may still drop: a duplicate of an already-stored row.

    Keeping exactly one copy of a program is the archive's documented dedup
    contract, so preservation is scoped to rows the loader *refused*.
    """
    path = tmp_path / "archive.jsonl"
    duplicate = _row("sha_dup")
    _write(path, [duplicate, duplicate])

    archive = ChampionArchive(path=str(path))
    assert [e.sha256 for e in archive.entries] == ["sha_dup"]
    assert archive.load_skipped == 0

    archive.save()
    assert len(_lines(path)) == 1


def test_skipped_row_with_undecodable_bytes_survives_a_save(tmp_path: Path) -> None:
    """Worst case for preservation: the raw bytes were unreadable in the first place.

    ``_load`` decodes with ``errors="replace"``, so the undecodable bytes are
    already U+FFFD by the time anything can be stashed -- the row still has to
    come back out (and stay counted on the next load) rather than disappear.
    """
    path = tmp_path / "archive.jsonl"
    path.write_bytes((_row("sha_ok") + "\n").encode("utf-8") + b"\xff\xfe\x00not utf-8 at all\n")

    archive = ChampionArchive(path=str(path))
    assert [e.sha256 for e in archive.entries] == ["sha_ok"]
    assert archive.load_skipped == 1

    archive.save()
    lines = _lines(path)
    assert len(lines) == 2, "the undecodable row disappeared on save"
    assert "not utf-8 at all" in lines[1]

    reloaded = ChampionArchive(path=str(path))
    assert reloaded.load_skipped == 1


def test_load_skipped_is_public_and_logged_once_per_path_and_count(tmp_path: Path, caplog) -> None:
    """Surfacing: the count is readable by callers and reported as a WARNING.

    The dedup half matters as much as the warning itself:
    ``merge_archive_to_recurrent`` constructs a ChampionArchive every cycle, and
    the preserved rows mean the condition persists forever, so a naive warning
    would print the same line every cycle of a long run.
    """
    path = tmp_path / "archive.jsonl"
    _write(path, [_row("sha_ok"), TRUNCATED])

    # explicit reset: the dedup set is process-global, so this test's
    # expectations must not depend on what an earlier test in this file did
    champion_archive._SKIP_WARNING_SEEN.clear()

    def warnings() -> list[object]:
        return [r for r in caplog.records if r.levelname == "WARNING"]

    with caplog.at_level(logging.WARNING, logger="champion_archive"):
        first = ChampionArchive(path=str(path))
        assert first.load_skipped == 1
        assert first.load_skipped == first._load_skipped

    logged = warnings()
    assert len(logged) == 1, "a damaged archive must be reported, not swallowed"
    message = logged[0].getMessage()
    assert str(path) in message
    assert "1 of 2 row(s)" in message

    # same file, same damage: a fresh instance must not repeat the line
    with caplog.at_level(logging.WARNING, logger="champion_archive"):
        ChampionArchive(path=str(path))
    assert len(warnings()) == 1, "read-path log spam: this fires every cycle otherwise"

    # new damage changes the count -> the key changes -> it warns again
    _write(path, [_row("sha_ok"), TRUNCATED, UNHASHABLE_SHA])
    with caplog.at_level(logging.WARNING, logger="champion_archive"):
        ChampionArchive(path=str(path))
    assert len(warnings()) == 2

    healthy = ChampionArchive(path=str(tmp_path / "fresh.jsonl"))
    assert healthy.load_skipped == 0


def test_read_failure_is_reported_loudly(tmp_path: Path, monkeypatch, caplog) -> None:
    """A mid-read OSError leaves a signal instead of a silent partial rewrite.

    ``_load`` swallows the OSError on purpose (an unreadable archive must not
    crash ``__init__``), but a later ``save()`` then persists only what was
    read - so the loss event, unlike its unknowable size, has to be reported.
    """
    path = tmp_path / "archive.jsonl"
    _write(path, [GOOD_1, GOOD_2])

    real_open = builtins.open

    def failing_open(file, *args, **kwargs):
        if str(file) == str(path):
            raise OSError("simulated mid-read failure")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)
    # clear through getattr: against pre-fix code the set does not exist, and
    # this test must still reach its own assertion instead of dying on
    # AttributeError before it can report the missing warning
    warned = getattr(champion_archive, "_IO_FAILURE_WARNED", None)
    if warned is not None:
        warned.clear()

    with caplog.at_level(logging.WARNING, logger="champion_archive"):
        archive = ChampionArchive(path=str(path))

    # open() stays patched until pytest teardown ON PURPOSE: the second
    # construction below has to hit the same OSError, otherwise the dedup
    # assertion would still pass if _IO_FAILURE_WARNED were deleted (round-2
    # reviewer proved that blind spot with a dedup-removing mutant).
    assert archive.entries == []
    assert archive.load_skipped == 0

    logged = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(logged) == 1, "a partial read must be reported, not swallowed"
    message = logged[0].getMessage()
    assert str(path) in message
    assert "read aborted" in message
    assert "0 row(s)" in message

    # once per path per process: a locked file must not warn every cycle
    with caplog.at_level(logging.WARNING, logger="champion_archive"):
        ChampionArchive(path=str(path))
    assert (
        len([r for r in caplog.records if r.levelname == "WARNING"]) == 1
    ), "the second failing load must be deduped, not repeated"


def test_program_readded_after_eviction_is_written_at_the_tail(tmp_path: Path) -> None:
    """An evicted row's slot must not be inherited by its re-added program.

    Positions used to live in a sha-keyed dict that survived eviction, so a
    re-added program was emitted at the dead row's index instead of the tail,
    contradicting save()'s "newest rows go last" contract.
    """
    path = tmp_path / "archive.jsonl"
    program = [0, 1]
    sha = _hash_program(program)
    _write(path, [_row(sha), NULL_GENERATION, GOOD_2])

    archive = ChampionArchive(path=str(path))
    assert archive.entries[0].sha256 == sha

    # _evict refuses to touch foundational levels, so force the capacity path:
    # at max_size=1 the loaded entry must go, taking its recorded position with it
    archive.max_size = 1
    archive._evict(new_level=7)
    assert sha not in [e.sha256 for e in archive.entries]
    archive.max_size = 10  # capacity restored: add() must not re-evict the re-add

    assert archive.add(program, level=7, fitness=9.0, cycle=9)
    archive.save()

    lines = _lines(path)
    assert NULL_GENERATION in lines, "the skipped row must still be preserved"
    assert len(lines) == 3
    assert lines[-1] == json.dumps(
        archive.entries[-1].to_dict()
    ), "a re-added program must be written at the tail, not at its evicted row's slot"


def test_skipped_row_keeps_surrounding_whitespace(tmp_path: Path) -> None:
    """``raw`` not ``strip()``-ed line: whitespace around a rejected row survives.

    ``save()`` claims to re-emit rejected rows verbatim; before the follow-up
    the stash held the stripped line, so a row written with trailing
    whitespace came back silently trimmed.
    """
    path = tmp_path / "archive.jsonl"
    padded = NULL_GENERATION + "\t   "
    _write(path, [GOOD_1, padded])

    archive = ChampionArchive(path=str(path))
    assert archive.load_skipped == 1

    archive.save()
    assert _lines(path) == [GOOD_1, padded], "trailing whitespace was trimmed on save"

    reloaded = ChampionArchive(path=str(path))
    assert reloaded.load_skipped == 1
