"""Contract tests for tests/live_results.py, the shared live-file reader.

The evolution daemon appends to the production ``multi_cycle_results.jsonl``
while the suite streams it, and ``scripts.dedupe_cycle_results.dedupe()``
rewrites it via ``os.replace``.  These tests pin the tolerant-read contract
that every converted test site relies on: absent file -> skip, torn line ->
ignored, split multi-byte char -> ignored, stale ``isfile`` -> retried, and
no row ever emitted twice.
"""

import json

import pytest

from tests import live_results
from tests.live_results import RESULTS_JSONL, live_champion_program, live_champion_row, live_result_rows


def _row(words: int, cycle: int, **extra):
    row = {
        "cycle": cycle,
        "final_level": 30,
        "champion_words": words,
        "champion_program": ["00000013"] * words,
    }
    row.update(extra)
    return row


def _write(tmp_path, text: str) -> None:
    (tmp_path / RESULTS_JSONL).write_text(text, encoding="utf-8")


def test_absent_file_skips_with_the_usual_reason(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pytest.skip.Exception, match="^needs live results jsonl$"):
        list(live_result_rows())
    with pytest.raises(pytest.skip.Exception, match="^needs live results jsonl$"):
        live_champion_row(60)


def test_torn_tail_line_is_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    good = json.dumps(_row(60, 1))
    torn = json.dumps(_row(60, 2, padding="x" * 4096))[:1663]  # no newline
    _write(tmp_path, good + "\n" + torn)
    assert [r["cycle"] for r in live_result_rows()] == [1]


def test_torn_line_between_complete_lines_is_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torn = json.dumps(_row(60, 9, padding="y" * 500))[:200]  # unterminated
    body = (
        json.dumps(_row(60, 1))
        + "\n"
        + json.dumps(_row(61, 2))
        + "\n"
        + torn
        + "\n"  # torn document closed by a newline: still unparsable
        + json.dumps(_row(60, 3))
        + "\n"
    )
    _write(tmp_path, body)
    assert [r["cycle"] for r in live_result_rows()] == [1, 2, 3]


def test_split_multibyte_character_at_eof_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    complete = json.dumps(_row(60, 1, note="é" * 40), ensure_ascii=False)
    torn = json.dumps(_row(60, 2, note="é" * 400), ensure_ascii=False)
    raw = torn.encode("utf-8")
    # Cut INSIDE the final multi-byte sequence, so the file ends on a bare
    # leading byte -- not after it.  Strict decoding of that tail must fail,
    # otherwise this test would not exercise errors="replace" at all (the
    # exact branch the daemon's mid-append split of a multi-byte char hits).
    cut = raw[: raw.rfind(b"\xc3") + 1]
    with pytest.raises(UnicodeDecodeError):
        cut.decode("utf-8")  # proves the character really is split in half
    (tmp_path / RESULTS_JSONL).write_bytes(complete.encode("utf-8") + b"\n" + cut)
    assert [r["cycle"] for r in live_result_rows()] == [1]


def test_complete_row_without_trailing_newline_is_yielded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, json.dumps(_row(60, 1)))  # file closed mid-append ...
    assert [r["cycle"] for r in live_result_rows()] == [1]  # ... but it PARSES


def test_non_object_json_line_is_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "[1, 2, 3]\n" + json.dumps(_row(60, 1)) + "\n42\n")
    assert [r["cycle"] for r in live_result_rows()] == [1]


def test_open_oserror_retries_three_times_then_skips(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, json.dumps(_row(60, 1)) + "\n")
    calls = {"n": 0}

    def always_broken(*_args, **_kwargs):
        calls["n"] += 1
        raise OSError(13, "Permission denied (probe)")

    monkeypatch.setattr(live_results, "open", always_broken, raising=False)
    with pytest.raises(pytest.skip.Exception, match="needs live results jsonl"):
        list(live_result_rows())
    assert calls["n"] == 3, f"expected 3 open attempts, got {calls['n']}"


def test_transient_oserror_is_retried_and_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = json.dumps(_row(60, 1)) + "\n" + json.dumps(_row(60, 2)) + "\n"
    _write(tmp_path, body)
    real_open = open
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(13, "stale isfile->open window (probe)")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(live_results, "open", flaky, raising=False)
    assert [r["cycle"] for r in live_result_rows()] == [1, 2]
    assert calls["n"] == 2


def test_mid_read_oserror_never_emits_duplicate_rows(tmp_path, monkeypatch):
    """A retried read must not hand accumulating callers the rows twice."""
    monkeypatch.chdir(tmp_path)
    body = "".join(json.dumps(_row(60, cycle)) + "\n" for cycle in (1, 2, 3))
    _write(tmp_path, body)
    lines = body.splitlines(keepends=True)
    calls = {"n": 0}

    class _FlakyHandle:
        """Context manager that dies mid-iteration on the first attempt."""

        def __init__(self, attempt: int):
            self._attempt = attempt
            self._lines = iter(lines)
            self._yielded = 0

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            if self._attempt == 1:
                self._yielded += 1
                if self._yielded > 2:
                    raise OSError(5, "simulated mid-read failure (probe)")
            return next(self._lines)

    def flaky_open(*args, **kwargs):
        calls["n"] += 1
        if args and str(args[0]) == RESULTS_JSONL:
            return _FlakyHandle(calls["n"])
        return open(*args, **kwargs)

    monkeypatch.setattr(live_results, "open", flaky_open, raising=False)
    rows = list(live_result_rows())
    assert calls["n"] == 2, f"expected 2 read attempts, got {calls['n']}"
    assert [r["cycle"] for r in rows] == [1, 2, 3], f"a re-yielded read duplicated rows: {[r['cycle'] for r in rows]}"


def test_blank_lines_and_whitespace_only_lines_are_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(
        tmp_path,
        "\n" + json.dumps(_row(60, 1)) + "\n\n   \n" + json.dumps(_row(60, 2)) + "\n",
    )
    assert [r["cycle"] for r in live_result_rows()] == [1, 2]


def test_champion_row_returns_last_match_and_none_without_skipping(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    body = "".join(
        json.dumps(row) + "\n"
        for row in (
            _row(60, 1),
            _row(61, 2),
            _row(60, 3, promotion_raw_fitness=12.5),
            {"cycle": 4, "final_level": 29, "champion_words": 60},
        )
    )
    _write(tmp_path, body)
    row = live_champion_row(60)
    assert row is not None
    assert row["cycle"] == 3  # LAST matching L30 row, not the first
    assert live_champion_row(999) is None  # callers keep their own assert/skip


def test_live_champion_program_decodes_the_last_matching_row(tmp_path, monkeypatch):
    """The shared probe lookup: hex words -> ints, whole program, no truncation."""
    monkeypatch.chdir(tmp_path)
    body = "".join(
        json.dumps(row) + "\n"
        for row in (
            _row(60, 1),
            _row(61, 2),
            _row(60, 3, champion_program=["00100073", "00000013"]),
        )
    )
    _write(tmp_path, body)
    prog = live_champion_program(60)
    assert prog == [0x00100073, 0x00000013]  # last match wins, words decoded as ints
    again = live_champion_program(60)
    assert again == prog  # same row -> same words on every call
    assert again is not prog  # but never a shared / cached list object


def test_live_champion_program_skips_on_absent_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(pytest.skip.Exception, match="^needs live results jsonl$"):
        live_champion_program(60)


def test_live_champion_program_fails_loudly_when_no_row_matches(tmp_path, monkeypatch):
    """A present file without a matching row must FAIL, never skip or return []."""
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, json.dumps(_row(61, 1)) + "\n")  # present, but no 60-word L30 row
    try:
        live_champion_program(60)
    except pytest.skip.Exception:
        # Skipped is a BaseException, so a bare pytest.raises(AssertionError)
        # would let this slip through as a green SKIP -- exactly the silent
        # degradation this helper exists to prevent.
        pytest.fail("live_champion_program SKIPPED on a present file; the probe must FAIL")
    except AssertionError:
        pass  # the loud failure the 24 probe call sites rely on
    else:
        pytest.fail("live_champion_program returned without failing on a missing row")
