#!/usr/bin/env python3
"""EE-W1b-FU3 (t_afa953da): every reader/writer of champions/archive.jsonl agrees on UTF-8.

t_670fe1e3 hardened ``ChampionArchive._load``. This module covers the two other
touch points the blast-radius audit found:

1. ``cron_run_10.py`` counted archive lines with an unguarded ``open()`` — one
   non-UTF8 byte raised ``UnicodeDecodeError`` straight out of ``main()`` and
   killed the cron run. The second, per-cycle copy of the same expression sat
   inside the cycle's ``try``, so it did not kill the script, but the handler's
   ``continue`` then dropped that cycle's archive logging, its meta-controller
   update and its saturation/scale check.
2. ``ChampionArchive.save()`` / ``_write_recurrent_entries()`` wrote with the
   *platform* default encoding while ``_load()`` reads UTF-8 — latent today
   (``json.dumps`` defaults to ``ensure_ascii=True`` and this host runs with
   PYTHONUTF8=1) but a real hazard on a cp1252-default platform combined with
   ``ensure_ascii=False``: either silent mojibake -- ``_load``'s
   ``errors="replace"`` keeps the row with U+FFFD-mangled data -- or a
   ``UnicodeEncodeError`` out of ``save()``, raised after ``open(path, "w")``
   has already truncated the archive.

The BOM hole in ``_load`` (item 3) has its regression test at the end of
``tests/unit/test_archive_scan_floor_probes.py`` next to the other ``_load``
guards.

The cron_run_10 assertions run in a subprocess on purpose: importing that
module does ``os.chdir(SCRIPT_DIR)``, which would move the whole pytest
session's cwd. The child still triggers run_multi_cycle's ``crash_fault.log``
truncation on import (11 other test modules already import it in-process), so
what the subprocess isolates is the chdir, not that side effect.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Non-ASCII on purpose: em dash, CJK, emoji. Kept as escapes so this file stays
# ASCII and its own encoding can never be the thing under test.
NON_ASCII_VALUE = "ok \u2014 \u4e2d\u6587 \U0001f600"

CRON_CHILD = r'''"""Import cron_run_10 for real and drive _count_archive_lines four ways."""
import sys

sys.path.insert(0, sys.argv[1])
import cron_run_10 as cron

path = sys.argv[2]

# Control: the unguarded reader this card replaced really does explode here.
try:
    with open(path, encoding="utf-8") as handle:
        handle.read()
except UnicodeDecodeError:
    print("CONTROL UnicodeDecodeError")
else:
    print("CONTROL no-raise")

print("COUNT", cron._count_archive_lines(path))
print("MISSING", cron._count_archive_lines(path + ".does-not-exist", default=7))


def _denied(*args, **kwargs):
    raise PermissionError("archive is locked")


cron.open = _denied  # module global shadows builtins inside the helper
print("IOERR", cron._count_archive_lines(path, default=11))
'''

WRITER_CHILD = r'''"""Write a non-ASCII archive on a NON-UTF8-default platform.

Run with PYTHONUTF8=0 + PYTHONCOERCECLOCALE=0 so locale.getpreferredencoding()
is the ANSI code page (cp1252 on this host) rather than UTF-8, and force
json.dumps to ensure_ascii=False so non-ASCII bytes actually reach the encoder:
the two halves of the latent hazard the card describes. Before the fix save()
raised UnicodeEncodeError here (or, if the value were encodable, mojibake'd it).
"""
import json as _json
import locale
import sys

sys.path.insert(0, sys.argv[1])
import champion_archive as ca

value = sys.argv[3]
print("PREFERRED", locale.getpreferredencoding(False))


class _NoAsciiEscape:
    """json module stand-in that lets non-ASCII through to the writer."""

    loads = staticmethod(_json.loads)
    dumps = staticmethod(lambda obj, **kw: _json.dumps(obj, ensure_ascii=False, **kw))


ca.json = _NoAsciiEscape

path = sys.argv[2]
archive = ca.ChampionArchive(path=path)
assert archive.add([0x11, 0x22, 0x33], level=3, fitness=12.5, cycle=9, trace_status=value)
archive.save()

recurrent = path + ".recurrent.jsonl"
assert archive.save_recurrent_data(recurrent, fitness_threshold=0.0) == 1

raw = open(path, "rb").read()
assert value.encode("utf-8") in raw, raw[:200]

reloaded = ca.ChampionArchive(path=path)
assert [e.trace_status for e in reloaded.entries] == [value], reloaded.entries
assert reloaded._load_skipped == 0

rec_raw = open(recurrent, "rb").read()
assert value.encode("utf-8") in rec_raw, rec_raw[:200]
print("OK")
'''


def _run_child(
    tmp_path: Path, source: str, args: list[str], env_overrides: dict | None = None
) -> subprocess.CompletedProcess:
    """Run a throwaway script with the interpreter pytest itself is using."""
    script = tmp_path / f"child_{abs(hash(source)) % 10**8}.py"
    script.write_text(source, encoding="utf-8")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        check=False,  # exit code asserted by the caller
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(tmp_path),
        env=env,
        timeout=300,
    )


def _child_line(proc: subprocess.CompletedProcess, prefix: str) -> str:
    for line in proc.stdout.splitlines():
        if line.startswith(prefix + " "):
            return line.split(" ", 1)[1]
    raise AssertionError(f"{prefix!r} missing from child output:\n{proc.stdout}\n{proc.stderr}")


def test_cron_run_10_counter_survives_non_utf8_archive_byte(tmp_path: Path):
    """Item 1: a 0xff byte in archive.jsonl must not kill cron_run_10's counter."""
    archive = tmp_path / "archive.jsonl"
    archive.write_bytes(b'{"n":1}\n\xff\xfe not utf-8\n{"n":2}\n\n{"n":3}\n')

    proc = _run_child(tmp_path, CRON_CHILD, [str(REPO_ROOT), str(archive)])

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    # the fixture really is undecodable, i.e. the pre-fix reader would have died here
    assert _child_line(proc, "CONTROL") == "UnicodeDecodeError"
    # 4 non-empty rows: 3 valid + the damaged one, which still counts
    assert _child_line(proc, "COUNT") == "4"
    assert _child_line(proc, "MISSING") == "7"  # absent archive -> caller's default
    # locked/unreadable archive -> logged warning, still the caller's default
    assert "WARNING: could not count archive lines" in proc.stdout
    assert _child_line(proc, "IOERR") == "11"


def test_cron_run_10_counts_through_the_helper_at_both_sites():
    """The whole point: neither raw reader of archive.jsonl may survive the fix."""
    src = (REPO_ROOT / "cron_run_10.py").read_text(encoding="utf-8")

    # initial size + per-cycle re-count
    assert src.count("archive_size = _count_archive_lines(") == 2
    assert "with open(archive_path)" not in src
    # the counting expression survives exactly once, inside the guarded helper
    assert src.count("sum(1 for line in f if line.strip())") == 1

    helper = src.split("def _count_archive_lines", 1)[1].split("def main", 1)[0]
    assert 'encoding="utf-8", errors="replace"' in helper
    assert "except OSError" in helper
    assert "sum(1 for line in f if line.strip())" in helper


def test_cron_counter_matches_a_binary_count_of_the_live_archive(tmp_path: Path):
    """Real-data oracle: the counter must agree with a byte-level count.

    The live file is snapshotted into tmp_path first, so the parent's byte
    count and the child's read see the same bytes (cron/other agents rewrite
    champions/archive.jsonl under us) and so the child script never lands in
    the repo's champions/ directory.
    """
    live = REPO_ROOT / "champions" / "archive.jsonl"
    if not live.is_file():
        pytest.skip("checkout has no champions/archive.jsonl")

    payload = live.read_bytes()
    snapshot = tmp_path / "archive.jsonl"
    snapshot.write_bytes(payload)
    expected = sum(1 for line in payload.splitlines() if line.strip())

    proc = _run_child(tmp_path, CRON_CHILD, [str(REPO_ROOT), str(snapshot)])

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    # matches the byte-level count whether or not the live file is clean UTF-8
    assert _child_line(proc, "COUNT") == str(expected)
    assert _child_line(proc, "CONTROL") in {"no-raise", "UnicodeDecodeError"}


def test_save_load_roundtrip_preserves_non_ascii_trace_status(tmp_path: Path, monkeypatch):
    """Item 2 (round-trip half): save() -> _load() keeps a non-ASCII value byte-for-byte.

    json.dumps is forced to ensure_ascii=False so real multi-byte UTF-8 has to
    cross the disk boundary instead of being escaped into pure ASCII. The
    platform-default-encoding *bite* lives in the subprocess test below
    (``test_save_writes_utf8_on_a_platform_default_platform``), which runs the
    same two writers under PYTHONUTF8=0; this test pins the round trip itself.
    """
    import champion_archive as ca
    from champion_archive import ChampionArchive

    class _NoAsciiEscape:
        loads = staticmethod(json.loads)
        dumps = staticmethod(lambda obj, **kw: json.dumps(obj, ensure_ascii=False, **kw))

    monkeypatch.setattr(ca, "json", _NoAsciiEscape)

    path = tmp_path / "archive.jsonl"
    writer = ChampionArchive(path=str(path))
    assert writer.add([1, 2, 3], level=4, fitness=99.5, cycle=17, trace_status=NON_ASCII_VALUE)

    writer.save()

    raw = path.read_bytes()
    # real multi-byte UTF-8 reached the file ...
    assert NON_ASCII_VALUE.encode("utf-8") in raw
    # ... and strict UTF-8 decodes it back to the exact value
    row = json.loads(raw.decode("utf-8").splitlines()[0])
    assert row["trace_status"].encode("utf-8") == NON_ASCII_VALUE.encode("utf-8")

    reader = ChampionArchive(path=str(path))
    assert reader._load_skipped == 0
    assert len(reader.entries) == 1
    assert reader.entries[0].trace_status == NON_ASCII_VALUE
    assert reader.entries[0].to_dict() == writer.entries[0].to_dict()

    # Both writers must pin their encoding instead of inheriting the locale
    # default -- this is what fails on the pre-fix file on ANY platform, where
    # open(..., "w") carries no encoding= at all.
    import inspect

    writer_src = inspect.getsource(ChampionArchive.save) + inspect.getsource(ChampionArchive._write_recurrent_entries)
    assert 'open(self.path, "w", encoding="utf-8")' in writer_src
    assert 'open(output_path, "w", encoding="utf-8")' in writer_src


def test_save_writes_utf8_on_a_platform_default_platform(tmp_path: Path):
    """Item 2's bite: a non-UTF-8 default + ensure_ascii=False must still round-trip."""
    archive = tmp_path / "archive.jsonl"
    proc = _run_child(
        tmp_path,
        WRITER_CHILD,
        [str(REPO_ROOT), str(archive), NON_ASCII_VALUE],
        {"PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LANG": "C", "LC_ALL": "C"},
    )

    # Platform-independent first: a child that dies (broken save(), import
    # error, API drift) has to fail on every host, never skip.
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OK" in proc.stdout
    assert NON_ASCII_VALUE.encode("utf-8") in archive.read_bytes()

    # Only the non-UTF-8-default claim is host-specific (cp1252 here,
    # ansi_x3.4-1968 on the Linux CI runner, possibly utf-8 on a Windows box
    # with the UTF-8 beta code page), so only it may skip.
    preferred = _child_line(proc, "PREFERRED")
    if preferred.lower().replace("-", "") in {"utf8", "utf_8"}:
        pytest.skip(f"platform default encoding is already {preferred}; hazard not reproducible here")
    assert "utf" not in preferred.lower(), f"a non-UTF-8 platform default is required, got {preferred!r}"
