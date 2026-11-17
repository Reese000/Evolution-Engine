#!/usr/bin/env python3
"""EE-W1b-FU4 (t_bd0760ea): every reader/writer of milestone1/recurrent_data.jsonl agrees on UTF-8.

``ChampionArchive._write_recurrent_entries()`` / ``save()`` were pinned to
``encoding="utf-8"`` by t_afa953da, but ``run_multi_cycle`` still opened the
same file with the *platform default* encoding at three sites:

* ``merge_archive_to_recurrent`` -- the dedup-hash load and the append,
* ``_validate_retrain_eligibility`` -- the example counter.

That is the mirror image of the hazard t_afa953da closed on the writer side.
It is latent today only because ``json.dumps(ensure_ascii=True)`` keeps the file
pure ASCII; the moment a non-ASCII value reaches the file (``ensure_ascii=False``,
or a cp1252-default host on the append path) the reader raises
``UnicodeDecodeError`` -- which ``except (OSError, PermissionError)`` does NOT
catch, so it escapes ``merge_archive_to_recurrent`` and kills the cron cycle
(same failure class t_afa953da fixed for cron_run_10's archive counter).

A fourth reader in the same file (``compute_memorization_generalization``'s
``recurrent_data_path`` -- currently uncalled anywhere in the repo) is pinned
too, so no ``open()`` of this file in ``run_multi_cycle.py`` is left on the
locale default and the source-level guard at the bottom of this module cannot
be evaded by a dormant site.

``errors="replace"`` on the two pure-read paths is deliberate and matches
``cron_run_10._count_archive_lines``: dedup is documented best-effort, and a
torn/partial line from a crash mid-append must degrade to a skipped row rather
than an uncaught decode error out of the cycle.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Non-ASCII on purpose: em dash, U+00C1 (kept as an escape below), CJK, emoji.
# Kept as escapes so this file stays pure ASCII and its own encoding can never
# be the thing under test. The U+00C1 is what makes the fixture undecodable on
# cp1252: its UTF-8 second byte 0x81 has no cp1252 mapping, so the pre-fix
# (locale-default) reader really does raise on this payload on any non-UTF-8
# default platform.
NON_ASCII_VALUE = "ok \u2014 \u00c1\u4e2d\u6587 \U0001f600"


class _NoAsciiEscape:
    """json module stand-in that lets non-ASCII through to the writer."""

    loads = staticmethod(json.loads)
    dumps = staticmethod(lambda obj, **kw: json.dumps(obj, ensure_ascii=False, **kw))


CHILD = r'''"""Drive run_multi_cycle's recurrent_data.jsonl sites on a platform whose
DEFAULT encoding is NOT UTF-8 (PYTHONUTF8=0 -> the ANSI code page).

argv: repo_root, tmp_root, non_ascii_value
"""
import json as _json
import locale
import os
import sys

repo, root, value = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, repo)

import champion_archive as ca
import run_multi_cycle as rmc

print("PREFERRED", locale.getpreferredencoding(False))


class _NoAsciiEscape:
    loads = staticmethod(_json.loads)
    dumps = staticmethod(lambda obj, **kw: _json.dumps(obj, ensure_ascii=False, **kw))


ca.json = _NoAsciiEscape
rmc.json = _NoAsciiEscape

champions = os.path.join(root, "champions")
recurrent_dir = os.path.join(root, "recurrent_data")
os.makedirs(champions, exist_ok=True)
os.makedirs(recurrent_dir, exist_ok=True)
archive_path = os.path.join(champions, "archive.jsonl")
recurrent_path = os.path.join(recurrent_dir, "recurrent_data.jsonl")

archive = ca.ChampionArchive(path=archive_path, max_size=100)
assert archive.add([0x100073], level=13, fitness=200.0, cycle=1, trace_status=value)
assert archive.add([0x100075], level=13, fitness=210.0, cycle=2, trace_status=value)
archive.save()
# Seed the file with ONLY the first entry, through the real writer, so the
# merge's append path still has something new (and non-ASCII) to write.
assert archive._write_recurrent_entries(recurrent_path, archive.entries[:1]) == 1

# --- controls: what the pre-fix (locale-default) reader would do -----------
try:
    with open(recurrent_path, encoding="utf-8") as fh:
        fh.read()
except UnicodeDecodeError:
    print("UTF8READ UnicodeDecodeError")
else:
    print("UTF8READ ok")

try:
    with open(recurrent_path) as fh:  # exactly the pre-fix expression
        fh.read()
except UnicodeDecodeError:
    print("CONTROL UnicodeDecodeError")
else:
    print("CONTROL no-raise")

# Point every recurrent-data global at the throwaway files FIRST, so none of
# the three calls below can touch the real milestone1/recurrent_data.jsonl.
rmc.PROJECT_ROOT = root
rmc.M1_RECURRENT = recurrent_dir
rmc.M1_RECURRENT_DATA = recurrent_path

# --- site 3: _validate_retrain_eligibility ---------------------------------
skip_result, count = rmc._validate_retrain_eligibility(1, None)
assert skip_result is None, skip_result
print("COUNT", count)

# --- sites 1 + 2: merge_archive_to_recurrent (dedup load, then append) -----
print("MERGE", rmc.merge_archive_to_recurrent(1))

with open(recurrent_path, "rb") as fh:
    raw = fh.read()
print("APPEND_UTF8", "ok" if value.encode("utf-8") in raw else "missing")
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


def _point_recurrent_paths(tmp_path: Path, monkeypatch, recurrent_path: str):
    """Redirect run_multi_cycle's three recurrent-data globals into tmp_path."""
    import run_multi_cycle as rmc

    monkeypatch.setattr(rmc, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(rmc, "M1_RECURRENT", str(tmp_path / "recurrent_data"))
    monkeypatch.setattr(rmc, "M1_RECURRENT_DATA", recurrent_path)
    return rmc


def test_dedup_load_reads_the_non_ascii_row_written_by_write_recurrent_entries(tmp_path, monkeypatch):
    """Round trip: _write_recurrent_entries -> merge_archive_to_recurrent's dedup load.

    The seed row carries a real multi-byte ``trace_status`` and the merge must
    read it back, recognise its sha256 and append nothing -- i.e. the dedup
    load path decodes the file the writer produced.
    """
    import champion_archive as ca
    from champion_archive import ChampionArchive

    monkeypatch.setattr(ca, "json", _NoAsciiEscape)

    os.makedirs(tmp_path / "champions", exist_ok=True)
    archive_path = str(tmp_path / "champions" / "archive.jsonl")
    archive = ChampionArchive(path=archive_path, max_size=100)
    assert archive.add([0x100073, 0x100074], level=13, fitness=200.0, cycle=1, trace_status=NON_ASCII_VALUE)
    archive.save()

    recurrent_dir = tmp_path / "recurrent_data"
    os.makedirs(recurrent_dir, exist_ok=True)
    recurrent_path = str(recurrent_dir / "recurrent_data.jsonl")

    # the writer under test
    assert archive._write_recurrent_entries(recurrent_path, list(archive.entries)) == 1

    raw = Path(recurrent_path).read_bytes()
    # real multi-byte UTF-8 reached the file, not \uXXXX escapes ...
    assert NON_ASCII_VALUE.encode("utf-8") in raw
    # ... and that payload is undecodable for a cp1252-default reader, i.e. it
    # is exactly what the pre-fix `open(recurrent_data)` would have rejected on
    # any non-UTF-8 default platform. On a UTF-8-default host (this one runs
    # with PYTHONUTF8=1) the pre-fix reader happens to survive, so the forced-
    # cp1252 subprocess test below -- not this round trip -- is the bite for
    # that site here.
    with pytest.raises(UnicodeDecodeError):
        raw.decode("cp1252")

    rmc = _point_recurrent_paths(tmp_path, monkeypatch, recurrent_path)

    # the dedup load path reads the non-ASCII row, matches the sha -> nothing new
    assert rmc.merge_archive_to_recurrent(cycle=1) == 0
    # nothing was appended, so the writer's bytes survived untouched
    assert Path(recurrent_path).read_bytes() == raw

    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["trace_status"] == NON_ASCII_VALUE
    assert rows[0]["save_reason"] == "champion_archive"


def test_merge_and_retrain_validate_survive_a_platform_default_encoding(tmp_path):
    """The bite: with a non-UTF-8 default encoding, all three sites still work.

    Runs in a subprocess under PYTHONUTF8=0 so ``open()`` without ``encoding=``
    uses the ANSI code page -- the exact configuration in which the pre-fix
    readers raise. The child also drives the append path with
    ``ensure_ascii=False`` so a real multi-byte value has to be *encoded* there.
    """
    proc = _run_child(
        tmp_path,
        CHILD,
        [str(REPO_ROOT), str(tmp_path), NON_ASCII_VALUE],
        {"PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LANG": "C", "LC_ALL": "C"},
    )

    # Platform-independent first: a child that dies (import error, API drift,
    # an uncaught decode/encode error) has to fail on every host, never skip.
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _child_line(proc, "UTF8READ") == "ok"
    assert _child_line(proc, "COUNT") == "1"  # site 3 read the seeded row
    assert int(_child_line(proc, "MERGE")) >= 1  # sites 1+2: dedup load + append
    assert _child_line(proc, "APPEND_UTF8") == "ok"  # append encoded UTF-8, not cp1252

    # Only the "a non-UTF-8 default" claim is host-specific (cp1252 here,
    # ansi_x3.4-1968 on a Linux CI runner, possibly utf-8 on a Windows box with
    # the UTF-8 beta code page), so only it may skip.
    preferred = _child_line(proc, "PREFERRED")
    if preferred.lower().replace("-", "") in {"utf8", "utf_8"}:
        pytest.skip(f"platform default encoding is already {preferred}; hazard not reproducible here")
    # the fixture really is undecodable for the pre-fix reader
    assert _child_line(proc, "CONTROL") == "UnicodeDecodeError"


def test_a_torn_or_corrupt_line_does_not_escape_the_dedup_load(tmp_path, monkeypatch):
    """errors=\"replace\": a byte a UTF-8 decoder rejects must not kill the merge.

    Crash-truncated appends leave exactly this kind of damage behind, and the
    surrounding ``except (OSError, PermissionError)`` does not catch a decode
    error. Pre-fix this test fails with UnicodeDecodeError on a UTF-8-default
    host; on a cp1252-default host those two bytes decode cleanly, so the
    forced-cp1252 subprocess test above carries that host class instead.
    """
    from champion_archive import ChampionArchive

    os.makedirs(tmp_path / "champions", exist_ok=True)
    archive_path = str(tmp_path / "champions" / "archive.jsonl")
    archive = ChampionArchive(path=archive_path, max_size=100)
    assert archive.add([0x100073], level=13, fitness=200.0, cycle=1)
    archive.save()

    recurrent_dir = tmp_path / "recurrent_data"
    os.makedirs(recurrent_dir, exist_ok=True)
    recurrent_path = recurrent_dir / "recurrent_data.jsonl"

    # a valid row, then a line no UTF-8 decoder accepts (0xff is never valid)
    recurrent_path.write_bytes(b'{"champion_sha256": "seed", "trace_status": "x"}\n{"torn": \xff\xfe\n')

    rmc = _point_recurrent_paths(tmp_path, monkeypatch, str(recurrent_path))

    # does not raise, and the good row still dedups while the new one appends
    added = rmc.merge_archive_to_recurrent(cycle=1)
    assert added >= 1
    # the counter reads through the damage instead of raising
    skip_result, count = rmc._validate_retrain_eligibility(1, None)
    assert skip_result is None
    assert count >= 3  # seed + torn + at least one appended copy

    decoded = recurrent_path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    for line in decoded:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # the deliberately torn row
    assert rows[0]["champion_sha256"] == "seed"
    assert rows[-1]["save_reason"] == "champion_archive"


def test_run_multi_cycle_pins_utf8_on_every_recurrent_data_open():
    """No ``open()`` of this file may stay on the locale default, now or later."""
    src = (REPO_ROOT / "run_multi_cycle.py").read_text(encoding="utf-8")

    calls = re.findall(r"open\([^()\n]*\)", src)
    recurrent_calls = [c for c in calls if "recurrent_data" in c]

    # the three sites this card names, plus the MemGen reader
    assert len(recurrent_calls) >= 3, recurrent_calls
    unpinned = [c for c in recurrent_calls if 'encoding="utf-8"' not in c]
    assert unpinned == [], f"recurrent_data.jsonl opened on the platform default: {unpinned}"

    # The regex above cannot see a nested-paren call (open(os.path.join(...)))
    # or one wrapped across lines, so re-check every SOURCE LINE that both
    # opens something and names recurrent data -- that form is style-proof.
    for lineno, line in enumerate(src.splitlines(), 1):
        if "open(" in line and "recurrent_data" in line:
            assert "encoding=" in line, (
                f"run_multi_cycle.py:{lineno} opens recurrent_data.jsonl on the " f"platform default: {line.strip()}"
            )

    # both pure-read paths tolerate torn bytes, not just one of them
    assert sum('errors="replace"' in c for c in recurrent_calls) == 2
