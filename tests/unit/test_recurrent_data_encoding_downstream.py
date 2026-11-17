#!/usr/bin/env python3
"""EE-W1b-FU5 (t_8feac984): every reader/writer of milestone1/recurrent_data.jsonl
*outside* run_multi_cycle.py agrees on UTF-8.

Follow-up to ``tests/unit/test_recurrent_data_encoding.py`` (t_bd0760ea), which
pinned the four run_multi_cycle.py sites.  The adversarial reviewer of that card
found the same locale-default hazard one step downstream:

* ``milestone1/data/loading.py`` -- the CANONICAL training reader
  (run_multi_cycle -> milestone1/train_model.py -> training/loop.py ->
  ``load_champion_traces``), with nothing around the open that catches
  ``UnicodeDecodeError``;
* ``evolution/recurrent_mixin.py`` -- the live append writer, the novelty-anchor
  reader and the eviction rewrite (``os.fdopen(fd, "w")``); the eviction's outer
  handler only catches ``(RuntimeError, OSError)``;
* ``milestone1/build_recurrent_dataset.py`` -- the writer that feeds the
  canonical file plus its two line counters (one has no handler at all);
* ``milestone1/evaluation/pipeline.py`` -- the ``recurrent_data_path`` reader;
* ``archive/augmentation_subprocess.py`` -- reads whatever ``data_path`` the
  caller passes. ``milestone1/training/loop.py`` builds that config from the
  canonical path, but the repo-root script it looks for does not exist (the
  surviving copy is this archive/ one, invoked by archive/profile_phases.py);
  the pin keeps the contract right either way.
* ``tests/unit/test_recurrent_data_filter.py`` -- the ``_read_jsonl`` helper.

Three kinds of test, so the suite bites on a UTF-8-default host as well as on a
cp1252 one:

1. round trips -- a REAL multi-byte UTF-8 row written and read back through the
   touched call of each module;
2. source guards -- every site is checked for ``encoding="utf-8"``, and the
   guards are themselves checked against the embedded pre-fix lines (the second
   half of every ``PIN_REVERTS`` pair), so a guard that silently stopped
   matching would fail;
3. forced non-UTF-8 default -- one child runs under ``PYTHONUTF8=0`` (ANSI code
   page) twice: against the working tree (all eight sites must work) and against
   a byte-exact pre-fix shadow produced by reverting each pin at run time (all
   eight must fail).  The legacy augmentation script gets the same before/after.

Non-ASCII text is kept as ``\\uXXXX`` escapes so this file stays pure ASCII and
its own encoding can never be the thing under test.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Em dash, U+00C1, CJK, emoji.  The U+00C1 is what makes the payload undecodable
# on a cp1252-default platform: its UTF-8 second byte 0x81 has no cp1252 mapping,
# so a locale-default reader really does raise on this row (same fixture the
# parent card used, so both suites bite on the same bytes).
NON_ASCII_VALUE = "ok \u2014 \u00c1\u4e2d\u6587 \U0001f600"
PIPELINE_LEVEL = "L0 \u4e2d\u6587"
SHA_ONE = "0" * 63 + "1"
SHA_TWO = "0" * 63 + "2"
PROGRAM = ["0x100073", "0x100074", "0x100075"]
WORDS = [0x100073, 0x100074, 0x100075]

# PYTHONUTF8=0 + no locale coercion -> the ANSI code page, i.e. a platform whose
# DEFAULT encoding is not UTF-8 (cp1252 on this host, ansi_x3.4-1968 on Linux CI).
ANSI_CODE_PAGE_ENV = {"PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LANG": "C", "LC_ALL": "C"}

SITES: dict[str, list[str]] = {
    "milestone1/data/loading.py": ["open(data_path"],
    "evolution/recurrent_mixin.py": ["open(jsonl_path", "fdopen(fd"],
    "milestone1/build_recurrent_dataset.py": ["open(path", "open(legacy_path"],
    "milestone1/evaluation/pipeline.py": ["open(recurrent_data_path"],
    "tests/unit/test_recurrent_data_filter.py": ["open(path"],
    "archive/augmentation_subprocess.py": ["open(data_path"],
}

# (file, the line this card pinned, the byte-exact PRE-FIX line it replaced).
# Used to build the pre-fix shadow and to prove the source guards bite.
PIN_REVERTS: dict[str, list[tuple[str, str]]] = {
    "milestone1/data/loading.py": [
        (
            '    with open(data_path, encoding="utf-8", errors="replace") as f:',
            "    with open(data_path) as f:",
        ),
    ],
    "evolution/recurrent_mixin.py": [
        (
            '                    with open(jsonl_path, encoding="utf-8") as f:',
            "                    with open(jsonl_path) as f:",
        ),
        (
            '            with open(jsonl_path, "a", encoding="utf-8") as f:',
            '            with open(jsonl_path, "a") as f:',
        ),
        # pre-card this one was already encoding-pinned; only errors="replace"
        # was missing, and nothing around it catches UnicodeDecodeError.
        (
            '            with open(jsonl_path, encoding="utf-8", errors="replace") as f:',
            '            with open(jsonl_path, encoding="utf-8") as f:',
        ),
        (
            '                with os.fdopen(fd, "w", encoding="utf-8") as f:',
            '                with os.fdopen(fd, "w") as f:',
        ),
    ],
    "milestone1/build_recurrent_dataset.py": [
        ('    with open(path, "a", encoding="utf-8") as f:', '    with open(path, "a") as f:'),
        (
            '        with open(legacy_path, "a", encoding="utf-8") as f:',
            '        with open(legacy_path, "a") as f:',
        ),
        ('    with open(path, encoding="utf-8", errors="replace") as f:', "    with open(path) as f:"),
        (
            '        with open(path, encoding="utf-8", errors="replace") as f:',
            "        with open(path) as f:",
        ),
    ],
    "milestone1/evaluation/pipeline.py": [
        (
            '        with open(recurrent_data_path, "r", encoding="utf-8", errors="replace") as f:',
            '        with open(recurrent_data_path, "r") as f:',
        ),
    ],
    "tests/unit/test_recurrent_data_filter.py": [
        ('    with open(path, encoding="utf-8") as f:', "    with open(path) as f:"),
    ],
    "archive/augmentation_subprocess.py": [
        ('            with open(data_path, encoding="utf-8") as f:', "            with open(data_path) as f:"),
    ],
}

# Sites whose handler would NOT catch UnicodeDecodeError, so errors="replace"
# is part of their contract (torn row -> skipped line, not an escaped crash).
REQUIRE_ERRORS_REPLACE = [
    '    with open(data_path, encoding="utf-8", errors="replace") as f:',
    '    with open(path, encoding="utf-8", errors="replace") as f:',
    '        with open(path, encoding="utf-8", errors="replace") as f:',
    '        with open(recurrent_data_path, "r", encoding="utf-8", errors="replace") as f:',
    '            with open(jsonl_path, encoding="utf-8", errors="replace") as f:',
]

CHILD_SITE_NAMES = (
    "LOADING",
    "BUILD_WRITE",
    "BUILD_READ",
    "MIXIN_WRITE",
    "MIXIN_NOVELTY",
    "MIXIN_EVICT",
    "PIPELINE",
    "FILTER",
)


class _NoAsciiEscape:
    """json module stand-in that lets non-ASCII through to a writer."""

    loads = staticmethod(json.loads)
    dumps = staticmethod(lambda obj, **kw: json.dumps(obj, ensure_ascii=False, **kw))


def _training_row() -> dict:
    """A row that survives every load_champion_traces() quality filter."""
    return {
        "fitness": 100.0,
        "task_score": 0.0,
        "level": 5,
        "generation": -1,
        "save_reason": "champion_trace",
        "champion_program": list(PROGRAM),
        "champion_sha256": SHA_ONE,
        "trace_status": NON_ASCII_VALUE,
    }


def _write_multi_byte_row(path: Path, obj: dict, *, append: bool = False) -> bytes:
    """Write one real multi-byte UTF-8 row (fixture writer, always pinned)."""
    chunk = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    if append:
        with open(path, "ab") as fh:
            fh.write(chunk)
    else:
        path.write_bytes(chunk)
    return path.read_bytes()


def _assert_multi_byte(raw: bytes, where: str) -> None:
    assert NON_ASCII_VALUE.encode("utf-8") in raw, f"{where}: multi-byte payload not on disk"
    with pytest.raises(UnicodeDecodeError):
        raw.decode("cp1252")


class _Tokenizer:
    """Enough of RV32IFieldTokenizer for load_champion_traces()."""

    def encode_trace(self, instrs: list) -> list[int]:
        assert instrs
        return [1, 2, 3]


def _decode_word(word: int) -> dict:
    return {"word": word}


def _stub_engine() -> SimpleNamespace:
    """Minimal engine for build_recurrent_dataset.collect_champion_snapshots()."""
    champion = SimpleNamespace(program=list(WORDS), generation=3, code_size=3, sha256=SHA_TWO, scores={})
    return SimpleNamespace(
        champion=champion,
        champion_fitness=200.0,
        current_task_level=5,
        champion_trace=[],
        completed_tasks=set(),
        champion_error=None,
    )


def _point_build_recurrent_dataset(tmp_path: Path, monkeypatch):
    """Redirect build_recurrent_dataset's module globals into tmp_path."""
    import milestone1.build_recurrent_dataset as brd

    out = tmp_path / "recurrent_data.jsonl"
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(brd, "RECURRENT_FILE", str(out))
    monkeypatch.setattr(brd, "RECURRENT_DIR", str(legacy_dir))
    monkeypatch.setattr(brd, "json", _NoAsciiEscape)
    return brd, out, legacy_dir


def _mixin_engine():
    from evolution.recurrent_mixin import RecurrentMixin

    class _E(RecurrentMixin):
        pass

    eng = _E()
    eng.champion_level = 5
    eng.current_generation = 7
    eng.generations_stalled = 0
    eng._saved_sha256s = {}
    eng.recurrent_examples_count = 0
    eng._log = lambda *a, **k: None
    return eng


class _PipelineStub:
    """Self for EvaluationManager._evaluate_generalisation (model never trained)."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(eval=lambda: None)

    def evaluate_quality(self, sequences, batch_size: int = 8):
        assert sequences
        return SimpleNamespace(accuracy=0.5)


def _load_filter_helper(path: Path | None = None):
    """Import test_recurrent_data_filter._read_jsonl (optionally a shadow copy)."""
    target = path or (REPO_ROOT / "tests" / "unit" / "test_recurrent_data_filter.py")
    spec = importlib.util.spec_from_file_location("_ee_w1b_fu5_recurrent_data_filter", target)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._read_jsonl


def _site_lines(src_text: str, rel: str) -> list[str]:
    """Every open()/fdopen() of recurrent_data.jsonl the guard is meant to own."""
    needles = SITES[rel]
    hits = [ln for ln in src_text.splitlines() if "open(" in ln and any(n in ln for n in needles)]
    # the mixin's line counter is deliberately binary and must be skipped
    return [ln.strip() for ln in hits if '"rb"' not in ln and "'rb'" not in ln]


def _unpinned_sites(src_text: str, rel: str) -> list[str]:
    """The subset of _site_lines() still sitting on the locale default."""
    return [ln for ln in _site_lines(src_text, rel) if 'encoding="utf-8"' not in ln]


def _write_pre_fix_shadow(dest: Path) -> dict[str, Path]:
    """Copy the six modules with every pin of this card reverted (byte-exact).

    Returns {flat_name: shadow_path}.  Raises if a pinned line is missing, which
    doubles as proof that each pin is exactly where PIN_REVERTS says it is.
    Line-wise matching matters here: the 4-space and 8-space counter lines of
    build_recurrent_dataset are substrings of one another.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for rel, reverts in PIN_REVERTS.items():
        original = (REPO_ROOT / rel).read_text(encoding="utf-8").split("\n")
        lines = list(original)
        for pinned, pre_fix in reverts:
            found = [i for i, ln in enumerate(lines) if ln == pinned]
            assert len(found) == 1, f"{rel}: pinned line not unique (found {found}): {pinned}"
            lines[found[0]] = pre_fix
        assert lines != original, f"{rel}: nothing reverted"
        flat = rel.replace("/", "__")
        target = dest / flat
        target.write_text("\n".join(lines), encoding="utf-8")
        written[flat] = target
    return written


# ---------------------------------------------------------------------------
# 1. milestone1/data/loading.py -- canonical training reader
# ---------------------------------------------------------------------------
def test_loading_reads_a_multi_byte_row(tmp_path: Path):
    """The retrain reader decodes the multi-byte row instead of raising."""
    from paths import ensure_project_on_path

    ensure_project_on_path()
    from data.loading import load_champion_traces

    data_path = tmp_path / "recurrent_data.jsonl"
    _write_multi_byte_row(data_path, _training_row())

    sequences, sample_weights, raw_dicts, entry_levels = load_champion_traces(
        str(data_path), _Tokenizer(), rv32i_decode=_decode_word
    )

    assert len(sequences) == 1, "row filtered out or the read died"
    assert entry_levels == [5]
    assert len(sample_weights) == 1
    assert raw_dicts and raw_dicts[0]
    # the payload really was multi-byte UTF-8, i.e. the pre-fix locale-default
    # reader would have raised on any non-UTF-8 default platform
    _assert_multi_byte(data_path.read_bytes(), "loading.py")


# ---------------------------------------------------------------------------
# 2. milestone1/build_recurrent_dataset.py -- writer + both line counters
# ---------------------------------------------------------------------------
def test_build_recurrent_dataset_writer_and_counters_round_trip(tmp_path: Path, monkeypatch):
    brd, out, legacy_dir = _point_build_recurrent_dataset(tmp_path, monkeypatch)

    # :127 primary append (and :134 legacy append) through the real writer
    written = brd.save_recurrent_data([_training_row()])
    assert written == str(out)

    _assert_multi_byte(out.read_bytes(), "build_recurrent_dataset:127")
    legacy = legacy_dir / "recurrent_data.jsonl"
    _assert_multi_byte(legacy.read_bytes(), "build_recurrent_dataset:134")

    # :158 count-read (no handler at all around it)
    assert brd.check_ready_for_retrain(min_examples=1) is True
    assert brd.check_ready_for_retrain(min_examples=99) is False

    # :185 count-read inside build_and_save (its handler is OSError-only)
    path = brd.build_and_save(_stub_engine())
    assert path == str(out)
    assert NON_ASCII_VALUE.encode("utf-8") in out.read_bytes()
    out.read_bytes().decode("utf-8")  # still a valid UTF-8 file after appends


# ---------------------------------------------------------------------------
# 3. evolution/recurrent_mixin.py -- append writer, novelty reader, eviction
# ---------------------------------------------------------------------------
def test_recurrent_mixin_append_and_novelty_reader_round_trip(tmp_path: Path, monkeypatch):
    import evolution.recurrent_mixin as rmm

    monkeypatch.setattr(rmm, "json", _NoAsciiEscape)
    eng = _mixin_engine()
    path = tmp_path / "recurrent_data.jsonl"

    # :560 live append writer
    saved = eng._write_recurrent_examples(
        [(WORDS, 123.0, {"status": NON_ASCII_VALUE, "instructions_count": 30}, SHA_TWO)],
        [1.0],
        [(5, 0.0)],
        str(path),
    )
    assert saved is True
    _assert_multi_byte(path.read_bytes(), "recurrent_mixin:560")

    # :428 novelty-anchor reader -- the row is read back through the touched call
    weights, applied = eng._apply_novelty_weights([(WORDS, 123.0, {}, SHA_TWO)], [1.0], str(path))
    assert applied is True
    assert weights and weights[0] == pytest.approx(1.0)
    assert eng._novelty_anchors == [WORDS]

    # and through the sha index reader as well
    assert SHA_TWO in eng._load_saved_sha256_index(str(path))


def test_recurrent_mixin_eviction_rewrite_keeps_utf8(tmp_path: Path, monkeypatch):
    """Eviction reads (:685) and republishes (:733) without losing encoding."""
    import evolution.recurrent_mixin as rmm

    monkeypatch.setattr(rmm, "json", _NoAsciiEscape)
    eng = _mixin_engine()
    path = tmp_path / "recurrent_data.jsonl"

    # MAX_RECURRENT_ENTRIES plain rows + one multi-byte high-fitness row
    rows = [
        {"champion_sha256": f"h{n:064d}", "fitness": 100.0, "champion_program": list(PROGRAM)}
        for n in range(rmm.MAX_RECURRENT_ENTRIES)
    ]
    rows.append(
        {
            "champion_sha256": SHA_TWO,
            "fitness": 200.0,
            "champion_program": list(PROGRAM),
            "trace_status": NON_ASCII_VALUE,
        }
    )
    path.write_bytes(b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in rows))
    assert path.read_bytes().count(b"\n") == rmm.MAX_RECURRENT_ENTRIES + 1

    eng._evict_recurrent_buffer(str(path))

    raw = path.read_bytes()
    assert raw.count(b"\n") == rmm.MAX_RECURRENT_ENTRIES, "eviction kept the wrong number of rows"
    # the highest-fitness row survived AND was republished as raw UTF-8 by the
    # os.fdopen(fd, "w") rewrite (the pre-fix site this test guards)
    assert NON_ASCII_VALUE.encode("utf-8") in raw
    kept_rows = [json.loads(ln) for ln in raw.decode("utf-8").splitlines() if ln.strip()]
    assert len(kept_rows) == rmm.MAX_RECURRENT_ENTRIES
    assert any(r.get("trace_status") == NON_ASCII_VALUE for r in kept_rows)


# ---------------------------------------------------------------------------
# 4. milestone1/evaluation/pipeline.py -- recurrent_data_path reader
# ---------------------------------------------------------------------------
def test_pipeline_generalisation_reads_a_multi_byte_row(tmp_path: Path):
    path = tmp_path / "recurrent_data.jsonl"
    _write_multi_byte_row(path, {"level": PIPELINE_LEVEL, "tokens": [1, 2, 3]})

    import milestone1.evaluation.pipeline as pl

    gen, mem, per_level = pl.EvaluationManager._evaluate_generalisation(_PipelineStub(), str(path))

    assert PIPELINE_LEVEL in per_level, "the multi-byte row was not decoded"
    assert per_level[PIPELINE_LEVEL] == pytest.approx(0.5)
    assert gen == pytest.approx(0.5)
    assert mem == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. tests/unit/test_recurrent_data_filter.py -- _read_jsonl helper
# ---------------------------------------------------------------------------
def test_recurrent_data_filter_helper_reads_a_multi_byte_row(tmp_path: Path):
    path = tmp_path / "recurrent_data.jsonl"
    _write_multi_byte_row(path, {"fitness": 100.0, "trace_status": NON_ASCII_VALUE})
    _write_multi_byte_row(path, {"fitness": 90.0, "trace_status": "ascii row"}, append=True)

    rows = _load_filter_helper()(str(path))

    assert len(rows) == 2
    assert rows[0]["trace_status"] == NON_ASCII_VALUE


# ---------------------------------------------------------------------------
# 6. Source-level guards (with a control proving they catch the pre-fix lines)
# ---------------------------------------------------------------------------
def test_every_recurrent_data_open_pins_utf8():
    """No open() of this file in the six modules may stay on the locale default."""
    counts: dict[str, int] = {}
    unpinned: list[str] = []

    for rel, needles in SITES.items():
        src = (REPO_ROOT / rel).read_text(encoding="utf-8")
        hits = _site_lines(src, rel)
        counts[rel] = len(hits)
        assert hits, f"{rel}: expected at least one {needles} site (moved or renamed?)"
        unpinned.extend(f"{rel}: {ln}" for ln in _unpinned_sites(src, rel))

    # build: primary append + legacy append + both counters.  mixin: the two
    # sha-index readers (already pinned), the novelty reader, the append writer,
    # the eviction reader and the eviction fdopen (the rb counter is excluded).
    assert counts["milestone1/build_recurrent_dataset.py"] == 4, counts
    assert counts["evolution/recurrent_mixin.py"] == 6, counts
    assert unpinned == [], "recurrent_data.jsonl opened on the platform default:\n" + "\n".join(unpinned)


def test_the_source_guard_flags_the_pre_fix_lines():
    """Control: the guard above really does flag every line this card changed.

    The pre-fix lines are embedded as the second half of each PIN_REVERTS pair
    (they are this card's own diff), so this fails if the guard's needles drift
    into something that no longer matches -- a green guard can never again mean
    "the pin is gone and nobody noticed".
    """
    flagged: list[str] = []
    total = 0
    for rel, reverts in PIN_REVERTS.items():
        pre_fix_lines = [pre for _pinned, pre in reverts if 'encoding="utf-8"' not in pre]
        # exactly one reverted line was encoding-pinned before this card
        # (recurrent_mixin's eviction reader); every other one was bare
        assert len(pre_fix_lines) == len(reverts) - (1 if rel == "evolution/recurrent_mixin.py" else 0), rel
        hits = _unpinned_sites("\n".join(pre_fix_lines), rel)
        total += len(pre_fix_lines)
        for line in pre_fix_lines:
            # _site_lines()/ _unpinned_sites() report stripped lines
            assert line.strip() in hits, f"{rel}: guard did not flag the pre-fix line {line!r}"
        flagged.extend(hits)

    # 11 of the 12 reverted lines were unpinned before this card; the 12th was
    # encoding-pinned but had no errors="replace", which
    # test_uncaught_readers_carry_errors_replace covers.
    assert len(flagged) == total == 11, (len(flagged), total)


def test_uncaught_readers_carry_errors_replace():
    """The five uncaught pure-read sites tolerate torn bytes, not just decoding."""
    all_lines: set[str] = set()
    for rel in SITES:
        all_lines.update((REPO_ROOT / rel).read_text(encoding="utf-8").splitlines())
    missing = [expected for expected in REQUIRE_ERRORS_REPLACE if expected not in all_lines]
    assert missing == [], f"expected pinned lines not found (formatting drift?): {missing}"


# ---------------------------------------------------------------------------
# 7. Forced non-UTF-8 default: all six modules, before and after
# ---------------------------------------------------------------------------
CHILD = r'''"""Drive every recurrent_data.jsonl reader/writer outside run_multi_cycle.py
on a platform whose DEFAULT encoding is NOT UTF-8 (PYTHONUTF8=0 -> ANSI CP).

argv: repo_root, tmp_root, non_ascii_value, module_source

module_source is "live" (the working tree) or a directory holding a byte-exact
pre-fix shadow of the same modules, so one child measures before AND after.
Each site prints SITE_<name> ok | SITE_<name> FAIL <Type>: <message>.
"""
import importlib
import importlib.util
import json as _json
import locale
import os
import sys
from types import SimpleNamespace

repo, root, value, module_source = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# Incidental prints (e.g. the pipeline echoing a non-ASCII level key) must not
# kill the child on a cp1252 stdout; the encoding under test is open()'s.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, repo)
from paths import ensure_project_on_path

ensure_project_on_path()

print("PREFERRED", locale.getpreferredencoding(False))
print("SOURCE", "live" if module_source == "live" else "pre-fix shadow")


class _NoAsciiEscape:
    loads = staticmethod(_json.loads)
    dumps = staticmethod(lambda obj, **kw: _json.dumps(obj, ensure_ascii=False, **kw))


PRE_NAMES = {
    "loading": "milestone1__data__loading.py",
    "build": "milestone1__build_recurrent_dataset.py",
    "mixin": "evolution__recurrent_mixin.py",
    "pipeline": "milestone1__evaluation__pipeline.py",
    "filter": "tests__unit__test_recurrent_data_filter.py",
}


def _load(name, live_path):
    if module_source == "live":
        return importlib.import_module(live_path)
    spec = importlib.util.spec_from_file_location(
        "shadow_" + name, os.path.join(module_source, PRE_NAMES[name])
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mod_loading = _load("loading", "data.loading")
mod_build = _load("build", "milestone1.build_recurrent_dataset")
mod_mixin = _load("mixin", "evolution.recurrent_mixin")
mod_pipeline = _load("pipeline", "milestone1.evaluation.pipeline")


def _filter_helper():
    if module_source == "live":
        path = os.path.join(repo, "tests", "unit", "test_recurrent_data_filter.py")
    else:
        path = os.path.join(module_source, PRE_NAMES["filter"])
    spec = importlib.util.spec_from_file_location("_ee_w1b_fu5_filter", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod._read_jsonl


def _utf8_write(path, obj):
    """Fixture writer: always real multi-byte UTF-8 on disk."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_json.dumps(obj, ensure_ascii=False) + "\n")


def _control(name, fn):
    """Run the exact pre-fix (locale default) expression and report what it does."""
    try:
        fn()
    except UnicodeDecodeError:
        out = "UnicodeDecodeError"
    except UnicodeEncodeError:
        out = "UnicodeEncodeError"
    else:
        out = "no-raise"
    print("CONTROL_" + name, out)


os.makedirs(root, exist_ok=True)
TRAIN = os.path.join(root, "recurrent_data.jsonl")
row = {
    "fitness": 100.0,
    "task_score": 0.0,
    "level": 5,
    "generation": -1,
    "save_reason": "champion_trace",
    "champion_program": ["0x100073", "0x100074", "0x100075"],
    "champion_sha256": "0" * 63 + "1",
    "trace_status": value,
}
_utf8_write(TRAIN, row)
with open(TRAIN, "rb") as fh:
    raw = fh.read()
print("MULTIBYTE", "ok" if value.encode("utf-8") in raw else "missing")


# --- controls: the exact pre-fix expressions -------------------------------
def _ctl_read_loading():          # pre-image of loading.py:56
    with open(TRAIN) as fh:
        for _ in fh:
            pass


def _ctl_read_build():            # pre-image of build_recurrent_dataset.py:158/:185
    with open(TRAIN) as fh:
        sum(1 for _ in fh)


def _ctl_read_mixin():            # pre-image of recurrent_mixin.py:428
    with open(TRAIN) as fh:
        for _ in fh:
            pass


def _ctl_read_pipeline():         # pre-image of evaluation/pipeline.py:570
    with open(TRAIN, "r") as fh:
        for _ in fh:
            pass


def _ctl_read_filter():           # pre-image of test_recurrent_data_filter.py:37
    with open(TRAIN) as fh:
        [_json.loads(ln) for ln in fh if ln.strip()]


def _ctl_write_build():           # pre-image of build_recurrent_dataset.py:127
    with open(TRAIN, "a") as fh:
        fh.write(_json.dumps(row, ensure_ascii=False) + "\n")


def _ctl_write_mixin():           # pre-image of recurrent_mixin.py:560
    with open(TRAIN, "a") as fh:
        fh.write(_json.dumps(row, ensure_ascii=False) + "\n")


for _name, _fn in (
    ("LOADING", _ctl_read_loading),
    ("BUILD_READ", _ctl_read_build),
    ("MIXIN_READ", _ctl_read_mixin),
    ("PIPELINE_READ", _ctl_read_pipeline),
    ("FILTER_READ", _ctl_read_filter),
    ("BUILD_WRITE", _ctl_write_build),
    ("MIXIN_WRITE", _ctl_write_mixin),
):
    _control(_name, _fn)


# --- the eight live sites --------------------------------------------------
class _Tok:
    def encode_trace(self, instrs):
        assert instrs
        return [1, 2, 3]


class _PipeStub:
    def __init__(self):
        self.model = SimpleNamespace(eval=lambda: None)

    def evaluate_quality(self, sequences, batch_size=8):
        assert sequences
        return SimpleNamespace(accuracy=0.5)


def _setup_build():
    mod_build.RECURRENT_FILE = TRAIN
    mod_build.RECURRENT_DIR = os.path.join(root, "legacy")
    os.makedirs(mod_build.RECURRENT_DIR, exist_ok=True)
    mod_build.json = _NoAsciiEscape
    return mod_build


def _setup_mixin():
    class _E(mod_mixin.RecurrentMixin):
        pass

    eng = _E()
    eng.champion_level = 5
    eng.current_generation = 7
    eng.generations_stalled = 0
    eng._saved_sha256s = {}
    eng.recurrent_examples_count = 0
    eng._log = lambda *a, **k: None
    mod_mixin.json = _NoAsciiEscape
    return eng


def site_loading():
    seqs, _w, _r, _l = mod_loading.load_champion_traces(
        TRAIN, _Tok(), rv32i_decode=lambda x: {"word": x}
    )
    if len(seqs) != 1:
        raise AssertionError("expected 1 sequence, got %d" % len(seqs))


def site_build_write():
    brd = _setup_build()
    written = brd.save_recurrent_data([row])
    with open(written, "rb") as fh:
        wraw = fh.read()
    if value.encode("utf-8") not in wraw:
        raise AssertionError("primary append is not UTF-8")
    legacy = os.path.join(brd.RECURRENT_DIR, "recurrent_data.jsonl")
    with open(legacy, "rb") as fh:
        lraw = fh.read()
    if value.encode("utf-8") not in lraw:
        raise AssertionError("legacy append is not UTF-8")
    champ = SimpleNamespace(
        program=[0x100073, 0x100074, 0x100075],
        generation=3,
        code_size=3,
        sha256="0" * 63 + "2",
        scores={},
    )
    eng = SimpleNamespace(
        champion=champ,
        champion_fitness=200.0,
        current_task_level=5,
        champion_trace=[],
        completed_tasks=set(),
        champion_error=None,
    )
    if brd.build_and_save(eng) != written:
        raise AssertionError("build_and_save returned a different path")


def site_build_read():
    brd = _setup_build()
    if not brd.check_ready_for_retrain(min_examples=1):
        raise AssertionError("no examples counted")


def site_mixin_write():
    eng = _setup_mixin()
    target = os.path.join(root, "mixin_write.jsonl")
    saved = eng._write_recurrent_examples(
        [
            (
                [0x100073, 0x100074, 0x100075],
                123.0,
                {"status": value, "instructions_count": 30},
                "0" * 63 + "2",
            )
        ],
        [1.0],
        [(5, 0.0)],
        target,
    )
    with open(target, "rb") as fh:
        mraw = fh.read()
    if not saved or value.encode("utf-8") not in mraw:
        raise AssertionError("append did not land as UTF-8")


def site_mixin_novelty():
    eng = _setup_mixin()
    target = os.path.join(root, "mixin_novelty.jsonl")
    _utf8_write(target, row)
    _weights, applied = eng._apply_novelty_weights(
        [([0x100073, 0x100074, 0x100075], 123.0, {}, "0" * 63 + "2")], [1.0], target
    )
    if not applied:
        raise AssertionError("novelty anchors not loaded (reader swallowed the decode error)")


def site_mixin_evict():
    limit = mod_mixin.MAX_RECURRENT_ENTRIES
    eng = _setup_mixin()
    target = os.path.join(root, "mixin_evict.jsonl")
    plain = [
        {"champion_sha256": "h%064d" % n, "fitness": 100.0, "champion_program": ["0x100073"] * 3}
        for n in range(limit)
    ]
    plain.append(
        {
            "champion_sha256": "0" * 63 + "9",
            "fitness": 200.0,
            "champion_program": ["0x100073"] * 3,
            "trace_status": value,
        }
    )
    with open(target, "w", encoding="utf-8") as fh:
        for r in plain:
            fh.write(_json.dumps(r, ensure_ascii=False) + "\n")
    eng._evict_recurrent_buffer(target)
    with open(target, "rb") as fh:
        eraw = fh.read()
    if eraw.count(b"\n") != limit:
        raise AssertionError("kept %d rows, expected %d" % (eraw.count(b"\n"), limit))
    if value.encode("utf-8") not in eraw:
        raise AssertionError("eviction rewrite dropped the UTF-8 payload")


def site_pipeline():
    target = os.path.join(root, "pipeline_recurrent_data.jsonl")
    _utf8_write(target, {"level": "L0 \u4e2d\u6587", "tokens": [1, 2, 3]})
    _gen, _mem, per_level = mod_pipeline.EvaluationManager._evaluate_generalisation(
        _PipeStub(), target
    )
    if "L0 \u4e2d\u6587" not in per_level:
        raise AssertionError("multi-byte level key not decoded")


def site_filter():
    target = os.path.join(root, "filter_recurrent_data.jsonl")
    _utf8_write(target, {"fitness": 100.0, "trace_status": value})
    rows = _filter_helper()(target)
    if not rows or rows[0].get("trace_status") != value:
        raise AssertionError("helper did not read the multi-byte value back")


for _name, _fn in (
    ("LOADING", site_loading),
    ("BUILD_WRITE", site_build_write),
    ("BUILD_READ", site_build_read),
    ("MIXIN_WRITE", site_mixin_write),
    ("MIXIN_NOVELTY", site_mixin_novelty),
    ("MIXIN_EVICT", site_mixin_evict),
    ("PIPELINE", site_pipeline),
    ("FILTER", site_filter),
):
    try:
        _fn()
    except BaseException as exc:  # noqa: BLE001 - report, never crash the sweep
        msg = str(exc).encode("ascii", "backslashreplace").decode("ascii")[:120]
        print("SITE_" + _name, "FAIL", type(exc).__name__ + ": " + msg)
    else:
        print("SITE_" + _name, "ok")

print("OK")
'''

PROBE_CHILD = "import locale\nprint('PREFERRED', locale.getpreferredencoding(False))\n"


def _run_child(tmp_path: Path, source: str, args: list[str], env_overrides: dict | None = None):
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


def _skip_if_utf8_default(preferred: str) -> None:
    """The hazard only exists where the default encoding is not UTF-8."""
    if preferred.lower().replace("-", "").replace("_", "") == "utf8":
        pytest.skip(f"platform default encoding is already {preferred}; hazard not reproducible here")


def _ansi_env() -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(ANSI_CODE_PAGE_ENV)
    return env


def _run_legacy_script(script: Path, data_path: Path) -> subprocess.CompletedProcess:
    cfg = json.dumps({"seed": 1, "max_sequences": 0, "data_path": str(data_path)})
    return subprocess.run(
        [sys.executable, str(script)],
        input=cfg,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        env=_ansi_env(),
        timeout=300,
    )


def test_pinned_modules_work_under_a_platform_default_encoding(tmp_path: Path):
    """All six modules work when the platform default encoding is not UTF-8.

    Runs under ``PYTHONUTF8=0`` so ``open()`` without ``encoding=`` uses the
    ANSI code page -- the exact configuration in which the pre-fix expressions
    raise.  This half is host-independent: a child that dies (import error, API
    drift, an uncaught decode/encode error) fails on every host, never skips.
    """
    proc = _run_child(
        tmp_path,
        CHILD,
        [str(REPO_ROOT), str(tmp_path / "live"), NON_ASCII_VALUE, "live"],
        ANSI_CODE_PAGE_ENV,
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _child_line(proc, "MULTIBYTE") == "ok"
    for name in CHILD_SITE_NAMES:
        assert _child_line(proc, f"SITE_{name}") == "ok", name
    assert proc.stdout.splitlines()[-1] == "OK"

    # CONTROL_* lines are the exact pre-fix expressions; they only hold where the
    # default encoding really is not UTF-8, so they may skip.
    _skip_if_utf8_default(_child_line(proc, "PREFERRED"))
    for name in ("LOADING", "BUILD_READ", "MIXIN_READ", "PIPELINE_READ", "FILTER_READ"):
        assert _child_line(proc, f"CONTROL_{name}") == "UnicodeDecodeError", name
    for name in ("BUILD_WRITE", "MIXIN_WRITE"):
        assert _child_line(proc, f"CONTROL_{name}") == "UnicodeEncodeError", name


def test_pre_fix_shadow_fails_every_site_under_a_platform_default_encoding(tmp_path: Path):
    """Before/after in one test: the same eight sites on the pre-fix modules.

    The shadow is produced by reverting each pin of this card into a copy of the
    working-tree file, so it is the byte-exact pre-fix state (sibling WIP
    intact) rather than an approximation.
    """
    shadow_dir = tmp_path / "pre_fix_shadow"
    written = _write_pre_fix_shadow(shadow_dir)
    assert len(written) == len(SITES)

    proc = _run_child(
        tmp_path,
        CHILD,
        [str(REPO_ROOT), str(tmp_path / "pre"), NON_ASCII_VALUE, str(shadow_dir)],
        ANSI_CODE_PAGE_ENV,
    )

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert _child_line(proc, "SOURCE") == "pre-fix shadow"
    _skip_if_utf8_default(_child_line(proc, "PREFERRED"))

    failures: list[str] = []
    for name in CHILD_SITE_NAMES:
        outcome = _child_line(proc, f"SITE_{name}")
        if not outcome.startswith("FAIL"):
            failures.append(f"{name} survived the pre-fix shadow: {outcome}")
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# 8. archive/augmentation_subprocess.py -- before/after on the legacy script
# ---------------------------------------------------------------------------
def test_augmentation_subprocess_reads_a_multi_byte_row(tmp_path: Path):
    """Under a non-UTF-8 default the legacy reader must not drop the row.

    Pre-fix the decode error is swallowed by the script's bare ``except``
    (``... could not load ...`` on stderr, ``count: 0``), so a lost row -- not a
    crash -- is the failure signal here.
    """
    data_path = tmp_path / "recurrent_data.jsonl"
    _write_multi_byte_row(data_path, {"tokens": [1, 2, 3], "weight": 1.0, "note": NON_ASCII_VALUE})

    proc = _run_legacy_script(REPO_ROOT / "archive" / "augmentation_subprocess.py", data_path)

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "could not load" not in proc.stderr, f"decode failure on the pinned reader: {proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["count"] >= 1
    assert payload["sequences"][0]["tokens"] == [1, 2, 3]


def test_augmentation_subprocess_pre_fix_shadow_drops_the_row(tmp_path: Path):
    """Before/after for the legacy script: the pre-fix copy loses the row."""
    shadow = _write_pre_fix_shadow(tmp_path / "pre_fix_shadow")
    data_path = tmp_path / "recurrent_data.jsonl"
    _write_multi_byte_row(data_path, {"tokens": [1, 2, 3], "weight": 1.0, "note": NON_ASCII_VALUE})

    probe = _run_child(tmp_path, PROBE_CHILD, [], ANSI_CODE_PAGE_ENV)
    _skip_if_utf8_default(_child_line(probe, "PREFERRED"))

    proc = _run_legacy_script(shadow["archive__augmentation_subprocess.py"], data_path)

    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "could not load" in proc.stderr, f"pre-fix script did not lose the row: {proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload["count"] == 0, payload
