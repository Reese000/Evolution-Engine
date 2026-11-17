"""Durability of the SHARED ``.agents/progress_controller_state.json``.

Two writers persist that one file:

* ``ProgressController.save()`` — curriculum keys (level, cycles_at_level,
  verify_score, raw_fitness, generations_stalled);
* ``terminal_stagnation._patch_progress_state()`` — terminal-efficiency keys
  (``terminal_efficiency_scan_floor_*``, ``terminal_efficiency_certified_words``,
  ``terminal_efficiency_stable_*``, ``terminal_efficiency_promote_cycle``).

``save()`` used to truncate-write in place while ``_patch_progress_state()``
used tmp + ``os.replace``, so an interleave could tear the file for readers or
silently drop the keys the other writer had just merged — and a torn read was
then returned as ``{}``, which re-arms the expensive per-cycle compression
sweeps and quiescence skips. Both writers now share
``evolution.progress_state_io`` (tmp + fsync + ``os.replace``, bounded retry),
and the read side distinguishes "file missing" from "present but unparseable".
"""

import errno
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from evolution import progress_state_io
from evolution import terminal_stagnation as ts
from evolution.progress_controller import ProgressController

TORN = '{"cycles_at_level": 7, "terminal_efficiency_scan_floor_wo'


@pytest.fixture()
def agents(tmp_path, monkeypatch):
    path = tmp_path / ".agents"
    path.mkdir()
    monkeypatch.setattr(ts, "_project_agents_dir", lambda: str(path))
    ts._invalidate_progress_state_cache()
    yield path
    ts._invalidate_progress_state_cache()


@pytest.fixture()
def state_path(agents):
    return agents / "progress_controller_state.json"


def _warnings(caplog):
    return [r for r in caplog.records if r.levelname == "WARNING"]


def _has_unreadable_warning(caplog) -> bool:
    return any("progress-controller state unreadable" in r.getMessage() for r in _warnings(caplog))


def _bump_mtime(path) -> None:
    """Force a distinct mtime so the reader's mtime cache cannot mask a tear."""
    st = os.stat(path)
    os.utime(path, (st.st_atime + 5.0, st.st_mtime + 5.0))


# --- read side: missing vs torn ------------------------------------------------


def test_read_state_distinguishes_missing_from_unparseable(tmp_path):
    path = str(tmp_path / "progress_controller_state.json")

    # (a) file missing is NOT an error — fresh install.
    assert progress_state_io.read_state(path) == ({}, None)

    # (b) present but torn/partial -> error, not a silent {}.
    with open(path, "w", encoding="utf-8") as f:
        f.write(TORN)
    state, err = progress_state_io.read_state(path)
    assert state == {}
    assert isinstance(err, (json.JSONDecodeError, OSError, ValueError))

    # (c) present, valid JSON, wrong shape -> error too.
    with open(path, "w", encoding="utf-8") as f:
        f.write("[1, 2, 3]")
    state, err = progress_state_io.read_state(path)
    assert state == {}
    assert err is not None

    # (d) present and valid -> data, no error.
    with open(path, "w", encoding="utf-8") as f:
        f.write("{}")
    assert progress_state_io.read_state(path) == ({}, None)


def test_torn_state_is_logged_and_keeps_last_good_values(agents, state_path, caplog):
    good = {"cycles_at_level": 7, "terminal_efficiency_scan_floor_words": 48}
    state_path.write_text(json.dumps(good), encoding="utf-8")

    with caplog.at_level("WARNING", logger="EvolutionEngine"):
        first = ts._read_progress_state()
        assert first["terminal_efficiency_scan_floor_words"] == 48

        # A truncate-write interleave (what save() used to do) leaves this.
        state_path.write_text(TORN, encoding="utf-8")
        _bump_mtime(state_path)
        torn = ts._read_progress_state()

    assert _has_unreadable_warning(caplog), "a torn state must be logged"
    # "does not silently become {}" — the last known-good parse survives, so
    # the terminal-efficiency floor keys are not dropped mid-write.
    assert torn == good
    assert "terminal_efficiency_scan_floor_words" in torn


def test_torn_state_without_cache_is_logged_not_silent(agents, state_path, caplog):
    state_path.write_text(TORN, encoding="utf-8")
    ts._invalidate_progress_state_cache()

    with caplog.at_level("WARNING", logger="EvolutionEngine"):
        result = ts._read_progress_state()

    assert result == {}  # nothing good has ever been read
    assert _has_unreadable_warning(caplog), "cold-start tear must still be logged"


def test_missing_state_file_is_silently_empty(agents, caplog):
    with caplog.at_level("WARNING", logger="EvolutionEngine"):
        assert ts._read_progress_state() == {}
    assert not _warnings(caplog), "an absent file is normal, not a failure"


def test_progress_controller_load_logs_torn_state(tmp_path, caplog):
    agents = tmp_path / ".agents"
    agents.mkdir()
    (agents / "progress_controller_state.json").write_text('{"level": 2', encoding="utf-8")

    with caplog.at_level("WARNING", logger="EvolutionEngine"):
        pc = ProgressController(str(tmp_path), stall_threshold=2)

    assert pc.state.level == 0  # keeps defaults rather than crashing
    assert _has_unreadable_warning(caplog)


def test_torn_state_warning_is_reported_once_per_bad_file(agents, state_path, caplog):
    state_path.write_text(json.dumps({"cycles_at_level": 3}), encoding="utf-8")
    with caplog.at_level("WARNING", logger="EvolutionEngine"):
        assert ts._read_progress_state()["cycles_at_level"] == 3
        state_path.write_text(TORN, encoding="utf-8")
        _bump_mtime(state_path)
        ts._read_progress_state()
        before = len(_warnings(caplog))
        ts._read_progress_state()  # same torn file again (cache hit)
        ts._read_progress_state()
    assert before >= 1
    assert len(_warnings(caplog)) == before, "read path runs every cycle: no spam"


# --- write side: one helper, atomic for both writers ----------------------------


def test_write_state_tmp_file_is_writer_unique(tmp_path, monkeypatch):
    path = str(tmp_path / "progress_controller_state.json")
    captured = {}
    real_replace = os.replace

    def spy(src, dst):
        captured["src"] = str(src)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    assert progress_state_io.write_state(path, {"a": 1}) is True
    # pid + thread id: concurrent writers must never share a tmp file, or one
    # writer's os.replace can move the other's half-written payload into place.
    assert captured["src"].endswith(f".{os.getpid()}.{threading.get_ident()}.tmp"), captured["src"]
    with open(path, encoding="utf-8") as f:
        assert json.loads(f.read()) == {"a": 1}


def test_progress_controller_save_retries_transient_replace_failures(tmp_path, monkeypatch, caplog):
    agents = tmp_path / ".agents"
    agents.mkdir()
    real_replace = os.replace
    state = {"calls": 0, "remaining": 2}

    def flaky(src, dst):
        if str(dst).endswith("progress_controller_state.json"):
            state["calls"] += 1
            if state["remaining"] > 0:
                state["remaining"] -= 1
                raise PermissionError("simulated transient write failure")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 21
    pc.state.cycles_at_level = 4
    with caplog.at_level("WARNING", logger="EvolutionEngine"):
        pc.save()

    assert state["calls"] == 3, "save() must use the same bounded 3-attempt retry"
    assert sleeps == [pytest.approx(0.05), pytest.approx(0.10)]
    data = json.loads((agents / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert data["level"] == 21
    assert data["cycles_at_level"] == 4
    assert list(agents.glob("*.tmp")) == [], "no stray tmp file left behind"


# --- the two writers round-tripping the same file -------------------------------


def test_save_and_patch_round_trip_preserves_both_writers_keys(tmp_path, agents, state_path):
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 21
    pc.state.cycles_at_level = 5
    pc.save()

    # terminal_stagnation writer merges its keys on top.
    ts.record_terminal_efficiency_scan_floor(6543, 48, 1.0)
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["level"] == 21
    assert data["terminal_efficiency_scan_floor_words"] == 48
    assert data["terminal_efficiency_scan_floor_cycle"] == 6543

    # controller writer again — must not clobber what the patcher merged.
    pc.state.cycles_at_level = 6
    pc.save()
    data = json.loads(state_path.read_text(encoding="utf-8"))
    assert data["level"] == 21
    assert data["cycles_at_level"] == 6
    assert data["terminal_efficiency_scan_floor_words"] == 48

    # and each reader sees the other writer's keys through its own API.
    assert ts._read_progress_state()["level"] == 21
    fresh = ProgressController(str(tmp_path), stall_threshold=2)
    assert fresh.state.level == 21
    assert fresh.state.cycles_at_level == 6


def test_concurrent_save_and_patch_never_tear_the_state_file(tmp_path, agents, state_path):
    """Both writers hammering the file concurrently: readers never see a torn
    (unparseable) document — the failure mode the truncate-write allowed."""
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 21
    pc.save()

    iterations = 40
    torn_reads: list[str] = []
    thread_errors: list[str] = []
    done = threading.Event()

    def write_save():
        try:
            for i in range(iterations):
                pc.state.cycles_at_level = i
                pc.save()
        except Exception as exc:  # noqa: BLE001 - failure surfaces in the assert below
            thread_errors.append(f"save: {exc!r}")
        finally:
            done.set()

    def write_patch():
        try:
            for i in range(iterations):
                ts._patch_progress_state({"terminal_efficiency_scan_floor_words": 48, "scan_patch_iter": i})
        except Exception as exc:  # noqa: BLE001 - failure surfaces in the assert below
            thread_errors.append(f"patch: {exc!r}")

    def read_loop():
        # Production read path: JSONDecodeError here would mean a writer tore
        # the file. Sharing-violation OSErrors (Windows replace vs open) are
        # transient and are NOT tears.
        while not done.is_set():
            _state, err = progress_state_io.read_state(str(state_path))
            if isinstance(err, json.JSONDecodeError):
                torn_reads.append(str(err))
            time.sleep(0.001)

    threads = [
        threading.Thread(target=write_save),
        threading.Thread(target=write_patch),
        threading.Thread(target=read_loop),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not thread_errors, thread_errors
    assert not torn_reads, f"readers saw torn JSON: {torn_reads[:3]}"
    # Final file is a complete document holding BOTH writers' key families:
    # STATE_LOCK serialises read-merge-write, so save()'s merge can never be
    # the stale one that clobbers the patcher's keys.
    final = json.loads(state_path.read_text(encoding="utf-8"))
    assert isinstance(final, dict)
    assert "level" in final
    assert "terminal_efficiency_scan_floor_words" in final, "lost update under concurrency"
    assert list(agents.glob("*.tmp")) == [], "tmp files must not leak"

    # Serialized round-trip afterwards: both writers' keys survive.
    pc.state.cycles_at_level = 99
    pc.save()
    ts._patch_progress_state({"terminal_efficiency_scan_floor_words": 48})
    final = json.loads(state_path.read_text(encoding="utf-8"))
    assert final["level"] == 21
    assert final["cycles_at_level"] == 99
    assert final["terminal_efficiency_scan_floor_words"] == 48


def test_interleaved_save_cannot_clobber_a_concurrent_patch(tmp_path, agents, state_path, monkeypatch):
    """Deterministic lost-update scenario (the reviewer's probe).

    Interleave: ``ProgressController.save()`` reads the file, then
    ``_patch_progress_state()`` commits the terminal-efficiency keys, then
    ``save()`` writes its (now stale) merge. Without
    ``progress_state_io.STATE_LOCK`` the patcher's keys are dropped; with it
    the patcher blocks until save()'s write lands and then merges on top.
    """
    pc = ProgressController(str(tmp_path), stall_threshold=2)
    pc.state.level = 21
    pc.save()

    paused = threading.Event()
    allow_write = threading.Event()
    flags = {"paused": False}
    real_read = progress_state_io.read_state

    def pausing_read(path):
        # Pause AFTER the snapshot: that is the stale merge that would clobber
        # a commit made in the window (the reviewer's probe).
        result = real_read(path)
        if not flags["paused"]:
            flags["paused"] = True
            paused.set()
            assert allow_write.wait(timeout=30), "harness never released save()"
        return result

    monkeypatch.setattr(progress_state_io, "read_state", pausing_read)

    errors: list[str] = []

    def do_save():
        try:
            pc.save()
        except Exception as exc:  # noqa: BLE001 - reported through `errors`
            errors.append(repr(exc))

    saver = threading.Thread(target=do_save)
    saver.start()
    assert paused.wait(timeout=30), "save() never reached its read"

    patcher = threading.Thread(
        target=ts._patch_progress_state,
        args=({"terminal_efficiency_scan_floor_words": 48},),
    )
    patcher.start()
    # Give an unlocked implementation every chance to commit in the window
    # (with STATE_LOCK the patcher is parked on the lock, which is the point).
    time.sleep(0.3)
    allow_write.set()
    saver.join(timeout=60)
    patcher.join(timeout=60)
    assert not saver.is_alive() and not patcher.is_alive()
    assert not errors, errors

    final = json.loads(state_path.read_text(encoding="utf-8"))
    assert final["level"] == 21, "save()'s own keys must survive"
    assert (
        final["terminal_efficiency_scan_floor_words"] == 48
    ), "save()'s stale merge clobbered the keys the patcher committed mid-flight"


# --- cross-process: STATE_LOCK only serialises threads of ONE process ----------

#: A whole writer process, run twice with different keys. Both children wait
#: for each other's start marker, then sleep inside their patched
#: ``read_state``, so BOTH processes snapshot the file BEFORE either merges —
#: the exact interleave that loses a key when nothing serialises the
#: read-modify-write across processes.
CROSS_PROCESS_CHILD = '''
"""One spawned writer process for the cross-process lock tests.

argv: state_path key value mode marker_dir idx repo_root

mode "lock"    -> progress_state_io exactly as shipped (advisory lock taken)
mode "no_lock" -> cross_process_lock neutralised = pre-fix behaviour
"""
import contextlib
import json
import os
import sys
import time

state_path, key, value, mode, marker_dir, idx, repo_root = sys.argv[1:8]
sys.path.insert(0, repo_root)

from evolution import progress_state_io as pio

if mode == "no_lock":

    @contextlib.contextmanager
    def _without_lock(*_args, **_kwargs):
        yield False

    pio.cross_process_lock = _without_lock

_real_read = pio.read_state

other = "0" if idx == "1" else "1"
marker_dir = os.path.abspath(marker_dir)

if mode == "no_lock":

    def _snapshot_then_pause(path):
        snapshot = _real_read(path)
        # Structural interleave for the control run: this writer may not merge
        # until its peer has snapshotted too, so BOTH reads provably precede
        # BOTH writes — no reliance on scheduling luck. This barrier is wrong
        # in lock mode: the peer would be parked on the lock and could never
        # write its marker.
        with open(os.path.join(marker_dir, "read_" + idx + ".done"), "w", encoding="utf-8") as fh:
            fh.write("1")
        peer_read = os.path.join(marker_dir, "read_" + other + ".done")
        deadline = time.time() + 30.0
        while not os.path.exists(peer_read) and time.time() < deadline:
            time.sleep(0.01)
        return snapshot

else:

    def _snapshot_then_pause(path):
        # No barrier here: with the lock held by one writer the other cannot
        # read at all until it is released, which is exactly the behaviour
        # under test. The short pause just widens the attempted interleave.
        snapshot = _real_read(path)
        time.sleep(0.1)
        return snapshot


pio.read_state = _snapshot_then_pause

with open(os.path.join(marker_dir, "start_" + idx + ".done"), "w", encoding="utf-8") as fh:
    fh.write("1")

peer = os.path.join(marker_dir, "start_" + other + ".done")
deadline = time.time() + 30.0
while not os.path.exists(peer) and time.time() < deadline:
    time.sleep(0.01)

written, err = pio.update_state(state_path, {key: value})
print(json.dumps({"written": bool(written), "read_error": str(err)}))
'''


def _run_two_writer_processes(state_path: str, mode: str, workdir) -> list[dict]:
    """Spawn two real OS processes, each doing one read → merge → write.

    Returns each child's reported ``{"written", "read_error"}``.
    """
    marker_dir = os.path.join(str(workdir), "markers")
    os.makedirs(marker_dir, exist_ok=True)
    script = os.path.join(str(workdir), "cross_process_writer.py")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(CROSS_PROCESS_CHILD)
    # tests/unit/test_*.py -> tests/unit -> tests -> repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # Mirror the acceptance run's `env -u PYTHONPATH`: a stray PYTHONPATH
    # pointing at a foreign site-packages must not shadow the project package.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

    procs = []
    for idx, key, value in (("0", "proc_a", "1"), ("1", "proc_b", "2")):
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    script,
                    state_path,
                    key,
                    value,
                    mode,
                    marker_dir,
                    idx,
                    repo_root,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(workdir),
                env=env,
            )
        )

    reports: list[dict] = []
    try:
        for proc in procs:
            out, err = proc.communicate(timeout=120)
            assert proc.returncode == 0, f"writer process failed (rc={proc.returncode}): {err[-4000:]}"
            reports.append(json.loads(out.strip().splitlines()[-1]))
    finally:
        for proc in procs:  # only the PIDs this test spawned
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=30)
    return reports


def _hold_lock_file(lock_file: str):
    """Take the lock the way a *foreign process* would (independent handle)."""
    handle = open(lock_file, "a+b")  # noqa: SIM115 - released by _release_lock_file
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def _release_lock_file(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _evolution_warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records if r.levelname == "WARNING" and r.name == progress_state_io.LOGGER_NAME
    ]


def test_two_processes_keep_both_writers_keys(tmp_path):
    """Two *processes* interleaving read → merge → write both survive.

    This is the residual gap t_66bf17f1 left open: STATE_LOCK serialises the
    two in-process writers only, so a second process could still snapshot the
    file, let the first commit, and clobber that commit with its stale merge.
    """
    state_path = str(tmp_path / "progress_controller_state.json")

    reports = _run_two_writer_processes(state_path, "lock", tmp_path)

    assert [r["written"] for r in reports] == [True, True], reports
    assert [r["read_error"] for r in reports] == ["None", "None"], reports
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert final.get("proc_a") == "1", f"process A's key was lost: {final}"
    assert final.get("proc_b") == "2", f"process B's key was lost: {final}"
    # The lock file is the sibling of the NORMALISED state path: two spellings
    # of one file must resolve to one lock (see lock_path).
    assert progress_state_io.lock_path(state_path) == (os.path.normcase(os.path.realpath(state_path)) + ".lock")
    assert os.path.isfile(progress_state_io.lock_path(state_path)), "sibling lock file"
    assert list(tmp_path.glob("*.tmp")) == [], "tmp files must not leak"


def test_identical_interleave_without_the_lock_loses_a_key(tmp_path):
    """Non-vacuity for the test above: neutralise cross_process_lock (i.e. the
    pre-fix, last-writer-wins behaviour) and the very same two processes drop
    one writer's keys — so preservation in the test above is the lock's work,
    not the harness's luck."""
    state_path = str(tmp_path / "progress_controller_state.json")

    reports = _run_two_writer_processes(state_path, "no_lock", tmp_path)
    # Both children must have really written: a silently dropped write would
    # leave one key too, without any interleaving having been demonstrated.
    assert [r["written"] for r in reports] == [True, True], reports
    assert [r["read_error"] for r in reports] == ["None", "None"], reports

    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    survivors = {k for k in ("proc_a", "proc_b") if k in final}
    assert len(survivors) == 1, f"without the cross-process lock exactly one writer must win, got {final}"


def test_lock_path_is_one_file_per_state_file(tmp_path):
    """One state file -> one lock: a different spelling of the same path must
    resolve to the same sibling lock, or two writers would 'serialise' on two
    different lock files and lose each other's keys anyway."""
    path = str(tmp_path / "progress_controller_state.json")
    try:
        relative = os.path.relpath(path)
    except ValueError:  # Windows: tmp_path and cwd on different drives
        pytest.skip("tmp_path and cwd are on different drives")
    assert relative != path, "the harness needs a genuinely different spelling"
    assert progress_state_io.lock_path(relative) == progress_state_io.lock_path(path)
    assert progress_state_io.lock_path(path).endswith(progress_state_io.LOCK_SUFFIX)


def test_held_lock_times_out_to_a_warning_instead_of_hanging(tmp_path, monkeypatch, caplog):
    """A foreign process holding the lock: bounded wait, WARNING, then today's
    behaviour (write anyway) — never a hang, never an exception."""
    path = str(tmp_path / "progress_controller_state.json")
    monkeypatch.setattr(progress_state_io, "CROSS_PROCESS_LOCK_TIMEOUT_SECONDS", 0.3)

    foreign = _hold_lock_file(progress_state_io.lock_path(path))
    try:
        started = time.monotonic()
        with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
            written, err = progress_state_io.update_state(path, {"level": 7})
        elapsed = time.monotonic() - started
    finally:
        _release_lock_file(foreign)

    # It really waited out the bound (otherwise the "lock" would be a no-op) …
    assert 0.3 <= elapsed < 5.0, f"bounded wait expected, took {elapsed:.3f}s"
    # … and degraded instead of raising or blocking.
    assert written is True and err is None, (written, err)
    warnings = _evolution_warnings(caplog)
    assert any("cross-process lock" in w and "proceeding without it" in w for w in warnings), warnings
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert final == {"level": 7}

    # Once the foreign holder lets go, the lock is taken silently again.
    caplog.clear()
    with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
        assert progress_state_io.update_state(path, {"cycles_at_level": 2}) == (
            True,
            None,
        )
    assert not _evolution_warnings(caplog), "an uncontended lock must stay silent"


def test_unopenable_lock_file_degrades_to_a_warning(tmp_path, caplog):
    """A broken lock path (here: a directory where the lock file belongs) must
    not stop the progress update either."""
    path = str(tmp_path / "progress_controller_state.json")
    os.mkdir(progress_state_io.lock_path(path))

    with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
        written, err = progress_state_io.update_state(path, {"level": 7})

    assert written is True and err is None, (written, err)
    warnings = _evolution_warnings(caplog)
    assert any("cross-process lock" in w and "cannot be opened" in w for w in warnings), warnings
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert final == {"level": 7}


def test_clean_updates_take_the_sibling_lock_without_warning(tmp_path, caplog):
    path = str(tmp_path / "progress_controller_state.json")
    with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
        assert progress_state_io.update_state(path, {"a": 1}) == (True, None)
        assert progress_state_io.update_state(path, {"b": 2}) == (True, None)
    assert not _evolution_warnings(caplog), "contended-lock warnings must be real only"
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert final == {"a": 1, "b": 2}
    assert list(tmp_path.glob("*.tmp")) == [], "tmp files must not leak"


def test_second_thread_is_refused_while_the_lock_is_held(tmp_path):
    """The OS lock belongs to the process, not to a Python object: the
    per-thread registry must never hand a second copy of an already-held lock
    to another thread (that would be a silent no-op lock)."""
    path = str(tmp_path / "progress_controller_state.json")
    outcome: dict[str, bool] = {}

    def other_thread_attempt():
        with progress_state_io.cross_process_lock(path, timeout=0.3) as held:
            outcome["held"] = held

    with progress_state_io.cross_process_lock(path) as held:
        assert held is True
        other = threading.Thread(target=other_thread_attempt)
        other.start()
        other.join(timeout=30)

    assert not other.is_alive()
    assert outcome == {"held": False}, f"a contended attempt must degrade, not be granted: {outcome!r}"


def test_nested_update_state_does_not_fight_its_own_lock(tmp_path, monkeypatch, caplog):
    """Reentrancy: the OS lock is owned by the process, so a nested
    update_state (the read-hook nesting STATE_LOCK documents) must not try to
    lock its own lock file — that would burn the full timeout and warn.

    Note the nested merge itself stays unsafe (the outer stale merge wins),
    exactly as it was before this card: nesting is allowed so nothing
    deadlocks, not so nested merges become atomic.
    """
    path = str(tmp_path / "progress_controller_state.json")
    real_read = progress_state_io.read_state
    entered = {"n": 0}

    def nesting_read(p):
        snapshot = real_read(p)
        if entered["n"] == 0:
            entered["n"] += 1
            progress_state_io.update_state(p, {"inner": 1})
        return snapshot

    monkeypatch.setattr(progress_state_io, "read_state", nesting_read)

    started = time.monotonic()
    with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
        written, err = progress_state_io.update_state(path, {"outer": 1})
    elapsed = time.monotonic() - started

    assert written is True and err is None, (written, err)
    assert elapsed < 3.0, f"nested call must not wait out the lock timeout (took {elapsed:.3f}s)"
    assert not [w for w in _evolution_warnings(caplog) if "cross-process lock" in w]
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert final == {"outer": 1}, final


def test_seed_failure_closes_the_lock_handle_and_degrades(tmp_path, monkeypatch, caplog):
    """A lock file that OPENS but cannot be seeded (disk full, read-only, AV)
    must not leak a descriptor: the handle is closed where the failure
    happens, and the progress update still lands."""
    path = str(tmp_path / "progress_controller_state.json")
    lock_file = progress_state_io.lock_path(path)
    state = {"closed": False}
    real_open = open

    class _Unseedable:
        """A real handle whose seed-byte write always fails."""

        def __init__(self, handle):
            self._handle = handle

        def seek(self, *args, **kwargs):
            return self._handle.seek(*args, **kwargs)

        def tell(self):
            return self._handle.tell()

        def write(self, *_args, **_kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        def flush(self):
            return self._handle.flush()

        def close(self):
            state["closed"] = True
            return self._handle.close()

    def faking_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if os.path.realpath(str(file)) == os.path.realpath(lock_file):
            return _Unseedable(handle)
        return handle

    monkeypatch.setattr(progress_state_io, "open", faking_open, raising=False)

    with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
        handle, acquired = progress_state_io._open_and_lock(lock_file, 5.0, "test")
        written, err = progress_state_io.update_state(path, {"a": 1})

    assert (handle, acquired) == (None, False)
    assert state["closed"] is True, "the opened lock handle must be closed here"
    warnings = _evolution_warnings(caplog)
    assert any("could not be seeded" in w for w in warnings), warnings
    assert written is True and err is None, (written, err)
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert final == {"a": 1}


def test_fatal_lock_error_degrades_without_burning_the_timeout(tmp_path, monkeypatch, caplog):
    """An error that can never mean 'another process holds the lock' must not
    cost the full timeout on every write — that cost is paid while STATE_LOCK
    is held, so it would serialise every in-process writer behind it."""
    path = str(tmp_path / "progress_controller_state.json")

    def always_invalid(_handle):
        raise OSError(errno.EINVAL, "Invalid argument")

    monkeypatch.setattr(progress_state_io, "_acquire_os_lock", always_invalid)

    started = time.monotonic()
    with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
        written, err = progress_state_io.update_state(path, {"a": 1})
    elapsed = time.monotonic() - started

    assert written is True and err is None, (written, err)
    assert elapsed < 1.0, (
        "must fail fast instead of waiting out the "
        f"{progress_state_io.CROSS_PROCESS_LOCK_TIMEOUT_SECONDS}s timeout "
        f"(took {elapsed:.3f}s)"
    )
    warnings = _evolution_warnings(caplog)
    assert any("cannot be locked" in w for w in warnings), warnings
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert final == {"a": 1}
    assert list(tmp_path.glob("*.tmp")) == [], "tmp files must not leak"


def test_nul_bearing_path_degrades_to_false_and_warning_instead_of_raising(tmp_path, caplog):
    """Never-raise end to end (N1): ``cross_process_lock`` already degraded on
    a path this interpreter refuses (``ValueError: embedded null character``)
    — the reviewer watched the lock log its WARNING and then ``write_state``'s
    tmp ``open()`` raise the very same ValueError straight out of
    ``update_state``. Both stages have to degrade now."""
    path = str(tmp_path / "bad\x00name.json")

    with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
        result = progress_state_io.update_state(path, {"level": 7})

    assert result == (False, None), result
    warnings = _evolution_warnings(caplog)
    assert any("cross-process lock" in w and "proceeding without it" in w for w in warnings), warnings
    assert any("write failed" in w and "null character" in w for w in warnings), warnings
    # Nothing can even be *named* with a NUL, so no tmp file was created …
    assert list(tmp_path.glob("*.tmp")) == [], "tmp files must not leak"
    # … the registry is clean, …
    assert progress_state_io._thread_held == {}, "the registry must be empty again"
    # … and a well-formed path still writes normally afterwards.
    good = str(tmp_path / "progress_controller_state.json")
    assert progress_state_io.update_state(good, {"level": 7}) == (True, None)
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert final == {"level": 7}


def test_nested_lock_under_degraded_outer_yields_false_without_waiting(tmp_path, monkeypatch):
    """A degraded outer acquire registers its context too, so a nested entry
    re-enters *that* context instead of opening a second one: it yields
    ``False`` — claiming a lock nobody holds would hide the lost-update window
    the caller is being told about — and pays no second bound."""
    path = str(tmp_path / "progress_controller_state.json")
    monkeypatch.setattr(progress_state_io, "CROSS_PROCESS_LOCK_TIMEOUT_SECONDS", 0.3)

    foreign = _hold_lock_file(progress_state_io.lock_path(path))
    try:
        started = time.monotonic()
        with progress_state_io.cross_process_lock(path) as outer:
            outer_elapsed = time.monotonic() - started
            nested_started = time.monotonic()
            with progress_state_io.cross_process_lock(path) as nested:
                nested_elapsed = time.monotonic() - nested_started
    finally:
        _release_lock_file(foreign)

    assert outer is False, "the foreign holder owns the lock, not us"
    assert nested is False, "a nested entry under a degraded outer must not claim a lock nobody holds"
    assert 0.3 <= outer_elapsed < 5.0, f"the outer call must really wait out the bound, took {outer_elapsed:.3f}s"
    assert nested_elapsed < 0.15, f"the nested entry must not re-attempt the lock (took {nested_elapsed:.3f}s)"
    assert progress_state_io._thread_held == {}, "the registry must be empty again"


def test_nested_update_state_after_degraded_outer_pays_one_timeout_not_two(tmp_path, monkeypatch, caplog):
    """N3: read-hook nesting under an outer acquire that had to degrade used to
    re-enter ``_open_and_lock`` and burn a *second* full timeout while
    ``STATE_LOCK`` stayed held behind it (0.300s outer + 0.312s inner measured
    on a 0.3s bound — i.e. ~10s on the production 5.0s). One bound, one wait,
    one warning now."""
    path = str(tmp_path / "progress_controller_state.json")
    monkeypatch.setattr(progress_state_io, "CROSS_PROCESS_LOCK_TIMEOUT_SECONDS", 0.3)
    real_read = progress_state_io.read_state
    entered = {"n": 0}
    nested_elapsed: dict[str, float] = {}
    nested_result: dict[str, object] = {}

    def nesting_read(p):
        snapshot = real_read(p)
        if entered["n"] == 0:
            entered["n"] += 1
            nested_started = time.monotonic()
            nested_result["value"] = progress_state_io.update_state(p, {"inner": 1})
            nested_elapsed["s"] = time.monotonic() - nested_started
        return snapshot

    monkeypatch.setattr(progress_state_io, "read_state", nesting_read)

    foreign = _hold_lock_file(progress_state_io.lock_path(path))
    try:
        started = time.monotonic()
        with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
            written, err = progress_state_io.update_state(path, {"outer": 1})
        elapsed = time.monotonic() - started
    finally:
        _release_lock_file(foreign)

    assert written is True and err is None, (written, err)
    assert entered["n"] == 1, "the nesting read hook must have fired"
    # The nested call must not wait out the bound again …
    assert nested_elapsed["s"] < 0.15, (
        "the nested update_state must not re-attempt the lock " f"(took {nested_elapsed['s']:.3f}s)"
    )
    # … so the whole thing costs one bounded wait, not two. This bound only
    # rules out an *unbounded* wait (0.6s is pre-fix's 0.300 + 0.312, so the
    # check is deliberately loose for a loaded box); the "one wait, not two"
    # proof is the single timeout WARNING below — two waits log two.
    assert 0.3 <= elapsed < 0.9, f"the outer wait must stay bounded, took {elapsed:.3f}s"
    timeouts = [w for w in _evolution_warnings(caplog) if "still held after" in w]
    assert len(timeouts) == 1, timeouts
    assert nested_result["value"] == (True, None), nested_result
    assert progress_state_io._thread_held == {}, "the registry must be empty again"
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    # Same as the acquired-outer nesting test: the nested merge is overwritten
    # by the outer one — nesting is allowed so nothing deadlocks, not to make
    # nested merges atomic.
    assert final == {"outer": 1}, final


def test_non_pathlike_path_type_error_is_the_documented_contract(tmp_path):
    """N2: ``path`` is typed ``str``. A non-path is a caller bug rather than
    I/O or lock trouble, so it raises ``TypeError`` before any file is
    touched — which is exactly what the ``update_state`` docstring now
    promises, including its two distinct sources."""
    # Not path-like at all -> TypeError from os.fspath.
    with pytest.raises(TypeError, match="expected str, bytes or os.PathLike object"):
        progress_state_io.update_state(123, {"a": 1})
    assert progress_state_io._thread_held == {}, "a type error must not leave a half-registered lock context behind"
    # A path-like object is a normal path and still writes.
    real_path = tmp_path / "progress_controller_state.json"
    assert progress_state_io.update_state(real_path, {"a": 1}) == (True, None)
    # ``bytes`` are rejected too — but from lock_path's str concat, not from
    # os.fspath: this module has never accepted bytes, and the docstring says
    # so rather than pretending os.fspath turns them away.
    real = os.fspath(real_path)
    with pytest.raises(TypeError, match="can't concat str to bytes"):
        progress_state_io.update_state(os.fsencode(real), {"b": 2})
    assert progress_state_io._thread_held == {}
    final = json.loads((tmp_path / "progress_controller_state.json").read_text(encoding="utf-8"))
    assert final == {"a": 1}, "the rejected call must not have written"


def test_non_json_state_still_raises_value_error_out_of_update_state(tmp_path):
    """The ``except ValueError`` this card added to ``write_state`` sits right
    next to ``json.dumps(state)``, which the module docstring deliberately
    lets raise for a non-JSON state. A circular state must therefore still
    escape ``update_state`` as ``ValueError``: swallowed, it would masquerade
    as a dropped write and hide a caller bug."""
    path = str(tmp_path / "progress_controller_state.json")
    state: dict = {}
    state["self"] = state

    with pytest.raises(ValueError, match="Circular reference"):
        progress_state_io.update_state(path, state)

    assert progress_state_io._thread_held == {}, "the raising call must still unwind the lock registry"
    assert not os.path.exists(path), "no partial file may be left behind"
    assert list(tmp_path.glob("*.tmp")) == [], "tmp files must not leak"


def test_nul_path_still_degrades_when_realpath_rejects_it_like_posix_does(tmp_path, monkeypatch, caplog):
    """Round-2 F1: only ``ntpath`` tolerates a NUL path — ``posixpath.realpath``
    stats every component and lets ``ValueError`` escape (its ``except OSError``
    around ``os.lstat``), so on Linux/macOS ``_lock_identity`` used to raise
    straight out of ``update_state`` before the lock stage or ``write_state``
    could degrade at all. Force the POSIX behaviour on any platform and demand
    the same ``(False, None)`` + WARNING outcome Windows already had."""
    path = str(tmp_path / "bad\x00name.json")

    def posix_like_realpath(p):
        os.lstat(p)  # exactly what posixpath.realpath does per component
        return p

    monkeypatch.setattr(os.path, "realpath", posix_like_realpath)

    # The fallback still has to yield a deterministic lock identity …
    assert progress_state_io._lock_identity(path) == os.path.normcase(os.path.abspath(path))
    # … and update_state must degrade end to end instead of raising.
    with caplog.at_level("WARNING", logger=progress_state_io.LOGGER_NAME):
        result = progress_state_io.update_state(path, {"level": 7})

    assert result == (False, None), result
    warnings = _evolution_warnings(caplog)
    assert any("cross-process lock" in w and "proceeding without it" in w for w in warnings), warnings
    assert any("write failed" in w and "null character" in w for w in warnings), warnings
    assert progress_state_io._thread_held == {}, "the registry must be empty again"
    assert list(tmp_path.glob("*.tmp")) == [], "tmp files must not leak"


def test_pathologically_nested_state_file_is_unreadable_not_a_crash(tmp_path):
    """Round-2 F2: a nested-but-valid document makes ``json.load`` raise
    ``RecursionError``, which ``read_state``'s except tuple used to miss — the
    error then escaped ``update_state`` instead of coming back as
    ``({}, error)``, contradicting read_state's own contract."""
    path = str(tmp_path / "progress_controller_state.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"k":' * 4000 + "1" + "}" * 4000)

    state, err = progress_state_io.read_state(path)
    assert state == {} and isinstance(err, RecursionError), (state, err)

    # Through update_state the never-raise contract holds: the merge starts
    # from {} (logged by the caller via read_error) and the write still lands.
    written, read_err = progress_state_io.update_state(path, {"a": 1})
    assert written is True and isinstance(read_err, RecursionError), (
        written,
        read_err,
    )
    assert progress_state_io._thread_held == {}, "the registry must be empty again"
    with open(path, encoding="utf-8") as f:
        final = json.load(f)
    assert final == {"a": 1}
