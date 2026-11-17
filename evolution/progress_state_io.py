"""Shared durability helpers for ``.agents/progress_controller_state.json``.

Several writers persist this one file, and they historically used different
durability:

* ``ProgressController.save()`` (evolution.progress_controller) wrote the
  curriculum keys with a plain truncate + ``json.dump`` — no tmp file, no
  ``os.replace``;
* ``_patch_progress_state()`` (evolution.terminal_stagnation) merged the
  terminal-efficiency keys (``terminal_efficiency_scan_floor_*``,
  ``terminal_efficiency_certified_words``, ``terminal_efficiency_stable_*``)
  with tmp + fsync + ``os.replace``.

A truncate-write can tear the file the atomic writer just committed (readers
then see partial JSON) or drop keys the atomic writer just merged; a torn read
is even worse, because callers that treat "unreadable" as ``{}`` silently
re-arm the expensive per-cycle compression sweeps and quiescence skips. Both
writers therefore go through :func:`read_state` / :func:`update_state` here.

Guarantees, and their limits:

* no torn documents for readers — every write is tmp + ``os.replace``, so a
  reader sees the previous complete file or the new complete file;
* no lost updates **between writers in this process** — :func:`update_state`
  serialises read → merge → write under :data:`STATE_LOCK`, so writer B can
  never merge a snapshot that predates writer A's commit and then clobber it;
* no lost updates **between writers in different processes** (engine vs
  daemon/CLI helper) — :func:`update_state` additionally takes an advisory OS
  lock on the sibling ``<state file>.lock`` for the same read → merge → write
  (``msvcrt.locking`` on Windows, ``fcntl.flock`` on POSIX), see
  :func:`cross_process_lock`.

Limits that remain, deliberately:

* the cross-process lock is *bounded* — :data:`CROSS_PROCESS_LOCK_TIMEOUT_SECONDS`
  caps how long a writer waits for a foreign holder. On timeout, when the lock
  file cannot be opened or seeded, or for an error that cannot mean contention
  at all (:data:`LOCK_FATAL_ERRNOS`, which does not even wait), the writer
  logs a ``WARNING`` and proceeds anyway, i.e. it degrades to the historic
  last-writer-wins behaviour instead of ever stalling a cycle on a wedged
  lock. Only genuine contention is polled, and only for that one bound;
* the lock is advisory: a writer that bypasses this module entirely is not
  serialised against it (all three known writers — ``ProgressController.save``,
  ``terminal_stagnation._patch_progress_state`` and
  ``run_multi_cycle_loop._maybe_apply_collapse_recovery`` — go through
  :func:`update_state`);
* readers never take it, and do not need to: every write is an atomic
  ``os.replace``, so a reader sees the old complete file or the new one.

Contract of :func:`write_state` (identical to the historic
``_patch_progress_state`` behaviour it was factored out of):

* atomic: unique tmp file (pid + thread id, so concurrent writers cannot
  steal — and partially overwrite — each other's tmp), fsync, ``os.replace``;
* bounded transient retry (3 attempts, 0.05s/0.10s backoff) for
  AV/locking-style ``OSError``; deterministic errnos (ENOENT/ENOTDIR/EISDIR)
  are never retried;
* never raises out of a production progress update — a dropped update logs a
  ``WARNING`` on logger ``EvolutionEngine`` and returns ``False``. That
  covers an unwritable file *and* a path this interpreter refuses to open
  (an embedded NUL, on POSIX as well as Windows — see
  :func:`_lock_identity`); what still raises is caller error only: a non-JSON
  or pathologically deep ``state`` (``TypeError``/``ValueError``/
  ``RecursionError`` from ``json.dumps``, which runs outside the guarded
  region) or a ``path`` that is not a path at all (``TypeError``, see
  :func:`update_state`). Unparseable *input* never raises: that comes back
  from :func:`read_state` as ``({}, error)``.
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import threading
import time
from collections.abc import Iterator
from typing import Any

#: Platform advisory-lock primitives, imported once at module load. Importing
#: them lazily inside the lock helpers would let an ``ImportError`` escape
#: :func:`update_state`, whose contract is to never raise for lock problems.
if os.name == "nt":
    import msvcrt
else:
    import fcntl

STATE_FILENAME = "progress_controller_state.json"
LOGGER_NAME = "EvolutionEngine"

#: Sibling file that carries the cross-process lock (``<state file>.lock``).
LOCK_SUFFIX = ".lock"

#: How long :func:`cross_process_lock` waits for a *foreign* process before it
#: gives up, logs a ``WARNING`` and proceeds unlocked (historic behaviour).
#: Bounded on purpose: a wedged lock holder must never stall an evolution
#: cycle, so degrading beats blocking.
CROSS_PROCESS_LOCK_TIMEOUT_SECONDS = 5.0

#: Poll interval while waiting for a foreign process's lock.
CROSS_PROCESS_LOCK_POLL_SECONDS = 0.02

#: Errors that can never mean "another process holds the lock" (bad/unsupported
#: lock target, read-only or lockless filesystem). Polling these out to the
#: full timeout would burn ``CROSS_PROCESS_LOCK_TIMEOUT_SECONDS`` on *every*
#: write while :data:`STATE_LOCK` is held, so they degrade immediately.
#: Contention itself surfaces as ``EACCES`` (``PermissionError`` on both
#: platforms) and keeps polling, which is the one case worth waiting for.
LOCK_FATAL_ERRNOS = frozenset(
    code
    for code in (
        getattr(errno, "EBADF", None),
        getattr(errno, "EINVAL", None),
        getattr(errno, "ENOLCK", None),
        getattr(errno, "ENOSYS", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "EROFS", None),
    )
    if code is not None
)

#: Errors where a retry cannot help (missing directory, broken path).
DETERMINISTIC_WRITE_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EISDIR})

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 0.05
_DEFAULT_LABEL = "progress-controller state"


def state_path(agents_dir: str) -> str:
    """Absolute path of the shared progress-controller state file."""
    return os.path.join(agents_dir, STATE_FILENAME)


def read_state(path: str) -> tuple[dict, BaseException | None]:
    """Read the state file, distinguishing "missing" from "unreadable".

    Returns ``(state, error)``:

    * ``({} , None)`` — file absent (fresh install, not a failure);
    * ``(data, None)`` — file parsed cleanly into a JSON object;
    * ``({} , error)`` — file present but torn/unparseable/not an object.

    Callers must NOT treat the third case as "no state" silently: dropping to
    ``{}`` re-arms the expensive per-cycle terminal-efficiency work, so it has
    to be observable (log it).
    """
    try:
        if not os.path.isfile(path):
            return {}, None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        # Raced with a writer's os.replace — same as "absent".
        return {}, None
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {}, exc
    except RecursionError as exc:
        # A pathologically nested but syntactically valid document makes
        # json.load exhaust the recursion limit. That is still "present but
        # unreadable", so it has to come back as ({}, error) like every other
        # parse failure instead of escaping update_state and contradicting
        # this function's contract.
        return {}, exc
    if not isinstance(data, dict):
        return {}, ValueError(f"expected a JSON object, got {type(data).__name__}")
    return data, None


#: Serialises read → merge → write for every writer of this file *inside one
#: process*. Reentrant so a caller may nest (e.g. a unit test that runs a
#: writer from inside another writer's read hook) without deadlocking.
STATE_LOCK = threading.RLock()


def _lock_identity(path) -> str:
    """Canonical identity of one state file — one file, one lock.

    Normalising (absolute + case-folded + symlink-resolved) matters twice: the
    lock file must be shared by every spelling of the same state file, and the
    reentrancy key below must match that spelling, or a nested call would try
    to lock its own lock file and time out instead of re-entering.

    ``realpath`` is guarded because only ``ntpath`` tolerates a path this
    interpreter rejects: ``posixpath.realpath`` stats every component and lets
    ``ValueError`` (embedded NUL) escape, which would raise straight out of
    :func:`update_state` on POSIX before any degradation could run. The
    fallback is a pure-string normalisation — such a path cannot be opened as
    a file anyway, so the identity only has to stay deterministic. ``TypeError``
    from :func:`os.fspath` is not caught: a ``path`` that is not a path at all
    is a caller bug, documented in :func:`update_state`.
    """
    resolved = os.fspath(path)
    try:
        resolved = os.path.realpath(resolved)
    except ValueError:
        resolved = os.path.abspath(resolved)
    return os.path.normcase(resolved)


def lock_path(path) -> str:
    """Sibling file carrying the cross-process lock for ``path``."""
    return _lock_identity(path) + LOCK_SUFFIX


#: ``(abs state path, thread ident) -> (nesting depth, really held)`` for the
#: lock context the thread already has open. The OS lock is owned by the
#: *process*, so a nested ``update_state`` from the same thread would
#: otherwise fight its own outer lock, burn the whole timeout and then warn
#: for nothing. The flag records whether the outer acquisition actually took
#: the OS lock: a degraded outer registers its own marker (depth 1,
#: ``really held = False``), so a nested call re-enters that context instead
#: of re-locking — it yields ``False`` (never ``True``, which would claim a
#: lock nobody holds) and never pays a second timeout while
#: :data:`STATE_LOCK` is held behind it. A second *thread* cannot reach here
#: while the first holds :data:`STATE_LOCK`, so per-thread bookkeeping is
#: enough and never lets two threads skip the real lock.
_registry_guard = threading.Lock()
_thread_held: dict[tuple[str, int], tuple[int, bool]] = {}


def _acquire_os_lock(handle) -> None:
    """Take the advisory lock (``OSError`` when another holder has it)."""
    if os.name == "nt":
        handle.seek(0)
        # LK_NBLCK is non-blocking, so contention surfaces as OSError here and
        # the caller decides whether to keep polling or to degrade.
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_os_lock(handle) -> None:
    """Drop the advisory lock; closing the handle would do it anyway."""
    try:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError):
        # Already released, or the handle was closed underneath us — either
        # way there is nothing left to unlock, and the caller still has to
        # get to its registry pop and its close().
        pass


def _open_and_lock(lock_file: str, timeout: float, label: str) -> tuple[Any, bool]:
    """Open ``lock_file`` and poll until the lock is taken or ``timeout``.

    Returns ``(handle, acquired)``. ``handle`` is ``None`` when the lock file
    could not be opened *or* could not be prepared (logged, and the handle that
    was opened is closed right there); ``acquired`` is ``False`` after a
    timeout, in which case the caller must still close the handle. Never
    raises for an ``OSError``/``ValueError`` — that is the "degrade to today's
    behaviour" path.
    """
    log = logging.getLogger(LOGGER_NAME)
    try:
        handle = open(lock_file, "a+b")  # noqa: SIM115 - closed on every path
    except (OSError, ValueError) as exc:
        log.warning(
            "%s: cross-process lock %s cannot be opened (%s) — proceeding " "without it (last-writer-wins, as before)",
            label,
            lock_file,
            exc,
        )
        return None, False

    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            # Belt and braces: some platforms/filesystems refuse to lock a
            # zero-length file, so give the sibling one byte of substance.
            handle.write(b"\0")
            handle.flush()
    except (OSError, ValueError) as exc:
        # Opened but unusable (full, read-only, AV): close it HERE, because
        # the caller only closes a handle it receives. Not closing it would
        # leak one descriptor per write in exactly the broken-disk situation
        # this degradation path exists to survive.
        try:
            handle.close()
        except (OSError, ValueError):
            pass
        log.warning(
            "%s: cross-process lock %s could not be seeded (%s) — proceeding "
            "without it (last-writer-wins, as before)",
            label,
            lock_file,
            exc,
        )
        return None, False

    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            _acquire_os_lock(handle)
            return handle, True
        except (OSError, ValueError) as exc:
            # Only genuine contention is worth waiting out: a lock target this
            # platform refuses to lock would otherwise cost the full timeout
            # on every single write, with STATE_LOCK held behind it.
            if not isinstance(exc, OSError) or getattr(exc, "errno", None) in LOCK_FATAL_ERRNOS:
                log.warning(
                    "%s: cross-process lock %s cannot be locked (%s) — "
                    "proceeding without it (last-writer-wins, as before)",
                    label,
                    lock_file,
                    exc,
                )
                return handle, False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning(
                    "%s: cross-process lock %s still held after %.1fs (%s) — "
                    "proceeding without it (last-writer-wins, as before)",
                    label,
                    lock_file,
                    timeout,
                    exc,
                )
                return handle, False
            time.sleep(min(CROSS_PROCESS_LOCK_POLL_SECONDS, remaining))


@contextlib.contextmanager
def cross_process_lock(path: str, *, timeout: float | None = None, label: str = _DEFAULT_LABEL) -> Iterator[bool]:
    """Advisory cross-process lock guarding one read → merge → write of ``path``.

    Yields ``True`` when this process holds the lock on the sibling
    :func:`lock_path` file, ``False`` when it had to proceed *without* it.
    Degradation cases — the lock still held by another process when the
    (bounded) timeout expires, a lock file that cannot be opened or seeded, or
    an error this platform can never recover from (:data:`LOCK_FATAL_ERRNOS`,
    which does not even wait out the timeout) — log a ``WARNING`` on logger
    ``EvolutionEngine`` and then continue, so a wedged lock costs one warning
    and one lost-update window (i.e. exactly the pre-existing behaviour),
    never a stalled cycle.

    Reentrant within a thread: nesting under an outer ``cross_process_lock``
    yields ``True`` again without re-locking, because the OS lock belongs to
    the process and would otherwise time out against itself. If the *outer*
    acquisition had to degrade, the nested call re-enters that degraded
    context instead of re-attempting the lock: it yields ``False`` — this
    process does not hold the lock, and claiming otherwise would hide the
    lost-update window the caller is being told about — and it costs no
    second :data:`CROSS_PROCESS_LOCK_TIMEOUT_SECONDS`. Nesting exists for
    read-hook nesting, which production writers do not do.
    """
    if timeout is None:
        timeout = CROSS_PROCESS_LOCK_TIMEOUT_SECONDS
    key = (_lock_identity(path), threading.get_ident())
    with _registry_guard:
        entry = _thread_held.get(key)
        if entry is not None:
            depth, really_held = entry
            _thread_held[key] = (depth + 1, really_held)
    if entry is not None:
        try:
            yield really_held
        finally:
            with _registry_guard:
                current = _thread_held.get(key)
                if current is not None and current[0] > 1:
                    _thread_held[key] = (current[0] - 1, really_held)
                else:
                    _thread_held.pop(key, None)
        return

    handle, acquired = _open_and_lock(lock_path(path), timeout, label)
    with _registry_guard:
        # Registered on BOTH outcomes: a degraded acquisition still marks this
        # context, so a nested call re-enters it (yielding False) instead of
        # paying a second full timeout behind STATE_LOCK. The marker is popped
        # by this frame's finally, like the acquired one.
        _thread_held[key] = (1, acquired)
    try:
        yield acquired
    finally:
        # Nested try/finally: a failure inside _release_os_lock must not skip
        # the registry pop — a leaked entry would make every later lock on this
        # thread a silent no-op — and neither must it skip the close.
        try:
            if handle is not None and acquired:
                _release_os_lock(handle)
        finally:
            with _registry_guard:
                _thread_held.pop(key, None)
            if handle is not None:
                with contextlib.suppress(OSError, ValueError):
                    handle.close()


def update_state(path: str, updates: dict, *, label: str = _DEFAULT_LABEL) -> tuple[bool, BaseException | None]:
    """Read → merge ``updates`` → atomic write, under both locks.

    Returns ``(written, read_error)``. ``read_error`` is non-``None`` when the
    pre-existing file could not be parsed, i.e. the merge started from ``}``
    and the previous keys could not be preserved — callers should log that.
    ``written`` is ``False`` when the write was dropped (already logged by
    :func:`write_state`).

    ``path`` must be a ``str`` (or an ``os.PathLike`` yielding one) — the
    signature's promise, and the only spelling :func:`lock_path` can build a
    lock name from. Anything else is a caller bug rather than an I/O or lock
    failure, and raises ``TypeError`` before a single file is touched: from
    :func:`os.fspath` for something that is not path-like at all (``int``,
    ``None``, …) and from :func:`lock_path` for ``bytes``, which this module
    has never accepted (``LOCK_SUFFIX`` is ``str``). The "never raise"
    contract covers *filesystem* trouble only — an unusable path such as an
    embedded NUL still degrades to ``(False, None)`` plus a ``WARNING``, it
    never escapes as an exception.

    The lock is what keeps two in-process writers from losing each other's
    keys: without it, writer A can read, writer B can commit, and A's write
    then clobbers B's keys with the stale merge. :data:`STATE_LOCK` covers
    threads of this process; :func:`cross_process_lock` covers writers in
    other processes, so two processes cannot interleave their snapshots
    either (unless the cross-process lock had to degrade — see
    :func:`cross_process_lock`).
    """
    with STATE_LOCK, cross_process_lock(path, label=label):
        state, err = read_state(path)
        state.update(updates)
        return write_state(path, state, label=label), err


def write_state(path: str, state: dict, *, label: str = _DEFAULT_LABEL) -> bool:
    """Atomically persist ``state`` to ``path`` (tmp + fsync + ``os.replace``).

    Returns ``True`` when the file now holds ``state``; ``False`` when the
    update was dropped (already logged as a ``WARNING``). Never raises for
    ``OSError`` — the historic contract is that a dead/unwritable state file
    must not take down a production progress update — and never for a path
    this interpreter refuses to open (e.g. an embedded NUL), which degrades
    the same way on the first attempt. ``TypeError``/``ValueError`` from
    serialising a non-JSON ``state`` do propagate, exactly as ``json.dump``
    did before: ``json.dumps`` runs outside the guarded region.

    ``path`` is validated by :func:`update_state` before this function is
    ever reached: called *directly* with a non-path it raises the raw
    ``TypeError`` from ``os.replace`` (after creating a tmp file), and a
    ``bytes`` path works only by accident of the tmp name being an
    f-string. The three production writers never call it directly.
    """
    payload = json.dumps(state, indent=2)
    # Unique per writer: a shared tmp name let two concurrent writers race
    # (writer B's os.replace could move writer A's half-written tmp into
    # place, which is precisely the torn file this module exists to prevent).
    tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    log = logging.getLogger(LOGGER_NAME)
    for attempt in range(_MAX_ATTEMPTS):
        # ``json.dumps(state)`` above is deliberately OUTSIDE this loop: a
        # non-JSON state must keep raising TypeError/ValueError at the caller,
        # so only the path-shaped ValueError this ``except`` can see — this
        # interpreter refuses to build a file name from ``path`` (embedded
        # NUL) — is degraded here. The write itself cannot be its source: the
        # payload is pure ASCII (``json.dumps`` defaults to ``ensure_ascii``),
        # so the ``UnicodeEncodeError`` a text handle can otherwise raise (a
        # ValueError subclass) is off the table.
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            return True
        except ValueError as exc:
            # An unusable path is as deterministic as ENOENT: retrying can
            # never make it writable, so degrade on the first attempt — same
            # WARNING + ``False`` contract as an unwritable file, never a
            # raise out of :func:`update_state`. No tmp file can be left
            # behind today (``tmp_path`` embeds ``path``, so ``open`` is what
            # failed), but the cleanup is mirrored anyway in case some future
            # platform refuses the path only at ``os.replace``.
            log.warning(
                "%s write failed for %s after %d attempt(s): %s",
                label,
                path,
                attempt + 1,
                exc,
            )
            with contextlib.suppress(OSError, ValueError):
                os.remove(tmp_path)
            return False
        except OSError as exc:
            if attempt >= _MAX_ATTEMPTS - 1 or getattr(exc, "errno", None) in DETERMINISTIC_WRITE_ERRNOS:
                log.warning(
                    "%s write failed for %s after %d attempt(s): %s",
                    label,
                    path,
                    attempt + 1,
                    exc,
                )
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                return False
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
    return False
