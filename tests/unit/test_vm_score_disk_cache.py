"""Disk-backed VM score cache survives probe-pool cold starts."""


def test_heldout_disk_cache_hit(tmp_path, monkeypatch):
    from evolution.terminal_stagnation import (
        _heldout_l30_verify_mean_cached,
        _vm_score_disk_put,
        _vm_score_disk_reset,
        _vm_score_program_key,
    )

    agents = tmp_path / ".agents"
    agents.mkdir()
    _vm_score_disk_reset()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    program = (0x00100093, 0x00200113, 0x00100073)
    disk_key = f"{_vm_score_program_key(program)}|8"
    _vm_score_disk_put("heldout_l30", disk_key, 0.88)
    _heldout_l30_verify_mean_cached.cache_clear()
    assert _heldout_l30_verify_mean_cached(program, 8) == 0.88


def _fresh_agents(tmp_path, monkeypatch) -> str:
    """Point the score memo at an empty directory and drop in-process state."""
    from evolution.terminal_stagnation import _vm_score_disk_reset

    agents = tmp_path / ".agents"
    agents.mkdir(exist_ok=True)
    _vm_score_disk_reset()
    monkeypatch.setattr(
        "evolution.terminal_stagnation._project_agents_dir",
        lambda: str(agents),
    )
    return str(agents)


def test_memo_hydrates_from_disk_when_fingerprint_matches(tmp_path, monkeypatch):
    """A restarted process must load stored scores, not silently drop them.

    This is the post-restart cold-start contract: the whole point of the disk
    memo is that a fresh interpreter re-hydrates it instead of re-scoring.
    """
    from evolution.terminal_stagnation import (
        _vm_score_disk_flush,
        _vm_score_disk_get,
        _vm_score_disk_put,
        _vm_score_disk_reset,
        _vm_score_source_fp,
    )

    _fresh_agents(tmp_path, monkeypatch)
    _vm_score_disk_put("heldout_l30", "00100093,00200113,00100073|8", 0.75)
    _vm_score_disk_flush()
    assert (tmp_path / ".agents" / "vm_score_disk_cache.json").is_file()

    _vm_score_disk_reset()  # what a brand-new process sees
    assert _vm_score_disk_get("heldout_l30", "00100093,00200113,00100073|8") == 0.75
    assert _vm_score_source_fp()


def test_memo_is_discarded_when_fingerprint_mismatches(tmp_path, monkeypatch):
    """A score written by other scoring code must never be served back."""
    import json
    import os

    from evolution.terminal_stagnation import (
        _vm_score_disk_flush,
        _vm_score_disk_get,
        _vm_score_disk_put,
        _vm_score_disk_reset,
    )

    agents = _fresh_agents(tmp_path, monkeypatch)
    _vm_score_disk_put("heldout_l30", "00100093,00200113,00100073|8", 0.75)
    _vm_score_disk_flush()
    path = os.path.join(agents, "vm_score_disk_cache.json")
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    payload["_version"] = "0" * 16
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

    _vm_score_disk_reset()
    assert _vm_score_disk_get("heldout_l30", "00100093,00200113,00100073|8") is None


def test_fingerprint_inputs_are_producers_not_the_whole_module():
    """Non-scoring edits to this 6k-line module must not invalidate the memo.

    Before this split the whole of ``evolution/terminal_stagnation.py`` was
    hashed, so any probe/gate/diagnostic edit made the next process load 0 of
    122,291 stored scores and re-score 1,137,353 VM calls in generation 0.
    """
    from evolution.terminal_stagnation import _VM_SCORE_FINGERPRINT_FUNCS, _vm_score_fingerprint_inputs

    labels = [label for label, _blob in _vm_score_fingerprint_inputs()]
    assert not any(label.endswith("terminal_stagnation.py") for label in labels), labels
    # The VM and the task set determine every score, so they stay whole-file.
    assert {"tasks_rv32i.py", "gpu_vm_farm.py"} <= set(labels), labels
    for name in _VM_SCORE_FINGERPRINT_FUNCS:
        assert f"func:{name}" in labels, (name, labels)
    # A producer that is not yet defined (fingerprint computed too early) would
    # be folded in by name only and stop reacting to real edits.
    assert not any(label.startswith("unreadable:") for label in labels), labels
    assert len(labels) == len(set(labels)), labels


def test_every_score_producer_in_module_is_fingerprinted():
    """Adding a new ``_vm_score_disk_put`` site without fingerprinting it fails."""
    import ast
    import inspect

    from evolution import terminal_stagnation as ts

    path = inspect.getsourcefile(ts)
    assert path, "could not locate terminal_stagnation.py source"
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    owners = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            owners[child] = node

    def top_level_statement(node):
        while node is not None and owners.get(node) is not tree:
            node = owners.get(node)
        return node

    producers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_vm_score_disk_put":
            owner = top_level_statement(node)
            if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                producers.add(owner.name)
            else:
                # Class methods / module-level writes are not in the list, so
                # this name can never match and the assertion below fails.
                producers.add(f"<top-level {type(owner).__name__ if owner is not None else 'None'}>")
    assert producers, "expected at least one in-module score producer"
    missing = producers - set(ts._VM_SCORE_FINGERPRINT_FUNCS)
    assert not missing, (
        f"score producers {sorted(missing)} write memo values but are not " "covered by _VM_SCORE_FINGERPRINT_FUNCS"
    )


def test_fingerprint_changes_when_a_producer_changes(monkeypatch):
    import evolution.terminal_stagnation as ts

    before = ts._vm_score_source_fingerprint()

    def _edited(program, n_seeds: int = 32) -> float:  # pragma: no cover - stub
        return 0.0

    monkeypatch.setattr(ts, "heldout_l30_verify_mean", _edited)
    after = ts._vm_score_source_fingerprint()
    assert after != before


def test_fingerprint_ignores_non_producer_edits(monkeypatch):
    """Probe selection / gate policy churn must not throw the memo away."""
    import evolution.terminal_stagnation as ts

    before = ts._vm_score_source_fingerprint()
    monkeypatch.setattr(ts, "select_terminal_probes_for_batch", lambda *a, **k: [])
    assert ts._vm_score_source_fingerprint() == before
