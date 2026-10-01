"""Tests for archive sidecar persistence."""

import json

from evolution.archive_persistence import (
    cap_strategy_archive_state,
    load_archive_from_result,
    load_archive_sidecar,
    save_archive_sidecar,
    slim_result_for_jsonl,
)


def test_sidecar_roundtrip(tmp_path):
    root = str(tmp_path)
    blob = {"strategy": {"records": [{"a": 1}]}, "qd": {}, "novelty": {}}
    path = save_archive_sidecar(root, 42, blob)
    loaded = load_archive_sidecar(root, 42)
    assert loaded == blob
    assert path.endswith("cycle_00042.json")


def test_slim_result_strips_inline_archive():
    big = {"cycle": 1, "final_level": 17, "archive_state": {"strategy": {"x": 1}}}
    slim = slim_result_for_jsonl(big)
    assert "archive_state" not in slim
    assert slim["final_level"] == 17


def test_load_from_sidecar_path(tmp_path):
    root = str(tmp_path)
    blob = {"strategy": {}}
    path = save_archive_sidecar(root, 5, blob)
    loaded = load_archive_from_result(root, {"archive_state_path": path, "cycle": 5})
    assert loaded == blob


def test_cap_strategy_archive_state_keeps_newest():
    blob = {"strategy": {"records": [{"i": i} for i in range(5000)]}, "qd": {}}
    capped = cap_strategy_archive_state(blob, 3200)
    assert capped is not None
    assert len(capped["strategy"]["records"]) == 3200
    assert capped["strategy"]["records"][0]["i"] == 1800
    assert capped["strategy"]["records"][-1]["i"] == 4999


def test_load_prefers_richer_nearby_sidecar(tmp_path):
    root = str(tmp_path)
    thin = {"strategy": {"records": [{"i": i} for i in range(100)]}}
    rich = {"strategy": {"records": [{"i": i} for i in range(8150)]}}
    save_archive_sidecar(root, 10, rich)
    save_archive_sidecar(root, 11, thin)
    loaded = load_archive_sidecar(root)
    assert loaded is not None
    assert len(loaded["strategy"]["records"]) == 8150


def test_load_keeps_capped_latest_instead_of_richer(tmp_path):
    root = str(tmp_path)
    rich = {"strategy": {"records": [{"i": i} for i in range(8150)]}}
    capped = {"strategy": {"records": [{"i": i} for i in range(1500)]}}
    save_archive_sidecar(root, 10, rich)
    save_archive_sidecar(root, 11, capped)
    loaded = load_archive_sidecar(root)
    assert loaded is not None
    assert len(loaded["strategy"]["records"]) == 1500
