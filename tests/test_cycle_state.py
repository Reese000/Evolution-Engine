#!/usr/bin/env python3
"""Tests for cycle_state.py — checksummed resume-state persistence.

Correct resume behavior depends on this: a silently-corrupted state must be
detected (checksum mismatch) and fall back to defaults rather than resuming
from garbage. This module had no direct test coverage.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cycle_state import SCHEMA_VERSION, CycleState, _compute_checksum, default_cycle_state


def _tmp_path() -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.unlink(path)  # start non-existent
    return path


class TestDefaultsAndChecksum:
    def test_default_state_shape(self):
        d = default_cycle_state()
        assert d["version"] == SCHEMA_VERSION
        assert d["current_cycle"] == 0
        assert d["best_fitness"] == 0.0
        assert d["champion_program"] == []

    def test_checksum_deterministic_and_sensitive(self):
        a = default_cycle_state()
        b = default_cycle_state()
        # start_timestamp differs by construction time; align it for equality.
        b["start_timestamp"] = a["start_timestamp"]
        assert _compute_checksum(a) == _compute_checksum(b)
        b["best_fitness"] = 123.4
        assert _compute_checksum(a) != _compute_checksum(b)


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_data(self):
        path = _tmp_path()
        try:
            cs = CycleState(path)
            cs.update(current_cycle=11, best_fitness=207.5, best_level=10, champion_program=[1, 2, 3])
            assert cs.save() is True

            cs2 = CycleState(path)
            assert cs2.load() is True
            assert cs2["current_cycle"] == 11
            assert cs2["best_fitness"] == 207.5
            assert cs2["best_level"] == 10
            assert cs2["champion_program"] == [1, 2, 3]
        finally:
            if os.path.isfile(path):
                os.unlink(path)


class TestIntegrity:
    def test_tampered_field_is_rejected(self):
        path = _tmp_path()
        try:
            cs = CycleState(path)
            cs.update(current_cycle=5, best_fitness=100.0)
            cs.save()

            # Tamper a checksummed field without updating the stored checksum.
            with open(path) as f:
                raw = json.load(f)
            raw["best_fitness"] = 999.0
            with open(path, "w") as f:
                json.dump(raw, f)

            cs2 = CycleState(path)
            assert cs2.load() is False, "tampered state must fail checksum"
            # Falls back to defaults, not the tampered value.
            assert cs2["best_fitness"] == 0.0
            assert cs2["current_cycle"] == 0
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_missing_file_returns_false(self):
        cs = CycleState("/nonexistent/dir/cycle_state.json")
        assert cs.load() is False
        assert cs["current_cycle"] == 0

    def test_corrupt_json_returns_false(self):
        path = _tmp_path()
        try:
            with open(path, "w") as f:
                f.write("{ not valid json ")
            cs = CycleState(path)
            assert cs.load() is False
            assert cs["current_cycle"] == 0
        finally:
            os.unlink(path)

    def test_old_version_rejected(self):
        path = _tmp_path()
        try:
            with open(path, "w") as f:
                json.dump({"version": 0, "current_cycle": 7}, f)
            cs = CycleState(path)
            assert cs.load() is False
            assert cs["current_cycle"] == 0
        finally:
            os.unlink(path)


class TestConvenience:
    def test_reset_preserves_start_timestamp(self):
        cs = CycleState(_tmp_path())
        original_start = cs["start_timestamp"]
        cs.update(current_cycle=9, best_fitness=50.0)
        cs.reset()
        assert cs["current_cycle"] == 0
        assert cs["best_fitness"] == 0.0
        assert cs["start_timestamp"] == original_start

    def test_item_access_and_update(self):
        cs = CycleState(_tmp_path())
        cs["best_level"] = 4
        assert cs["best_level"] == 4
        assert cs.get("best_level") == 4
        assert cs.get("missing_key", "default") == "default"
        cs.update(current_cycle=3, total_errors=2)
        assert cs["current_cycle"] == 3
        assert cs["total_errors"] == 2
