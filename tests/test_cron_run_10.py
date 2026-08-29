#!/usr/bin/env python3
"""Tests for cron_run_10 helpers (previously untested).

find_latest_state drives resume (picks the highest cycle that has archive_state
across result files); win_to_wsl is a pure path converter.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cron_run_10


class TestWinToWsl:
    def test_empty_and_none(self):
        assert cron_run_10.win_to_wsl("") == ""
        assert cron_run_10.win_to_wsl(None) is None

    def test_already_wsl(self):
        assert cron_run_10.win_to_wsl("/mnt/c/x") == "/mnt/c/x"

    def test_windows_drive_conversion(self):
        assert cron_run_10.win_to_wsl("C:\\Users\\foo") == "/mnt/c/Users/foo"
        assert cron_run_10.win_to_wsl("D:\\a\\b") == "/mnt/d/a/b"

    def test_relative_passthrough(self):
        assert cron_run_10.win_to_wsl("foo/bar") == "foo/bar"


class TestFindLatestState:
    def test_none_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cron_run_10, "PROJECT_ROOT", str(tmp_path))
        assert cron_run_10.find_latest_state() is None

    def test_picks_highest_cycle_with_archive_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cron_run_10, "PROJECT_ROOT", str(tmp_path))
        path = os.path.join(str(tmp_path), "multi_cycle_results.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"cycle": 1, "archive_state": {"qd": {}}}) + "\n")
            f.write(json.dumps({"cycle": 5, "archive_state": {"qd": {}}}) + "\n")
            # Higher cycle but NO archive_state -> must be ignored.
            f.write(json.dumps({"cycle": 9}) + "\n")
        best = cron_run_10.find_latest_state()
        assert best is not None
        assert best["cycle"] == 5

    def test_ignores_malformed_lines(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cron_run_10, "PROJECT_ROOT", str(tmp_path))
        path = os.path.join(str(tmp_path), "multi_cycle_results.jsonl")
        with open(path, "w") as f:
            f.write("{ not json\n")
            f.write("\n")
            f.write(json.dumps({"cycle": 3, "archive_state": {"qd": {}}}) + "\n")
        best = cron_run_10.find_latest_state()
        assert best is not None and best["cycle"] == 3
