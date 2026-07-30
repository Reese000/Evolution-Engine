"""Tests for ChampionArchive integration into the training pipeline.

Tests the merge_archive_to_recurrent() function that feeds accumulated
champion programs into recurrent_data.jsonl for training.
"""

import json
import os
import sys
import tempfile

import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "milestone1"))


class TestMergeArchiveToRecurrent:
    """Tests for merge_archive_to_recurrent integration."""

    def test_merge_adds_archive_entries(self, tmp_path):
        """Archive entries are appended to recurrent_data.jsonl."""
        from champion_archive import ChampionArchive

        # Create a temporary archive with some entries (must be under champions/ subdir)
        os.makedirs(tmp_path / "champions", exist_ok=True)
        archive_path = str(tmp_path / "champions" / "archive.jsonl")
        archive = ChampionArchive(path=archive_path, max_size=100)
        archive.add([0x100073], level=13, fitness=200.0, cycle=1)
        archive.add([0x100073, 0x100073], level=12, fitness=150.0, cycle=2)
        archive.save()

        # Create recurrent data directory
        recurrent_dir = str(tmp_path / "recurrent_data")
        os.makedirs(recurrent_dir, exist_ok=True)
        recurrent_path = os.path.join(recurrent_dir, "recurrent_data.jsonl")

        # Mock the paths in run_multi_cycle
        import run_multi_cycle as rmc

        orig_project = rmc.PROJECT_ROOT
        orig_recurrent = rmc.M1_RECURRENT
        orig_recurrent_data = rmc.M1_RECURRENT_DATA
        try:
            rmc.PROJECT_ROOT = str(tmp_path)
            rmc.M1_RECURRENT = recurrent_dir
            rmc.M1_RECURRENT_DATA = recurrent_path

            added = rmc.merge_archive_to_recurrent(cycle=1)
        finally:
            rmc.PROJECT_ROOT = orig_project
            rmc.M1_RECURRENT = orig_recurrent
            rmc.M1_RECURRENT_DATA = orig_recurrent_data

        assert added == 2
        assert os.path.isfile(recurrent_path)

        with open(recurrent_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 2
        assert lines[0]["level"] == 13
        assert lines[0]["save_reason"] == "champion_archive"
        assert lines[1]["level"] == 12

    def test_merge_deduplicates_by_sha256(self, tmp_path):
        """Already-existing entries are not duplicated."""
        from champion_archive import ChampionArchive

        os.makedirs(tmp_path / "champions", exist_ok=True)
        archive_path = str(tmp_path / "champions" / "archive.jsonl")
        archive = ChampionArchive(path=archive_path, max_size=100)
        archive.add([0x100073], level=13, fitness=200.0, cycle=1)
        archive.save()

        recurrent_dir = str(tmp_path / "recurrent_data")
        os.makedirs(recurrent_dir, exist_ok=True)
        recurrent_path = os.path.join(recurrent_dir, "recurrent_data.jsonl")

        # Pre-populate with the same entry
        from champion_archive import _hash_program

        sha = _hash_program([0x100073])
        with open(recurrent_path, "w") as f:
            f.write(json.dumps({"champion_sha256": sha, "level": 13}) + "\n")

        import run_multi_cycle as rmc

        orig_project = rmc.PROJECT_ROOT
        orig_recurrent = rmc.M1_RECURRENT
        orig_recurrent_data = rmc.M1_RECURRENT_DATA
        try:
            rmc.PROJECT_ROOT = str(tmp_path)
            rmc.M1_RECURRENT = recurrent_dir
            rmc.M1_RECURRENT_DATA = recurrent_path

            added = rmc.merge_archive_to_recurrent(cycle=1)
        finally:
            rmc.PROJECT_ROOT = orig_project
            rmc.M1_RECURRENT = orig_recurrent
            rmc.M1_RECURRENT_DATA = orig_recurrent_data

        assert added == 0  # No new entries added (deduplicated)

        with open(recurrent_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 1  # Only the original entry

    def test_merge_empty_archive(self, tmp_path):
        """Empty archive adds nothing."""
        import run_multi_cycle as rmc

        os.makedirs(tmp_path / "champions", exist_ok=True)
        archive_path = str(tmp_path / "champions" / "archive.jsonl")
        # Don't create the archive file

        recurrent_dir = str(tmp_path / "recurrent_data")
        os.makedirs(recurrent_dir, exist_ok=True)
        recurrent_path = os.path.join(recurrent_dir, "recurrent_data.jsonl")

        orig_project = rmc.PROJECT_ROOT
        orig_recurrent = rmc.M1_RECURRENT
        orig_recurrent_data = rmc.M1_RECURRENT_DATA
        try:
            rmc.PROJECT_ROOT = str(tmp_path)
            rmc.M1_RECURRENT = recurrent_dir
            rmc.M1_RECURRENT_DATA = recurrent_path

            added = rmc.merge_archive_to_recurrent(cycle=1)
        finally:
            rmc.PROJECT_ROOT = orig_project
            rmc.M1_RECURRENT = orig_recurrent
            rmc.M1_RECURRENT_DATA = orig_recurrent_data

        assert added == 0

    def test_merge_creates_recurrent_dir(self, tmp_path):
        """merge_archive_to_recurrent creates the recurrent data directory if missing."""
        from champion_archive import ChampionArchive

        os.makedirs(tmp_path / "champions", exist_ok=True)
        archive_path = str(tmp_path / "champions" / "archive.jsonl")
        archive = ChampionArchive(path=archive_path, max_size=100)
        archive.add([0x100073], level=5, fitness=100.0, cycle=1)
        archive.save()

        recurrent_dir = str(tmp_path / "nonexistent" / "recurrent_data")
        recurrent_path = os.path.join(recurrent_dir, "recurrent_data.jsonl")

        import run_multi_cycle as rmc

        orig_project = rmc.PROJECT_ROOT
        orig_recurrent = rmc.M1_RECURRENT
        orig_recurrent_data = rmc.M1_RECURRENT_DATA
        try:
            rmc.PROJECT_ROOT = str(tmp_path)
            rmc.M1_RECURRENT = recurrent_dir
            rmc.M1_RECURRENT_DATA = recurrent_path

            added = rmc.merge_archive_to_recurrent(cycle=1)
        finally:
            rmc.PROJECT_ROOT = orig_project
            rmc.M1_RECURRENT = orig_recurrent
            rmc.M1_RECURRENT_DATA = orig_recurrent_data

        assert added == 1
        assert os.path.isdir(recurrent_dir)

    def test_merge_preserves_existing_data(self, tmp_path):
        """Existing recurrent data entries are preserved after merge."""
        from champion_archive import ChampionArchive

        os.makedirs(tmp_path / "champions", exist_ok=True)
        archive_path = str(tmp_path / "champions" / "archive.jsonl")
        archive = ChampionArchive(path=archive_path, max_size=100)
        archive.add([0x100073], level=13, fitness=200.0, cycle=1)
        archive.save()

        recurrent_dir = str(tmp_path / "recurrent_data")
        os.makedirs(recurrent_dir, exist_ok=True)
        recurrent_path = os.path.join(recurrent_dir, "recurrent_data.jsonl")

        # Pre-populate with different entries
        with open(recurrent_path, "w") as f:
            f.write(json.dumps({"champion_sha256": "aaa", "level": 0}) + "\n")
            f.write(json.dumps({"champion_sha256": "bbb", "level": 1}) + "\n")

        import run_multi_cycle as rmc

        orig_project = rmc.PROJECT_ROOT
        orig_recurrent = rmc.M1_RECURRENT
        orig_recurrent_data = rmc.M1_RECURRENT_DATA
        try:
            rmc.PROJECT_ROOT = str(tmp_path)
            rmc.M1_RECURRENT = recurrent_dir
            rmc.M1_RECURRENT_DATA = recurrent_path

            added = rmc.merge_archive_to_recurrent(cycle=1)
        finally:
            rmc.PROJECT_ROOT = orig_project
            rmc.M1_RECURRENT = orig_recurrent
            rmc.M1_RECURRENT_DATA = orig_recurrent_data

        assert added == 1

        with open(recurrent_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 3  # 2 original + 1 from archive
        assert lines[0]["champion_sha256"] == "aaa"
        assert lines[1]["champion_sha256"] == "bbb"
        assert lines[2]["level"] == 13

    def test_merge_idempotent(self, tmp_path):
        """Calling merge twice doesn't duplicate entries."""
        from champion_archive import ChampionArchive

        os.makedirs(tmp_path / "champions", exist_ok=True)
        archive_path = str(tmp_path / "champions" / "archive.jsonl")
        archive = ChampionArchive(path=archive_path, max_size=100)
        archive.add([0x100073], level=13, fitness=200.0, cycle=1)
        archive.save()

        recurrent_dir = str(tmp_path / "recurrent_data")
        os.makedirs(recurrent_dir, exist_ok=True)
        recurrent_path = os.path.join(recurrent_dir, "recurrent_data.jsonl")

        import run_multi_cycle as rmc

        orig_project = rmc.PROJECT_ROOT
        orig_recurrent = rmc.M1_RECURRENT
        orig_recurrent_data = rmc.M1_RECURRENT_DATA
        try:
            rmc.PROJECT_ROOT = str(tmp_path)
            rmc.M1_RECURRENT = recurrent_dir
            rmc.M1_RECURRENT_DATA = recurrent_path

            added1 = rmc.merge_archive_to_recurrent(cycle=1)
            added2 = rmc.merge_archive_to_recurrent(cycle=2)
        finally:
            rmc.PROJECT_ROOT = orig_project
            rmc.M1_RECURRENT = orig_recurrent
            rmc.M1_RECURRENT_DATA = orig_recurrent_data

        assert added1 == 1
        assert added2 == 0  # Second call adds nothing (deduplicated)
