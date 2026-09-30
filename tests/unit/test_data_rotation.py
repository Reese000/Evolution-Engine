#!/usr/bin/env python3
"""Tests for data_rotation module.

Validates:
  - rotate_checkpoints: sacred file preservation, keep_n newest, empty dir, .regressed prioritization
  - rotate_recurrent_data: under-limit skip, over-limit rotation, backup limiting, missing file
  - rotate_archives: keep_n newest cycle dirs, empty state
  - rotate_reports: keep_n newest progress_report files
  - run_all_rotations: integration test returning correct summary counts
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ══════════════════════════════════════════════════════════════════════
# Checkpoint rotation tests
# ══════════════════════════════════════════════════════════════════════


class TestRotateCheckpoints(unittest.TestCase):
    """Tests for rotate_checkpoints."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._ckpt_dir = os.path.join(self._tmpdir, "checkpoints")

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_pt(self, name, mtime=None):
        """Create a fake .pt file and optionally set its mtime."""
        path = os.path.join(self._ckpt_dir, name)
        with open(path, "wb") as f:
            f.write(b"\x00" * 1024)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path

    @patch("data_rotation.M1_CHECKPOINTS", new_callable=lambda: None)
    def test_rotate_checkpoints_keeps_sacred(self, _):
        import data_rotation as dr

        dr.M1_CHECKPOINTS = self._ckpt_dir
        os.makedirs(self._ckpt_dir, exist_ok=True)

        # Sacred files
        self._make_pt("checkpoint_best.pt", mtime=1000.0)
        self._make_pt("checkpoint_current.pt", mtime=2000.0)

        # 10 regular files with increasing mtimes (10 = newest)
        for i in range(1, 11):
            self._make_pt(f"model_epoch{i:02d}.pt", mtime=float(i * 100))

        deleted = dr.rotate_checkpoints(keep_n=5)

        remaining = set(os.listdir(self._ckpt_dir))
        self.assertIn("checkpoint_best.pt", remaining)
        self.assertIn("checkpoint_current.pt", remaining)
        # 5 regular files should remain (the 5 newest: epoch06 through epoch10)
        regular = [f for f in remaining if f not in ("checkpoint_best.pt", "checkpoint_current.pt")]
        self.assertEqual(len(regular), 5)
        self.assertEqual(len(deleted), 5)

    @patch("data_rotation.M1_CHECKPOINTS", new_callable=lambda: None)
    def test_rotate_checkpoints_keeps_n_newest(self, _):
        import data_rotation as dr

        dr.M1_CHECKPOINTS = self._ckpt_dir
        os.makedirs(self._ckpt_dir, exist_ok=True)

        for i in range(1, 11):
            self._make_pt(f"model_{i:02d}.pt", mtime=float(i * 100))

        deleted = dr.rotate_checkpoints(keep_n=5)

        remaining = sorted(os.listdir(self._ckpt_dir))
        self.assertEqual(len(remaining), 5)
        # Newest 5 should be model_06 through model_10
        self.assertEqual(remaining, [f"model_{i:02d}.pt" for i in range(6, 11)])
        self.assertEqual(len(deleted), 5)

    @patch("data_rotation.M1_CHECKPOINTS", new_callable=lambda: None)
    def test_rotate_checkpoints_empty_dir(self, _):
        import data_rotation as dr

        dr.M1_CHECKPOINTS = self._ckpt_dir
        os.makedirs(self._ckpt_dir, exist_ok=True)

        deleted = dr.rotate_checkpoints(keep_n=5)
        self.assertEqual(deleted, [])
        self.assertEqual(os.listdir(self._ckpt_dir), [])

    @patch("data_rotation.M1_CHECKPOINTS", new_callable=lambda: None)
    def test_rotate_checkpoints_regression_files(self, _):
        import data_rotation as dr

        dr.M1_CHECKPOINTS = self._ckpt_dir
        os.makedirs(self._ckpt_dir, exist_ok=True)

        # Sacred file
        self._make_pt("checkpoint_best.pt", mtime=5000.0)

        # 5 regular .pt files
        for i in range(1, 6):
            self._make_pt(f"model_{i:02d}.pt", mtime=float(i * 100))

        # 3 .pt.regressed files (older mtimes)
        for i in range(1, 4):
            path = os.path.join(self._ckpt_dir, f"model_{i:02d}.pt.regressed")
            with open(path, "wb") as f:
                f.write(b"\x00" * 1024)
            os.utime(path, (float(i * 10), float(i * 10)))

        # 8 non-sacred files total, keep_n=5 => delete 3
        # .regressed files should be deleted first
        deleted = dr.rotate_checkpoints(keep_n=5)

        remaining = set(os.listdir(self._ckpt_dir))
        self.assertIn("checkpoint_best.pt", remaining)

        # All 3 .regressed files should be deleted
        regressed_remaining = [f for f in remaining if f.endswith(".regressed")]
        self.assertEqual(len(regressed_remaining), 0)

        # All 5 regular .pt files should remain (not sacred, not regressed)
        regular_remaining = [
            f
            for f in remaining
            if f.endswith(".pt")
            and not f.endswith(".regressed")
            and f not in ("checkpoint_best.pt", "checkpoint_current.pt")
        ]
        self.assertEqual(len(regular_remaining), 5)

        self.assertEqual(len(deleted), 3)


# ══════════════════════════════════════════════════════════════════════
# Recurrent data rotation tests
# ══════════════════════════════════════════════════════════════════════


class TestRotateRecurrentData(unittest.TestCase):
    """Tests for rotate_recurrent_data."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._recurrent_dir = os.path.join(self._tmpdir, "recurrent")

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_recurrent(self, content=b"line1\nline2\n"):
        """Create recurrent_data.jsonl with given content."""
        os.makedirs(self._recurrent_dir, exist_ok=True)
        path = os.path.join(self._recurrent_dir, "recurrent_data.jsonl")
        with open(path, "wb") as f:
            f.write(content)
        return path

    @patch("data_rotation.M1_RECURRENT", new_callable=lambda: None)
    def test_rotate_recurrent_data_under_limit(self, _):
        import data_rotation as dr

        dr.M1_RECURRENT = self._recurrent_dir
        self._make_recurrent(b"small data\n")

        result = dr.rotate_recurrent_data(max_mb=100)
        self.assertFalse(result["rotated"])
        # Original file should still exist
        self.assertTrue(os.path.isfile(os.path.join(self._recurrent_dir, "recurrent_data.jsonl")))

    @patch("data_rotation.M1_RECURRENT", new_callable=lambda: None)
    def test_rotate_recurrent_data_over_limit(self, _):
        import data_rotation as dr

        dr.M1_RECURRENT = self._recurrent_dir

        # Create file just over 100MB (100MB + 1 byte)
        data_path = self._make_recurrent(b"")
        with open(data_path, "r+b") as f:
            f.seek(100 * 1024 * 1024)  # 100MB
            f.write(b"\n")
            f.truncate()

        result = dr.rotate_recurrent_data(max_mb=100)
        self.assertTrue(result["rotated"])
        self.assertIn("backup_path", result)
        self.assertTrue(os.path.isfile(result["backup_path"]))
        # New empty file should exist
        main_path = os.path.join(self._recurrent_dir, "recurrent_data.jsonl")
        self.assertTrue(os.path.isfile(main_path))
        self.assertEqual(os.path.getsize(main_path), 0)

    @patch("data_rotation.M1_RECURRENT", new_callable=lambda: None)
    def test_rotate_recurrent_data_limits_backups(self, _):
        import data_rotation as dr

        dr.M1_RECURRENT = self._recurrent_dir
        os.makedirs(self._recurrent_dir, exist_ok=True)

        # Create 5 existing backups
        for i in range(1, 6):
            backup_name = f"recurrent_data.jsonl.2024010{i}_120000"
            path = os.path.join(self._recurrent_dir, backup_name)
            with open(path, "wb") as f:
                f.write(b"backup data\n")

        # Create main file over limit
        main_path = os.path.join(self._recurrent_dir, "recurrent_data.jsonl")
        with open(main_path, "wb") as f:
            f.seek(100 * 1024 * 1024)
            f.write(b"\n")
            f.truncate()

        dr.rotate_recurrent_data(max_mb=100)

        backups = [
            f
            for f in os.listdir(self._recurrent_dir)
            if f.startswith("recurrent_data.jsonl.") and not f.endswith(".jsonl")
        ]
        # Should keep max 3 backups
        self.assertLessEqual(len(backups), 3)

    @patch("data_rotation.M1_RECURRENT", new_callable=lambda: None)
    def test_rotate_recurrent_data_missing_file(self, _):
        import data_rotation as dr

        dr.M1_RECURRENT = self._recurrent_dir
        os.makedirs(self._recurrent_dir, exist_ok=True)

        result = dr.rotate_recurrent_data(max_mb=100)
        self.assertFalse(result["rotated"])


# ══════════════════════════════════════════════════════════════════════
# Archive rotation tests
# ══════════════════════════════════════════════════════════════════════


class TestRotateArchives(unittest.TestCase):
    """Tests for rotate_archives."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("data_rotation.PROJECT_ROOT", new_callable=lambda: None)
    def test_rotate_archives_keeps_n_newest(self, _):
        import data_rotation as dr

        dr.PROJECT_ROOT = self._tmpdir

        # Create 25 cycle directories with increasing mtimes
        for i in range(1, 26):
            cycle_dir = os.path.join(self._tmpdir, f"cycle_{i:03d}")
            os.makedirs(cycle_dir)
            # Touch a file inside so it has a meaningful mtime
            fpath = os.path.join(cycle_dir, "data.txt")
            with open(fpath, "w") as f:
                f.write(f"cycle {i}")
            os.utime(fpath, (float(i * 100), float(i * 100)))

        deleted = dr.rotate_archives(keep_n=20)

        remaining = [d for d in os.listdir(self._tmpdir) if d.startswith("cycle_")]
        self.assertEqual(len(remaining), 20)
        self.assertEqual(len(deleted), 5)

    @patch("data_rotation.PROJECT_ROOT", new_callable=lambda: None)
    def test_rotate_archives_empty(self, _):
        import data_rotation as dr

        dr.PROJECT_ROOT = self._tmpdir

        deleted = dr.rotate_archives(keep_n=20)
        self.assertEqual(deleted, [])


# ══════════════════════════════════════════════════════════════════════
# Report rotation tests
# ══════════════════════════════════════════════════════════════════════


class TestRotateReports(unittest.TestCase):
    """Tests for rotate_reports."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("data_rotation.PROJECT_ROOT", new_callable=lambda: None)
    def test_rotate_reports_keeps_n_newest(self, _):
        import data_rotation as dr

        dr.PROJECT_ROOT = self._tmpdir

        # Create the protected file (progress_report.json) — must not be deleted
        protected = os.path.join(self._tmpdir, "progress_report.json")
        with open(protected, "w") as f:
            f.write("{}")

        # Create 60 report files with increasing mtimes
        for i in range(1, 61):
            report_path = os.path.join(self._tmpdir, f"progress_report_cycle_{i}.json")
            with open(report_path, "w") as f:
                f.write(f'{{"cycle": {i}}}')
            os.utime(report_path, (float(i * 100), float(i * 100)))

        deleted = dr.rotate_reports(keep_n=50)

        remaining = [f for f in os.listdir(self._tmpdir) if f.startswith("progress_report")]
        # 50 numbered reports + 1 protected progress_report.json = 51
        self.assertIn("progress_report.json", remaining)
        self.assertEqual(len(remaining) - 1, 50)  # 50 numbered reports
        self.assertEqual(len(deleted), 10)


# ══════════════════════════════════════════════════════════════════════
# Integration: run_all_rotations
# ══════════════════════════════════════════════════════════════════════


class TestRunAllRotations(unittest.TestCase):
    """Integration test for run_all_rotations."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._ckpt_dir = os.path.join(self._tmpdir, "checkpoints")
        self._recurrent_dir = os.path.join(self._tmpdir, "recurrent")
        os.makedirs(self._ckpt_dir, exist_ok=True)
        os.makedirs(self._recurrent_dir, exist_ok=True)

    def tearDown(self):
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    @patch("data_rotation.PROJECT_ROOT", new_callable=lambda: None)
    @patch("data_rotation.M1_RECURRENT", new_callable=lambda: None)
    @patch("data_rotation.M1_CHECKPOINTS", new_callable=lambda: None)
    def test_run_all_rotations(self, _, _2, _3):
        import data_rotation as dr

        dr.M1_CHECKPOINTS = self._ckpt_dir
        dr.M1_RECURRENT = self._recurrent_dir
        dr.PROJECT_ROOT = self._tmpdir

        # Create checkpoint files: 8 regular .pt files
        for i in range(1, 9):
            path = os.path.join(self._ckpt_dir, f"model_{i:02d}.pt")
            with open(path, "wb") as f:
                f.write(b"\x00" * 1024)
            os.utime(path, (float(i * 100), float(i * 100)))

        # Create recurrent data file under limit (no rotation expected)
        rpath = os.path.join(self._recurrent_dir, "recurrent_data.jsonl")
        with open(rpath, "w") as f:
            f.write('{"cycle": 1}\n')

        # Create archive dirs: 3 cycle dirs (all under keep_n=20)
        for i in range(1, 4):
            os.makedirs(os.path.join(self._tmpdir, f"cycle_{i:03d}"))

        # Create report files: 5 numbered reports (under keep_n=50)
        for i in range(1, 6):
            path = os.path.join(self._tmpdir, f"progress_report_cycle_{i}.json")
            with open(path, "w") as f:
                f.write(f'{{"cycle": {i}}}')

        result = dr.run_all_rotations()

        self.assertIsInstance(result, dict)
        self.assertIn("checkpoints", result)
        self.assertIn("recurrent", result)
        self.assertIn("archives", result)
        self.assertIn("reports", result)

        # Checkpoints: 8 created, keep_n=5 => deleted 3
        self.assertEqual(result["checkpoints"], 3)
        # Recurrent: under limit => rotated=0
        self.assertEqual(result["recurrent"], 0)
        # Archives: 3 created, keep_n=20 => deleted 0
        self.assertEqual(result["archives"], 0)
        # Reports: 5 created, keep_n=50 => deleted 0
        self.assertEqual(result["reports"], 0)


if __name__ == "__main__":
    unittest.main()
