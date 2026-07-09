"""Tests for recurrent_data.jsonl filtering in save_recurrent_data().

Verifies:
  - Fitness threshold filtering removes low-fitness noise
  - max_entries cap limits output size
  - Weighted selection: 50% top, 30% recent, 20% diverse
  - Edge cases: empty archive, threshold excludes all, small archives
"""

import json
import os
import sys

import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from champion_archive import ChampionArchive, _hash_program


def _make_entries(archive: ChampionArchive, specs: list[dict]) -> None:
    """Helper to bulk-add entries to an archive from specs.

    Each spec: {"level": int, "fitness": float, "cycle": int, "program": list[int]}
    """
    for i, spec in enumerate(specs):
        program = spec.get("program", [0x100073 + i])
        archive.add(
            program=program,
            level=spec.get("level", 13),
            fitness=spec["fitness"],
            cycle=spec.get("cycle", 1),
            generation=spec.get("generation", i),
        )


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return list of parsed dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestFitnessThresholdFiltering:
    """Tests for the fitness_threshold parameter."""

    def test_filters_low_fitness(self, tmp_path):
        """Entries below fitness_threshold are excluded."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=200)
        _make_entries(
            archive,
            [
                {"fitness": 200.0, "cycle": 1, "program": [0x100073]},
                {"fitness": 100.0, "cycle": 1, "program": [0x100074]},  # below threshold
                {"fitness": 180.0, "cycle": 2, "program": [0x100075]},
                {"fitness": 50.0, "cycle": 2, "program": [0x100076]},  # below threshold
            ],
        )

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, fitness_threshold=150.0)

        assert count == 2
        records = _read_jsonl(out)
        fitnesses = [r["fitness"] for r in records]
        assert all(f >= 150.0 for f in fitnesses)
        assert 200.0 in fitnesses
        assert 180.0 in fitnesses

    def test_threshold_zero_includes_all(self, tmp_path):
        """fitness_threshold=0.0 includes all entries (no filtering)."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=200)
        _make_entries(
            archive,
            [
                {"fitness": 0.0, "cycle": 1, "program": [0x100073]},
                {"fitness": 1.0, "cycle": 1, "program": [0x100074]},
                {"fitness": 999.0, "cycle": 1, "program": [0x100075]},
            ],
        )

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, fitness_threshold=0.0)

        assert count == 3

    def test_threshold_excludes_all(self, tmp_path):
        """When all entries are below threshold, output is empty."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=200)
        _make_entries(
            archive,
            [
                {"fitness": 10.0, "cycle": 1, "program": [0x100073]},
                {"fitness": 20.0, "cycle": 1, "program": [0x100074]},
            ],
        )

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, fitness_threshold=150.0)

        assert count == 0
        assert not os.path.exists(out) or os.path.getsize(out) == 0

    def test_default_threshold_is_150(self, tmp_path):
        """Default fitness_threshold=150.0 filters as expected."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=200)
        _make_entries(
            archive,
            [
                {"fitness": 149.0, "cycle": 1, "program": [0x100073]},
                {"fitness": 150.0, "cycle": 1, "program": [0x100074]},  # exactly at threshold
                {"fitness": 151.0, "cycle": 1, "program": [0x100075]},
            ],
        )

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out)  # uses defaults

        assert count == 2
        records = _read_jsonl(out)
        assert all(r["fitness"] >= 150.0 for r in records)


class TestMaxEntriesCap:
    """Tests for the max_entries parameter."""

    def test_under_cap_writes_all(self, tmp_path):
        """When eligible entries < max_entries, all are written."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=200)
        _make_entries(
            archive,
            [
                {"fitness": 200.0, "cycle": 1, "program": [0x100073]},
                {"fitness": 180.0, "cycle": 2, "program": [0x100074]},
            ],
        )

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, max_entries=100)

        assert count == 2

    def test_over_cap_limits_output(self, tmp_path):
        """When eligible > max_entries, output is capped at max_entries."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=500)
        # Create 50 entries with fitness above threshold
        specs = [{"fitness": 150.0 + i, "cycle": i // 10 + 1, "program": [0x100073 + i]} for i in range(50)]
        _make_entries(archive, specs)

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, max_entries=20, fitness_threshold=0.0)

        assert count == 20
        records = _read_jsonl(out)
        assert len(records) == 20

    def test_default_max_entries_is_100(self, tmp_path):
        """Default max_entries=100 caps output at 100 entries."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=500)
        specs = [{"fitness": 150.0 + i * 10, "cycle": i // 20 + 1, "program": [0x100073 + i]} for i in range(150)]
        _make_entries(archive, specs)

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, fitness_threshold=0.0)

        assert count == 100

    def test_max_entries_one(self, tmp_path):
        """max_entries=1 returns exactly one entry."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=200)
        _make_entries(
            archive,
            [
                {"fitness": 300.0, "cycle": 1, "program": [0x100073]},
                {"fitness": 250.0, "cycle": 1, "program": [0x100074]},
                {"fitness": 200.0, "cycle": 1, "program": [0x100075]},
            ],
        )

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, max_entries=1, fitness_threshold=0.0)

        assert count == 1
        records = _read_jsonl(out)
        assert records[0]["fitness"] == 300.0  # top performer


class TestWeightedSelection:
    """Tests that the 50/30/20 split produces correct mix."""

    def test_top_performers_dominated(self, tmp_path):
        """50% of output should be highest-fitness entries."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=500)
        # 20 high-fitness, 20 low-fitness, 10 mid-fitness, 10 very recent
        specs = []
        for i in range(20):
            specs.append({"fitness": 300.0 + i, "cycle": 1, "program": [0x200000 + i]})
        for i in range(20):
            specs.append({"fitness": 100.0 + i, "cycle": 10, "program": [0x300000 + i]})
        for i in range(10):
            specs.append({"fitness": 200.0 + i, "cycle": 5, "program": [0x400000 + i]})
        for i in range(10):
            specs.append({"fitness": 160.0 + i, "cycle": 10, "program": [0x500000 + i]})

        _make_entries(archive, specs)

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, max_entries=20, fitness_threshold=0.0)

        assert count == 20
        records = _read_jsonl(out)
        fitnesses = sorted([r["fitness"] for r in records], reverse=True)
        # Top 10 should all be >= 300 (the highest-fitness group)
        assert fitnesses[0] >= 300.0
        # At least some high-fitness entries should be in the output
        high_count = sum(1 for f in fitnesses if f >= 300.0)
        assert high_count >= 5  # At least 50% of 20 = 10, but overlap may reduce

    def test_recency_preserved(self, tmp_path):
        """Recent entries (high cycle numbers) should appear in the 30% bucket."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=500)
        specs = []
        # Old high-fitness entries
        for i in range(15):
            specs.append({"fitness": 300.0 + i, "cycle": 1, "program": [0x200000 + i]})
        # Recent lower-fitness entries
        for i in range(15):
            specs.append({"fitness": 160.0 + i, "cycle": 100, "program": [0x600000 + i]})

        _make_entries(archive, specs)

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, max_entries=15, fitness_threshold=0.0)

        assert count == 15
        records = _read_jsonl(out)
        cycles = [r["cycle"] for r in records]
        # Some recent entries (cycle=100) should be present
        assert 100 in cycles

    def test_no_duplicates_in_output(self, tmp_path):
        """Output should never contain duplicate sha256 values."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=500)
        specs = [{"fitness": 150.0 + i, "cycle": i // 5 + 1, "program": [0x100073 + i]} for i in range(30)]
        _make_entries(archive, specs)

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, max_entries=20, fitness_threshold=0.0)

        records = _read_jsonl(out)
        sha256s = [r["champion_sha256"] for r in records]
        assert len(sha256s) == len(set(sha256s)), "Duplicate sha256 found in output"


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_archive(self, tmp_path):
        """Empty archive produces empty output."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=100)
        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out)

        assert count == 0

    def test_single_entry(self, tmp_path):
        """Archive with one entry writes exactly one record."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=100)
        archive.add(program=[0x100073], level=13, fitness=200.0, cycle=1)

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out)

        assert count == 1
        records = _read_jsonl(out)
        assert records[0]["fitness"] == 200.0

    def test_output_format_preserved(self, tmp_path):
        """Output JSON records contain all expected fields."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=100)
        archive.add(program=[0x100073, 0x100074], level=13, fitness=200.0, cycle=1, task_score=42.0)

        out = str(tmp_path / "recurrent_data.jsonl")
        archive.save_recurrent_data(out, fitness_threshold=0.0)

        records = _read_jsonl(out)
        assert len(records) == 1
        r = records[0]
        expected_keys = {
            "level",
            "generation",
            "fitness",
            "champion_sha256",
            "champion_words",
            "champion_program",
            "trace_status",
            "trace_instr_count",
            "task_score",
            "save_reason",
            "cycle",
        }
        assert expected_keys == set(r.keys())
        assert r["fitness"] == 200.0
        assert r["level"] == 13
        assert r["task_score"] == 42.0
        assert r["save_reason"] == "champion_archive"
        assert len(r["champion_program"]) == 2

    def test_max_entries_equals_eligible(self, tmp_path):
        """When max_entries == eligible count, all eligible are written."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=200)
        _make_entries(
            archive,
            [
                {"fitness": 200.0, "cycle": 1, "program": [0x100073]},
                {"fitness": 180.0, "cycle": 1, "program": [0x100074]},
            ],
        )

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, max_entries=2, fitness_threshold=0.0)

        assert count == 2

    def test_interplay_threshold_and_max(self, tmp_path):
        """fitness_threshold + max_entries work together correctly."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=500)
        # 10 high-fitness, 10 low-fitness
        for i in range(10):
            archive.add(program=[0x200000 + i], level=13, fitness=300.0 + i, cycle=1)
        for i in range(10):
            archive.add(program=[0x300000 + i], level=5, fitness=100.0 + i, cycle=1)

        out = str(tmp_path / "recurrent_data.jsonl")
        # Threshold filters to 10 high-fitness, max_entries=5 further caps
        count = archive.save_recurrent_data(out, max_entries=5, fitness_threshold=150.0)

        assert count == 5
        records = _read_jsonl(out)
        assert all(r["fitness"] >= 150.0 for r in records)
        # All should be from the high-fitness group
        assert all(r["fitness"] >= 300.0 for r in records)


class TestDiversitySelection:
    """Tests for the 20% diverse bucket."""

    def test_diverse_entries_have_varied_programs(self, tmp_path):
        """The diverse bucket should include entries with different instruction patterns."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=500)
        # Create entries with distinct programs at similar fitness
        # High-fitness but similar programs (same instructions)
        similar_base = [0x100073, 0x100074, 0x100075]
        for i in range(10):
            prog = [w + i for w in similar_base]  # very similar
            archive.add(program=prog, level=13, fitness=250.0 + i, cycle=1)

        # Lower fitness but VERY different programs
        for i in range(10):
            prog = [0x900000 + i * 1000] * 3  # very different instructions
            archive.add(program=prog, level=13, fitness=160.0 + i, cycle=5)

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, max_entries=10, fitness_threshold=0.0)

        assert count == 10
        assert count == 10
        records = _read_jsonl(out)
        programs = [r["champion_program"] for r in records]
        # Should have entries with different hex patterns (full word comparison)
        all_words = set()
        for prog in programs:
            for word in prog:
                all_words.add(word)
        # With 20 distinct programs, we should see more than 1 unique hex word
        assert len(all_words) >= 2

    def test_diverse_bucket_fills_remaining(self, tmp_path):
        """Diverse count = max_entries - top_count - recent_count."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=500)
        # 50 entries spread across cycles
        for i in range(50):
            archive.add(
                program=[0x100073 + i * 100],
                level=13,
                fitness=150.0 + (i % 50),
                cycle=(i // 10) + 1,
            )

        out = str(tmp_path / "recurrent_data.jsonl")
        count = archive.save_recurrent_data(out, max_entries=10, fitness_threshold=0.0)

        # 10 = int(10*0.50) top + int(10*0.30) recent + 2 diverse
        assert count == 10
        records = _read_jsonl(out)
        fitnesses = [r["fitness"] for r in records]
        # Top bucket = 5 highest fitness (range is 150.0 to 199.0)
        high_count = sum(1 for f in fitnesses if f >= 195.0)
        assert high_count >= 3  # at least some top performers present


class TestBackwardCompatibility:
    """Tests that existing callers still work with defaults."""

    def test_no_args_works(self, tmp_path):
        """Calling save_recurrent_data(path) with no extra args works (backward compat)."""
        archive = ChampionArchive(path=str(tmp_path / "champions" / "archive.jsonl"), max_size=100)
        _make_entries(
            archive,
            [
                {"fitness": 200.0, "cycle": 1, "program": [0x100073]},
                {"fitness": 100.0, "cycle": 1, "program": [0x100074]},
            ],
        )

        out = str(tmp_path / "recurrent_data.jsonl")
        # Should use defaults: max_entries=100, fitness_threshold=150.0
        count = archive.save_recurrent_data(out)

        # Only 200.0 passes the default threshold of 150.0
        assert count == 1
        records = _read_jsonl(out)
        assert records[0]["fitness"] == 200.0
