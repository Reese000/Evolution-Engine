"""Integration tests for Evolution Engine core flows.

Tests end-to-end functionality:
  1. Full evolution cycle (init -> mutate -> select -> evaluate -> repeat)
  2. CycleState checkpoint save/restore roundtrip
  3. Config validation (invalid configs are rejected)

These tests use lightweight mocks and stubs — no GPU or full model needed.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time

import pytest

# Path setup
from paths import ensure_project_on_path

ensure_project_on_path()

from config import Config, EvolutionSection, RetrainSection

# ══════════════════════════════════════════════════════════════════════
# 1. Full evolution cycle integration
# ══════════════════════════════════════════════════════════════════════


class TestEvolutionCycleIntegration:
    """Integration test: init -> mutate -> select -> evaluate -> repeat."""

    def test_mini_evolution_cycle(self):
        """Run a minimal evolution cycle: init population, mutate, select, evaluate."""
        from mutation_proposer import EBREAK, generate_seed_program, validate_mutation
        from qd_archive import MAPElitesArchive

        rng = random.Random(42)
        pop_size = 10

        # 1. Initialize population
        population = []
        for _ in range(pop_size):
            prog = generate_seed_program(length=5, rng=rng)
            population.append(prog)

        assert len(population) == pop_size

        # 2. Mutate (each program gets a mutation)
        mutated = []
        for prog in population:
            # Simple mutation: duplicate last instruction before EBREAK
            ebreak_idx = len(prog) - 1
            mutated_prog = prog[:ebreak_idx] + [prog[ebreak_idx - 1]] + prog[ebreak_idx:]
            mutated.append(mutated_prog)

        assert len(mutated) == pop_size

        # 3. Evaluate (dummy fitness: length-based)
        scored = []
        for prog in mutated:
            valid, _ = validate_mutation(prog)
            fitness = float(len(prog)) if valid else 0.0
            scored.append((fitness, len(prog), prog, {}))

        assert len(scored) == pop_size

        # 4. Select (tournament selection via simple max)
        scored.sort(key=lambda x: x[0], reverse=True)
        survivors = scored[: pop_size // 2]
        assert len(survivors) == pop_size // 2

        # 5. Archive top solutions
        archive = MAPElitesArchive(bins=5)
        for fitness, instr_count, prog, trace in scored[:3]:
            descriptor = [0.5, 0.5]  # dummy descriptor
            archive.add_solution(descriptor, fitness, prog, trace)

        assert archive.size >= 1
        assert archive.coverage() > 0.0

    def test_multi_generation_evolution(self):
        """Run multiple generations and verify population evolves."""
        from mutation_proposer import EBREAK, generate_seed_program, validate_mutation
        from qd_archive import MAPElitesArchive

        rng = random.Random(42)
        pop_size = 20
        n_generations = 5

        # Initialize
        population = []
        for _ in range(pop_size):
            prog = generate_seed_program(length=3, rng=rng)
            population.append(prog)

        archive = MAPElitesArchive(bins=5)

        for gen in range(n_generations):
            # Mutate
            mutated = []
            for prog in population:
                # Random mutation: add an instruction before EBREAK
                if rng.random() < 0.5 and len(prog) > 1:
                    pos = rng.randint(0, len(prog) - 2)
                    new_instr = generate_seed_program(length=1, rng=rng)[0]
                    mutated_prog = prog[:pos] + [new_instr] + prog[pos:]
                else:
                    mutated_prog = list(prog)
                mutated.append(mutated_prog)

            # Evaluate
            scored = []
            for prog in mutated:
                valid, _ = validate_mutation(prog)
                # Fitness: prefer longer valid programs (simple proxy)
                fitness = len(prog) * 10.0 if valid else 0.0
                scored.append((fitness, len(prog), prog, {}))

            # Select top half
            scored.sort(key=lambda x: x[0], reverse=True)
            survivors = scored[: pop_size // 2]

            # Fill rest with copies of survivors
            new_pop = [list(s[2]) for s in survivors]
            while len(new_pop) < pop_size:
                parent = survivors[rng.randint(0, len(survivors) - 1)]
                new_pop.append(list(parent[2]))
            population = new_pop

            # Archive best
            if scored:
                best = scored[0]
                archive.add_solution([gen / n_generations, 0.5], best[0], best[2], {})

        # After 5 generations, archive should have solutions
        assert archive.size >= 1
        assert archive.coverage() > 0.0

        # All programs should be valid
        for prog in population:
            valid, err = validate_mutation(prog)
            assert valid, f"Invalid program after evolution: {err}"


# ══════════════════════════════════════════════════════════════════════
# 2. Checkpoint save/restore roundtrip (CycleState)
# ══════════════════════════════════════════════════════════════════════


class TestCycleStateRoundtrip:
    """Integration test: CycleState save/load preserves data integrity."""

    def test_save_load_roundtrip(self):
        """Save a CycleState, load it back, verify all fields match."""
        from cycle_state import CycleState

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cycle_state.json")
            cs = CycleState(path)

            # Set custom values
            cs.update(
                current_cycle=15,
                best_fitness=42.5,
                best_level=7,
                champion_program=[1, 2, 3, 4, 5],
                diversity_score=0.35,
                consecutive_stalls=3,
                cycles_completed=15,
                population_size=30,
                metrics_summary={"fitness_trend": "improving", "val_loss_last": 0.01},
            )

            # Save
            assert cs.save() is True
            assert os.path.isfile(path)

            # Load into a fresh instance
            cs2 = CycleState(path)
            assert cs2.load() is True

            # Verify all fields preserved
            assert cs2["current_cycle"] == 15
            assert cs2["best_fitness"] == 42.5
            assert cs2["best_level"] == 7
            assert cs2["champion_program"] == [1, 2, 3, 4, 5]
            assert cs2["diversity_score"] == 0.35
            assert cs2["consecutive_stalls"] == 3
            assert cs2["cycles_completed"] == 15
            assert cs2["population_size"] == 30
            assert cs2["metrics_summary"]["fitness_trend"] == "improving"

    def test_checksum_detects_corruption(self):
        """Tampering with the file causes checksum mismatch on load."""
        from cycle_state import CycleState

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cycle_state.json")
            cs = CycleState(path)
            cs.update(current_cycle=10, best_fitness=5.0)
            cs.save()

            # Tamper with the file
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            data["best_fitness"] = 999.0
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            # Load should detect corruption
            cs2 = CycleState(path)
            result = cs2.load()
            # Either returns False (checksum mismatch) or data is reset
            # Either way, the tampered value should NOT be trusted
            if result:
                # If it loaded, the data should be the default (not tampered)
                assert cs2["best_fitness"] != 999.0

    def test_load_missing_file_returns_false(self):
        """Loading a nonexistent file returns False with default state."""
        from cycle_state import CycleState

        cs = CycleState("/nonexistent/path/cycle_state.json")
        assert cs.load() is False
        assert cs["current_cycle"] == 0

    def test_save_load_preserves_timestamps(self):
        """Save/load roundtrip preserves start_timestamp."""
        from cycle_state import CycleState

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cycle_state.json")
            cs = CycleState(path)
            original_ts = cs["start_timestamp"]
            cs.update(current_cycle=5)
            cs.save()

            cs2 = CycleState(path)
            cs2.load()
            assert cs2["start_timestamp"] == original_ts


# ══════════════════════════════════════════════════════════════════════
# 3. Config validation
# ══════════════════════════════════════════════════════════════════════


class TestConfigValidation:
    """Integration test: invalid configs are rejected at startup."""

    def test_valid_config_loads_without_error(self):
        """Default config loads and validates without raising."""
        cfg = Config(config_path=None)
        assert cfg.evolution.population_size >= 1
        assert 0.0 <= cfg.evolution.mutation_rate <= 1.0

    def test_negative_population_size_rejected(self):
        """population_size < 1 raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"evolution": {"population_size": -5}}, f)
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="population_size must be >= 1"):
                Config(config_path=config_path)
        finally:
            os.unlink(config_path)

    def test_mutation_rate_out_of_range_rejected(self):
        """mutation_rate > 1.0 raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"evolution": {"mutation_rate": 1.5}}, f)
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="mutation_rate must be in"):
                Config(config_path=config_path)
        finally:
            os.unlink(config_path)

    def test_negative_temperature_rejected(self):
        """temperature_start < 0 raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"evolution": {"temperature_start": -1.0}}, f)
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="temperature_start must be in"):
                Config(config_path=config_path)
        finally:
            os.unlink(config_path)

    def test_negative_retrain_lr_rejected(self):
        """retrain_lr <= 0 raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"retrain": {"retrain_lr": -0.001}}, f)
            f.flush()
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="retrain_lr must be > 0"):
                Config(config_path=config_path)
        finally:
            os.unlink(config_path)

    def test_config_warnings_for_small_population(self):
        """population_size < 10 produces a validation warning."""
        evo = EvolutionSection()
        evo.population_size = 5
        warnings = evo.validate_config()
        assert any("population_size" in w for w in warnings)

    def test_config_warnings_for_no_archives(self):
        """All archives disabled produces a warning."""
        evo = EvolutionSection()
        evo.use_novelty_archive = False
        evo.use_qd_archive = False
        evo.use_strategy_archive = False
        warnings = evo.validate_config()
        assert any("No diversity mechanisms" in w for w in warnings)

    def test_config_to_dict_roundtrip(self):
        """Config.to_dict produces serializable dict with all sections."""
        cfg = Config(config_path=None)
        d = cfg.to_dict()
        assert "evolution" in d
        assert "retrain" in d
        assert "meta_controller" in d
        assert "autonomous_runner" in d
        assert isinstance(d["evolution"]["population_size"], int)
        assert isinstance(d["retrain"]["retrain_lr"], float)
