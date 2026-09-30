"""Round 8 improvements — convergence quality and diversity trend detection.

Tests for:
  1. Diversity trend detection catches early erosion before absolute threshold
  2. Stochastic perturbation uses correct EBREAK constant (no NameError)
  3. Diversity trend edge cases: no false positives, no division by zero
  4. MetaController diversity trend in tune_evolution_params triggers injection
  5. Single-pass stagnation detection correctness across all indicators
  6. StrategyArchive _rebuild_indices preserves all data
  7. CycleState checksum roundtrip integrity
  8. StrategyArchive record_batch bulk index maintenance
  9. MetaController telemetry cache invalidation after update_cycle
  10. Diversity trend does not fire when first diversity is zero
"""

import os
import random
import tempfile
from unittest.mock import patch

import pytest

# ── Diversity trend detection ──────────────────────────────────────────


class TestDiversityTrendDetection:
    """Verify detect_stagnation catches early diversity erosion."""

    def _make_controller(self):
        from meta_controller import MetaController

        return MetaController(checkpoint_dir="/tmp/mc_div_test")

    def test_trend_detected_when_erosion_40pct(self):
        """Diversity dropping >40% across window with low current diversity
        should trigger DIVERSITY_COLLAPSE even if min is above absolute threshold."""
        from meta_controller import StagnationResult, TelemetrySnapshot

        mc = self._make_controller()
        # Diversity starts at 0.20, ends at 0.08 (60% drop)
        # Absolute threshold is 0.05 (default), current is 0.08 > 0.05
        # But 0.08 < 0.05 * 1.5 = 0.075... so trend threshold not met
        # Let's use: first=0.30, last=0.15, min=0.10, dc_thresh=0.05
        # 0.15 < 0.30 * 0.6 = 0.18 (yes), 0.15 < 0.05 * 1.5 = 0.075 (no)
        # Need: last < first * 0.6 AND last < dc_thresh * 1.5
        # first=0.30, last=0.05, dc_thresh=0.05: 0.05 < 0.18 yes, 0.05 < 0.075 yes
        # But 0.05 == dc_thresh so the absolute threshold ALREADY fires
        # Use: first=0.30, last=0.06, min_above=0.07, dc_thresh=0.05
        # 0.06 < 0.18 yes, 0.06 < 0.075 yes -> trend fires, but abs min=0.07 > 0.05
        # Need diversity to go below dc_thresh for abs to fire, so set min above.
        # The trend checks first/last diversity, not min.
        snaps = [
            TelemetrySnapshot(
                cycle=1,
                fitness=-1.0,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.30,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=2,
                fitness=-0.9,
                level=5,
                champion_words=10,
                val_loss=1.9,
                diversity=0.20,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=3,
                fitness=-0.8,
                level=5,
                champion_words=10,
                val_loss=1.8,
                diversity=0.12,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=4,
                fitness=-0.7,
                level=5,
                champion_words=10,
                val_loss=1.7,
                diversity=0.06,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
        ]
        for s in snaps:
            mc.update_cycle(s)
        result = mc.detect_stagnation()
        # first.diversity=0.30, last.diversity=0.06
        # 0.06 < 0.30 * 0.6 = 0.18 -> yes
        # 0.06 < 0.05 * 1.5 = 0.075 -> yes -> trend fires
        assert result == StagnationResult.DIVERSITY_COLLAPSE

    def test_no_false_positive_stable_diversity(self):
        """Stable diversity should NOT trigger diversity collapse."""
        from meta_controller import StagnationResult, TelemetrySnapshot

        mc = self._make_controller()
        for c in range(1, 6):
            snap = TelemetrySnapshot(
                cycle=c,
                fitness=-1.0 + c * 0.1,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.30,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        result = mc.detect_stagnation()
        assert result == StagnationResult.NORMAL

    def test_trend_does_not_fire_when_first_diversity_zero(self):
        """When first diversity is 0, division avoidance should prevent trend."""
        from meta_controller import StagnationResult, TelemetrySnapshot

        mc = self._make_controller()
        snaps = [
            TelemetrySnapshot(
                cycle=1,
                fitness=-1.0,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.0,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=2,
                fitness=-0.9,
                level=5,
                champion_words=10,
                val_loss=1.9,
                diversity=0.0,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=3,
                fitness=-0.8,
                level=5,
                champion_words=10,
                val_loss=1.8,
                diversity=0.0,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=4,
                fitness=-0.7,
                level=5,
                champion_words=10,
                val_loss=1.7,
                diversity=0.0,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
        ]
        for s in snaps:
            mc.update_cycle(s)
        result = mc.detect_stagnation()
        # first=0.0 -> trend check skipped (div_first > 0 is False)
        # abs threshold: div_min=0.0 < 0.05 -> fires
        assert result == StagnationResult.DIVERSITY_COLLAPSE

    def test_no_false_positive_mild_decline(self):
        """A mild diversity decline (<40%) should not trigger trend detection."""
        from meta_controller import StagnationResult, TelemetrySnapshot

        mc = self._make_controller()
        snaps = [
            TelemetrySnapshot(
                cycle=1,
                fitness=-1.0,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.30,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=2,
                fitness=-0.9,
                level=5,
                champion_words=10,
                val_loss=1.9,
                diversity=0.25,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=3,
                fitness=-0.8,
                level=5,
                champion_words=10,
                val_loss=1.8,
                diversity=0.20,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=4,
                fitness=-0.7,
                level=5,
                champion_words=10,
                val_loss=1.7,
                diversity=0.20,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
        ]
        for s in snaps:
            mc.update_cycle(s)
        result = mc.detect_stagnation()
        # first=0.30, last=0.20: 0.20 < 0.18? No -> trend doesn't fire
        # abs: min=0.20 > 0.05 -> doesn't fire
        assert result == StagnationResult.NORMAL


# ── Stochastic perturbation EBREAK fix ────────────────────────────────


class TestStochasticPerturbationEBREAKFix:
    """Verify stochastic_perturbation uses correct EBREAK constant,
    not an undefined ebreak_val variable."""

    def test_stochastic_perturbation_runs_without_error(self):
        """stochastic_perturbation should not raise NameError."""
        from mutation_proposer import EBREAK, DirectedMutator, make_rv32i

        rng = random.Random(42)
        prog = [make_rv32i("ADDI", rd=5, rs1=0, imm=10), EBREAK]
        # Should run without NameError on ebreak_val
        result, _desc = DirectedMutator.stochastic_perturbation(prog, rng)
        assert len(result) > 0
        assert result[-1] == EBREAK  # should always have EBREAK at end

    def test_stochastic_perturbation_adds_ebreak_if_missing(self):
        """When input program lacks EBREAK, stochastic_perturbation adds one."""
        from mutation_proposer import EBREAK, DirectedMutator, make_rv32i

        rng = random.Random(42)
        prog = [make_rv32i("ADDI", rd=5, rs1=0, imm=10)]
        result, _desc = DirectedMutator.stochastic_perturbation(prog, rng)
        assert len(result) > 0
        assert result[-1] == EBREAK

    def test_stochastic_perturbation_empty_program(self):
        """Empty program should be regenerated with EBREAK."""
        from mutation_proposer import EBREAK, DirectedMutator

        rng = random.Random(42)
        result, desc = DirectedMutator.stochastic_perturbation([], rng)
        assert len(result) == 2  # one random instr + EBREAK
        assert result[-1] == EBREAK
        assert "regenerate" in desc


# ── Single-pass stagnation detection ──────────────────────────────────


class TestSinglePassStagnation:
    """Verify the single-pass stagnation detection correctly classifies
    all four indicators from one iteration."""

    def test_fitness_plateau_only(self):
        """Flat fitness with enough cycles should trigger FITNESS_PLATEAU."""
        from meta_controller import MetaController, StagnationResult, TelemetrySnapshot

        mc = MetaController(checkpoint_dir="/tmp/mc_sp_test", fitness_plateau_cycles=3)
        snaps = [
            TelemetrySnapshot(
                cycle=1,
                fitness=-2.0,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.3,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=2,
                fitness=-2.0,
                level=6,
                champion_words=10,
                val_loss=1.9,
                diversity=0.3,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=3,
                fitness=-2.0,
                level=7,
                champion_words=10,
                val_loss=1.8,
                diversity=0.3,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=4,
                fitness=-2.0,
                level=7,
                champion_words=10,
                val_loss=1.7,
                diversity=0.3,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
        ]
        for s in snaps:
            mc.update_cycle(s)
        result = mc.detect_stagnation()
        # fitness: cycles_since_fitness_improvement=3 >= 3, range=0.0 < 0.1 -> plateau
        # level: cycles_since_level_up=1 < 5 -> no stall
        # val_loss: decreasing -> no regression
        # diversity: 0.3 > 0.05 -> no collapse
        assert result == StagnationResult.FITNESS_PLATEAU

    def test_compound_stall_multiple_indicators(self):
        """Multiple indicators should trigger COMPOUND_STALL."""
        from meta_controller import MetaController, StagnationResult, TelemetrySnapshot

        mc = MetaController(
            checkpoint_dir="/tmp/mc_compound_test",
            fitness_plateau_cycles=3,
            level_stall_cycles=3,
            diversity_collapse_threshold=0.30,
        )
        snaps = [
            TelemetrySnapshot(
                cycle=1,
                fitness=-2.0,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.30,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=2,
                fitness=-2.0,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.20,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=3,
                fitness=-2.0,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.15,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
            TelemetrySnapshot(
                cycle=4,
                fitness=-2.0,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.10,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            ),
        ]
        for s in snaps:
            mc.update_cycle(s)
        result = mc.detect_stagnation()
        # fitness: 3 cycles no improvement, range=0 -> plateau (ind_fitness=True)
        # level: 3 cycles no level up -> stall (ind_level=True)
        # diversity: min=0.10 < 0.30 -> collapse (ind_diversity=True)
        # >= 2 indicators -> COMPOUND_STALL
        assert result == StagnationResult.COMPOUND_STALL


# ── StrategyArchive _rebuild_indices ──────────────────────────────────


class TestStrategyArchiveRebuildIndices:
    """Verify _rebuild_indices preserves all data and indices."""

    def test_rebuild_preserves_record_count(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for i in range(100):
            sa.record("RANDOM" if i % 2 == 0 else "HYBRID", level=i % 5, success=i % 3 == 0, generation=i)
        assert sa.total_records() == 100
        sa._rebuild_indices()
        assert sa.total_records() == 100

    def test_rebuild_preserves_success_rates(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for i in range(50):
            sa.record("RANDOM", level=0, success=True, generation=i)
        rate_before = sa.success_rate("RANDOM", level=0)
        sa._rebuild_indices()
        rate_after = sa.success_rate("RANDOM", level=0)
        assert abs(rate_before - rate_after) < 1e-9

    def test_rebuild_preserves_level_strategies(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        sa.record("RANDOM", level=0, success=True)
        sa.record("HYBRID", level=1, success=True)
        sa._rebuild_indices()
        best_l0 = sa.best_for_level(0)
        best_l1 = sa.best_for_level(1)
        assert best_l0 is not None
        assert best_l0[0] == "RANDOM"
        assert best_l1 is not None
        assert best_l1[0] == "HYBRID"

    def test_rebuild_after_load(self):
        """After loading from a file, indices should be rebuilt."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        for i in range(20):
            sa.record("TASK_AWARE", level=i % 3, success=i % 4 == 0, generation=i)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            sa.save(path)
            sa2 = StrategyArchive()
            sa2.load(path)
            assert sa2.total_records() == 20
            assert sa2.success_rate("TASK_AWARE", level=0) > 0
        finally:
            os.unlink(path)


# ── StrategyArchive record_batch index maintenance ────────────────────


class TestStrategyArchiveRecordBatch:
    """Verify record_batch maintains indices correctly."""

    def test_batch_index_consistency(self):
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        batch = [
            {"strategy": "RANDOM", "level": 0, "success": True, "gen": 0},
            {"strategy": "HYBRID", "level": 1, "success": False, "gen": 0},
            {"strategy": "RANDOM", "level": 0, "success": False, "gen": 1},
        ]
        sa.record_batch(batch)
        assert sa.total_records() == 3
        # Per-strategy index should have 2 RANDOM records
        assert len(sa._strategy_record_idx.get("RANDOM", set())) == 2
        assert len(sa._strategy_record_idx.get("HYBRID", set())) == 1
        # Level strategies
        assert "RANDOM" in sa._level_strategies.get(0, set())
        assert "HYBRID" in sa._level_strategies.get(1, set())

    def test_batch_partial_records(self):
        """Batch with missing keys should be reconstructed."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive()
        batch = [{"strategy": "DIRECTED", "level": 2}]
        sa.record_batch(batch)
        assert sa.total_records() == 1
        rec = sa._records[0]
        assert rec["strategy"] == "DIRECTED"
        assert rec["level"] == 2
        assert rec["success"] is False  # default

    def test_batch_triggers_trim(self):
        """Batch that exceeds max_records should trigger trim."""
        from strategy_archive import StrategyArchive

        sa = StrategyArchive(max_records=10)
        batch = [{"strategy": "RANDOM", "level": 0, "success": True, "gen": i} for i in range(15)]
        sa.record_batch(batch)
        assert sa.total_records() == 10  # trimmed to max
        # Indices should still be consistent
        total_idx = sum(len(v) for v in sa._strategy_record_idx.values())
        assert total_idx == 10


# ── MetaController cache invalidation ─────────────────────────────────


class TestMetaControllerCacheInvalidation:
    """Verify telemetry cache is invalidated after update_cycle."""

    def test_cache_cleared_on_new_cycle(self):
        """After update_cycle, detect_stagnation should recompute (not use stale cache)."""
        from meta_controller import MetaController, TelemetrySnapshot

        mc = MetaController(checkpoint_dir="/tmp/mc_cache_test")
        # Need 2+ snapshots for detect_stagnation to return non-NORMAL and set cache
        for c in range(1, 4):
            snap = TelemetrySnapshot(
                cycle=c,
                fitness=-1.0,
                level=5,
                champion_words=10,
                val_loss=2.0,
                diversity=0.3,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        mc.detect_stagnation()
        # Cache is set after detect_stagnation with >= 2 snapshots
        assert mc._cached_stagnation is not None
        assert mc._cached_stagnation_len == 3
        # New cycle should invalidate cache (set _cached_stagnation_len to -1)
        snap4 = TelemetrySnapshot(
            cycle=4,
            fitness=-1.0,
            level=5,
            champion_words=10,
            val_loss=2.0,
            diversity=0.3,
            novelty_size=5,
            qd_occupied=5,
            qd_total=100,
            traces_count=20,
            retrained=True,
            generations_completed=50,
        )
        mc.update_cycle(snap4)
        # After update_cycle, _cached_stagnation_len should be invalidated (-1)
        assert mc._cached_stagnation_len == -1
        mc.detect_stagnation()
        # Now cache is updated to length 4
        assert mc._cached_stagnation_len == 4


# ── CycleState checksum integrity ─────────────────────────────────────


class TestCycleStateChecksum:
    """Verify CycleState checksum roundtrip integrity."""

    def test_save_load_roundtrip(self):
        from cycle_state import CycleState

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            cs = CycleState(path)
            cs.update(current_cycle=5, best_fitness=-0.5, best_level=3)
            cs.save()

            cs2 = CycleState(path)
            ok = cs2.load()
            assert ok is True
            assert cs2["current_cycle"] == 5
            assert cs2["best_fitness"] == -0.5
            assert cs2["best_level"] == 3

    def test_corrupted_file_fails_load(self):
        from cycle_state import CycleState

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            cs = CycleState(path)
            cs.save()
            # Corrupt the file
            with open(path, "w") as f:
                f.write("not valid json {{{")
            cs2 = CycleState(path)
            ok = cs2.load()
            assert ok is False

    def test_missing_file_returns_false(self):
        from cycle_state import CycleState

        cs = CycleState("/nonexistent/path/state.json")
        ok = cs.load()
        assert ok is False

    def test_tampered_checksum_detected(self):
        from cycle_state import CycleState

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            cs = CycleState(path)
            cs.update(current_cycle=3)
            cs.save()
            # Tamper with a field but keep valid JSON
            import json

            with open(path) as f:
                data = json.load(f)
            data["current_cycle"] = 999
            with open(path, "w") as f:
                json.dump(data, f)
            cs2 = CycleState(path)
            ok = cs2.load()
            assert ok is False


# ── MetaController detect_stagnation normal state ──────────────────────


class TestDetectStagnationNormal:
    """Verify NORMAL classification when all indicators are fine."""

    def test_normal_state(self):
        from meta_controller import MetaController, StagnationResult, TelemetrySnapshot

        mc = MetaController(checkpoint_dir="/tmp/mc_normal_test")
        for c in range(1, 6):
            snap = TelemetrySnapshot(
                cycle=c,
                fitness=-2.0 + c * 0.2,
                level=c,
                champion_words=10,
                val_loss=2.0 - c * 0.1,
                diversity=0.3,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        result = mc.detect_stagnation()
        assert result == StagnationResult.NORMAL

    def test_single_snapshot_returns_normal(self):
        from meta_controller import MetaController, StagnationResult, TelemetrySnapshot

        mc = MetaController(checkpoint_dir="/tmp/mc_single_test")
        snap = TelemetrySnapshot(
            cycle=1,
            fitness=-1.0,
            level=5,
            champion_words=10,
            val_loss=2.0,
            diversity=0.3,
            novelty_size=5,
            qd_occupied=5,
            qd_total=100,
            traces_count=20,
            retrained=True,
            generations_completed=50,
        )
        mc.update_cycle(snap)
        result = mc.detect_stagnation()
        assert result == StagnationResult.NORMAL


# ── MetaEvolution config validation ────────────────────────────────────


class TestMetaEvolutionConfigValidation:
    """Verify MetaEvolutionEngine validates seed configs."""

    def test_invalid_seed_rejected(self):
        from meta_evolution import MetaEvolutionConfig, MetaEvolutionEngine

        # Mock the seeded genomes to include a config with out-of-range values.
        # Must include all required keys but with one value out of range.
        with patch(
            "meta_evolution._SEEDED_META_GENOMES",
            [
                {
                    "mutation_rate": 2.0,  # out of range [0, 1]
                    "crossover_rate": 0.3,
                    "elite_ratio": 0.1,
                    "tournament_size": 5,
                    "population_size": 50,
                    "stall_threshold": 3,
                    "novelty_weight": 0.3,
                    "strategy_type_weights": {
                        "RANDOM": 0.25,
                        "TASK_AWARE": 0.25,
                        "DIRECTED": 0.25,
                        "PERTURBATION_CAPPED": 0.25,
                    },
                }
            ],
        ):
            config = MetaEvolutionConfig(meta_population_size=5)
            with pytest.raises(ValueError, match="out of range"):
                MetaEvolutionEngine(meta_config=config, rng=random.Random(42))

    def test_missing_seed_keys_rejected(self):
        from meta_evolution import MetaEvolutionConfig, MetaEvolutionEngine

        with patch("meta_evolution._SEEDED_META_GENOMES", [{"mutation_rate": 0.3}]):  # missing many keys
            config = MetaEvolutionConfig(meta_population_size=5)
            with pytest.raises(ValueError, match="missing keys"):
                MetaEvolutionEngine(meta_config=config, rng=random.Random(42))


# ── compute_fallback for regression tiers ──────────────────────────────


class TestTuneEvolutionRegressionTiers:
    """Verify tune_evolution_params regression tier escalation."""

    def _make_controller_with_telemetry(self, n_cycles, fitness_values):
        from meta_controller import MetaController, TelemetrySnapshot

        mc = MetaController(checkpoint_dir="/tmp/mc_reg_test", val_loss_regression_ratio=1.5, fitness_plateau_cycles=10)
        for c, f in zip(range(1, n_cycles + 1), fitness_values):
            # val_loss starts low (good), then spikes (regression)
            if c <= 1:
                vl = 1.0
            else:
                vl = 2.0  # > 1.0 * 1.5 = 1.5 -> regression detected
            snap = TelemetrySnapshot(
                cycle=c,
                fitness=f,
                level=5,
                champion_words=10,
                val_loss=vl,
                diversity=0.3,
                novelty_size=5,
                qd_occupied=5,
                qd_total=100,
                traces_count=20,
                retrained=True,
                generations_completed=50,
            )
            mc.update_cycle(snap)
        return mc

    def test_val_loss_regression_tier1(self):
        """First tier of val_loss regression uses moderate elite_ratio."""
        # Fitness improving (not plateau) so only val_loss regression fires.
        # plateau_cycles=10 to avoid fitness_plateau detection.
        mc = self._make_controller_with_telemetry(4, [-1.0, -0.9, -0.8, -0.7])
        params = mc.tune_evolution_params(4)
        # val_loss goes 1.0 -> 2.0, ratio 2.0/1.0 = 2.0 > 1.5 -> regression
        # consecutive == 1 (only 1 tuning event so far) -> tier 1
        assert "elite_ratio" in params
        assert params["elite_ratio"] == 0.15
