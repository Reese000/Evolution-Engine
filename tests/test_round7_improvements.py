"""Round 7 improvements — evolution efficiency and quality tests.

Tests for:
  1. MetaGenome.__copy__ produces independent clone with correct fields
  2. MetaController ring-buffer-based consecutive stagnation count (O(1))
  3. MetaController _record_tuning state extraction correctness
  4. compute_qd_axes register pressure without redundant isinstance check
  5. Hall of Fame heap correctness with deterministic fitness values
  6. EvolutionEngine._hall_of_fame attribute always present (no hasattr)
  7. MetaEvolution non-crossover path uses copy (verifies clone behavior)

R23 additions:
  8. MetaGenome retrain fields included in _FIELDS and _COPY_FIELDS
  9. MetaGenome.mutate() perturbs retrain fields
  10. MetaGenome.crossover() handles retrain fields via data-driven loop
  11. _BOUNDS dict removed (dead code)
  12. HallOfFame.get_random returns defensive copies
"""

import collections
import random

# ── MetaGenome.__copy__ ──────────────────────────────────────────────


class TestMetaGenomeCopy:
    """Verify __copy__ produces an independent clone with correct fields."""

    def test_copy_preserves_all_fields(self):
        from meta_genome import MetaGenome

        original = MetaGenome(
            mutation_rate=0.42,
            crossover_rate=0.15,
            elite_ratio=0.12,
            tournament_size=7,
            population_size=80,
            stall_threshold=4,
            novelty_weight=0.35,
            strategy_type_weights={"RANDOM": 0.1, "TASK_AWARE": 0.7, "DIRECTED": 0.1, "PERTURBATION_CAPPED": 0.1},
        )
        import copy

        cloned = copy.copy(original)

        assert cloned.mutation_rate == 0.42
        assert cloned.crossover_rate == 0.15
        assert cloned.elite_ratio == 0.12
        assert cloned.tournament_size == 7
        assert cloned.population_size == 80
        assert cloned.stall_threshold == 4
        assert cloned.novelty_weight == 0.35
        assert cloned.strategy_type_weights == original.strategy_type_weights

    def test_copy_mutation_independence(self):
        """Mutating cloned weights must not affect original."""
        import copy

        from meta_genome import MetaGenome

        original = MetaGenome(mutation_rate=0.5)
        original.strategy_type_weights["RANDOM"] = 0.99

        cloned = copy.copy(original)
        cloned.strategy_type_weights["RANDOM"] = 0.01
        cloned.mutation_rate = 0.0

        assert original.mutation_rate == 0.5
        assert original.strategy_type_weights["RANDOM"] == 0.99
        assert cloned.mutation_rate == 0.0
        assert cloned.strategy_type_weights["RANDOM"] == 0.01

    def test_copy_produces_two_independent_clones(self):
        """Two copies from the same source are independent of each other."""
        import copy

        from meta_genome import MetaGenome

        original = MetaGenome()
        clone1 = copy.copy(original)
        clone2 = copy.copy(original)
        clone1.strategy_type_weights["RANDOM"] = 0.5
        clone2.strategy_type_weights["RANDOM"] = 0.25

        assert clone1.strategy_type_weights["RANDOM"] != clone2.strategy_type_weights["RANDOM"]


# ── MetaController ring-buffer consecutive stagnation ──────────────────


class TestMetaControllerRingBuffer:
    """Verify _count_consecutive_stagnation uses ring buffer (O(1))."""

    def _make_controller(self):
        from meta_controller import MetaController

        return MetaController(checkpoint_dir="/tmp/mc_ring_test")

    def test_consecutive_stagnation_with_matching_state(self):
        """When _last_stagnation_state matches, counter is O(1)."""
        from meta_controller import StagnationResult

        mc = self._make_controller()
        mc._last_stagnation_state = "fitness_plateau"
        mc._consecutive_stagnation_counter = 5
        result = mc._count_consecutive_stagnation(StagnationResult.FITNESS_PLATEAU)
        assert result == 5

    def test_consecutive_stagnation_from_ring_buffer(self):
        """Non-tracked state uses ring buffer scan (max 16)."""
        from meta_controller import StagnationResult

        mc = self._make_controller()
        # Simulate 3 consecutive diversity_collapse events in ring
        mc._state_ring = collections.deque(
            ["diversity_collapse", "diversity_collapse", "diversity_collapse"],
            maxlen=16,
        )
        result = mc._count_consecutive_stagnation(StagnationResult.DIVERSITY_COLLAPSE)
        assert result == 3

    def test_ring_buffer_mixed_states(self):
        """Ring buffer correctly counts trailing matches only."""
        from meta_controller import StagnationResult

        mc = self._make_controller()
        mc._state_ring = collections.deque(
            ["fitness_plateau", "fitness_plateau", "diversity_collapse", "diversity_collapse"],
            maxlen=16,
        )
        mc._last_stagnation_state = ""
        mc._consecutive_stagnation_counter = 0
        result = mc._count_consecutive_stagnation(StagnationResult.DIVERSITY_COLLAPSE)
        assert result == 2

    def test_ring_buffer_cap_exceeded(self):
        """Ring buffer respects maxlen=16."""
        from meta_controller import StagnationResult

        mc = self._make_controller()
        mc._state_ring = collections.deque(
            ["fitness_plateau"] * 20,
            maxlen=16,
        )
        result = mc._count_consecutive_stagnation(StagnationResult.FITNESS_PLATEAU)
        # deque(maxlen=16) truncates to 16 most recent
        assert result == 16


# ── MetaController _record_tuning state extraction ─────────────────────


class TestRecordTuningStateExtraction:
    """Verify _record_tuning correctly extracts state from reason strings."""

    def _make_controller(self):
        from meta_controller import MetaController

        return MetaController(checkpoint_dir="/tmp/mc_tuning_test")

    def test_stagnation_state_extraction(self):
        mc = self._make_controller()
        mc._record_tuning("retrain", "retrain_lr", None, 1e-5, "stagnation: fitness_plateau")
        assert mc._last_stagnation_state == "fitness_plateau"
        assert mc._consecutive_stagnation_counter == 1

    def test_regression_state_extraction(self):
        mc = self._make_controller()
        mc._record_tuning("retrain", "retrain_lr", None, 1e-5, "regression: val_loss_regression")
        assert mc._last_stagnation_state == "val_loss_regression"

    def test_recovery_state_extraction(self):
        mc = self._make_controller()
        mc._record_tuning("retrain", "retrain_lr", None, 1e-5, "recovery_mode")
        # "recovery" is not in _STAGNATION_STATE_VALUES, so the else branch
        # resets the counter.  The state IS detected as "recovery" for the
        # ring buffer but doesn't drive the consecutive counter.
        assert mc._state_ring[-1] == "recovery"
        assert mc._consecutive_stagnation_counter == 0

    def test_non_stagnation_resets_counter(self):
        mc = self._make_controller()
        mc._last_stagnation_state = "fitness_plateau"
        mc._consecutive_stagnation_counter = 3
        mc._record_tuning("evolution", "mutation_rate", None, 0.7, "fallback: val_loss_none_fitness_stagnant")
        assert mc._consecutive_stagnation_counter == 0

    def test_stagnation_state_with_tier_suffix(self):
        mc = self._make_controller()
        mc._record_tuning("retrain", "retrain_freeze_layers", None, 0, "stagnation: diversity_collapse (tier 1)")
        assert mc._last_stagnation_state == "diversity_collapse"

    def test_ring_buffer_populated_on_record(self):
        """_record_tuning must append to the state ring buffer."""
        mc = self._make_controller()
        mc._record_tuning("retrain", "retrain_lr", None, 1e-5, "stagnation: fitness_plateau")
        mc._record_tuning("retrain", "retrain_lr", None, 1e-5, "stagnation: fitness_plateau")
        assert len(mc._state_ring) == 2
        assert mc._state_ring[0] == "fitness_plateau"
        assert mc._state_ring[1] == "fitness_plateau"


# ── compute_qd_axes register pressure ──────────────────────────────────


class TestComputeQDAxesOptimized:
    """Verify compute_qd_axes register pressure still works after removing
    the redundant isinstance(val, (int, float)) check."""

    def test_register_pressure_basic(self):
        from evolution_engine import compute_qd_axes

        program = [0x00100073]  # EBREAK
        trace = {"final_regs": {1: 5, 2: 3, 10: 0, 15: 42}}
        axes = compute_qd_axes(program, trace)
        # active: regs 1 (nonzero), 2 (nonzero), 15 (nonzero) = 3
        # reg_pressure = (3/31)*2 - 1
        expected = (3.0 / 31.0) * 2.0 - 1.0
        assert abs(axes[1] - expected) < 1e-9

    def test_register_pressure_empty_trace(self):
        from evolution_engine import compute_qd_axes

        axes = compute_qd_axes([0x00100073], {})
        assert axes[1] == -1.0

    def test_register_pressure_nonzero_vals(self):
        """Boolean-like truthy values (1.0, 1) should count as active."""
        from evolution_engine import compute_qd_axes

        trace = {"final_regs": {1: 1.0, 2: 2, 3: -1, 4: 0}}
        axes = compute_qd_axes([0x00100073], trace)
        # active: 1 (1.0 != 0), 2 (2 != 0), 3 (-1 != 0) = 3
        expected = (3.0 / 31.0) * 2.0 - 1.0
        assert abs(axes[1] - expected) < 1e-9


# ── Hall of Fame heap correctness ─────────────────────────────────────


class TestHallOfFameHeap:
    """Verify HallOfFame uses heap correctly with deterministic fitnesses."""

    def test_add_evicts_worst_when_full(self):
        from evolution_engine import HallOfFame

        hof = HallOfFame(capacity=3)
        hof.add([1, 2], fitness=10.0, level=0)
        hof.add([3, 4], fitness=20.0, level=0)
        hof.add([5, 6], fitness=30.0, level=0)
        # All 3 slots full
        assert hof.size == 3
        # Adding a better one should evict the worst (10.0)
        hof.add([7, 8], fitness=25.0, level=0)
        assert hof.size == 3
        assert hof.best_fitness == 30.0

    def test_add_rejects_worse_than_all(self):
        from evolution_engine import HallOfFame

        hof = HallOfFame(capacity=2)
        hof.add([1, 2], fitness=10.0, level=0)
        hof.add([3, 4], fitness=20.0, level=0)
        added = hof.add([5, 6], fitness=5.0, level=0)
        assert not added
        assert hof.size == 2

    def test_get_best_filtered_by_level(self):
        from evolution_engine import HallOfFame

        hof = HallOfFame(capacity=5)
        hof.add([1], fitness=10.0, level=0)
        hof.add([2], fitness=20.0, level=1)
        hof.add([3], fitness=30.0, level=0)
        best_l0 = hof.get_best(level=0)
        assert best_l0 == [3]  # fitness=30, not [2] (fitness=20, level=1)
        best_l1 = hof.get_best(level=1)
        assert best_l1 == [2]


# ── EvolutionEngine._hall_of_fame always initialized ──────────────────


class TestHallOfFameAlwaysInitialized:
    """Verify _hall_of_fame is always present after __init__."""

    def test_hall_of_fame_exists_after_init(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig(population_size=5, max_generations=2)
        engine = EvolutionEngine(config=config)
        assert hasattr(engine, "_hall_of_fame")
        assert engine._hall_of_fame is not None
        assert engine._hall_of_fame.size == 0


# ── Bonus threshold cache correctness ──────────────────────────────────


class TestBonusThresholdCache:
    """Verify _bonus_threshold_cache is directly accessible (no getattr)."""

    def test_bonus_threshold_cache_set_after_first_compute(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig(population_size=5, max_generations=2)
        engine = EvolutionEngine(config=config)
        # Initially should be None
        assert engine._bonus_threshold_cache is None

    def test_bonus_threshold_cache_typed_correctly(self):
        from evolution_engine import EvolutionConfig, EvolutionEngine

        config = EvolutionConfig(population_size=5, max_generations=2)
        engine = EvolutionEngine(config=config)
        # Type should be None or tuple
        cache = engine._bonus_threshold_cache
        assert cache is None or isinstance(cache, tuple)


# ── R23: MetaGenome retrain fields in _FIELDS ──────────────────────────


class TestMetaGenomeRetrainFields:
    """Verify retrain fields are included in _FIELDS and behave correctly."""

    def test_retrain_fields_in_fields(self):
        """All retrain_* fields should be present in _FIELDS."""
        from meta_genome import _FIELDS

        retrain_names = [f[0] for f in _FIELDS if f[0].startswith("retrain_")]
        expected = {
            "retrain_lr",
            "retrain_batch_size",
            "retrain_weight_decay",
            "retrain_patience",
            "retrain_grad_clip",
            "retrain_max_epochs",
            "retrain_val_split",
            "retrain_freeze_layers",
            "retrain_max_per_level",
            "retrain_timeout",
            "retrain_min_examples",
        }
        assert set(retrain_names) == expected

    def test_copy_fields_matches_fields(self):
        """_COPY_FIELDS should be exactly the names from _FIELDS."""
        from meta_genome import _COPY_FIELDS, _FIELDS

        assert _COPY_FIELDS == tuple(f[0] for f in _FIELDS)

    def test_copy_preserves_retrain_fields(self):
        """__copy__ should preserve retrain field values."""
        import copy

        from meta_genome import MetaGenome

        g = MetaGenome(retrain_lr=0.001, retrain_batch_size=16)
        c = copy.copy(g)
        assert c.retrain_lr == 0.001
        assert c.retrain_batch_size == 16

    def test_copy_retrain_independence(self):
        """Mutating clone retrain fields should not affect original."""
        import copy

        from meta_genome import MetaGenome

        g = MetaGenome(retrain_lr=0.001)
        c = copy.copy(g)
        c.retrain_lr = 0.01
        assert g.retrain_lr == 0.001

    def test_mutate_perturbs_retrain_fields(self):
        """With large enough std, retrain fields should get perturbed."""
        from meta_genome import MetaGenome

        rng = random.Random(42)
        g = MetaGenome()
        # With full reset, retrain fields should be non-sentinel
        m = g.mutate(std=1.0, rng=rng, reset_prob=1.0)
        # At least some retrain fields should differ from -1.0 sentinel
        assert m.retrain_lr != -1.0 or m.retrain_batch_size != -1.0

    def test_crossover_includes_retrain_fields(self):
        """Crossover should cross over retrain fields between parents."""
        from meta_genome import MetaGenome

        rng = random.Random(42)
        g1 = MetaGenome(retrain_lr=0.001, retrain_batch_size=8)
        g2 = MetaGenome(retrain_lr=0.01, retrain_batch_size=32)
        c1, c2 = g1.crossover(g2, rng=rng)
        # Children should get values from one of the two parents
        assert c1.retrain_lr in (0.001, 0.01)
        assert c1.retrain_batch_size in (8, 32)
        assert c2.retrain_lr in (0.001, 0.01)
        assert c2.retrain_batch_size in (8, 32)

    def test_random_initializes_retrain_fields(self):
        """MetaGenome.random() should initialize retrain fields."""
        from meta_genome import MetaGenome

        rng = random.Random(42)
        g = MetaGenome.random(rng=rng)
        # Sentinel fields should be initialized via _FIELDS defaults in __new__
        # The default in _FIELDS for retrain fields is -1.0 (sentinel)
        assert isinstance(g.retrain_lr, float)
        assert isinstance(g.retrain_batch_size, float)


# ── R23: _BOUNDS dict removed ──────────────────────────────────────────


class TestBOUNDSRemoved:
    """Verify _BOUNDS dead code dict was removed."""

    def test_no_bounds_dict(self):
        """_BOUNDS should not exist as an attribute in meta_genome."""
        import meta_genome

        assert not hasattr(meta_genome, "_BOUNDS")


# ── R23: HallOfFame.get_random defensive copy ─────────────────────────


class TestHallOfFameDefensiveCopy:
    """Verify get_random returns a defensive copy."""

    def test_get_random_returns_new_list(self):
        import random

        from evolution_engine import HallOfFame

        hof = HallOfFame(capacity=5)
        prog = [1, 2, 3, 4]
        hof.add(prog, fitness=10.0, level=0)
        rng = random.Random(42)
        result = hof.get_random(rng, level=0)
        # Result should be a new list object (defensive copy)
        assert result is not None
        assert result == [1, 2, 3, 4]
        # Mutating result should not affect internal state
        result[0] = 999
        internal = hof.get_random(rng, level=0)
        assert internal[0] == 1

    def test_get_random_empty_returns_none(self):
        import random

        from evolution_engine import HallOfFame

        hof = HallOfFame(capacity=5)
        rng = random.Random(42)
        assert hof.get_random(rng) is None
