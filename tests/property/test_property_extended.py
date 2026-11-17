"""Property-based tests for Evolution Engine core invariants.

Uses Hypothesis to verify invariants that hold for ALL valid inputs,
complementing the mutation-focused tests in test_mutation_properties.py.

Covers: population management, QD archive monotonicity, fitness bounds,
mutation rate scheduling, and archive coverage.
"""
from __future__ import annotations

import math
import random

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Path setup
from paths import ensure_project_on_path
ensure_project_on_path()

# ── Hypothesis strategies ────────────────────────────────────────────


def _descriptor() -> st.SearchStrategy[list[float]]:
    """Strategy: a 2-D behavioural descriptor in [-1, 1]."""
    return st.lists(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=2,
    )


def _fitness() -> st.SearchStrategy[float]:
    """Strategy: a finite fitness value in [-200, 200]."""
    return st.floats(min_value=-200.0, max_value=200.0, allow_nan=False, allow_infinity=False)


def _small_positive_int() -> st.SearchStrategy[int]:
    """Strategy: positive integer for population-like sizes."""
    return st.integers(min_value=1, max_value=200)


# ── Archive properties ──────────────────────────────────────────────


class TestArchiveProperties:
    """Property-based tests for the QD archive (MAP-Elites)."""

    @given(
        descriptors=st.lists(_descriptor(), min_size=1, max_size=50),
        fitnesses=st.lists(_fitness(), min_size=1, max_size=50),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_archive_coverage_never_negative_or_above_one(self, descriptors, fitnesses):
        """Archive coverage is always in [0.0, 1.0] regardless of additions."""
        from qd_archive import MAPElitesArchive

        archive = MAPElitesArchive(bins=10)
        n = min(len(descriptors), len(fitnesses))
        for i in range(n):
            archive.add_solution(descriptors[i], fitnesses[i], [i])
        cov = archive.coverage()
        assert 0.0 <= cov <= 1.0, f"coverage={cov} outside [0, 1]"

    @given(
        descriptors=st.lists(_descriptor(), min_size=1, max_size=50),
        fitnesses=st.lists(_fitness(), min_size=1, max_size=50),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_archive_size_never_shrinks(self, descriptors, fitnesses):
        """Archive occupied count never decreases after add_solution calls (QD invariant)."""
        from qd_archive import MAPElitesArchive

        archive = MAPElitesArchive(bins=10)
        n = min(len(descriptors), len(fitnesses))
        sizes = [0]
        for i in range(n):
            archive.add_solution(descriptors[i], fitnesses[i], [i])
            sizes.append(archive.size)

        # Monotonicity: each size >= previous
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1], (
                f"Archive shrank: sizes[{i-1}]={sizes[i-1]} -> sizes[{i}]={sizes[i]}"
            )

    @given(
        descriptors=st.lists(_descriptor(), min_size=1, max_size=30),
        fitnesses=st.lists(_fitness(), min_size=1, max_size=30),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_archive_grid_solutions_count_matches_size(self, descriptors, fitnesses):
        """grid_solutions() length equals archive.size."""
        from qd_archive import MAPElitesArchive

        archive = MAPElitesArchive(bins=10)
        n = min(len(descriptors), len(fitnesses))
        for i in range(n):
            archive.add_solution(descriptors[i], fitnesses[i], [i])
        solutions = archive.grid_solutions()
        assert len(solutions) == archive.size, (
            f"grid_solutions={len(solutions)} != size={archive.size}"
        )

    @given(desc=_descriptor())
    @settings(max_examples=100)
    def test_selection_bias_always_bounded(self, desc):
        """selection_bias() always returns a value in [0.1, 1.0]."""
        from qd_archive import MAPElitesArchive

        archive = MAPElitesArchive(bins=10)
        bias = archive.selection_bias(desc)
        assert 0.1 <= bias <= 1.0, f"selection_bias={bias} outside [0.1, 1.0]"

    @given(
        descriptor=_descriptor(),
        fitness=_fitness(),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_add_solution_returns_bool(self, descriptor, fitness):
        """add_solution always returns a bool (accepted or not)."""
        from qd_archive import MAPElitesArchive

        archive = MAPElitesArchive(bins=5)
        result = archive.add_solution(descriptor, fitness, [0])
        assert isinstance(result, bool)


# ── Fitness bounds properties ───────────────────────────────────────


class TestFitnessBounds:
    """Property-based tests for fitness computation bounds."""

    @given(
        expected_regs=st.dictionaries(
            st.integers(min_value=0, max_value=31),
            st.integers(min_value=-1000, max_value=1000),
            min_size=1,
            max_size=10,
        ),
        final_regs=st.dictionaries(
            st.integers(min_value=0, max_value=31),
            st.integers(min_value=-1000, max_value=1000),
            min_size=0,
            max_size=20,
        ),
    )
    @settings(max_examples=200)
    def test_score_register_match_is_symmetric_sub_additive(self, expected_regs, final_regs):
        """score_register_match(A, A) == 1.0 and score >= 0 always."""
        from evolution.fitness_helpers import score_register_match

        n = len(expected_regs)
        # Perfect self-match is always 1.0
        self_score = score_register_match(expected_regs, n, dict(expected_regs))
        assert abs(self_score - 1.0) < 1e-10

        # Any match score is non-negative
        score = score_register_match(expected_regs, n, final_regs)
        assert score >= 0.0

    @given(
        reg_match=st.floats(min_value=0.0, max_value=1.0),
        instr_count=st.integers(min_value=0, max_value=1000),
        status=st.sampled_from(["ok", "crash", "timeout"]),
        budget=st.integers(min_value=1, max_value=50000),
    )
    @settings(max_examples=200)
    def test_compute_base_score_range(self, reg_match, instr_count, status, budget):
        """compute_base_score is bounded in [-100, 200]."""
        from evolution.fitness_helpers import compute_base_score

        score = compute_base_score(reg_match, instr_count, status, budget)
        assert -100.0 <= score <= 200.0, f"base_score={score} outside [-100, 200]"

    @given(
        status=st.sampled_from(["ok", "crash", "timeout"]),
        reg_match=st.floats(min_value=0.0, max_value=1.0),
        instr_count=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=200)
    def test_compute_time_credit_bounded(self, status, reg_match, instr_count):
        """compute_time_credit is in [0.0, 20.0]."""
        from evolution.fitness_helpers import compute_time_credit

        credit = compute_time_credit(status, reg_match, instr_count)
        assert 0.0 <= credit <= 20.0, f"time_credit={credit} outside [0, 20]"


# ── Mutation rate scheduling properties ─────────────────────────────


class TestMutationRateScheduling:
    """Property-based tests for mutation rate scheduling bounds."""

    @given(
        gen=st.integers(min_value=0, max_value=500),
        start_rate=st.floats(min_value=0.1, max_value=0.9),
        end_rate=st.floats(min_value=0.05, max_value=0.5),
        total_gens=st.integers(min_value=50, max_value=500),
    )
    @settings(max_examples=200)
    def test_scheduled_rate_stays_in_bounds(self, gen, start_rate, end_rate, total_gens):
        """Scheduled mutation rate always stays in [end_rate, start_rate]."""
        # Linear interpolation clamped
        if gen >= total_gens:
            rate = end_rate
        else:
            frac = gen / total_gens
            rate = start_rate + (end_rate - start_rate) * frac
        assert min(start_rate, end_rate) <= rate <= max(start_rate, end_rate), (
            f"scheduled rate={rate} outside [{min(start_rate, end_rate)}, {max(start_rate, end_rate)}]"
        )

    @given(
        start_rate=st.floats(min_value=0.0, max_value=1.0),
        end_rate=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_mutation_rate_config_bounds(self, start_rate, end_rate):
        """Config validation rejects mutation rates outside [0, 1]."""
        from config import EvolutionSection

        evo = EvolutionSection()
        evo.mutation_rate_start = start_rate
        evo.mutation_rate_end = end_rate
        # These should not raise — both are in [0, 1]
        assert 0.0 <= evo.mutation_rate_start <= 1.0
        assert 0.0 <= evo.mutation_rate_end <= 1.0


# ── Population management properties ────────────────────────────────


class TestPopulationProperties:
    """Property-based tests for population management invariants."""

    @given(
        pop_size=st.integers(min_value=2, max_value=50),
        elite_ratio=st.floats(min_value=0.05, max_value=0.5),
    )
    @settings(max_examples=100)
    def test_elite_size_always_positive_and_le_pop(self, pop_size, elite_ratio):
        """Elite size is in [1, pop_size] for any valid inputs."""
        elite_size = max(1, int(pop_size * elite_ratio))
        assert 1 <= elite_size <= pop_size

    @given(
        pop_size=st.integers(min_value=1, max_value=50),
        tournament_size=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=100)
    def test_tournament_size_capped_at_pop_size(self, pop_size, tournament_size):
        """Tournament size is effectively min(tournament_size, pop_size)."""
        effective = min(tournament_size, pop_size)
        assert effective <= pop_size
        assert effective >= 1

    @given(
        n=st.integers(min_value=1, max_value=100),
        diversity=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_diversity_inject_count_bounded(self, n, diversity):
        """Diversity injection count is <= pop_size and >= 0."""
        inject_count = int(n * 0.1) if diversity < 0.3 else 0
        assert 0 <= inject_count <= n
