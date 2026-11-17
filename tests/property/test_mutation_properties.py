"""Property-based tests for Evolution Engine mutation operators.

Uses Hypothesis to verify invariants that hold for ALL valid inputs,
not just the examples we think to test. These catch edge cases that
example-based tests miss.
"""
from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

# Path setup
from paths import ensure_project_on_path

ensure_project_on_path()

from rv32i_mask import decode as rv32i_decode
from rv32i_mask import encode as rv32i_encode

from mutation_proposer import (
    ALL_INSTRUCTION_NAMES,
    EBREAK,
    MAX_PROGRAM_LEN,
    NOP,
    DirectedMutator,
    find_ebreak_index,
    generate_seed_program,
    has_ebreak,
    make_random_rv32i,
    make_rv32i,
    simplify_program,
    trim_dead_code,
    validate_mutation,
)

# ── Hypothesis strategies ────────────────────────────────────────────


def _rv32i_instruction_name():
    """Strategy: any valid RV32I instruction name."""
    return st.sampled_from(ALL_INSTRUCTION_NAMES)


def _rv32i_register():
    """Strategy: valid RV32I register index (0-31)."""
    return st.integers(min_value=0, max_value=31)


def _rv32i_immediate():
    """Strategy: reasonable immediate value for most instructions."""
    return st.integers(min_value=-2048, max_value=2047)


def _valid_instruction():
    """Strategy: a single valid RV32I instruction as 32-bit int."""
    return st.builds(
        make_rv32i,
        name=_rv32i_instruction_name(),
        rd=_rv32i_register(),
        rs1=_rv32i_register(),
        rs2=_rv32i_register(),
        imm=_rv32i_immediate(),
    )


def _valid_program():
    """Strategy: a list of valid RV32I instructions ending with EBREAK."""
    return st.lists(_valid_instruction(), min_size=1, max_size=50).map(
        lambda prog: prog + [EBREAK]
    )


def _valid_program_no_ebreak():
    """Strategy: a list of valid RV32I instructions (no EBREAK terminator)."""
    return st.lists(_valid_instruction(), min_size=0, max_size=30)


def _seed_length():
    """Strategy: reasonable seed program length."""
    return st.integers(min_value=1, max_value=100)


# ── Mutation operator properties ────────────────────────────────────


class TestMutationOperators:
    """Property-based tests for mutation operators."""

    @given(program=_valid_program())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_trim_dead_code_preserves_validity(self, program):
        """trim_dead_code output is always a valid program (all instructions decodable)."""
        trimmed = trim_dead_code(program)
        assert isinstance(trimmed, list)
        valid, err = validate_mutation(trimmed)
        assert valid, f"trim_dead_code produced invalid program: {err}"

    @given(program=_valid_program())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_trim_dead_code_never_grows(self, program):
        """trim_dead_code output length <= input length."""
        trimmed = trim_dead_code(program)
        assert len(trimmed) <= len(program)

    @given(program=_valid_program())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_trim_dead_code_preserves_ebreak_at_end(self, program):
        """trim_dead_code output always ends with EBREAK."""
        trimmed = trim_dead_code(program)
        assert len(trimmed) > 0
        assert trimmed[-1] == EBREAK, "trim_dead_code should preserve EBREAK at end"

    @given(program=_valid_program())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_simplify_program_output_ends_with_ebreak(self, program):
        """simplify_program output always ends with EBREAK."""
        simplified = simplify_program(program)
        assert len(simplified) > 0
        assert simplified[-1] == EBREAK

    @given(program=_valid_program())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_simplify_program_output_is_valid(self, program):
        """simplify_program output is always a valid program."""
        simplified = simplify_program(program)
        valid, err = validate_mutation(simplified)
        assert valid, f"simplify_program produced invalid program: {err}"

    @given(program=_valid_program())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_simplify_program_never_grows_beyond_max(self, program):
        """simplify_program output is at most MAX_PROGRAM_LEN."""
        simplified = simplify_program(program)
        assert len(simplified) <= MAX_PROGRAM_LEN

    @given(length=_seed_length())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_generate_seed_program_ends_with_ebreak(self, length):
        """generate_seed_program always produces a program ending with EBREAK."""
        rng = random.Random(42)
        prog = generate_seed_program(length=length, rng=rng)
        assert len(prog) > 0
        assert prog[-1] == EBREAK

    @given(length=_seed_length())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_generate_seed_program_all_instructions_valid(self, length):
        """generate_seed_program output: every instruction is decodable."""
        rng = random.Random(42)
        prog = generate_seed_program(length=length, rng=rng)
        valid, err = validate_mutation(prog)
        assert valid, f"generate_seed_program produced invalid instruction: {err}"

    @given(length=_seed_length())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_generate_seed_program_length_respected(self, length):
        """generate_seed_program produces a program of the requested length."""
        rng = random.Random(42)
        prog = generate_seed_program(length=length, rng=rng)
        assert len(prog) == max(length, 2)  # at least [instr, EBREAK]

    @given(
        program=st.lists(_valid_instruction(), min_size=0, max_size=30),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_stochastic_perturbation_preserves_validity(self, program):
        """stochastic_perturbation output is always a valid program."""
        rng = random.Random(42)
        result, desc = DirectedMutator.stochastic_perturbation(program, rng)
        assert isinstance(result, list)
        assert len(result) > 0
        valid, err = validate_mutation(result)
        assert valid, f"stochastic_perturbation produced invalid program: {err}\n  desc={desc}"

    @given(
        program=st.lists(_valid_instruction(), min_size=2, max_size=30),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_stochastic_perturbation_ends_with_ebreak(self, program):
        """stochastic_perturbation output always ends with EBREAK."""
        rng = random.Random(42)
        result, desc = DirectedMutator.stochastic_perturbation(program, rng)
        assert result[-1] == EBREAK, f"stochastic_perturbation output doesn't end with EBREAK: {desc}"

    @given(
        program=st.lists(_valid_instruction(), min_size=0, max_size=20),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_stochastic_perturbation_length_bounded(self, program):
        """stochastic_perturbation output is at most MAX_PROGRAM_LEN."""
        rng = random.Random(42)
        result, _desc = DirectedMutator.stochastic_perturbation(program, rng)
        assert len(result) <= MAX_PROGRAM_LEN

    @given(
        name=_rv32i_instruction_name(),
        rd=_rv32i_register(),
        rs1=_rv32i_register(),
        rs2=_rv32i_register(),
        imm=_rv32i_immediate(),
    )
    @settings(max_examples=200)
    def test_make_rv32i_roundtrip(self, name, rd, rs1, rs2, imm):
        """make_rv32i encoding roundtrips through decode."""
        encoded = make_rv32i(name=name, rd=rd, rs1=rs1, rs2=rs2, imm=imm)
        assert isinstance(encoded, int)
        decoded = rv32i_decode(encoded, validate=True)
        assert decoded["name"] == name

    @given(program=_valid_program_no_ebreak())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_ebreak_index_properties(self, program):
        """find_ebreak_index properties: None when absent, valid index when present."""
        # Without EBREAK
        idx = find_ebreak_index(program)
        assert idx is None

        # With EBREAK appended
        program_with_ebreak = program + [EBREAK]
        idx = find_ebreak_index(program_with_ebreak)
        assert idx is not None
        assert idx == len(program)  # EBREAK is last element

    @given(program=_valid_program())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_has_ebreak_consistency(self, program):
        """has_ebreak is consistent with find_ebreak_index."""
        assert has_ebreak(program) == (find_ebreak_index(program) is not None)


# ── Fitness function properties ─────────────────────────────────────


class TestFitnessProperties:
    """Property-based tests for fitness computation functions."""

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
    def test_score_register_match_always_in_range(self, expected_regs, final_regs):
        """score_register_match always returns a value in [0.0, 1.0]."""
        from evolution.fitness_helpers import score_register_match

        expected_total = len(expected_regs)
        score = score_register_match(expected_regs, expected_total, final_regs)
        assert 0.0 <= score <= 1.0, f"score_register_match returned {score}, expected [0.0, 1.0]"

    @given(
        expected_regs=st.dictionaries(
            st.integers(min_value=0, max_value=31),
            st.integers(min_value=1, max_value=100),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=200)
    def test_score_register_match_perfect_match(self, expected_regs):
        """score_register_match returns 1.0 when all registers match exactly."""
        from evolution.fitness_helpers import score_register_match

        expected_total = len(expected_regs)
        score = score_register_match(expected_regs, expected_total, dict(expected_regs))
        assert abs(score - 1.0) < 1e-10, f"Perfect match should score 1.0, got {score}"

    @given(
        reg_match=st.floats(min_value=0.0, max_value=1.0),
        instr_count=st.integers(min_value=0, max_value=1000),
        status=st.sampled_from(["ok", "crash", "timeout"]),
        budget=st.integers(min_value=1, max_value=50000),
    )
    @settings(max_examples=200)
    def test_compute_base_score_bounded_below(self, reg_match, instr_count, status, budget):
        """compute_base_score: score >= -100 (bloat penalty has reasonable floor)."""
        from evolution.fitness_helpers import compute_base_score

        score = compute_base_score(reg_match, instr_count, status, budget)
        # Base score floor: reg_match*100 - max_bloat_penalty
        # max bloat_penalty for 1000 instr = (1000-50)*0.05 = 47.5
        # worst case: 0*100 - 47.5 = -47.5
        assert score >= -100.0, f"compute_base_score too low: {score}"

    @given(
        reg_match=st.floats(min_value=0.0, max_value=1.0),
        instr_count=st.integers(min_value=0, max_value=1000),
        status=st.sampled_from(["ok", "crash", "timeout"]),
        budget=st.integers(min_value=1, max_value=50000),
    )
    @settings(max_examples=200)
    def test_compute_base_score_bounded_above(self, reg_match, instr_count, status, budget):
        """compute_base_score: score <= ~180 (max: 100 + 50 solved + 10 exec)."""
        from evolution.fitness_helpers import compute_base_score

        score = compute_base_score(reg_match, instr_count, status, budget)
        assert score <= 200.0, f"compute_base_score too high: {score}"

    @given(
        status=st.sampled_from(["ok", "crash", "timeout"]),
        reg_match=st.floats(min_value=0.0, max_value=1.0),
        instr_count=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=200)
    def test_compute_time_credit_non_negative(self, status, reg_match, instr_count):
        """compute_time_credit is always >= 0."""
        from evolution.fitness_helpers import compute_time_credit

        credit = compute_time_credit(status, reg_match, instr_count)
        assert credit >= 0.0, f"compute_time_credit returned negative: {credit}"

    @given(
        status=st.sampled_from(["ok", "crash", "timeout"]),
        reg_match=st.floats(min_value=0.0, max_value=1.0),
        instr_count=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=200)
    def test_compute_time_credit_bounded(self, status, reg_match, instr_count):
        """compute_time_credit is always <= 20.0."""
        from evolution.fitness_helpers import compute_time_credit

        credit = compute_time_credit(status, reg_match, instr_count)
        assert credit <= 20.0, f"compute_time_credit exceeded max: {credit}"

    @given(
        mt_weight=st.floats(min_value=0.0, max_value=5.0),
    )
    @settings(max_examples=100)
    def test_compute_multi_task_score_none_results(self, mt_weight):
        """compute_multi_task_score returns 0.0 for None results."""
        from evolution.fitness_helpers import compute_multi_task_score

        score = compute_multi_task_score(None, mt_weight=mt_weight)
        assert score == 0.0

    @given(
        mt_weight=st.floats(min_value=0.0, max_value=5.0),
    )
    @settings(max_examples=100)
    def test_compute_multi_task_score_empty_results(self, mt_weight):
        """compute_multi_task_score returns 0.0 for empty results."""
        from evolution.fitness_helpers import compute_multi_task_score

        score = compute_multi_task_score([], mt_weight=mt_weight)
        assert score == 0.0


# ── Program structure properties ────────────────────────────────────


class TestProgramStructure:
    """Property-based tests for program structural invariants."""

    @given(program=_valid_program())
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_find_ebreak_index_last_ebreak(self, program):
        """find_ebreak_index returns the index of the LAST EBREAK."""
        idx = find_ebreak_index(program)
        if idx is not None:
            assert program[idx] == EBREAK
            # No EBREAK after this index
            for i in range(idx + 1, len(program)):
                assert program[i] != EBREAK

    @given(
        prog=st.lists(_valid_instruction(), min_size=1, max_size=20),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_trim_dead_code_preserves_prefix(self, prog):
        """trim_dead_code preserves the prefix up to and including the last EBREAK."""
        # Find last EBREAK
        ebreak_idx = None
        for i in range(len(prog) - 1, -1, -1):
            if prog[i] == EBREAK:
                ebreak_idx = i
                break

        if ebreak_idx is not None:
            trimmed = trim_dead_code(prog)
            assert trimmed == prog[:ebreak_idx + 1]

    @given(length=st.integers(min_value=1, max_value=50))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_generate_seed_program_deterministic_with_seed(self, length):
        """Same seed produces identical programs."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        prog1 = generate_seed_program(length=length, rng=rng1)
        prog2 = generate_seed_program(length=length, rng=rng2)
        assert prog1 == prog2

    @given(
        name=_rv32i_instruction_name(),
        rd=_rv32i_register(),
        rs1=_rv32i_register(),
        imm=_rv32i_immediate(),
    )
    @settings(max_examples=200)
    def test_make_rv32i_positive_encoding(self, name, rd, rs1, imm):
        """make_rv32i always produces a non-negative 32-bit value."""
        encoded = make_rv32i(name=name, rd=rd, rs1=rs1, imm=imm)
        assert 0 <= encoded <= 0xFFFFFFFF


# ── Selection / population properties ───────────────────────────────


class TestSelectionProperties:
    """Property-based tests for selection and population management."""

    @given(
        population=st.lists(
            st.tuples(
                st.floats(min_value=-200.0, max_value=200.0),
                st.integers(min_value=0, max_value=1000),
                _valid_program(),
                st.dictionaries(st.text(), st.text()),
            ),
            min_size=2,
            max_size=50,
        ),
        elite_size=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_tournament_select_returns_from_population(self, population, elite_size):
        """Tournament selection always returns individuals from the original population."""
        engine = _make_engine_for_selection()
        n_scored = len(population)
        elite_size = min(elite_size, n_scored)

        selected = list(engine._tournament_select(population, elite_size, n_scored))
        assert len(selected) <= elite_size

        pop_indices = set(range(n_scored))
        for item, idx in selected:
            assert idx in pop_indices, f"Selected index {idx} not in population [0, {n_scored})"

    @given(
        n=st.integers(min_value=2, max_value=50),
        elite_ratio=st.floats(min_value=0.05, max_value=0.5),
    )
    @settings(max_examples=100)
    def test_elite_size_computation(self, n, elite_ratio):
        """Elite size = max(1, int(population_size * elite_ratio))."""
        elite_size = max(1, int(n * elite_ratio))
        assert elite_size >= 1
        assert elite_size <= n

    @given(
        diversity=st.floats(min_value=0.0, max_value=1.0),
        base_tourney=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=100)
    def test_adaptive_tournament_size_bounds(self, diversity, base_tourney):
        """Adaptive tournament size is bounded by [1, base+2]."""
        if diversity >= 0.6:
            adaptive = base_tourney + 2
        elif diversity >= 0.4:
            adaptive = base_tourney + 1
        elif diversity >= 0.2:
            adaptive = base_tourney
        else:
            adaptive = max(2, base_tourney - 1)

        assert adaptive >= 2
        assert adaptive <= base_tourney + 2


def _make_engine_for_selection():
    """Create a minimal engine for testing selection methods."""
    from evolution_engine import EvolutionConfig, EvolutionEngine

    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine.config = EvolutionConfig()
    engine.rng = random.Random(42)
    engine._current_diversity = 0.5
    engine.config.tournament_size = 3
    return engine
