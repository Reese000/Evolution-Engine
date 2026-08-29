#!/usr/bin/env python3
"""Integration test for L14-L30 curriculum levels.

Tests that the VM can execute reference solutions for L14-L16 (the ones
that work with EBREAK-terminated programs), that the evolution loop can
initialize with max_level=30, and that the curriculum properly reports
31 levels.
"""
from __future__ import annotations

import random

import pytest

from paths import ensure_project_on_path

ensure_project_on_path()

from evolution_engine import EvolutionConfig, EvolutionEngine
from tasks_rv32i import LEVELS, get_task_level, reference_solution
from vm_farm import EvalResult, _run_one_evaluation

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def engine_config():
    """Create an EvolutionConfig with max_level=30."""
    cfg = EvolutionConfig()
    cfg.max_level = 30
    cfg.auto_curriculum = True
    return cfg


@pytest.fixture
def vm_kernel():
    """Resolve the safety-kernel path (empty string = pure Python VM)."""
    import os

    from milestone1 import MILESTONE0_DIR  # type: ignore

    candidates = [
        os.path.join(MILESTONE0_DIR, "safety-kernel", "target", "release", "safety-kernel"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""


# ═══════════════════════════════════════════════════════════════════════
# 1. Curriculum reports 31 levels
# ═══════════════════════════════════════════════════════════════════════


class TestCurriculumLevelCount:
    """Verify that the curriculum has exactly 31 levels (L0-L30)."""

    def test_levels_dict_has_31_entries(self):
        """LEVELS dict should contain exactly 31 entries (L0 through L30)."""
        assert len(LEVELS) == 31

    def test_all_levels_0_through_30_present(self):
        """Every integer from 0 to 30 should be a key in LEVELS."""
        for i in range(31):
            assert i in LEVELS, f"Level {i} missing from LEVELS dict"

    def test_get_task_level_returns_class(self):
        """get_task_level should return a TaskLevel subclass for each level."""
        for i in range(31):
            cls = get_task_level(i)
            assert hasattr(cls, "generate_instance"), f"Level {i} missing generate_instance"
            assert hasattr(cls, "verify"), f"Level {i} missing verify"
            assert hasattr(cls, "reference_solution"), f"Level {i} missing reference_solution"

    def test_generate_all_tasks_returns_31_instances(self):
        """generate_all_tasks should return 31 task instances."""
        from tasks_rv32i import generate_all_tasks
        instances = generate_all_tasks(seed=42)
        assert len(instances) == 31

    def test_instance_level_numbers_match(self):
        """Each generated instance should report its level number correctly."""
        for i in range(31):
            task = get_task_level(i)(seed=42)
            inst = task.generate_instance()
            assert inst["level"] == i, f"Instance at index {i} reports level={inst['level']}"


# ═══════════════════════════════════════════════════════════════════════
# 2. EvolutionEngine initializes with max_level=30
# ═══════════════════════════════════════════════════════════════════════


class TestEvolutionEngineMaxLevel30:
    """Verify that EvolutionEngine accepts max_level=30 configuration."""

    def test_evolution_config_max_level_30(self, engine_config):
        """EvolutionConfig should accept max_level=30 without error."""
        assert engine_config.max_level == 30

    def test_engine_init_with_max_level_30(self, engine_config):
        """EvolutionEngine.__init__ should succeed with max_level=30."""
        engine = EvolutionEngine(config=engine_config)
        assert engine.config.max_level == 30
        assert engine.champion_level == engine_config.curriculum_level

    def test_engine_curriculum_level_clamped_at_construction(self):
        """Setting curriculum_level above max_level gets clamped during __post_init__."""
        # Construct with curriculum_level=35, max_level=30
        # __post_init__ should clamp curriculum_level to 29
        cfg = EvolutionConfig(curriculum_level=35, max_level=30)
        assert cfg.curriculum_level == 29
        assert cfg.max_level == 30

    def test_engine_curriculum_level_valid_after_init(self, engine_config):
        """Engine's champion_level should match the config's curriculum_level."""
        engine = EvolutionEngine(config=engine_config)
        assert engine.champion_level == engine_config.curriculum_level

    def test_engine_state_initialized(self, engine_config):
        """Engine should have all state variables initialized after __init__."""
        engine = EvolutionEngine(config=engine_config)
        assert engine.config is not None
        assert hasattr(engine, "_hall_of_fame")
        assert hasattr(engine, "_fitness_velocity")
        assert engine.config.max_level == 30


# ═══════════════════════════════════════════════════════════════════════
# 3. Reference solutions for L14-L16
# ═══════════════════════════════════════════════════════════════════════


def _encode_reference_solution(level: int) -> list[int]:
    """Encode a reference solution dict list into 32-bit instruction words.

    Uses the same make_rv32i function that the rest of the engine uses,
    so we exercise the same encode path.
    """
    from mutation_proposer import EBREAK, make_rv32i

    ref = reference_solution(level)
    program = []
    for instr_dict in ref:
        encoded = make_rv32i(
            name=instr_dict["name"],
            rd=instr_dict.get("rd", 0),
            rs1=instr_dict.get("rs1", 0),
            rs2=instr_dict.get("rs2", 0),
            imm=instr_dict.get("imm", 0),
        )
        program.append(encoded)
    # Reference solutions end with EBREAK for the VM to halt
    if program[-1] != EBREAK:
        program.append(EBREAK)
    return program


class TestReferenceSolutionEncoding:
    """Test that reference solutions for L14-L16 can be encoded."""

    def test_l14_reference_encodes(self):
        """L14 reference solution should encode without error."""
        program = _encode_reference_solution(14)
        assert len(program) > 0
        # Should be valid 32-bit unsigned integers
        for word in program:
            assert 0 <= word <= 0xFFFFFFFF

    def test_l15_reference_encodes(self):
        """L15 reference solution should encode without error."""
        program = _encode_reference_solution(15)
        assert len(program) > 0

    def test_l16_reference_encodes(self):
        """L16 reference solution should encode without error."""
        program = _encode_reference_solution(16)
        assert len(program) > 0

    def test_l14_ref_has_ebreak(self):
        """L14 reference program should end with EBREAK."""
        program = _encode_reference_solution(14)
        assert program[-1] == 0x00100073, "Reference should end with EBREAK"

    def test_l15_ref_has_ebreak(self):
        """L15 reference program should end with EBREAK."""
        program = _encode_reference_solution(15)
        assert program[-1] == 0x00100073

    def test_l16_ref_has_ebreak(self):
        """L16 reference program should end with EBREAK."""
        program = _encode_reference_solution(16)
        assert program[-1] == 0x00100073


# ═══════════════════════════════════════════════════════════════════════
# 4. VM execution of L14-L16 reference solutions
# ═══════════════════════════════════════════════════════════════════════


class TestReferenceSolutionExecution:
    """Execute L14-L16 reference solutions on the VM and verify output.

    These tests run the actual RV32I VM (safety-kernel or pure-Python),
    which requires the milestone0 module to be importable.
    """

    @pytest.mark.parametrize("level", [14, 15, 16])
    def test_reference_solution_executes_on_vm(self, level):
        """Each reference solution should execute successfully on the VM."""
        rng = random.Random(42)
        level_cls = get_task_level(level)
        task = level_cls(seed=rng.randint(0, 2**31 - 1))
        inst = task.generate_instance()

        program = _encode_reference_solution(level)
        inputs = inst["inputs"]

        result = _run_one_evaluation(
            kernel_path="",
            program=program,
            budget=10000,
            init_regs=inputs.get("registers", {}),
            init_mem=inputs.get("memory", {}),
        )
        assert result is not None, "VM returned None result"
        assert result.success, f"L{level} reference failed: {result.error}"

    @pytest.mark.parametrize("level", [14, 15, 16])
    def test_reference_solution_produces_correct_output(self, level):
        """Each reference solution should produce the expected register values."""
        rng = random.Random(42)
        level_cls = get_task_level(level)
        task = level_cls(seed=rng.randint(0, 2**31 - 1))
        inst = task.generate_instance()

        program = _encode_reference_solution(level)
        inputs = inst["inputs"]

        result = _run_one_evaluation(
            kernel_path="",
            program=program,
            budget=10000,
            init_regs=inputs.get("registers", {}),
            init_mem=inputs.get("memory", {}),
        )
        assert result.success, f"L{level} reference failed: {result.error}"

        expected = inst["expected_output"]["registers"]
        # Verify the key output register matches
        output_regs = result.registers
        for reg, exp_val in expected.items():
            actual_val = output_regs.get(reg, 0)
            assert actual_val == exp_val, (
                f"L{level} register x{reg}: expected {exp_val:#x}, got {actual_val:#x}"
            )

    @pytest.mark.parametrize("level", [14, 15, 16])
    def test_reference_solution_verify_score(self, level):
        """The verify() function should return 1.0 for the reference output."""
        rng = random.Random(42)
        level_cls = get_task_level(level)
        task = level_cls(seed=rng.randint(0, 2**31 - 1))
        inst = task.generate_instance()

        program = _encode_reference_solution(level)
        inputs = inst["inputs"]

        result = _run_one_evaluation(
            kernel_path="",
            program=program,
            budget=10000,
            init_regs=inputs.get("registers", {}),
            init_mem=inputs.get("memory", {}),
        )
        assert result.success

        output = {"registers": result.registers, "memory": result.memory}
        score = task.verify(output, inst)
        assert score >= 1.0, f"L{level} verify score {score} < 1.0 for reference solution"

    def test_l14_correct_output_register(self):
        """L14 reference should produce x17 = 2*x1 + x2 + 100."""
        rng = random.Random(42)
        task = get_task_level(14)(seed=rng.randint(0, 2**31 - 1))
        inst = task.generate_instance()
        program = _encode_reference_solution(14)
        inputs = inst["inputs"]

        result = _run_one_evaluation(
            kernel_path="", program=program, budget=10000,
            init_regs=inputs.get("registers", {}), init_mem=inputs.get("memory", {}),
        )
        assert result.success
        # x17 should hold the final output
        v1 = inputs["registers"].get(1, 0)
        v2 = inputs["registers"].get(2, 0)
        expected_x17 = (2 * v1 + v2 + 100) & 0xFFFFFFFF
        assert result.registers.get(17, 0) == expected_x17

    def test_l16_correct_output_register(self):
        """L16 reference should produce x19 = 4*x1 + 8*x2."""
        rng = random.Random(42)
        task = get_task_level(16)(seed=rng.randint(0, 2**31 - 1))
        inst = task.generate_instance()
        program = _encode_reference_solution(16)
        inputs = inst["inputs"]

        result = _run_one_evaluation(
            kernel_path="", program=program, budget=10000,
            init_regs=inputs.get("registers", {}), init_mem=inputs.get("memory", {}),
        )
        assert result.success
        v1 = inputs["registers"].get(1, 0)
        v2 = inputs["registers"].get(2, 0)
        expected_x19 = ((v1 << 2) + (v2 << 3)) & 0xFFFFFFFF
        assert result.registers.get(19, 0) == expected_x19
