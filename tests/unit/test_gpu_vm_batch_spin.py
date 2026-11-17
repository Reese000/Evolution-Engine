"""Batch VM must match scalar eval when a spinner shares the batch."""

from gpu_vm_farm import _execute_vm, _execute_vm_batch_numpy
from mutation_proposer import EBREAK, make_rv32i


def test_spinner_does_not_change_linear_result():
    linear = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=4, rs1=1, rs2=2),
        EBREAK,
    ]
    spinner = [
        make_rv32i("BEQ", rs1=0, rs2=0, imm=0),
        EBREAK,
    ]
    inputs = {"registers": {1: 3, 2: 5}, "memory": {}}
    budget = 4000
    batch = _execute_vm_batch_numpy([linear, spinner], [inputs, inputs], budget)
    scalar_lin = _execute_vm(linear, inputs["registers"], inputs["memory"], budget)
    scalar_spin = _execute_vm(spinner, inputs["registers"], inputs["memory"], budget)
    assert batch[0]["registers"] == scalar_lin["registers"]
    assert batch[0]["cycles"] == scalar_lin["cycles"]
    assert batch[1]["cycles"] == scalar_spin["cycles"]
    assert batch[1]["cycles"] == budget


def test_two_insn_spinner_timeout_matches_scalar():
    spin2 = [
        make_rv32i("ADDI", rd=7, rs1=0, imm=1),
        make_rv32i("BEQ", rs1=0, rs2=0, imm=-4),
        EBREAK,
    ]
    linear = [
        make_rv32i("ADD", rd=4, rs1=1, rs2=2),
        EBREAK,
    ]
    inputs = {"registers": {1: 3, 2: 5}, "memory": {}}
    budget = 4000
    batch = _execute_vm_batch_numpy([linear, spin2], [inputs, inputs], budget)
    scalar = _execute_vm(spin2, inputs["registers"], inputs["memory"], budget)
    assert batch[1]["cycles"] == budget
    assert batch[1]["cycles"] == scalar["cycles"]
    assert batch[1]["registers"] == scalar["registers"]
    assert batch[0]["registers"] == _execute_vm(linear, inputs["registers"], inputs["memory"], budget)["registers"]


def test_six_insn_x0_spin_timeouts_without_scalar_replay():
    nops = [make_rv32i("ADDI", rd=0, rs1=0, imm=0) for _ in range(5)]
    spin6 = nops + [make_rv32i("BEQ", rs1=0, rs2=0, imm=-20), EBREAK]
    linear = [make_rv32i("ADD", rd=4, rs1=1, rs2=2), EBREAK]
    inputs = {"registers": {1: 3, 2: 5}, "memory": {}}
    budget = 4000
    batch = _execute_vm_batch_numpy([linear, spin6], [inputs, inputs], budget)
    scalar = _execute_vm(spin6, inputs["registers"], inputs["memory"], budget)
    assert batch[1]["cycles"] == budget
    assert batch[1]["cycles"] == scalar["cycles"]
    assert batch[1]["registers"] == scalar["registers"]


def test_duplicate_programs_match_unique_eval():
    prog = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=4, rs1=1, rs2=2),
        EBREAK,
    ]
    spinner = [
        make_rv32i("BEQ", rs1=0, rs2=0, imm=0),
        EBREAK,
    ]
    inputs = {"registers": {1: 7, 2: 9}, "memory": {}}
    budget = 2500
    batch = _execute_vm_batch_numpy(
        [prog, spinner, prog, spinner],
        [inputs, inputs, inputs, inputs],
        budget,
    )
    assert batch[0]["registers"] == batch[2]["registers"]
    assert batch[0]["cycles"] == batch[2]["cycles"]
    assert batch[1]["cycles"] == budget
    assert batch[3]["cycles"] == budget
    scalar = _execute_vm(prog, inputs["registers"], inputs["memory"], budget)
    assert batch[0]["registers"] == scalar["registers"]


def test_bounded_loop_batch_matches_scalar():
    loop = [
        make_rv32i("ADDI", rd=12, rs1=0, imm=0),
        make_rv32i("ADDI", rd=16, rs1=0, imm=0),
        make_rv32i("SLTIU", rd=19, rs1=2, imm=16),
        make_rv32i("BEQ", rs1=19, rs2=0, imm=24),
        make_rv32i("ADDI", rd=12, rs1=12, imm=1),
        make_rv32i("ADDI", rd=16, rs1=16, imm=1),
        make_rv32i("BLT", rs1=16, rs2=2, imm=-8),
        EBREAK,
    ]
    spinner = [make_rv32i("BEQ", rs1=0, rs2=0, imm=0), EBREAK]
    inputs = {"registers": {1: 4, 2: 10}, "memory": {}}
    budget = 4000
    batch = _execute_vm_batch_numpy([loop, spinner], [inputs, inputs], budget)
    scalar = _execute_vm(loop, inputs["registers"], inputs["memory"], budget)
    assert batch[0]["registers"] == scalar["registers"]
    assert batch[0]["cycles"] == scalar["cycles"]
    assert batch[1]["cycles"] == budget


def test_scalar_tight_spin_reports_budget_cycles():
    spin = [
        make_rv32i("BEQ", rs1=0, rs2=0, imm=0),
        EBREAK,
    ]
    out = _execute_vm(spin, {1: 3, 2: 5}, {}, 4000)
    assert out["cycles"] == 4000


def test_scalar_counting_loop_still_halts():
    loop = [
        make_rv32i("ADDI", rd=12, rs1=0, imm=0),
        make_rv32i("ADDI", rd=16, rs1=0, imm=0),
        make_rv32i("SLTIU", rd=19, rs1=2, imm=16),
        make_rv32i("BEQ", rs1=19, rs2=0, imm=24),
        make_rv32i("ADDI", rd=12, rs1=12, imm=1),
        make_rv32i("ADDI", rd=16, rs1=16, imm=1),
        make_rv32i("BLT", rs1=16, rs2=2, imm=-8),
        EBREAK,
    ]
    out = _execute_vm(loop, {1: 4, 2: 10}, {}, 4000)
    assert out["cycles"] < 4000
    assert out["registers"].get(12) == 10


def test_vector_spin_cap_ignores_long_outlier():
    from gpu_vm_farm import _vector_spin_cap

    short = [1] * 66
    long = [1] * 400
    budget = 15500
    mixed = _vector_spin_cap([short, long], budget)
    assert mixed == _vector_spin_cap([short], budget)
    assert mixed == 330
    assert _vector_spin_cap([long], budget) == 480


def test_long_mutant_does_not_change_short_result():
    linear = [
        make_rv32i("ADDI", rd=3, rs1=1, imm=0),
        make_rv32i("ADD", rd=4, rs1=1, rs2=2),
        EBREAK,
    ]
    long_spin = [make_rv32i("ADDI", rd=0, rs1=0, imm=0)] * 400 + [
        make_rv32i("BEQ", rs1=0, rs2=0, imm=0),
        EBREAK,
    ]
    long_halt = [make_rv32i("ADDI", rd=5, rs1=1, imm=7)] + [0] * 200 + [EBREAK]
    inputs = {"registers": {1: 3, 2: 5}, "memory": {}}
    budget = 4000
    batch = _execute_vm_batch_numpy([linear, long_spin, long_halt], [inputs, inputs, inputs], budget)
    scalar = _execute_vm(linear, inputs["registers"], inputs["memory"], budget)
    assert batch[0]["registers"] == scalar["registers"]
    assert batch[0]["cycles"] == scalar["cycles"]
    assert batch[1]["cycles"] == budget
    scalar_halt = _execute_vm(long_halt, inputs["registers"], inputs["memory"], budget)
    assert batch[2]["registers"] == scalar_halt["registers"]
