"""Data augmentation utilities.

Register-renaming augmentation, synthetic data injection, and reference
solution injection — extracted from the monolithic train_model.py.
"""

from __future__ import annotations

import os
import random
import sys

import torch

# Ensure milestone1/ and parent are on sys.path so sibling modules resolve.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
_LOGIT_MASK_DIR = os.path.join(BASE_DIR, "..", "milestone0", "logit-masking")
if os.path.isdir(_LOGIT_MASK_DIR) and _LOGIT_MASK_DIR not in sys.path:
    sys.path.insert(0, _LOGIT_MASK_DIR)


# ── Register renaming data augmentation ───────────────────────────────
def register_rename_augment(
    instr_dicts: list[dict],
    k: int = 8,
    rng: random.Random | None = None,
) -> list[list[dict]]:
    """Generate K register-renamed variants of a program trace.

    RV32I programs are permutation-invariant w.r.t. most register
    assignments: swapping x1-x10, x2-x11, etc. preserves semantics
    (x0 is never permuted -- it is hardwired to zero).

    Only architecturally meaningful register fields are remapped
    per instruction format:
      R-type: rd, rs1, rs2
      I-type: rd, rs1
      S-type: rs1, rs2
      B-type: rs1, rs2
      U-type: rd only
      J-type: rd only

    Args:
        instr_dicts: list of instruction dicts (as returned by rv32i_decode).
        k:           number of random permutations to generate.
        rng:         optional Random instance for reproducibility.

    Returns:
        List of k new lists, each a register-renamed copy of instr_dicts.
        Empty if the trace has fewer than 2 instructions.
    """
    if len(instr_dicts) < 2:
        return []

    if rng is None:
        rng = random

    # Register fields that are architecturally meaningful per format.
    _RD_FORMATS = {"R", "I", "U", "J"}
    _RS1_FORMATS = {"R", "I", "S", "B"}
    _RS2_FORMATS = {"R", "S", "B"}

    augmented: list[list[dict]] = []

    # Check upfront: skip traces that write to x0 (writes to x0 are
    # discarded by hardware, so renaming around x0 writes is pointless).
    for instr in instr_dicts:
        fmt = instr.get("fmt", instr.get("format", ""))
        if fmt in _RD_FORMATS and instr.get("rd", 0) == 0:
            return []

    for _ in range(k):
        # Build a random permutation of registers 1..31.
        # x0 is never included (hardwired to zero).
        perm = list(range(1, 32))
        rng.shuffle(perm)

        new_program: list[dict] = []

        for instr in instr_dicts:
            new_instr = dict(instr)  # shallow copy

            # Determine format -- the decode function uses 'fmt' as the key
            fmt = instr.get("fmt", instr.get("format", ""))

            # Remap rd if this format uses a destination register
            if fmt in _RD_FORMATS:
                rd = instr.get("rd", 0)
                if rd != 0:
                    new_instr["rd"] = perm[rd - 1]

            # Remap rs1 if this format uses a source register 1
            if fmt in _RS1_FORMATS:
                rs1 = instr.get("rs1", 0)
                if rs1 != 0:
                    new_instr["rs1"] = perm[rs1 - 1]

            # Remap rs2 if this format uses a source register 2
            if fmt in _RS2_FORMATS:
                rs2 = instr.get("rs2", 0)
                if rs2 != 0:
                    new_instr["rs2"] = perm[rs2 - 1]

            new_program.append(new_instr)

        augmented.append(new_program)

    return augmented


def inject_synthetic_data(
    sequences: list[torch.Tensor],
    sample_weights: list[float],
    *,
    seed: int = 42,
    max_sequences: int = 5000,
) -> None:
    """Generate and append synthetic training sequences.

    Wraps the SyntheticDataGenerator to produce synthetic sequences
    covering instruction patterns, data flow chains, subsequences from
    references, and template instantiations.

    Synthetic data gets a uniform weight of 0.5 (lower than references).
    Appends directly to the input lists (mutable pattern).

    Args:
        sequences:      mutable list of tokenized tensors (extended in-place).
        sample_weights: mutable list of floats (extended in-place).
        seed:           RNG seed for reproducibility.
        max_sequences:  Cap on synthetic sequences (default 5000).
    """
    # Belt-and-suspenders: hard-cap at 5000 to prevent training slowness
    max_sequences = min(max_sequences, 5000)
    try:
        from synthetic_data import SyntheticDataGenerator

        synth_gen = SyntheticDataGenerator(seed=seed)
        synthetic_seqs = synth_gen.generate_all()
        synth_weight = 0.5
        if len(synthetic_seqs) > max_sequences:
            synthetic_seqs = synthetic_seqs[:max_sequences]
        for seq in synthetic_seqs:
            sequences.append(torch.tensor(seq, dtype=torch.long))
            sample_weights.append(synth_weight)
        print(f"[Train] Synthetic data: +{len(synthetic_seqs)} sequences (weight={synth_weight})")
        del synth_gen, synthetic_seqs
    except Exception as e:
        print(f"[Train] Synthetic data generation failed: {e}")


def inject_reference_solutions(
    tokenizer,
    sequences: list[torch.Tensor],
    sample_weights: list[float],
    ref_copies: dict[int, int] | None = None,
    *,
    rv32i_decode=None,
) -> None:
    """Inject reference solutions for ALL levels L0-L11.

    Teaches the model correct program structures regardless of what
    the champion has achieved. Complexity-scaled copy counts: simpler
    levels get fewer copies, harder levels get more.

    Appends directly to the input lists (mutable pattern).

    Args:
        tokenizer:    An RV32IFieldTokenizer instance.
        sequences:    mutable list of tokenized tensors (extended in-place).
        sample_weights: mutable list of floats (extended in-place).
        ref_copies:   dict mapping level -> number of copies to inject.
                      Defaults to the original hardcoded schedule.
        rv32i_decode: The decode function; if None, imports from rv32i_mask.
    """
    if ref_copies is None:
        ref_copies = {
            0: 5,
            1: 5,
            2: 5,
            3: 5,
            4: 5,
            5: 10,
            6: 10,
            7: 10,
            8: 20,
            9: 30,
            10: 30,
            11: 30,
            12: 40,
            13: 50,
        }

    if rv32i_decode is None:
        from rv32i_mask import decode as rv32i_decode

    ref_injected = 0
    try:
        from mutation_proposer import _encode_inst
        from tasks_rv32i import LEVELS, get_task_level

        inject_levels = sorted(LEVELS.keys())  # ALL levels L0-L11
        for level in inject_levels:
            try:
                level_cls = get_task_level(level)
                ref = level_cls.reference_solution()
                if not ref:
                    continue
                # Encode reference to words
                ref_words = []
                for inst in ref:
                    w = _encode_inst(inst)
                    if w is not None:
                        ref_words.append(w)
                ref_words.append(0x00100073)  # EBREAK
                # Decode to instruction dicts -> tokenize
                ref_dicts = []
                for w in ref_words:
                    try:
                        d = rv32i_decode(w)
                        if d:
                            ref_dicts.append(d)
                    except (ValueError, KeyError):
                        continue
                if not ref_dicts:
                    # R46: For levels without reference solutions (L7-L13),
                    # generate synthetic programs from the task instance.
                    # This fills the training data gap that causes val_loss
                    # regression when the curriculum tries these levels.
                    try:
                        inst = level_cls(seed=42).generate_instance()
                        expected = inst.get("expected_output", {}).get("registers", {})
                        if expected:
                            # Generate a simple program: load expected values into registers
                            synth_words = []
                            for reg_idx, reg_val in sorted(expected.items()):
                                if reg_val != 0:
                                    # ADDI rd, x0, imm
                                    synth_words.append(0x00000013 | (reg_idx << 7) | ((reg_val & 0xFFF) << 20))
                            if synth_words:
                                synth_words.append(0x00100073)  # EBREAK
                                synth_dicts = []
                                for w in synth_words:
                                    try:
                                        d = rv32i_decode(w)
                                        if d:
                                            synth_dicts.append(d)
                                    except (ValueError, KeyError):
                                        continue
                                if synth_dicts:
                                    ref_tokens = tokenizer.encode_trace(synth_dicts)
                                    if ref_tokens and len(ref_tokens) >= 2:
                                        copies = ref_copies.get(level, 10)
                                        for _ in range(copies):
                                            sequences.append(torch.tensor(ref_tokens, dtype=torch.long))
                                            sample_weights.append(0.8)  # synthetic: slightly lower weight
                                        ref_injected += 1
                    except Exception:
                        continue
                    continue
                ref_tokens = tokenizer.encode_trace(ref_dicts)
                if ref_tokens and len(ref_tokens) >= 2:
                    copies = ref_copies.get(level, 10)
                    for _ in range(copies):
                        sequences.append(torch.tensor(ref_tokens, dtype=torch.long))
                        sample_weights.append(1.0)  # reference solutions are highest quality
                    ref_injected += 1
            except Exception:
                continue
    except ImportError:
        pass

    if ref_injected > 0:
        print(f"[Train] Injected {ref_injected} reference solutions (L0-L11)")
