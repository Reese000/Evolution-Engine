#!/usr/bin/env python3
"""
rv32i_mask.py — RV32I Instruction Encoding / Decoding
======================================================

Provides the canonical RV32I (base integer) instruction encoding table
used by the Evolution Engine's transformer model and mutation proposer.

Exports:
    get_valid_combinations() -> list of (opcode, funct3, funct7, name, fmt)
    encode(name, rd=0, rs1=0, rs2=0, imm=0, validate=True) -> int
    decode(word, validate=True) -> dict
    is_valid(word) -> bool
    get_encodings() -> dict mapping name -> RV32IEncoding
    RV32IEncoding  -- namedtuple for instruction encoding metadata
    DC             -- don't-care sentinel for unused fields
    Opcode         -- opcode constants
"""

from typing import Dict, List, NamedTuple, Optional, Tuple

# =====================================================================
#  CONSTANTS
# =====================================================================

DC = -1  # Don't care sentinel for unused fields


class Opcode:
    """RV32I opcode constants (bits[6:0])."""

    LUI = 0x37
    AUIPC = 0x17
    JAL = 0x6F
    JALR = 0x67
    BRANCH = 0x63
    LOAD = 0x03
    STORE = 0x23
    OP_IMM = 0x13
    OP = 0x33
    FENCE = 0x0F
    SYSTEM = 0x73


# =====================================================================
#  ENCODING TABLE
# =====================================================================

# Format constants
_R = "R"
_I = "I"
_S = "S"
_B = "B"
_U = "U"
_J = "J"

# Each entry: (opcode, funct3, funct7, name, format)
# funct3=-1 means "not used for this instruction"
# funct7=-1 means "not used for this instruction"
_INSTRUCTIONS = [
    # ── R-type (opcode=0x33) ──────────────────────────────────────
    (Opcode.OP, 0, 0, "ADD", _R),
    (Opcode.OP, 0, 0x20, "SUB", _R),
    (Opcode.OP, 1, 0, "SLL", _R),
    (Opcode.OP, 2, 0, "SLT", _R),
    (Opcode.OP, 3, 0, "SLTU", _R),
    (Opcode.OP, 4, 0, "XOR", _R),
    (Opcode.OP, 5, 0, "SRL", _R),
    (Opcode.OP, 5, 0x20, "SRA", _R),
    (Opcode.OP, 6, 0, "OR", _R),
    (Opcode.OP, 7, 0, "AND", _R),
    # ── I-type arithmetic (opcode=0x13) ───────────────────────────
    (Opcode.OP_IMM, 0, -1, "ADDI", _I),
    (Opcode.OP_IMM, 2, -1, "SLTI", _I),
    (Opcode.OP_IMM, 3, -1, "SLTIU", _I),
    (Opcode.OP_IMM, 4, -1, "XORI", _I),
    (Opcode.OP_IMM, 6, -1, "ORI", _I),
    (Opcode.OP_IMM, 7, -1, "ANDI", _I),
    (Opcode.OP_IMM, 1, 0, "SLLI", _I),
    (Opcode.OP_IMM, 5, 0, "SRLI", _I),
    (Opcode.OP_IMM, 5, 0x20, "SRAI", _I),
    # ── I-type load (opcode=0x03) ─────────────────────────────────
    (Opcode.LOAD, 0, -1, "LB", _I),
    (Opcode.LOAD, 1, -1, "LH", _I),
    (Opcode.LOAD, 2, -1, "LW", _I),
    (Opcode.LOAD, 4, -1, "LBU", _I),
    (Opcode.LOAD, 5, -1, "LHU", _I),
    # ── S-type (opcode=0x23) ──────────────────────────────────────
    (Opcode.STORE, 0, -1, "SB", _S),
    (Opcode.STORE, 1, -1, "SH", _S),
    (Opcode.STORE, 2, -1, "SW", _S),
    # ── B-type (opcode=0x63) ──────────────────────────────────────
    (Opcode.BRANCH, 0, -1, "BEQ", _B),
    (Opcode.BRANCH, 1, -1, "BNE", _B),
    (Opcode.BRANCH, 4, -1, "BLT", _B),
    (Opcode.BRANCH, 5, -1, "BGE", _B),
    (Opcode.BRANCH, 6, -1, "BLTU", _B),
    (Opcode.BRANCH, 7, -1, "BGEU", _B),
    # ── U-type ────────────────────────────────────────────────────
    (Opcode.LUI, -1, -1, "LUI", _U),
    (Opcode.AUIPC, -1, -1, "AUIPC", _U),
    # ── J-type (opcode=0x6F) ──────────────────────────────────────
    (Opcode.JAL, -1, -1, "JAL", _J),
    # ── I-type jump (opcode=0x67) ─────────────────────────────────
    (Opcode.JALR, 0, -1, "JALR", _I),
    # ── I-type fence (opcode=0x0F) ────────────────────────────────
    (Opcode.FENCE, 0, -1, "FENCE", _I),
    (Opcode.FENCE, 1, -1, "FENCE.I", _I),
    # ── I-type system (opcode=0x73) ───────────────────────────────
    # ECALL and EBREAK share opcode=0x73, funct3=0
    # They are distinguished by the immediate: ECALL=0, EBREAK=1
    (Opcode.SYSTEM, 0, 0, "ECALL", _I),
    (Opcode.SYSTEM, 0, 1, "EBREAK", _I),
    # CSR instructions (Zicsr extension, commonly included)
    (Opcode.SYSTEM, 1, -1, "CSRRW", _I),
    (Opcode.SYSTEM, 2, -1, "CSRRS", _I),
    (Opcode.SYSTEM, 3, -1, "CSRRC", _I),
    (Opcode.SYSTEM, 5, -1, "CSRRWI", _I),
    (Opcode.SYSTEM, 6, -1, "CSRRSI", _I),
    (Opcode.SYSTEM, 7, -1, "CSRRCI", _I),
]


# =====================================================================
#  NAMED TUPLE FOR ENCODING METADATA
# =====================================================================


class RV32IEncoding(NamedTuple):
    """Metadata for a single RV32I instruction encoding."""

    opcode: int
    funct3: int
    funct7: int
    name: str
    fmt: str


# =====================================================================
#  PUBLIC API
# =====================================================================


def get_valid_combinations() -> list[tuple[int, int, int, str, str]]:
    """Return list of (opcode, funct3, funct7, name, format) for all
    valid RV32I instruction encodings.

    This is the canonical encoding table consumed by
    RV32IVocab._build_instruction_map().
    """
    return list(_INSTRUCTIONS)


def get_encodings() -> dict[str, RV32IEncoding]:
    """Return a dict mapping instruction name -> RV32IEncoding metadata."""
    result = {}
    for opcode, funct3, funct7, name, fmt in _INSTRUCTIONS:
        result[name] = RV32IEncoding(
            opcode=opcode,
            funct3=funct3,
            funct7=funct7,
            name=name,
            fmt=fmt,
        )
    return result


def _sign_extend(value: int, bits: int) -> int:
    """Sign-extend an unsigned integer of *bits* width to Python int."""
    mask = (1 << bits) - 1
    value = value & mask
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _split_imm_s(imm: int) -> tuple[int, int]:
    """Split a 12-bit signed immediate into S-type fields:
    imm[11:5] and imm[4:0].
    """
    imm = imm & 0xFFF  # 12-bit unsigned representation
    return (imm >> 5) & 0x7F, imm & 0x1F


def _split_imm_b(imm: int) -> tuple[int, int, int, int]:
    """Split a 13-bit signed immediate (must be even) into B-type fields:
    imm[12], imm[10:5], imm[4:1], imm[11].

    Returns (bit12, bits_10_5, bits_4_1, bit11).
    """
    imm = imm & 0x1FFF  # 13-bit unsigned representation (bit 12 is sign)
    bit12 = (imm >> 12) & 1
    bit11 = (imm >> 11) & 1
    bits_10_5 = (imm >> 5) & 0x3F
    bits_4_1 = (imm >> 1) & 0xF
    return bit12, bits_10_5, bits_4_1, bit11


def _split_imm_j(imm: int) -> tuple[int, int, int, int]:
    """Split a 21-bit signed immediate into J-type fields:
    imm[20], imm[10:1], imm[11], imm[19:12].

    Returns (bit20, bits_10_1, bit11, bits_19_12).
    """
    imm = imm & 0x1FFFFF  # 21-bit unsigned representation (bit 20 is sign)
    bit20 = (imm >> 20) & 1
    bits_19_12 = (imm >> 12) & 0xFF
    bit11 = (imm >> 11) & 1
    bits_10_1 = (imm >> 1) & 0x3FF
    return bit20, bits_10_1, bit11, bits_19_12


def encode(name: str, rd: int = 0, rs1: int = 0, rs2: int = 0, imm: int = 0, validate: bool = True) -> int:
    """Encode an RV32I instruction to a 32-bit word.

    Args:
        name:     Instruction mnemonic (e.g. 'ADDI', 'LW', 'BEQ').
        rd:       Destination register index (0-31).
        rs1:      Source register 1 index (0-31).
        rs2:      Source register 2 index (0-31).
        imm:      Immediate value (signed, width depends on format).
        validate: If True, raise ValueError for invalid instructions.

    Returns:
        32-bit integer encoding of the instruction.
    """
    # Look up encoding metadata
    enc = None
    for opcode, funct3, funct7, iname, fmt in _INSTRUCTIONS:
        if iname == name:
            enc = (opcode, funct3, funct7, fmt)
            break

    if enc is None:
        if validate:
            raise ValueError(f"Unknown instruction: {name}")
        return 0

    opcode, funct3, funct7, fmt = enc

    # Build the 32-bit word
    word = opcode & 0x7F  # bits[6:0]

    if fmt == _R:
        # R-type: funct7 | rs2 | rs1 | funct3 | rd | opcode
        word |= (rd & 0x1F) << 7
        word |= (funct3 & 0x7) << 12
        word |= (rs1 & 0x1F) << 15
        word |= (rs2 & 0x1F) << 20
        word |= (funct7 & 0x7F) << 25

    elif fmt == _I:
        # Special handling for shift instructions (SLLI, SRLI, SRAI)
        if name in ("SLLI", "SRLI", "SRAI"):
            shamt = imm & 0x1F
            word |= (rd & 0x1F) << 7
            word |= (funct3 & 0x7) << 12
            word |= (rs1 & 0x1F) << 15
            word |= (shamt & 0x1F) << 20
            word |= (funct7 & 0x7F) << 25
        # Special handling for ECALL/EBREAK
        elif name == "ECALL":
            word |= 0  # imm = 0
        elif name == "EBREAK":
            word |= 1 << 20  # imm = 1, bit 20 set
        # CSR instructions: imm field is CSR address (or zimm for *I variants)
        elif name in ("CSRRW", "CSRRS", "CSRRC", "CSRRWI", "CSRRSI", "CSRRCI"):
            word |= (rd & 0x1F) << 7
            word |= (funct3 & 0x7) << 12
            word |= (rs1 & 0x1F) << 15
            word |= (imm & 0xFFF) << 20
        else:
            # Standard I-type: imm[11:0] | rs1 | funct3 | rd | opcode
            word |= (rd & 0x1F) << 7
            word |= (funct3 & 0x7) << 12
            word |= (rs1 & 0x1F) << 15
            word |= (imm & 0xFFF) << 20

    elif fmt == _S:
        # S-type: imm[11:5] | rs2 | rs1 | funct3 | imm[4:0] | opcode
        hi, lo = _split_imm_s(imm)
        word |= (lo & 0x1F) << 7
        word |= (funct3 & 0x7) << 12
        word |= (rs1 & 0x1F) << 15
        word |= (rs2 & 0x1F) << 20
        word |= (hi & 0x7F) << 25

    elif fmt == _B:
        # B-type: imm[12|10:5] | rs2 | rs1 | funct3 | imm[4:1|11] | opcode
        bit12, bits_10_5, bits_4_1, bit11 = _split_imm_b(imm)
        word |= (bit11 & 1) << 7
        word |= (bits_4_1 & 0xF) << 8
        word |= (funct3 & 0x7) << 12
        word |= (rs1 & 0x1F) << 15
        word |= (rs2 & 0x1F) << 20
        word |= (bits_10_5 & 0x3F) << 25
        word |= (bit12 & 1) << 31

    elif fmt == _U:
        # U-type: imm[31:12] | rd | opcode
        word |= (rd & 0x1F) << 7
        word |= (imm & 0xFFFFF) << 12

    elif fmt == _J:
        # J-type: imm[20|10:1|11|19:12] | rd | opcode
        bit20, bits_10_1, bit11, bits_19_12 = _split_imm_j(imm)
        word |= (rd & 0x1F) << 7
        word |= (bits_19_12 & 0xFF) << 12
        word |= (bit11 & 1) << 20
        word |= (bits_10_1 & 0x3FF) << 21
        word |= (bit20 & 1) << 31

    if validate and not is_valid(word):
        raise ValueError(f"Encoded word 0x{word:08X} for {name} is not a recognized " f"RV32I instruction")

    return word


def decode(word: int, validate: bool = True) -> dict | None:
    """Decode a 32-bit word into an instruction dictionary.

    Args:
        word:     32-bit integer (unsigned).
        validate: If True, raise ValueError for unrecognized instructions.

    Returns:
        Dict with keys: 'name', 'format', 'opcode', 'funct3', 'funct7',
        'rd', 'rs1', 'rs2', 'imm'.

        For R-type: rs2, rs1, rd are populated; imm=0.
        For I-type: rs1, rd, imm are populated; rs2=0.
        For S-type: rs2, rs1, imm are populated; rd=0.
        For B-type: rs2, rs1, imm are populated; rd=0.
        For U-type: rd, imm are populated; rs1=rs2=0.
        For J-type: rd, imm are populated; rs1=rs2=0.

        Returns None if decode fails and validate is False.
    """
    word = word & 0xFFFFFFFF

    opcode = word & 0x7F
    rd = (word >> 7) & 0x1F
    funct3 = (word >> 12) & 0x7
    rs1 = (word >> 15) & 0x1F
    rs2 = (word >> 20) & 0x1F
    funct7 = (word >> 25) & 0x7F

    # Try to find matching instruction
    best_match = None

    for enc_opcode, enc_funct3, enc_funct7, name, fmt in _INSTRUCTIONS:
        if enc_opcode != opcode:
            continue

        if fmt == _R:
            if enc_funct3 == funct3 and enc_funct7 == funct7:
                best_match = (name, fmt, enc_funct3, enc_funct7)
                break

        elif fmt == _U:
            # U-type: no funct3 distinction (LUI vs AUIPC differentiated by opcode)
            if enc_funct3 == -1:
                best_match = (name, fmt, funct3, 0)
                break

        elif fmt == _J:
            # J-type: JAL (opcode=0x6F)
            if enc_funct3 == -1:
                best_match = (name, fmt, funct3, 0)
                break

        elif name in ("SLLI", "SRLI", "SRAI"):
            # Shift instructions: use funct3 + funct7 to distinguish
            if enc_funct3 == funct3 and enc_funct7 == funct7:
                best_match = (name, fmt, enc_funct3, enc_funct7)
                break

        elif name == "ECALL":
            # ECALL: opcode=0x73, funct3=0, imm=0 (bit 20 = 0)
            if enc_funct3 == funct3 and not (word & (1 << 20)):
                best_match = (name, fmt, enc_funct3, 0)
                break

        elif name == "EBREAK":
            # EBREAK: opcode=0x73, funct3=0, imm=1 (bit 20 = 1)
            if enc_funct3 == funct3 and (word & (1 << 20)):
                best_match = (name, fmt, enc_funct3, 1)
                break

        elif enc_funct3 == funct3 or enc_funct3 == -1:
            # Standard match by funct3
            best_match = (name, fmt, enc_funct3 if enc_funct3 >= 0 else funct3, 0)
            # For I/S/B types with matching funct3, continue to find exact match
            if enc_funct3 == funct3:
                break

    if best_match is None:
        if validate:
            raise ValueError(f"Cannot decode instruction word: 0x{word:08X}")
        return None

    name, fmt, decoded_funct3, decoded_funct7 = best_match

    # Extract immediate based on format
    if fmt == _R:
        imm = 0
    elif fmt == _I:
        if name in ("SLLI", "SRLI", "SRAI"):
            imm = (word >> 20) & 0x1F
            # Sign-extend SRAI shift amount (top bit of funct7)
        elif name == "ECALL":
            imm = 0
        elif name == "EBREAK":
            imm = 1
        else:
            imm = _sign_extend((word >> 20) & 0xFFF, 12)
    elif fmt == _S:
        hi = (word >> 25) & 0x7F
        lo = (word >> 7) & 0x1F
        imm = _sign_extend((hi << 5) | lo, 12)
    elif fmt == _B:
        bit11 = (word >> 7) & 1
        bits_4_1 = (word >> 8) & 0xF
        bit12 = (word >> 31) & 1
        bits_10_5 = (word >> 25) & 0x3F
        imm_unsigned = (bit12 << 12) | (bits_10_5 << 5) | (bit11 << 11) | (bits_4_1 << 1)
        imm = _sign_extend(imm_unsigned, 13)
    elif fmt == _U:
        imm = _sign_extend((word >> 12) & 0xFFFFF, 20)
    elif fmt == _J:
        bits_19_12 = (word >> 12) & 0xFF
        bit11 = (word >> 20) & 1
        bits_10_1 = (word >> 21) & 0x3FF
        bit20 = (word >> 31) & 1
        imm_unsigned = (bit20 << 20) | (bits_19_12 << 12) | (bit11 << 11) | (bits_10_1 << 1)
        imm = _sign_extend(imm_unsigned, 21)

    return {
        "name": name,
        "format": fmt,
        "opcode": opcode,
        "funct3": decoded_funct3,
        "funct7": decoded_funct7,
        "rd": rd,
        "rs1": rs1,
        "rs2": rs2,
        "imm": imm,
    }


def is_valid(word: int) -> bool:
    """Check if a 32-bit word is a recognized RV32I instruction.

    Returns True if the word decodes to a known instruction.
    """
    try:
        result = decode(word, validate=False)
        return result is not None
    except Exception:
        return False
