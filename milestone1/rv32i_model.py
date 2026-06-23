#!/usr/bin/env python3
"""
rv32i_model.py — RV32I-Aware Micro-Model
=========================================
A decoder-only transformer with field-level tokenization for RISC-V RV32I
(47 instructions, 6 encoding formats, 32 registers).

Adapted from simulation/micro_model.py for the real RV32I ISA.

Architecture:
  - Hidden dim 256, 8 layers, 8 heads, d_ff=1024
  - Field-level vocabulary: 47 instruction types + 32 registers + 8 funct3 +
    128 funct7 + 1500 immediate values + special tokens = 2048 vocab
  - Causal attention mask, pre-norm LayerNorm, residual connections
  - Tied embedding/head weights for parameter efficiency
  - Logit masking enforces valid instruction field sequences per format

Key differences from simulation (16-bit TinyVM) to real RV32I:
  - 47 instruction types (not 16)
  - 6 encoding formats: R, I, S, B, U, J (not 3)
  - 32 registers x0-x31 (not 16)
  - 32-bit instruction words (not 16-bit)
  - Full 12-bit immediates, 20-bit upper immediates
  - X0 hardwired to zero

Spec reference: RISC-V Unprivileged Spec v20191213
"""

import math
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import RV32I encoding table for building instruction mapping
sys.path.insert(0, '../milestone0/logit-masking')
from rv32i_mask import get_valid_combinations, is_valid, encode as rv_encode, decode as rv_decode


# =====================================================================
#  VOCABULARY CONSTANTS
# =====================================================================

class RV32IVocab:
    """Token ID constants for the RV32I field-level vocabulary.

    Token layout (2048 total):
        PAD          = 0
        SEP          = 1
        CLS          = 2
        MASK         = 3
        inst_0..46   = 4..50       (47 instruction types)
        reg_0..31    = 51..82      (32 registers x0-x31)
        funct3_0..7  = 83..90      (8 funct3 values)
        funct7_0..127 = 91..218    (128 funct7 values, for logit masking)
        imm_0..1499  = 219..1718   (1500 immediate values)
        NO_REG       = 1719
        Reserved     = 1720..2047  (328 for future expansion)
    """
    # Special tokens
    PAD = 0
    SEP = 1
    CLS = 2
    MASK = 3

    # Token ranges
    INST_START = 4       # 4..50   (47 instruction types)
    INST_COUNT = 47
    REG_START = 51       # 51..82  (32 registers x0-x31)
    REG_COUNT = 32
    FUNCT3_START = 83    # 83..90  (8 funct3 values)
    FUNCT3_COUNT = 8
    FUNCT7_START = 91    # 91..218 (128 funct7 values)
    FUNCT7_COUNT = 128
    IMM_START = 219      # 219..1718 (1500 immediate values)
    IMM_COUNT = 1500
    NO_REG = 1719
    VOCAB_SIZE = 2048

    # Fields per instruction (fixed-length encoding)
    # Positions: [inst, funct3, rd, rs1, rs2, imm, SEP]
    TOKENS_PER_INST = 7

    # Format constants
    R_FORMAT = 'R'
    I_FORMAT = 'I'
    S_FORMAT = 'S'
    B_FORMAT = 'B'
    U_FORMAT = 'U'
    J_FORMAT = 'J'

    # Instruction mapping cache (lazily built from rv32i_mask)
    _INSTRUCTION_NAMES: List[str] = []          # token_index -> name
    _NAME_TO_TOKEN: Dict[str, int] = {}         # name -> token_id
    _NAME_TO_FORMAT: Dict[str, str] = {}        # name -> format
    _NAME_TO_FUNCT3: Dict[str, int] = {}        # name -> funct3 value
    _NAME_TO_FUNCT7: Dict[str, int] = {}        # name -> funct7 value
    _NAME_TO_OPCODE: Dict[str, int] = {}        # name -> opcode
    _TOKEN_TO_NAME: Dict[int, str] = {}         # token_id -> name
    _BUILT = False

    @classmethod
    def _build_instruction_map(cls):
        """Build instruction-to-token mapping from RV32I encoding table."""
        if cls._BUILT:
            return

        combos = get_valid_combinations()
        # Sort by (opcode, name) for deterministic ordering
        combos_sorted = sorted(combos, key=lambda x: (x[0], x[3]))

        for i, (opcode, funct3, funct7, name, fmt) in enumerate(combos_sorted):
            token_id = cls.INST_START + i
            cls._INSTRUCTION_NAMES.append(name)
            cls._NAME_TO_TOKEN[name] = token_id
            cls._NAME_TO_FORMAT[name] = fmt
            cls._NAME_TO_FUNCT3[name] = funct3 if funct3 >= 0 else 0
            cls._NAME_TO_FUNCT7[name] = funct7 if funct7 >= 0 else 0
            cls._NAME_TO_OPCODE[name] = opcode
            cls._TOKEN_TO_NAME[token_id] = name

        cls._BUILT = True

    @classmethod
    def get_instruction_names(cls) -> List[str]:
        cls._build_instruction_map()
        return list(cls._INSTRUCTION_NAMES)

    @classmethod
    def get_instruction_token(cls, name: str) -> int:
        cls._build_instruction_map()
        return cls._NAME_TO_TOKEN[name]

    @classmethod
    def get_instruction_name(cls, token_id: int) -> str:
        cls._build_instruction_map()
        return cls._TOKEN_TO_NAME.get(token_id, "UNKNOWN")

    @classmethod
    def get_format(cls, name_or_token: str | int) -> str:
        cls._build_instruction_map()
        if isinstance(name_or_token, int):
            name = cls._TOKEN_TO_NAME.get(name_or_token, "UNKNOWN")
        else:
            name = name_or_token
        return cls._NAME_TO_FORMAT.get(name, cls.R_FORMAT)

    @classmethod
    def get_funct3(cls, name_or_token: str | int) -> int:
        cls._build_instruction_map()
        if isinstance(name_or_token, int):
            name = cls._TOKEN_TO_NAME.get(name_or_token, "UNKNOWN")
        else:
            name = name_or_token
        return cls._NAME_TO_FUNCT3.get(name, 0)

    @classmethod
    def get_opcode(cls, name_or_token: str | int) -> int:
        cls._build_instruction_map()
        if isinstance(name_or_token, int):
            name = cls._TOKEN_TO_NAME.get(name_or_token, "UNKNOWN")
        else:
            name = name_or_token
        return cls._NAME_TO_OPCODE.get(name, 0)

    @classmethod
    def is_instruction(cls, token: int) -> bool:
        return cls.INST_START <= token < cls.INST_START + cls.INST_COUNT

    @classmethod
    def is_register(cls, token: int) -> bool:
        return cls.REG_START <= token < cls.REG_START + cls.REG_COUNT

    @classmethod
    def is_funct3(cls, token: int) -> bool:
        return cls.FUNCT3_START <= token < cls.FUNCT3_START + cls.FUNCT3_COUNT

    @classmethod
    def is_funct7(cls, token: int) -> bool:
        return cls.FUNCT7_START <= token < cls.FUNCT7_START + cls.FUNCT7_COUNT

    @classmethod
    def is_imm(cls, token: int) -> bool:
        return cls.IMM_START <= token < cls.IMM_START + cls.IMM_COUNT

    @classmethod
    def is_special(cls, token: int) -> bool:
        return token in (cls.PAD, cls.SEP, cls.CLS, cls.MASK, cls.NO_REG)

    @classmethod
    def register_token(cls, reg: int) -> int:
        """Get token for register x0-x31."""
        return cls.REG_START + max(0, min(reg, cls.REG_COUNT - 1))

    @classmethod
    def funct3_token(cls, f3: int) -> int:
        """Get token for funct3 value 0-7."""
        return cls.FUNCT3_START + max(0, min(f3, cls.FUNCT3_COUNT - 1))

    @classmethod
    def funct7_token(cls, f7: int) -> int:
        """Get token for funct7 value 0-127."""
        return cls.FUNCT7_START + max(0, min(f7, cls.FUNCT7_COUNT - 1))

    @classmethod
    def imm_token(cls, imm: int) -> int:
        """Get token for immediate value.

        Maps signed immediate values to token IDs with an offset bias.
        Values outside [-749, 750] are clamped.
        """
        # Bias to make signed values non-negative
        bias = cls.IMM_COUNT // 2  # 750
        idx = imm + bias
        if idx < 0:
            idx = 0
        elif idx >= cls.IMM_COUNT:
            idx = cls.IMM_COUNT - 1
        return cls.IMM_START + idx

    @classmethod
    def imm_value(cls, token: int) -> int:
        """Convert imm token back to signed immediate value."""
        idx = token - cls.IMM_START
        bias = cls.IMM_COUNT // 2
        return idx - bias


# =====================================================================
#  FIELD TOKENIZER
# =====================================================================

class RV32IFieldTokenizer:
    """Field-level tokenizer for RV32I instructions.

    Encodes each instruction as 7 tokens:
        [inst_type, funct3, rd, rs1, rs2, imm, SEP]

    Format-dependent field mapping:
        R-type: [inst, funct3, rd,   rs1, rs2,   NO_REG, SEP]
        I-type: [inst, funct3, rd,   rs1, NO_REG, imm,    SEP]
        S-type: [inst, funct3, NO_REG, rs1, rs2,   imm,    SEP]
        B-type: [inst, funct3, NO_REG, rs1, rs2,   imm,    SEP]
        U-type: [inst, NO_REG, rd,   NO_REG, NO_REG, imm,    SEP]
        J-type: [inst, NO_REG, rd,   NO_REG, NO_REG, imm,    SEP]
    """

    def __init__(self):
        self.vocab = RV32IVocab
        # Ensure instruction map is built
        self.vocab._build_instruction_map()

    def encode_instruction(self, inst: Dict) -> List[int]:
        """Encode a single instruction dict to token IDs.

        Args:
            inst: dict with keys 'name' and optionally 'rd', 'rs1', 'rs2',
                  'imm'. Fields unused by the instruction's format may be
                  absent or None.

        Returns:
            List of 7 token IDs.
        """
        name = inst['name']
        fmt = self.vocab.get_format(name)
        tokens = []

        # Position 0: instruction type
        tokens.append(self.vocab.get_instruction_token(name))

        # Position 1: funct3 (or NO_REG for U/J)
        if fmt in (self.vocab.U_FORMAT, self.vocab.J_FORMAT):
            tokens.append(self.vocab.NO_REG)
        else:
            f3 = self.vocab.get_funct3(name)
            tokens.append(self.vocab.funct3_token(f3))

        # Position 2: rd (or NO_REG for S/B)
        if fmt in (self.vocab.S_FORMAT, self.vocab.B_FORMAT):
            tokens.append(self.vocab.NO_REG)
        else:
            rd = inst.get('rd', 0)
            tokens.append(self.vocab.register_token(rd))

        # Position 3: rs1 (or NO_REG for U/J)
        if fmt in (self.vocab.U_FORMAT, self.vocab.J_FORMAT):
            tokens.append(self.vocab.NO_REG)
        else:
            rs1 = inst.get('rs1', 0)
            tokens.append(self.vocab.register_token(rs1))

        # Position 4: rs2 (or NO_REG for I/U/J)
        if fmt in (self.vocab.I_FORMAT, self.vocab.U_FORMAT, self.vocab.J_FORMAT):
            tokens.append(self.vocab.NO_REG)
        else:
            rs2 = inst.get('rs2', 0)
            tokens.append(self.vocab.register_token(rs2))

        # Position 5: imm (or NO_REG for R)
        if fmt == self.vocab.R_FORMAT:
            tokens.append(self.vocab.NO_REG)
        else:
            imm = inst.get('imm', 0)
            tokens.append(self.vocab.imm_token(imm))

        # Position 6: SEP
        tokens.append(self.vocab.SEP)

        return tokens

    def encode_trace(self, trace: List[Dict]) -> List[int]:
        """Encode a list of instruction dicts to a flat token sequence.

        Args:
            trace: list of instruction dicts

        Returns:
            Flat list of token IDs for the entire trace.
        """
        tokens = []
        for inst in trace:
            tokens.extend(self.encode_instruction(inst))
        return tokens

    def decode_tokens(self, tokens: List[int]) -> Optional[Dict]:
        """Decode a flat token sequence back to at most one instruction dict.

        Takes the first full instruction (7 tokens ending in SEP) from the
        sequence, ignoring leading special tokens.

        Args:
            tokens: list of token IDs

        Returns:
            Instruction dict with keys name, format, rd, rs1, rs2, imm,
            or None if decoding fails.
        """
        # Strip leading special tokens
        clean = [t for t in tokens
                 if t not in (self.vocab.PAD, self.vocab.CLS, self.vocab.MASK)]

        # Need at least 7 tokens for one instruction
        if len(clean) < self.vocab.TOKENS_PER_INST:
            return None

        inst_toks = clean[:self.vocab.TOKENS_PER_INST]

        # Check final token is SEP
        if inst_toks[-1] != self.vocab.SEP:
            return None

        # Decode instruction type (position 0)
        inst_tok = inst_toks[0]
        if not self.vocab.is_instruction(inst_tok):
            return None
        name = self.vocab.get_instruction_name(inst_tok)
        fmt = self.vocab.get_format(name)

        result = {
            'name': name,
            'format': fmt,
        }

        # Position 1: funct3 (token or NO_REG)
        f3_tok = inst_toks[1]
        if f3_tok == self.vocab.NO_REG:
            result['funct3'] = None
        elif self.vocab.is_funct3(f3_tok):
            result['funct3'] = f3_tok - self.vocab.FUNCT3_START
        else:
            result['funct3'] = None

        # Position 2: rd
        rd_tok = inst_toks[2]
        if rd_tok == self.vocab.NO_REG:
            result['rd'] = None
        elif self.vocab.is_register(rd_tok):
            result['rd'] = rd_tok - self.vocab.REG_START
        else:
            result['rd'] = None

        # Position 3: rs1
        rs1_tok = inst_toks[3]
        if rs1_tok == self.vocab.NO_REG:
            result['rs1'] = None
        elif self.vocab.is_register(rs1_tok):
            result['rs1'] = rs1_tok - self.vocab.REG_START
        else:
            result['rs1'] = None

        # Position 4: rs2
        rs2_tok = inst_toks[4]
        if rs2_tok == self.vocab.NO_REG:
            result['rs2'] = None
        elif self.vocab.is_register(rs2_tok):
            result['rs2'] = rs2_tok - self.vocab.REG_START
        else:
            result['rs2'] = None

        # Position 5: imm
        imm_tok = inst_toks[5]
        if imm_tok == self.vocab.NO_REG:
            result['imm'] = None
        elif self.vocab.is_imm(imm_tok):
            result['imm'] = self.vocab.imm_value(imm_tok)
        else:
            result['imm'] = None

        return result


# =====================================================================
#  CAUSAL SELF-ATTENTION
# =====================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with pre-computed causal mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer (filled on first forward)
        self.register_buffer("causal_mask", None, persistent=False)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Build or retrieve the causal attention mask."""
        if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            )
            self.register_buffer("causal_mask", mask, persistent=False)
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, d_model)"""
        B, T, D = x.shape
        device = x.device

        # QKV projection
        qkv = self.qkv(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, D)

        # Reshape to multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        # Apply causal mask
        mask = self._build_causal_mask(T, device)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)

        # Softmax + dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum
        y = attn @ v  # (B, H, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y


# =====================================================================
#  TRANSFORMER BLOCK (Pre-Norm)
# =====================================================================

class TransformerBlock(nn.Module):
    """Decoder transformer block with pre-norm LayerNorm.

    Sequence:
        x = x + SelfAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# =====================================================================
#  RV32I TRANSFORMER — Full Decoder-Only Model
# =====================================================================

class RV32ITransformer(nn.Module):
    """Decoder-only transformer for RV32I instruction prediction.

    Architecture:
        - Token embedding + learned position embedding
        - 8 transformer blocks (pre-norm, causal attention, SwiGLU FFN)
        - Tied embedding/head weights (weight tying)
        - Logit masking for format-dependent field prediction

    Args:
        vocab_size: number of tokens in the vocabulary
        d_model: embedding/hidden dimension
        n_layers: number of transformer blocks
        n_heads: number of attention heads
        d_ff: feed-forward hidden dimension
        max_seq_len: maximum sequence length (for position embeddings)
        dropout: dropout probability
        tie_weights: share weights between embedding and output head
    """

    def __init__(
        self,
        vocab_size: int = RV32IVocab.VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 8,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Ensure vocab is built
        RV32IVocab._build_instruction_map()

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=RV32IVocab.PAD)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Output head (LM head for next-token prediction)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        """Initialize weights with small normal noise."""
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
        # Reset padding embedding to near zero
        with torch.no_grad():
            self.token_embedding.weight[RV32IVocab.PAD].zero_()

    def forward(
        self,
        input_ids: Tensor,
    ) -> Tensor:
        """Forward pass producing logits for next-token prediction.

        Args:
            input_ids: (batch, seq_len) token IDs

        Returns:
            logits: (batch, seq_len, vocab_size) — logits for next token
                    at each position
        """
        B, T = input_ids.shape
        device = input_ids.device

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
            )

        # Token embeddings
        x = self.token_embedding(input_ids)  # (B, T, d_model)

        # Position embeddings
        pos = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embedding(pos)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.ln_f(x)

        # LM head
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    @torch.no_grad()
    def _get_logit_mask(self, input_ids: Tensor) -> Tensor:
        """Build a logit mask for the next token after input_ids.

        The mask ensures only valid field transitions are allowed based on
        RV32I instruction formats.

        Args:
            input_ids: (batch, seq_len) current sequence

        Returns:
            mask: (batch, vocab_size) — True for allowed tokens
        """
        B, T = input_ids.shape
        device = input_ids.device
        vocab = RV32IVocab

        # Initialize all tokens as allowed
        mask = torch.ones(B, vocab.VOCAB_SIZE, dtype=torch.bool, device=device)

        for b in range(B):
            seq = input_ids[b]

            # Find last non-pad token
            non_pad = (seq != vocab.PAD).nonzero(as_tuple=True)[0]
            if len(non_pad) == 0:
                # Empty sequence → start with instruction token
                allowed = torch.zeros(vocab.VOCAB_SIZE, dtype=torch.bool, device=device)
                allowed[vocab.INST_START:vocab.INST_START + vocab.INST_COUNT] = True
                mask[b] = allowed
                continue

            last_idx = non_pad[-1].item()
            last_token = seq[last_idx].item()

            # Count tokens since last SEP or CLS to determine position
            boundary_positions = (
                (seq[:last_idx + 1] == vocab.SEP) | (seq[:last_idx + 1] == vocab.CLS)
            ).nonzero(as_tuple=True)[0]

            if len(boundary_positions) == 0:
                inst_start = 0
            else:
                inst_start = boundary_positions[-1].item() + 1

            num_inst_tokens = last_idx - inst_start + 1

            allowed = torch.zeros(vocab.VOCAB_SIZE, dtype=torch.bool, device=device)

            # After SEP or CLS → start new instruction
            if last_token in (vocab.SEP, vocab.CLS):
                allowed[vocab.INST_START:vocab.INST_START + vocab.INST_COUNT] = True
                mask[b] = allowed
                continue

            if num_inst_tokens == 0:
                allowed[vocab.INST_START:vocab.INST_START + vocab.INST_COUNT] = True
                mask[b] = allowed
                continue

            first_tok = seq[inst_start].item()

            if num_inst_tokens == 1:
                # Position 0 (inst) was just placed. Next is funct3 (position 1)
                # OR SEP (end instruction early)
                if vocab.is_instruction(first_tok):
                    fmt = vocab.get_format(first_tok)
                    if fmt in (vocab.U_FORMAT, vocab.J_FORMAT):
                        # U/J: no funct3 → NO_REG
                        allowed[vocab.NO_REG] = True
                    else:
                        # R/I/S/B: funct3 value determined by instruction
                        f3_val = vocab.get_funct3(first_tok)
                        allowed[vocab.funct3_token(f3_val)] = True
                allowed[vocab.SEP] = True

            elif num_inst_tokens == 2:
                # Position 1 (funct3) was just placed. Next is rd (position 2)
                if vocab.is_instruction(first_tok):
                    fmt = vocab.get_format(first_tok)
                    if fmt in (vocab.S_FORMAT, vocab.B_FORMAT):
                        # S/B: no rd → NO_REG
                        allowed[vocab.NO_REG] = True
                    else:
                        # R/I/U/J: rd is a register
                        allowed[vocab.REG_START:vocab.REG_START + vocab.REG_COUNT] = True
                allowed[vocab.SEP] = True

            elif num_inst_tokens == 3:
                # Position 2 (rd/NO_REG) was just placed. Next is rs1 (position 3)
                if vocab.is_instruction(first_tok):
                    fmt = vocab.get_format(first_tok)
                    if fmt in (vocab.U_FORMAT, vocab.J_FORMAT):
                        # U/J: no rs1 → NO_REG
                        allowed[vocab.NO_REG] = True
                    else:
                        # R/I/S/B: rs1 is a register
                        allowed[vocab.REG_START:vocab.REG_START + vocab.REG_COUNT] = True
                allowed[vocab.SEP] = True

            elif num_inst_tokens == 4:
                # Position 3 (rs1/NO_REG) was just placed. Next is rs2 (position 4)
                if vocab.is_instruction(first_tok):
                    fmt = vocab.get_format(first_tok)
                    if fmt in (vocab.I_FORMAT, vocab.U_FORMAT, vocab.J_FORMAT):
                        # I/U/J: no rs2 → NO_REG
                        allowed[vocab.NO_REG] = True
                    else:
                        # R/S/B: rs2 is a register
                        allowed[vocab.REG_START:vocab.REG_START + vocab.REG_COUNT] = True
                allowed[vocab.SEP] = True

            elif num_inst_tokens == 5:
                # Position 4 (rs2/NO_REG) was just placed. Next is imm (position 5)
                if vocab.is_instruction(first_tok):
                    fmt = vocab.get_format(first_tok)
                    if fmt == vocab.R_FORMAT:
                        # R: no imm → NO_REG
                        allowed[vocab.NO_REG] = True
                    else:
                        # I/S/B/U/J: imm
                        allowed[vocab.IMM_START:vocab.IMM_START + vocab.IMM_COUNT] = True
                allowed[vocab.SEP] = True

            elif num_inst_tokens == 6:
                # Position 5 (imm/NO_REG) was just placed. Next must be SEP.
                allowed[vocab.SEP] = True

            else:
                # Past a full instruction → start new one
                allowed[vocab.INST_START:vocab.INST_START + vocab.INST_COUNT] = True

            mask[b] = allowed

        return mask

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: int = RV32IVocab.SEP,
    ) -> Tensor:
        """Autoregressive generation with logit masking.

        Args:
            input_ids: (batch, seq_len) prompt tokens to continue from
            max_new_tokens: maximum number of new tokens to generate
            temperature: sampling temperature (>1 more random, <1 more greedy)
            top_k: if set, only sample from top-k tokens
            eos_token_id: stop when this token is produced

        Returns:
            (batch, seq_len + generated_tokens) completed sequences
        """
        self.eval()
        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            curr_seq = input_ids[:, -self.max_seq_len:]

            # Forward pass
            logits = self.forward(curr_seq)
            next_logits = logits[:, -1, :]  # (B, V)

            # Apply logit mask
            mask = self._get_logit_mask(curr_seq)
            next_logits = next_logits.masked_fill(~mask, float('-inf'))

            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)), dim=-1)
                next_logits[next_logits < top_k_vals[:, -1:]] = float('-inf')

            # Sample
            if temperature == 0.0:
                next_tokens = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                if torch.isnan(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Check EOS for all batch items
            done = (next_tokens == eos_token_id).all().item()
            if done:
                break

        return input_ids

    def apply_logit_mask(self, logits: Tensor, input_ids: Tensor) -> Tensor:
        """Apply logit masking to raw logits.

        Args:
            logits: (batch, seq_len, vocab_size) raw logits
            input_ids: (batch, seq_len) context tokens

        Returns:
            Masked logits (batch, seq_len, vocab_size)
        """
        B, T, V = logits.shape

        for t in range(T):
            ctx = input_ids[:, :t + 1]
            mask = self._get_logit_mask(ctx)
            logits[:, t, :] = logits[:, t, :].masked_fill(~mask, float('-inf'))

        return logits


# =====================================================================
#  TRAINING FUNCTIONS
# =====================================================================

def compute_loss(
    model: RV32ITransformer,
    input_ids: Tensor,
    labels: Optional[Tensor] = None,
    ignore_index: int = RV32IVocab.PAD,
) -> Tensor:
    """Compute cross-entropy loss for next-token prediction.

    Args:
        model: the transformer model
        input_ids: (batch, seq_len) input token IDs
        labels: optional (batch, seq_len) target token IDs
        ignore_index: token ID to ignore in loss computation

    Returns:
        scalar cross-entropy loss
    """
    logits = model(input_ids)  # (B, T, V)

    if labels is None:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
    else:
        shift_logits = logits[:, :-1, :].contiguous() if logits.shape[1] > 1 else logits
        shift_labels = labels.contiguous()

    if shift_logits.shape[1] == 0:
        return torch.tensor(0.0, requires_grad=True, device=input_ids.device)

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    return loss


def train_step(
    model: RV32ITransformer,
    batch: List[List[int]],
    optimizer: torch.optim.Optimizer,
    ignore_index: int = RV32IVocab.PAD,
) -> float:
    """Single training step on a batch of token sequences.

    Args:
        model: the transformer model
        batch: list of token ID lists
        optimizer: PyTorch optimizer
        ignore_index: token ID to ignore in loss

    Returns:
        loss value (float)
    """
    model.train()

    # Pad batch to same length
    max_len = max(len(seq) for seq in batch)
    padded = []
    for seq in batch:
        pad_len = max_len - len(seq)
        padded.append(seq + [RV32IVocab.PAD] * pad_len)

    input_ids = torch.tensor(padded, dtype=torch.long)

    optimizer.zero_grad()
    loss = compute_loss(model, input_ids, ignore_index=ignore_index)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return loss.item()


# =====================================================================
#  UTILITY: Count Parameters
# =====================================================================

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model.
    Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory_footprint(total_params: int, dtype_bytes: int = 4) -> Dict[str, float]:
    """Estimate memory footprint of the model.

    Args:
        total_params: total number of parameters
        dtype_bytes: bytes per parameter (4 for float32, 2 for float16/bfloat16)

    Returns:
        dict with keys 'parameters_mb', 'gradients_mb', 'optimizer_mb', 'total_mb'
    """
    params_mb = total_params * dtype_bytes / (1024 * 1024)
    grads_mb = params_mb  # same size as parameters for gradients
    # AdamW: 2 states per param (exp_avg, exp_avg_sq)
    optimizer_mb = params_mb * 2
    total_mb = params_mb + grads_mb + optimizer_mb
    return {
        'parameters_mb': round(params_mb, 2),
        'gradients_mb': round(grads_mb, 2),
        'optimizer_mb': round(optimizer_mb, 2),
        'total_mb': round(total_mb, 2),
    }


# =====================================================================
#  SELF-TEST
# =====================================================================

def _run_self_test():
    """Instantiate the model, run through dummy data, verify shapes."""
    print("=" * 64)
    print("RV32I Micro-Model Self-Test")
    print("=" * 64)

    # --- 1. Instantiate tokenizer ---
    print("\n[1] Instantiating RV32IFieldTokenizer...")
    tokenizer = RV32IFieldTokenizer()
    num_instructions = len(RV32IVocab.get_instruction_names())
    print(f"    Instruction types: {num_instructions}")
    print(f"    Vocab size: {RV32IVocab.VOCAB_SIZE}")

    # Print all 47 instructions grouped by format
    print("\n[1b] Instruction encodings by format:")
    by_fmt: Dict[str, List[str]] = {}
    for name in RV32IVocab.get_instruction_names():
        fmt = RV32IVocab.get_format(name)
        by_fmt.setdefault(fmt, []).append(name)
    for fmt in [RV32IVocab.R_FORMAT, RV32IVocab.I_FORMAT, RV32IVocab.S_FORMAT,
                RV32IVocab.B_FORMAT, RV32IVocab.U_FORMAT, RV32IVocab.J_FORMAT]:
        if fmt in by_fmt:
            names = by_fmt[fmt]
            print(f"    {fmt}-type ({len(names)}): {', '.join(names)}")

    # --- 2. Encode a few instructions ---
    print("\n[2] Testing instruction encoding...")
    insts = [
        # ADD x1, x2, x3  (R-type)
        {'name': 'ADD', 'rd': 1, 'rs1': 2, 'rs2': 3},
        # ADDI x4, x5, 42  (I-type)
        {'name': 'ADDI', 'rd': 4, 'rs1': 5, 'imm': 42},
        # SW x2, 16(x1)  (S-type)
        {'name': 'SW', 'rs1': 1, 'rs2': 2, 'imm': 16},
        # BEQ x6, x7, 8   (B-type)
        {'name': 'BEQ', 'rs1': 6, 'rs2': 7, 'imm': 8},
        # LUI x8, 0x12345 (U-type)
        {'name': 'LUI', 'rd': 8, 'imm': 0x12345},
        # JAL x1, 0x1234  (J-type)
        {'name': 'JAL', 'rd': 1, 'imm': 0x1234},
        # NOP via ADDI x0, x0, 0
        {'name': 'ADDI', 'rd': 0, 'rs1': 0, 'imm': 0},
    ]

    trace_tokens = []
    for inst in insts:
        toks = tokenizer.encode_instruction(inst)
        trace_tokens.extend(toks)
        decoded = tokenizer.decode_tokens(toks)
        fmt = RV32IVocab.get_format(inst['name'])
        print(f"    {inst['name']:6s} [{fmt}]: {toks} → {decoded}")

    print(f"\n    Full trace ({len(insts)} insts, {len(trace_tokens)} tokens)")
    print(f"    Trace tokens: {trace_tokens}")

    # --- 3. Model instantiation ---
    print("\n[3] Instantiating RV32ITransformer...")
    model = RV32ITransformer(
        vocab_size=RV32IVocab.VOCAB_SIZE,
        d_model=256,
        n_layers=8,
        n_heads=8,
        d_ff=1024,
        max_seq_len=2048,
        dropout=0.1,
        tie_weights=True,
    )

    total_params, trainable_params = count_parameters(model)
    print(f"    Total parameters:     {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    print(f"    Parameter efficiency: {total_params / RV32IVocab.VOCAB_SIZE:.0f} params/vocab")

    # Memory footprint
    mem = estimate_memory_footprint(total_params)
    print(f"    Memory (float32):")
    print(f"      Parameters: {mem['parameters_mb']} MB")
    print(f"      Gradients:  {mem['gradients_mb']} MB")
    print(f"      Optimizer:  {mem['optimizer_mb']} MB")
    print(f"      Total:      {mem['total_mb']} MB")

    # --- 4. Forward pass ---
    print("\n[4] Forward pass on dummy batch...")
    batch_size = 2
    seq_len = len(trace_tokens)
    input_ids = torch.tensor([trace_tokens] * batch_size, dtype=torch.long)
    print(f"    Input shape: {input_ids.shape}")

    logits = model(input_ids)
    print(f"    Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, RV32IVocab.VOCAB_SIZE), \
        f"Expected ({batch_size}, {seq_len}, {RV32IVocab.VOCAB_SIZE}), got {logits.shape}"
    print(f"    Logit range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # --- 5. Test logit masking ---
    print("\n[5] Testing logit masking...")
    with torch.no_grad():
        masked = model.apply_logit_mask(logits.clone(), input_ids)
        has_inf = torch.isinf(masked).any()
        print(f"    Mask applied (inf values present): {has_inf.item()}")

        mask_single = model._get_logit_mask(input_ids[:1, :6])
        allowed_counts = mask_single.sum(dim=-1).tolist()
        print(f"    Allowed tokens after each of first 6 positions: {allowed_counts}")

    # --- 6. Test generation ---
    print("\n[6] Testing generation (single step)...")
    prompt_tok = RV32IVocab.get_instruction_token('ADD')
    prompt = torch.tensor([[RV32IVocab.CLS, prompt_tok]], dtype=torch.long)
    print(f"    Prompt shape: {prompt.shape}")
    print(f"    Prompt tokens: [{RV32IVocab.CLS}, {prompt_tok} (ADD)]")

    with torch.no_grad():
        output = model.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.8,
            top_k=20,
        )
    print(f"    Generated shape: {output.shape}")
    gen_tokens = output[0].tolist()
    print(f"    Generated tokens: {gen_tokens}")

    # Decode the generated instruction
    decoded = tokenizer.decode_tokens(gen_tokens[1:])  # Skip CLS
    print(f"    Decoded: {decoded}")

    # --- 7. Test training step ---
    print("\n[7] Testing training step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch_data = [trace_tokens, trace_tokens]
    loss_val = train_step(model, batch_data, optimizer)
    print(f"    Loss after single step: {loss_val:.4f}")
    assert loss_val > 0, "Loss should be positive"

    # --- 8. Verify total parameter count is in expected range ---
    print("\n[8] Verification...")
    expected_range = (5_000_000, 12_000_000)
    in_range = expected_range[0] <= total_params <= expected_range[1]
    print(f"    Parameter count {total_params:,} in range {expected_range}: {in_range}")
    assert in_range, f"Parameter count {total_params} outside expected range {expected_range}"

    # --- Summary ---
    print("\n" + "=" * 64)
    print("SELF-TEST PASSED")
    print(f"    Model params:   {total_params:,}")
    print(f"    Forward output: {logits.shape}")
    print(f"    Loss after 1 step: {loss_val:.4f}")
    print(f"    Memory (fp32):  {mem['total_mb']} MB total")
    print("=" * 64)


if __name__ == "__main__":
    _run_self_test()
