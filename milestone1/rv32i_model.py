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
    128 funct7 + 48 immediate nibble tokens + special tokens = 2048 vocab
  - Causal attention mask, pre-norm RMSNorm, residual connections
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


import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Import RV32I encoding table for building instruction mapping
try:
    from rv32i_mask import decode as rv_decode
    from rv32i_mask import encode as rv_encode
    from rv32i_mask import get_valid_combinations, is_valid
except ImportError:
    # Local fallback: define stubs so the module can be imported standalone
    def get_valid_combinations():
        return []

    def is_valid(word):
        return False

    def rv_encode(inst):
        return 0

    def rv_decode(word):
        return None

    encode = rv_encode
    decode = rv_decode


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
        IMM_NIBBLE_0..15 = 219..234  (16 nibble values for imm pos 0 / high)
        IMM_NIBBLE_16..31 = 235..250 (16 nibble values for imm pos 1 / mid)
        IMM_NIBBLE_32..47 = 251..266 (16 nibble values for imm pos 2 / low)
        NO_REG       = 267
        Reserved     = 268..2047  (1780 for future expansion)
    """

    # Special tokens
    PAD = 0
    SEP = 1
    CLS = 2
    MASK = 3

    # Token ranges
    INST_START = 4  # 4..50   (47 instruction types)
    INST_COUNT = 47
    REG_START = 51  # 51..82  (32 registers x0-x31)
    REG_COUNT = 32
    FUNCT3_START = 83  # 83..90  (8 funct3 values)
    FUNCT3_COUNT = 8
    FUNCT7_START = 91  # 91..218 (128 funct7 values)
    FUNCT7_COUNT = 128

    # Structured Immediate Nibble Encoding
    # A 12-bit signed immediate [-2048..2047] is decomposed into 3 nibbles
    # (4 bits each). Each nibble position gets its own vocab range so the
    # model can learn position-specific nibble distributions.
    IMM_NIBBLE_START = 219  # First nibble token (pos 0 = high nibble)
    IMM_NIBBLE_COUNT = 48  # 16 values x 3 positions = 48 total nibble tokens
    IMM_POS0_START = 219  # High nibble:   tokens 219..234
    IMM_POS1_START = 235  # Mid nibble:    tokens 235..250
    IMM_POS2_START = 251  # Low nibble:    tokens 251..266

    # Backward-compat aliases (IMM_START/IMM_COUNT now point at nibble tokens)
    IMM_START = IMM_NIBBLE_START  # 219
    IMM_COUNT = IMM_NIBBLE_COUNT  # 48

    NO_REG = 267
    VOCAB_SIZE = 2048

    # Fields per instruction (fixed-length encoding with 3 nibble tokens)
    # Positions: [inst, funct3, rd, rs1, rs2, imm_high, imm_mid, imm_low, SEP]
    TOKENS_PER_INST = 9

    # Format constants
    R_FORMAT = "R"
    I_FORMAT = "I"
    S_FORMAT = "S"
    B_FORMAT = "B"
    U_FORMAT = "U"
    J_FORMAT = "J"

    # Instruction mapping cache (lazily built from rv32i_mask)
    _INSTRUCTION_NAMES: list[str] = []  # token_index -> name
    _NAME_TO_TOKEN: dict[str, int] = {}  # name -> token_id
    _NAME_TO_FORMAT: dict[str, str] = {}  # name -> format
    _NAME_TO_FUNCT3: dict[str, int] = {}  # name -> funct3 value
    _NAME_TO_FUNCT7: dict[str, int] = {}  # name -> funct7 value
    _NAME_TO_OPCODE: dict[str, int] = {}  # name -> opcode
    _TOKEN_TO_NAME: dict[int, str] = {}  # token_id -> name
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
            cls._NAME_TO_FUNCT3[name] = max(funct3, 0)
            cls._NAME_TO_FUNCT7[name] = max(funct7, 0)
            cls._NAME_TO_OPCODE[name] = opcode
            cls._TOKEN_TO_NAME[token_id] = name

        cls._BUILT = True

    @classmethod
    def get_instruction_names(cls) -> list[str]:
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
        """Check if token is an immediate nibble token (any position)."""
        return cls.IMM_NIBBLE_START <= token < cls.IMM_NIBBLE_START + cls.IMM_NIBBLE_COUNT

    @classmethod
    def is_imm_nibble(cls, token: int) -> bool:
        """Alias for is_imm — check if token is an immediate nibble token."""
        return cls.is_imm(token)

    @classmethod
    def imm_to_nibble_tokens(cls, imm: int) -> list[int]:
        """Encode a signed immediate as 3 nibble tokens.

        Args:
            imm: signed integer to encode (will be clamped to 12-bit signed
                 range [-2048, 2047]).

        Returns:
            List of 3 token IDs [high_nibble, mid_nibble, low_nibble].
            Each token is positioned in its own vocab range
            (POS0 for high, POS1 for mid, POS2 for low).
        """
        # Clamp to 12-bit signed range
        imm = max(-2048, min(2047, imm))
        # Convert to unsigned 12-bit representation
        unsigned = imm & 0xFFF
        # Extract 3 nibbles (4 bits each): high, mid, low
        nibble_high = (unsigned >> 8) & 0xF  # bits [11:8]
        nibble_mid = (unsigned >> 4) & 0xF  # bits [7:4]
        nibble_low = unsigned & 0xF  # bits [3:0]
        return [
            cls.IMM_POS0_START + nibble_high,
            cls.IMM_POS1_START + nibble_mid,
            cls.IMM_POS2_START + nibble_low,
        ]

    @classmethod
    def nibble_tokens_to_imm(cls, tokens: list[int]) -> int:
        """Decode 3 nibble tokens back to a signed immediate value.

        Args:
            tokens: list of 3 token IDs [high, mid, low].

        Returns:
            Signed integer in [-2048, 2047].
        """
        nibble_high = tokens[0] - cls.IMM_POS0_START
        nibble_mid = tokens[1] - cls.IMM_POS1_START
        nibble_low = tokens[2] - cls.IMM_POS2_START
        # Reconstruct unsigned 12-bit value
        unsigned = (nibble_high << 8) | (nibble_mid << 4) | nibble_low
        # Sign-extend from 12 bits
        if unsigned >= 2048:
            return unsigned - 4096
        return unsigned

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


# =====================================================================
#  FIELD TOKENIZER
# =====================================================================


class RV32IFieldTokenizer:
    """Field-level tokenizer for RV32I instructions.

    Encodes each instruction as 9 tokens:
        [inst_type, funct3, rd, rs1, rs2, imm_high, imm_mid, imm_low, SEP]

    Format-dependent field mapping:
        R-type: [inst, funct3, rd,   rs1, rs2,   NO_REG, NO_REG, NO_REG, SEP]
        I-type: [inst, funct3, rd,   rs1, NO_REG, imm_h,  imm_m,  imm_l,  SEP]
        S-type: [inst, funct3, NO_REG, rs1, rs2,   imm_h,  imm_m,  imm_l,  SEP]
        B-type: [inst, funct3, NO_REG, rs1, rs2,   imm_h,  imm_m,  imm_l,  SEP]
        U-type: [inst, NO_REG, rd,   NO_REG, NO_REG, imm_h,  imm_m,  imm_l,  SEP]
        J-type: [inst, NO_REG, rd,   NO_REG, NO_REG, imm_h,  imm_m,  imm_l,  SEP]
    """

    def __init__(self):
        self.vocab = RV32IVocab
        # Ensure instruction map is built
        self.vocab._build_instruction_map()

    def encode_instruction(self, inst: dict) -> list[int]:
        """Encode a single instruction dict to token IDs.

        Args:
            inst: dict with keys 'name' and optionally 'rd', 'rs1', 'rs2',
                  'imm'. Fields unused by the instruction's format may be
                  absent or None.

        Returns:
            List of 9 token IDs.
        """
        name = inst["name"]
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
            rd = inst.get("rd", 0)
            tokens.append(self.vocab.register_token(rd))

        # Position 3: rs1 (or NO_REG for U/J)
        if fmt in (self.vocab.U_FORMAT, self.vocab.J_FORMAT):
            tokens.append(self.vocab.NO_REG)
        else:
            rs1 = inst.get("rs1", 0)
            tokens.append(self.vocab.register_token(rs1))

        # Position 4: rs2 (or NO_REG for I/U/J)
        if fmt in (self.vocab.I_FORMAT, self.vocab.U_FORMAT, self.vocab.J_FORMAT):
            tokens.append(self.vocab.NO_REG)
        else:
            rs2 = inst.get("rs2", 0)
            tokens.append(self.vocab.register_token(rs2))

        # Positions 5,6,7: imm as 3 nibble tokens (or 3x NO_REG for R)
        if fmt == self.vocab.R_FORMAT:
            tokens.append(self.vocab.NO_REG)
            tokens.append(self.vocab.NO_REG)
            tokens.append(self.vocab.NO_REG)
        else:
            imm = inst.get("imm", 0)
            tokens.extend(self.vocab.imm_to_nibble_tokens(imm))

        # Position 8: SEP
        tokens.append(self.vocab.SEP)

        return tokens

    def encode_trace(self, trace: list[dict]) -> list[int]:
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

    def decode_tokens(self, tokens: list[int]) -> dict | None:
        """Decode a flat token sequence back to at most one instruction dict.

        Takes the first full instruction (9 tokens ending in SEP) from the
        sequence, ignoring leading special tokens.

        Args:
            tokens: list of token IDs

        Returns:
            Instruction dict with keys name, format, rd, rs1, rs2, imm,
            or None if decoding fails.
        """
        # Strip leading special tokens
        clean = [t for t in tokens if t not in (self.vocab.PAD, self.vocab.CLS, self.vocab.MASK)]

        # Need at least 9 tokens for one instruction
        if len(clean) < self.vocab.TOKENS_PER_INST:
            return None

        inst_toks = clean[: self.vocab.TOKENS_PER_INST]

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
            "name": name,
            "format": fmt,
        }

        # Position 1: funct3 (token or NO_REG)
        f3_tok = inst_toks[1]
        if f3_tok == self.vocab.NO_REG:
            result["funct3"] = None
        elif self.vocab.is_funct3(f3_tok):
            result["funct3"] = f3_tok - self.vocab.FUNCT3_START
        else:
            result["funct3"] = None

        # Position 2: rd
        rd_tok = inst_toks[2]
        if rd_tok == self.vocab.NO_REG:
            result["rd"] = None
        elif self.vocab.is_register(rd_tok):
            result["rd"] = rd_tok - self.vocab.REG_START
        else:
            result["rd"] = None

        # Position 3: rs1
        rs1_tok = inst_toks[3]
        if rs1_tok == self.vocab.NO_REG:
            result["rs1"] = None
        elif self.vocab.is_register(rs1_tok):
            result["rs1"] = rs1_tok - self.vocab.REG_START
        else:
            result["rs1"] = None

        # Position 4: rs2
        rs2_tok = inst_toks[4]
        if rs2_tok == self.vocab.NO_REG:
            result["rs2"] = None
        elif self.vocab.is_register(rs2_tok):
            result["rs2"] = rs2_tok - self.vocab.REG_START
        else:
            result["rs2"] = None

        # Positions 5,6,7: imm as 3 nibble tokens
        imm_high_tok = inst_toks[5]
        imm_mid_tok = inst_toks[6]
        imm_low_tok = inst_toks[7]

        if imm_high_tok == self.vocab.NO_REG and imm_mid_tok == self.vocab.NO_REG and imm_low_tok == self.vocab.NO_REG:
            result["imm"] = None
        elif self.vocab.is_imm(imm_high_tok) and self.vocab.is_imm(imm_mid_tok) and self.vocab.is_imm(imm_low_tok):
            result["imm"] = self.vocab.nibble_tokens_to_imm([imm_high_tok, imm_mid_tok, imm_low_tok])
        else:
            result["imm"] = None

        return result


# =====================================================================
#  ROTARY POSITION EMBEDDINGS (RoPE)
# =====================================================================


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for relative positional encoding.

    Encodes positions as rotation matrices applied to Q and K, enabling
    the model to generalize to unseen sequence lengths and providing
    relative position information in attention.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        # Precompute frequency bands: theta_i = base^(-2i/head_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Precompute cos/sin cache for all positions
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return (
            self.cos_cached[:seq_len].to(device),
            self.sin_cached[:seq_len].to(device),
        )


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the hidden dims: [x1, x2] -> [-x2, x1]."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary embeddings to Q and K tensors.

    Args:
        q: (B, H, T, head_dim)
        k: (B, H, T, head_dim)
        cos: (T, head_dim) -- cosine values for each position
        sin: (T, head_dim) -- sine values for each position
    """
    # Reshape cos/sin for broadcasting: (1, 1, T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# =====================================================================
#  CAUSAL SELF-ATTENTION
# =====================================================================


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and pre-computed causal mask."""

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 2048, use_rope: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.use_rope = use_rope

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # RoPE
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)

        # Causal mask (filled on first forward)
        self.causal_mask = None

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Build or retrieve the causal attention mask."""
        if self.causal_mask is None or self.causal_mask.shape[-1] < seq_len:
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)
            self.register_buffer("causal_mask", mask, persistent=False)
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x: Tensor, kv_cache: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """x: (batch, seq_len, d_model)

        Args:
            x: input tensor (B, T, D)
            kv_cache: optional (cached_k, cached_v) each (B, H, cache_len, head_dim)

        Returns:
            y: output tensor (B, T, D)
            new_cache: (k, v) updated cache, each (B, H, cache_len + T, head_dim)
        """
        B, T, D = x.shape
        device = x.device

        # QKV projection
        qkv = self.qkv(x)  # (B, T, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, D)

        # Reshape to multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.use_rope:
            if kv_cache is not None:
                # Incremental decoding: apply RoPE to positions cache_len..cache_len+T-1
                cache_len = kv_cache[0].shape[2]
                cos_full, sin_full = self.rotary(cache_len + T, device)
                # q and k_new use the NEW positions only
                cos_new = cos_full[cache_len : cache_len + T].unsqueeze(0).unsqueeze(0)
                sin_new = sin_full[cache_len : cache_len + T].unsqueeze(0).unsqueeze(0)
                q = q * cos_new + _rotate_half(q) * sin_new
                # Apply RoPE to new k BEFORE concatenating with cached k
                k = k * cos_new + _rotate_half(k) * sin_new
                # Concatenate cached K/V with new K/V
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            else:
                # Full sequence: apply RoPE normally
                cos, sin = self.rotary(T, device)
                q, k = _apply_rope(q, k, cos, sin)

        new_cache = (k, v)

        # Flash Attention via PyTorch SDPA
        # When using KV cache: q has length T, k/v have length cache_len + T.
        # We must NOT use is_causal=True here because SDPA's causal mask is
        # based on absolute position indices — with cache_len prefix, the new
        # query at index 0 would only attend to k/v index 0 (the oldest cached
        # token) instead of all positions.  Instead, for single-token decoding
        # (T=1), all positions are valid; for multi-token, we need the causal
        # mask to start from position cache_len.  The simplest correct approach:
        # set is_causal=False (no mask) and let the model handle it.  This is
        # safe because during KV-cache inference we only ever query the LAST
        # token position, so all prior positions are valid context.
        use_causal = kv_cache is None
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=use_causal,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        y = self.out_proj(y)
        y = self.resid_dropout(y)
        return y, new_cache


# =====================================================================
#  TRANSFORMER BLOCK (Pre-Norm, SwiGLU)
# =====================================================================


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network (LLaMA/Mistral-style).

    FFN_SwiGLU(x) = (Swish(xW1) (xW3)) W2
    Uses 3 weight matrices instead of 2, so d_ff is 2/3 of standard FFN
    to maintain the same parameter count.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA/Mistral-style)."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


class TransformerBlock(nn.Module):
    """Decoder transformer block with pre-norm RMSNorm and SwiGLU FFN.

    Sequence:
        x = x + SelfAttention(RMSNorm(x))
        x = x + SwiGLU_FFN(RMSNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
        use_rope: bool = True,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len=max_seq_len, use_rope=use_rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x: Tensor, kv_cache: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


# =====================================================================
#  RV32I TRANSFORMER — Full Decoder-Only Model
# =====================================================================


class RV32ITransformer(nn.Module):
    """Decoder-only transformer for RV32I instruction prediction.

    Architecture:
        - Token embedding + RoPE positional encoding
        - N transformer blocks (pre-norm, causal attention with RoPE, SwiGLU FFN)
        - Tied embedding/head weights (weight tying)
        - Logit masking for format-dependent field prediction

    Args:
        vocab_size: number of tokens in the vocabulary
        d_model: embedding/hidden dimension
        n_layers: number of transformer blocks
        n_heads: number of attention heads
        d_ff: feed-forward hidden dimension (for SwiGLU, this is the intermediate dim)
        max_seq_len: maximum sequence length
        dropout: dropout probability
        tie_weights: share weights between embedding and output head
        use_rope: use Rotary Position Embeddings (recommended)
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
        use_rope: bool = True,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.use_checkpointing = use_checkpointing
        self.tokens_per_inst = 9

        # Ensure vocab is built
        RV32IVocab._build_instruction_map()

        # Token embedding (no learned position embedding — RoPE handles positions)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=RV32IVocab.PAD)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len=max_seq_len, use_rope=use_rope)
                for _ in range(n_layers)
            ]
        )

        # Final layer norm (RMSNorm for ~10% speedup)
        self.lm_head_norm = RMSNorm(d_model)

        # Token type embeddings: instruction, register, funct3, funct7, immediate, special
        self.token_type_embedding = nn.Embedding(6, d_model)
        self._type_id_special = 0  # PAD, SEP, CLS, MASK
        self._type_id_inst = 1  # instruction tokens
        self._type_id_reg = 2  # register tokens
        self._type_id_funct3 = 3  # funct3 tokens
        self._type_id_funct7 = 4  # funct7 tokens
        self._type_id_imm = 5  # immediate tokens

        # Output head (LM head for next-token prediction)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        """GPT-2 style scaled initialization.

        Residual projections are scaled by 1/sqrt(2*n_layers) to prevent
        gradient explosion in deep transformers (14+ layers).
        """
        residual_scale = 1.0 / (2.0 * self.n_layers) ** 0.5

        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue  # skip 1D params (RMSNorm weights, biases)
            if "out_proj" in name or "w2" in name:
                # Residual-path projections: scaled down
                nn.init.normal_(p, mean=0.0, std=0.02 * residual_scale)
            elif "qkv" in name or "w1" in name or "w3" in name:
                # Input projections: normal init
                nn.init.normal_(p, mean=0.0, std=0.02)
            else:
                nn.init.normal_(p, mean=0.0, std=0.02)
        # Reset padding embedding to zero
        with torch.no_grad():
            self.token_embedding.weight[RV32IVocab.PAD].zero_()

    def forward(
        self,
        input_ids: Tensor,
        noise_std: float = 0.0,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
    ):
        """Forward pass producing logits for next-token prediction.

        Args:
            input_ids: (batch, seq_len) token IDs
            noise_std: std dev of Gaussian noise added to embeddings
                       during training. 0.0 disables (default).
            kv_caches: optional list of (k, v) cache tensors per layer.
                       When provided, enables incremental decoding — only the
                       last token in input_ids is processed (prefill on first call).

        Returns:
            When kv_caches is None: logits (batch, seq_len, vocab_size)
            When kv_caches is provided: (logits, updated_caches)
                logits: (batch, 1, vocab_size) — logits for next token
                updated_caches: list of (k, v) tuples, one per layer
        """
        _B, T = input_ids.shape
        _device = input_ids.device

        if kv_caches is not None:
            # ── Incremental decoding path ──────────────────────────────
            # Only process the last token (or the full input on first call)
            if T > 1:
                # First call (prefill): run full sequence, build caches
                logits, caches = self._forward_with_caches(input_ids, noise_std, None)
                return logits, caches
            else:
                # True incremental: T == 1, use provided caches
                return self._forward_with_caches(input_ids, noise_std, kv_caches)
        else:
            # ── Standard forward (training / evaluation) ────────────────
            # Returns just logits for backward compatibility with training code
            logits, _ = self._forward_full(input_ids, noise_std)
            return logits

    def _forward_with_caches(
        self,
        input_ids: Tensor,
        noise_std: float,
        kv_caches: list[tuple[Tensor, Tensor]] | None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Forward pass that always returns (logits, caches).

        When kv_caches is None, caches are built from scratch.
        When kv_caches is provided, caches are updated incrementally.
        """
        _B, T = input_ids.shape
        _device = input_ids.device

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_embedding(input_ids)

        # Token type embeddings
        type_ids = torch.full_like(input_ids, self._type_id_special)
        vocab = RV32IVocab
        inst_mask = (input_ids >= vocab.INST_START) & (input_ids < vocab.INST_START + vocab.INST_COUNT)
        type_ids[inst_mask] = self._type_id_inst
        reg_mask = (input_ids >= vocab.REG_START) & (input_ids < vocab.REG_START + vocab.REG_COUNT)
        type_ids[reg_mask] = self._type_id_reg
        f3_mask = (input_ids >= vocab.FUNCT3_START) & (input_ids < vocab.FUNCT3_START + vocab.FUNCT3_COUNT)
        type_ids[f3_mask] = self._type_id_funct3
        f7_mask = (input_ids >= vocab.FUNCT7_START) & (input_ids < vocab.FUNCT7_START + vocab.FUNCT7_COUNT)
        type_ids[f7_mask] = self._type_id_funct7
        imm_mask = (input_ids >= vocab.IMM_START) & (input_ids < vocab.IMM_START + vocab.IMM_COUNT)
        type_ids[imm_mask] = self._type_id_imm

        x = x + self.token_type_embedding(type_ids)

        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        updated_caches = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            if self.use_checkpointing and self.training:

                def _block_fn(x_in, c=cache_i):
                    return block(x_in, kv_cache=c)

                x, new_cache = torch.utils.checkpoint.checkpoint(_block_fn, x, use_reentrant=False)
            else:
                x, new_cache = block(x, kv_cache=cache_i)
            updated_caches.append(new_cache)

        x = self.lm_head_norm(x)
        logits = self.lm_head(x)
        return logits, updated_caches

    def _forward_full(
        self,
        input_ids: Tensor,
        noise_std: float = 0.0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Full forward pass returning (logits, caches) for cache building."""
        _B, T = input_ids.shape
        _device = input_ids.device

        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_embedding(input_ids)

        type_ids = torch.full_like(input_ids, self._type_id_special)
        vocab = RV32IVocab
        inst_mask = (input_ids >= vocab.INST_START) & (input_ids < vocab.INST_START + vocab.INST_COUNT)
        type_ids[inst_mask] = self._type_id_inst
        reg_mask = (input_ids >= vocab.REG_START) & (input_ids < vocab.REG_START + vocab.REG_COUNT)
        type_ids[reg_mask] = self._type_id_reg
        f3_mask = (input_ids >= vocab.FUNCT3_START) & (input_ids < vocab.FUNCT3_START + vocab.FUNCT3_COUNT)
        type_ids[f3_mask] = self._type_id_funct3
        f7_mask = (input_ids >= vocab.FUNCT7_START) & (input_ids < vocab.FUNCT7_START + vocab.FUNCT7_COUNT)
        type_ids[f7_mask] = self._type_id_funct7
        imm_mask = (input_ids >= vocab.IMM_START) & (input_ids < vocab.IMM_START + vocab.IMM_COUNT)
        type_ids[imm_mask] = self._type_id_imm

        x = x + self.token_type_embedding(type_ids)

        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        updated_caches = []
        if self.use_checkpointing and self.training:
            for block in self.blocks:

                def _block_fn(x_in, _block=block):
                    return _block(x_in)

                x, new_cache = torch.utils.checkpoint.checkpoint(_block_fn, x, use_reentrant=False)
                updated_caches.append(new_cache)
        else:
            for block in self.blocks:
                x, new_cache = block(x)
                updated_caches.append(new_cache)

        x = self.lm_head_norm(x)
        logits = self.lm_head(x)
        return logits, updated_caches

    def forward_with_cache(
        self,
        input_ids: Tensor,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Forward pass with KV-cache support.

        First call: pass full input_ids, kv_caches=None -> returns (logits, new_caches)
        Subsequent calls: pass only the new token(s), kv_caches=previous_caches -> returns (logits, updated_caches)

        Args:
            input_ids: (B, seq_len) — full sequence on first call, single token after
            kv_caches: list of (k, v) per layer, or None for first call

        Returns:
            (logits, updated_caches) — logits is (B, seq_len, V), caches is list of (k, v)
        """
        if kv_caches is None:
            # First call: full forward, build caches
            return self._forward_full(input_ids)
        else:
            # Incremental call
            return self._forward_with_caches(input_ids, 0.0, kv_caches)

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
        B, _T = input_ids.shape
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
                allowed[vocab.INST_START : vocab.INST_START + vocab.INST_COUNT] = True
                mask[b] = allowed
                continue

            last_idx = non_pad[-1].item()
            last_token = seq[last_idx].item()

            # Count tokens since last SEP or CLS to determine position
            boundary_positions = ((seq[: last_idx + 1] == vocab.SEP) | (seq[: last_idx + 1] == vocab.CLS)).nonzero(
                as_tuple=True
            )[0]

            if len(boundary_positions) == 0:
                inst_start = 0
            else:
                inst_start = boundary_positions[-1].item() + 1

            num_inst_tokens = last_idx - inst_start + 1

            allowed = torch.zeros(vocab.VOCAB_SIZE, dtype=torch.bool, device=device)

            # After SEP or CLS → start new instruction
            if last_token in (vocab.SEP, vocab.CLS):
                allowed[vocab.INST_START : vocab.INST_START + vocab.INST_COUNT] = True
                mask[b] = allowed
                continue

            if num_inst_tokens == 0:
                allowed[vocab.INST_START : vocab.INST_START + vocab.INST_COUNT] = True
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
                        allowed[vocab.REG_START : vocab.REG_START + vocab.REG_COUNT] = True
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
                        allowed[vocab.REG_START : vocab.REG_START + vocab.REG_COUNT] = True
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
                        allowed[vocab.REG_START : vocab.REG_START + vocab.REG_COUNT] = True
                allowed[vocab.SEP] = True

            elif num_inst_tokens == 5:
                # Position 4 (rs2/NO_REG) was just placed. Next is imm_high (position 5)
                if vocab.is_instruction(first_tok):
                    fmt = vocab.get_format(first_tok)
                    if fmt == vocab.R_FORMAT:
                        # R: no imm -> first NO_REG of triple
                        allowed[vocab.NO_REG] = True
                    else:
                        # I/S/B/U/J: imm high nibble (16 possible values)
                        allowed[vocab.IMM_POS0_START : vocab.IMM_POS0_START + 16] = True
                allowed[vocab.SEP] = True

            elif num_inst_tokens == 6:
                # Position 5 (imm_high/NO_REG) was just placed. Next is imm_mid (position 6)
                if vocab.is_instruction(first_tok):
                    fmt = vocab.get_format(first_tok)
                    if fmt == vocab.R_FORMAT:
                        # R: second NO_REG of triple
                        allowed[vocab.NO_REG] = True
                    else:
                        # I/S/B/U/J: imm mid nibble (16 possible values)
                        allowed[vocab.IMM_POS1_START : vocab.IMM_POS1_START + 16] = True
                allowed[vocab.SEP] = True

            elif num_inst_tokens == 7:
                # Position 6 (imm_mid/NO_REG) was just placed. Next is imm_low (position 7)
                if vocab.is_instruction(first_tok):
                    fmt = vocab.get_format(first_tok)
                    if fmt == vocab.R_FORMAT:
                        # R: third NO_REG of triple
                        allowed[vocab.NO_REG] = True
                    else:
                        # I/S/B/U/J: imm low nibble (16 possible values)
                        allowed[vocab.IMM_POS2_START : vocab.IMM_POS2_START + 16] = True
                allowed[vocab.SEP] = True

            elif num_inst_tokens == 8:
                # Position 7 (imm_low/NO_REG) was just placed. Next must be SEP.
                allowed[vocab.SEP] = True

            else:
                # Past a full instruction → start new one
                allowed[vocab.INST_START : vocab.INST_START + vocab.INST_COUNT] = True

            mask[b] = allowed

        return mask

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
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
        _batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            curr_seq = input_ids[:, -self.max_seq_len :]

            # Forward pass
            logits = self.forward(curr_seq)
            next_logits = logits[:, -1, :]  # (B, V)

            # Apply logit mask
            mask = self._get_logit_mask(curr_seq)
            next_logits = next_logits.masked_fill(~mask, float("-inf"))

            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)), dim=-1)
                next_logits[next_logits < top_k_vals[:, -1:]] = float("-inf")

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
        _B, T, _V = logits.shape

        for t in range(T):
            ctx = input_ids[:, : t + 1]
            mask = self._get_logit_mask(ctx)
            logits[:, t, :] = logits[:, t, :].masked_fill(~mask, float("-inf"))

        return logits


# =====================================================================
#  POSITION-AWARE LOSS WEIGHTING
# =====================================================================


def _build_position_weights(tokens_per_inst: int, seq_len: int, device: torch.device) -> Tensor:
    """Build per-position weight tensor for loss weighting.

    Gives 1.5x weight to position 0 (instruction type) of each instruction
    group within the flat token sequence, and 1.0 weight elsewhere. This
    encourages the model to focus on getting the instruction opcode right,
    which is the most critical field for program semantics.

    Args:
        tokens_per_inst: number of tokens per instruction in the encoding
        seq_len:         total sequence length to produce weights for
        device:          torch device for the output tensor

    Returns:
        (seq_len,) weight tensor with 1.5 at instruction-type positions
    """
    w = torch.ones(seq_len, device=device)
    # Position 0 of every instruction group gets 1.5x weight
    w[::tokens_per_inst] = 1.5
    return w


# =====================================================================
#  TRAINING FUNCTIONS
# =====================================================================


def compute_loss(
    model: RV32ITransformer,
    input_ids: Tensor,
    labels: Tensor | None = None,
    ignore_index: int = RV32IVocab.PAD,
    boundary_mask: Tensor | None = None,
    label_smoothing: float = 0.0,
    noise_std: float = 0.0,
    sample_weights: Tensor | None = None,
) -> Tensor:
    """Compute position-aware weighted cross-entropy loss for next-token prediction.

    Applies 1.5x weight to position 0 (instruction type) of each instruction
    group to encourage accurate opcode prediction, which is the most
    semantically critical field.

    Args:
        model: the transformer model
        input_ids: (batch, seq_len) input token IDs
        labels: optional (batch, seq_len) target token IDs
        ignore_index: token ID to ignore in loss computation (PAD)
        boundary_mask: optional (batch, seq_len) bool tensor, True where loss
            should be computed.  Used by sequence packing to mask out
            positions at sub-sequence boundaries where attention leaks.
        label_smoothing: label smoothing factor (0.0 = none). When > 0,
            mixes the one-hot target with uniform distribution over vocab.
        noise_std: std dev of Gaussian noise added to embeddings during
            training. Passed through to model.forward().
        sample_weights: optional (batch_size,) tensor of per-sample weights
            for reward-weighted loss (Phase 5).

    Returns:
        scalar weighted cross-entropy loss
    """
    logits = model(input_ids, noise_std=noise_std)  # (B, T, V)

    tokens_per_inst = getattr(model, "tokens_per_inst", 7)

    if labels is None:
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        # Shift boundary mask by 1 to align with shifted labels
        shift_boundary = boundary_mask[:, 1:].contiguous() if boundary_mask is not None else None
    else:
        shift_logits = logits[:, :-1, :].contiguous() if logits.shape[1] > 1 else logits
        shift_labels = labels.contiguous()
        shift_boundary = boundary_mask.contiguous() if boundary_mask is not None else None

    if shift_logits.shape[1] == 0:
        return torch.tensor(0.0, requires_grad=True, device=input_ids.device)

    B, T, V = shift_logits.shape

    # Build per-position weight vector (shifted by -1 to align with labels)
    # For label position t, the original flat position was t, so the
    # instruction-group position within the flat sequence is t.
    weights = _build_position_weights(tokens_per_inst, T, shift_logits.device)

    # Per-token cross-entropy (no reduction) for manual weighting
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    )  # (B * T,)

    # Build mask for non-ignored positions
    active_mask = (shift_labels.view(-1) != ignore_index).float()

    # Apply boundary mask (from sequence packing) if provided
    if shift_boundary is not None:
        active_mask = active_mask * shift_boundary.view(-1).float()

    # Apply position weights and compute weighted mean
    # Broadcast weights across batch: (T,) -> (B*T,) by repeating for each batch element
    weights_expanded = weights.unsqueeze(0).expand(B, -1).reshape(-1)
    if sample_weights is not None:
        # Expand per-sample weights to per-token: (B,) -> (B, 1) -> (B, T) -> (B*T,)
        sw = sample_weights.unsqueeze(1).expand(B, T).reshape(-1)
        weighted_loss = per_token_loss * weights_expanded * active_mask * sw
        total_weight = (active_mask * sw).sum()
    else:
        weighted_loss = per_token_loss * weights_expanded * active_mask
        total_weight = active_mask.sum()

    if total_weight == 0:
        return torch.tensor(0.0, requires_grad=True, device=input_ids.device)

    loss = weighted_loss.sum() / total_weight
    return loss


def compute_loss_with_scheduled_sampling(
    model: RV32ITransformer,
    input_ids: Tensor,
    teacher_prob: float = 0.5,
    noise_std: float = 0.0,
    labels: Tensor | None = None,
    ignore_index: int = RV32IVocab.PAD,
    boundary_mask: Tensor | None = None,
    sample_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Compute loss with scheduled sampling (Phase 3).

    With probability ``teacher_prob`` the full sequence is teacher-forced.
    Otherwise a corruption point is chosen, teacher forcing is applied up to
    that point, 1-5 tokens are generated autoregressively from the model's
    own distribution, and the remaining sequence continues in teacher-forced
    mode.

    Corruption points are aligned to instruction boundaries
    (multiples of tokens_per_inst).

    Args:
        model: the transformer model
        input_ids: (batch, seq_len) input token IDs
        teacher_prob: probability of using pure teacher forcing
        noise_std: embedding noise std dev (passed to compute_loss)
        labels: optional (batch, seq_len) target token IDs
        ignore_index: PAD token id
        boundary_mask: optional (batch, seq_len) bool tensor
        sample_weights: optional (batch_size,) per-sample weights
        label_smoothing: label smoothing factor (passed to compute_loss)

    Returns:
        scalar weighted cross-entropy loss
    """
    # Pure teacher-forcing path
    if torch.rand(1).item() < teacher_prob:
        return compute_loss(
            model,
            input_ids,
            labels=labels,
            ignore_index=ignore_index,
            boundary_mask=boundary_mask,
            noise_std=noise_std,
            sample_weights=sample_weights,
            label_smoothing=label_smoothing,
        )

    B, seq_len = input_ids.shape
    device = input_ids.device
    tokens_per_inst = getattr(model, "tokens_per_inst", 9)

    # --- Choose a corruption point at an instruction boundary ---
    # Usable boundaries: multiples of tokens_per_inst in [10%, 80%) of seq_len
    min_inst = max(1, int(seq_len * 0.10 / tokens_per_inst))
    max_inst = max(min_inst + 1, int(seq_len * 0.80 / tokens_per_inst))
    corrupt_inst = torch.randint(min_inst, max_inst, (1,)).item()
    corrupt_pos = corrupt_inst * tokens_per_inst  # token boundary

    # How many tokens to generate autoregressively (1-5, clamped)
    n_gen = min(torch.randint(1, 6, (1,)).item(), seq_len - corrupt_pos)

    # --- Run teacher-forced forward up to the corruption point ---
    prefix = input_ids[:, :corrupt_pos]  # (B, corrupt_pos)
    with torch.no_grad():
        _prefix_logits = model(prefix, noise_std=noise_std)  # (B, corrupt_pos, V)

    # --- Generate n_gen tokens autoregressively from the model ---
    generated = []
    cur_input = prefix  # (B, corrupt_pos)

    for step in range(n_gen):
        step_logits = model(cur_input, noise_std=noise_std)  # (B, cur_T, V)
        next_logits = step_logits[:, -1, :]  # (B, V)

        # Apply logit mask for structural validity
        logit_mask = model._get_logit_mask(cur_input)  # (B, V)
        next_logits = next_logits.masked_fill(~logit_mask, float("-inf"))

        # Temperature sampling
        probs = F.softmax(next_logits / 0.9, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
        generated.append(next_token)

        # Append generated token to input for next step
        cur_input = torch.cat([cur_input, next_token.unsqueeze(1)], dim=1)

    # --- Assemble the corrupted input ---
    gen_tokens = torch.stack(generated, dim=1)  # (B, n_gen)
    suffix_start = corrupt_pos + n_gen
    suffix = input_ids[:, suffix_start:]  # (B, remaining)

    corrupted_input = torch.cat([prefix, gen_tokens, suffix], dim=1)

    # Pad or truncate to original seq_len
    if corrupted_input.shape[1] > seq_len:
        corrupted_input = corrupted_input[:, :seq_len]
    elif corrupted_input.shape[1] < seq_len:
        pad_len = seq_len - corrupted_input.shape[1]
        pad = torch.full((B, pad_len), ignore_index, dtype=torch.long, device=device)
        corrupted_input = torch.cat([corrupted_input, pad], dim=1)

    return compute_loss(
        model,
        corrupted_input,
        labels=labels,
        ignore_index=ignore_index,
        boundary_mask=boundary_mask,
        noise_std=noise_std,
        sample_weights=sample_weights,
        label_smoothing=label_smoothing,
    )


def train_step(
    model: RV32ITransformer,
    batch: list[list[int]],
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


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model.
    Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory_footprint(total_params: int, dtype_bytes: int = 4) -> dict[str, float]:
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
        "parameters_mb": round(params_mb, 2),
        "gradients_mb": round(grads_mb, 2),
        "optimizer_mb": round(optimizer_mb, 2),
        "total_mb": round(total_mb, 2),
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
    by_fmt: dict[str, list[str]] = {}
    for name in RV32IVocab.get_instruction_names():
        fmt = RV32IVocab.get_format(name)
        by_fmt.setdefault(fmt, []).append(name)
    for fmt in [
        RV32IVocab.R_FORMAT,
        RV32IVocab.I_FORMAT,
        RV32IVocab.S_FORMAT,
        RV32IVocab.B_FORMAT,
        RV32IVocab.U_FORMAT,
        RV32IVocab.J_FORMAT,
    ]:
        if fmt in by_fmt:
            names = by_fmt[fmt]
            print(f"    {fmt}-type ({len(names)}): {', '.join(names)}")

    # --- 2. Encode a few instructions ---
    print("\n[2] Testing instruction encoding...")
    insts = [
        # ADD x1, x2, x3  (R-type)
        {"name": "ADD", "rd": 1, "rs1": 2, "rs2": 3},
        # ADDI x4, x5, 42  (I-type)
        {"name": "ADDI", "rd": 4, "rs1": 5, "imm": 42},
        # SW x2, 16(x1)  (S-type)
        {"name": "SW", "rs1": 1, "rs2": 2, "imm": 16},
        # BEQ x6, x7, 8   (B-type)
        {"name": "BEQ", "rs1": 6, "rs2": 7, "imm": 8},
        # LUI x8, 0x12345 (U-type)
        {"name": "LUI", "rd": 8, "imm": 0x12345},
        # JAL x1, 0x1234  (J-type)
        {"name": "JAL", "rd": 1, "imm": 0x1234},
        # NOP via ADDI x0, x0, 0
        {"name": "ADDI", "rd": 0, "rs1": 0, "imm": 0},
    ]

    trace_tokens = []
    for inst in insts:
        toks = tokenizer.encode_instruction(inst)
        trace_tokens.extend(toks)
        decoded = tokenizer.decode_tokens(toks)
        fmt = RV32IVocab.get_format(inst["name"])
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
    print("    Memory (float32):")
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
    assert logits.shape == (
        batch_size,
        seq_len,
        RV32IVocab.VOCAB_SIZE,
    ), f"Expected ({batch_size}, {seq_len}, {RV32IVocab.VOCAB_SIZE}), got {logits.shape}"
    print(f"    Logit range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # --- 5. Test logit masking ---
    print("\n[5] Testing logit masking...")
    with torch.no_grad():
        masked = model.apply_logit_mask(logits.clone(), input_ids)
        has_inf = torch.isinf(masked).any()
        print(f"    Mask applied (inf values present): {has_inf.item()}")

        # Check allowed counts at key positions in the first instruction (9 tokens)
        mask_single = model._get_logit_mask(input_ids[:1, :8])
        allowed_counts = mask_single.sum(dim=-1).tolist()
        print(f"    Allowed tokens after each of first 8 positions: {allowed_counts}")

    # --- 5b. Test nibble encoding roundtrip ---
    print("\n[5b] Testing nibble encoding roundtrip...")
    test_imms = [0, 1, -1, 42, -42, 100, -100, 255, -255, 1000, -1000, 2047, -2048, 0x1234]
    for imm_val in test_imms:
        nib_tokens = RV32IVocab.imm_to_nibble_tokens(imm_val)
        recovered = RV32IVocab.nibble_tokens_to_imm(nib_tokens)
        clamped = max(-2048, min(2047, imm_val))
        status = "OK" if recovered == clamped else "FAIL"
        print(f"    imm={imm_val:6d} -> tokens={nib_tokens} -> recovered={recovered:6d} [{status}]")
        assert recovered == clamped, f"Roundtrip failed for {imm_val}"

    # Verify is_imm_nibble works
    assert RV32IVocab.is_imm_nibble(RV32IVocab.IMM_POS0_START)
    assert RV32IVocab.is_imm_nibble(RV32IVocab.IMM_POS2_START + 15)
    assert not RV32IVocab.is_imm_nibble(RV32IVocab.NO_REG)
    print("    is_imm_nibble checks passed")

    # --- 6. Test generation ---
    print("\n[6] Testing generation (single step)...")
    prompt_tok = RV32IVocab.get_instruction_token("ADD")
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
