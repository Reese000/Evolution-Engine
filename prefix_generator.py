#!/usr/bin/env python3
"""
prefix_generator.py — Model-Guided Instruction Completion
==========================================================

Loads a trained RV32I transformer checkpoint and provides a complete()
method that generates new instruction sequences from prefix contexts.

Handles the architecture mismatch between the old checkpoint (GELU FFN,
learned position embeddings) and the current model (SwiGLU FFN, RoPE)
by mapping compatible weights and randomly initializing new components.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Ensure milestone1 is importable
_milestone1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "milestone1")
if _milestone1 not in sys.path:
    sys.path.insert(0, _milestone1)

from rv32i_model import RV32ITransformer, RV32IFieldTokenizer, RV32IVocab
from rv32i_mask import decode as rv_decode

logger = logging.getLogger(__name__)


# =====================================================================
#  KEY MAPPING: Old checkpoint (GELU FFN) -> New model (SwiGLU + RoPE)
# =====================================================================

def _build_key_map(n_layers: int) -> Dict[str, str]:
    """Build a mapping from old checkpoint keys to new model keys.

    Old architecture (in checkpoint):
        blocks.N.ffn.0 = Linear(d_model, d_ff, bias=True)   # GELU gate
        blocks.N.ffn.3 = Linear(d_ff, d_model, bias=True)   # GELU output
        blocks.N.ln1 has weight + bias (LayerNorm)
        blocks.N.ln2 has weight + bias (LayerNorm)
        pos_embedding = learned position embeddings
        ln_f = final LayerNorm (weight + bias)

    New architecture (current model):
        blocks.N.ffn.w1 = Linear(d_model, d_ff, bias=False)  # SwiGLU gate
        blocks.N.ffn.w2 = Linear(d_ff, d_model, bias=False)  # SwiGLU down
        blocks.N.ffn.w3 = Linear(d_model, d_ff, bias=False)  # SwiGLU up (NEW)
        blocks.N.ln1 = RMSNorm (weight only, no bias)
        blocks.N.ln2 = RMSNorm (weight only, no bias)
        RoPE replaces learned position embeddings
        lm_head_norm = final RMSNorm (weight only)
    """
    key_map = {}

    for i in range(n_layers):
        prefix_old = f"blocks.{i}"
        prefix_new = f"blocks.{i}"

        # Attention weights: identical between architectures
        key_map[f"{prefix_old}.attn.qkv.weight"] = f"{prefix_new}.attn.qkv.weight"
        key_map[f"{prefix_old}.attn.out_proj.weight"] = f"{prefix_new}.attn.out_proj.weight"

        # LayerNorm 1: drop bias (RMSNorm has no bias)
        key_map[f"{prefix_old}.ln1.weight"] = f"{prefix_new}.ln1.weight"

        # LayerNorm 2: drop bias
        key_map[f"{prefix_old}.ln2.weight"] = f"{prefix_new}.ln2.weight"

        # FFN: GELU -> SwiGLU mapping
        # Old ffn.0 (Linear with bias) -> New ffn.w1 (Linear, no bias)
        key_map[f"{prefix_old}.ffn.0.weight"] = f"{prefix_new}.ffn.w1.weight"
        # Old ffn.3 (Linear with bias) -> New ffn.w2 (Linear, no bias)
        key_map[f"{prefix_old}.ffn.3.weight"] = f"{prefix_new}.ffn.w2.weight"
        # ffn.w3 (SwiGLU up projection) is NEW — no mapping, stays random

    # Token embedding: identical
    key_map["token_embedding.weight"] = "token_embedding.weight"

    # LM head: identical (tied to token_embedding)
    key_map["lm_head.weight"] = "lm_head.weight"

    # Final norm: ln_f -> lm_head_norm (drop bias)
    key_map["ln_f.weight"] = "lm_head_norm.weight"

    # Dropped keys (no mapping):
    # - pos_embedding.weight (replaced by RoPE)
    # - ln1.bias, ln2.bias, ln_f.bias (RMSNorm has no bias)
    # - ffn.0.bias, ffn.3.bias (SwiGLU layers have no bias)

    return key_map


def _infer_model_dims(state_dict: dict) -> dict:
    """Infer model dimensions from the state dict shapes.

    The checkpoint's config metadata may not match the actual saved weights
    (e.g., d_model=512 in config but 256 in weights). We trust the shapes.
    """
    dims = {}

    # vocab_size from token_embedding
    te = state_dict.get("token_embedding.weight")
    if te is not None:
        dims["vocab_size"] = te.shape[0]
        dims["d_model"] = te.shape[1]
    else:
        dims["vocab_size"] = RV32IVocab.VOCAB_SIZE
        dims["d_model"] = 256

    # n_layers: count blocks.N.* keys
    layer_nums = set()
    for k in state_dict:
        if k.startswith("blocks.") and ".attn.qkv.weight" in k:
            layer_num = int(k.split(".")[1])
            layer_nums.add(layer_num)
    dims["n_layers"] = max(layer_nums) + 1 if layer_nums else 8

    # d_ff from ffn.0.weight shape (old checkpoint)
    ffn_key = f"blocks.0.ffn.0.weight"
    if ffn_key in state_dict:
        dims["d_ff"] = state_dict[ffn_key].shape[0]
    else:
        dims["d_ff"] = dims["d_model"] * 4

    # n_heads: infer from qkv weight (3*d_model x d_model)
    qkv_key = "blocks.0.attn.qkv.weight"
    if qkv_key in state_dict:
        # qkv is (3*d_model, d_model), so d_model is the second dim
        # Common head dims: 32, 64
        d_model = dims["d_model"]
        for nh in [8, 4, 16, 2]:
            if d_model % nh == 0 and d_model // nh >= 16:
                dims["n_heads"] = nh
                break
        else:
            dims["n_heads"] = 8
    else:
        dims["n_heads"] = 8

    dims["max_seq_len"] = 1024  # safe default

    return dims


# =====================================================================
#  PREFIX GENERATOR
# =====================================================================

class PrefixGenerator:
    """Generates RV32I instruction completions from prefix contexts.

    Loads a trained transformer checkpoint and uses it to autoregressively
    generate new instruction sequences, conditioned on a prefix of existing
    instructions.
    """

    def __init__(
        self,
        checkpoint_path: str,
        temperature: float = 0.9,
        max_new_tokens: int = 64,
        device: Optional[str] = None,
        verbose: bool = True,
        **kwargs,
    ):
        """Initialize PrefixGenerator from a checkpoint file.

        Args:
            checkpoint_path: path to the .pt checkpoint file
            temperature: default sampling temperature
            max_new_tokens: default max tokens to generate per completion
            device: 'cuda', 'cpu', or None for auto-detect
        """
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.tokenizer = RV32IFieldTokenizer()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # torch.compile for inference (20-40% speedup)
        # Requires Triton — skip on Windows where Triton is unavailable.
        if self.device.type == "cuda":
            try:
                import triton  # noqa: F401
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("torch.compile enabled for inference (reduce-overhead)")
            except ImportError:
                logger.info("Triton not available — skipping torch.compile for inference")
            except Exception as e:
                logger.warning(f"torch.compile failed (falling back to eager): {e}")
        logger.info(
            f"PrefixGenerator loaded from {checkpoint_path} "
            f"(device={self.device}, temp={temperature})"
        )

    def _load_model(self, checkpoint_path: str) -> RV32ITransformer:
        """Load model from checkpoint, handling architecture migration."""
        if not os.path.isfile(checkpoint_path):
            logger.warning(
                f"Checkpoint not found: {checkpoint_path}, "
                f"creating fresh model"
            )
            return self._create_fresh_model()

        try:
            ckpt = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, creating fresh model")
            return self._create_fresh_model()

        state_dict = ckpt.get("model_state_dict", ckpt)

        # Infer dimensions from actual weight shapes
        dims = _infer_model_dims(state_dict)
        logger.info(
            f"Inferred model dims: d_model={dims['d_model']}, "
            f"n_layers={dims['n_layers']}, d_ff={dims['d_ff']}, "
            f"vocab_size={dims['vocab_size']}"
        )

        # Build model with NEW architecture (SwiGLU + RoPE)
        try:
            model = RV32ITransformer(
                vocab_size=dims["vocab_size"],
                d_model=dims["d_model"],
                n_layers=dims["n_layers"],
                n_heads=dims["n_heads"],
                d_ff=dims["d_ff"],
                max_seq_len=dims["max_seq_len"],
                dropout=0.0,  # no dropout for inference
                tie_weights=True,
                use_rope=True,
            )
        except Exception as e:
            logger.warning(f"Failed to create model with inferred dims: {e}")
            return self._create_fresh_model()

        # Map old keys to new keys
        key_map = _build_key_map(dims["n_layers"])

        mapped_sd = {}
        unmapped_old = []
        for old_key, tensor in state_dict.items():
            if old_key in key_map:
                new_key = key_map[old_key]
                mapped_sd[new_key] = tensor
            else:
                unmapped_old.append(old_key)

        if unmapped_old:
            logger.info(
                f"Dropped {len(unmapped_old)} old keys (expected): "
                f"{unmapped_old[:5]}{'...' if len(unmapped_old) > 5 else ''}"
            )

        # Load with strict=False — new architecture has extra keys
        # (ffn.w3.*, rotary.*, token_type_embedding.*) that stay random
        result = model.load_state_dict(mapped_sd, strict=False)
        if result.missing_keys:
            logger.info(
                f"Missing keys (new architecture, randomly initialized): "
                f"{result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}"
            )
        if result.unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {result.unexpected_keys}")

        model = model.to(self.device)
        logger.info(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} params)")
        return model

    def _create_fresh_model(self) -> RV32ITransformer:
        """Create a fresh small model as fallback."""
        logger.info("Creating fresh fallback model (d_model=256, 8 layers)")
        model = RV32ITransformer(
            vocab_size=RV32IVocab.VOCAB_SIZE,
            d_model=256,
            n_layers=8,
            n_heads=8,
            d_ff=1024,
            max_seq_len=1024,
            dropout=0.0,
            tie_weights=True,
            use_rope=True,
        )
        return model.to(self.device)

    def _hex_to_inst_dict(self, hex_str: str) -> Optional[Dict]:
        """Convert a hex instruction string to an instruction dict."""
        try:
            word = int(hex_str, 16)
            return rv_decode(word, validate=False)
        except (ValueError, TypeError):
            return None

    def _inst_dict_to_hex(self, inst: Dict) -> str:
        """Convert an instruction dict to a hex string.

        Uses the tokenizer to encode, then reconstructs the 32-bit word
        from the decoded fields.
        """
        try:
            from rv32i_mask import encode as rv_encode
            name = inst.get("name", "ADDI")
            rd = inst.get("rd", 0) or 0
            rs1 = inst.get("rs1", 0) or 0
            rs2 = inst.get("rs2", 0) or 0
            imm = inst.get("imm", 0) or 0
            word = rv_encode(name, rd=rd, rs1=rs1, rs2=rs2, imm=imm, validate=False)
            return f"0x{word & 0xFFFFFFFF:08x}"
        except Exception:
            # Fallback: NOP (ADDI x0, x0, 0)
            return "0x00000013"

    def _tokens_to_hex_list(self, token_ids: List[int]) -> List[str]:
        """Convert a list of generated token IDs to hex instruction strings.

        Parses the flat token stream into instruction groups (9 tokens each,
        ending with SEP), decodes each group to an instruction dict, then
        converts to hex.
        """
        vocab = RV32IVocab
        hex_instructions = []
        i = 0

        while i < len(token_ids):
            # Skip non-instruction tokens (PAD, CLS, etc.)
            if token_ids[i] not in range(vocab.INST_START, vocab.INST_START + vocab.INST_COUNT):
                i += 1
                continue

            # Collect 9 tokens for one instruction
            inst_tokens = token_ids[i : i + vocab.TOKENS_PER_INST]
            if len(inst_tokens) < vocab.TOKENS_PER_INST:
                break

            # Decode this instruction
            inst_dict = self.tokenizer.decode_tokens(inst_tokens)
            if inst_dict is not None:
                hex_str = self._inst_dict_to_hex(inst_dict)
                hex_instructions.append(hex_str)

            i += vocab.TOKENS_PER_INST

        return hex_instructions

    @torch.no_grad()
    def _generate_with_top_p(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """Generate tokens with top-k, top-p, and repetition penalty.

        This replaces the model's built-in generate() to support top_p
        and repetition_penalty, which the base model doesn't implement.

        Args:
            input_ids: (batch, seq_len) prompt tokens
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_k: keep only top-k tokens
            top_p: nucleus sampling threshold
            repetition_penalty: penalty for previously generated tokens

        Returns:
            (batch, seq_len + generated_tokens) completed sequences
        """
        self.model.eval()
        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            curr_seq = input_ids[:, -self.model.max_seq_len :]

            # Forward pass
            logits = self.model(curr_seq)
            next_logits = logits[:, -1, :]  # (B, V)

            # Apply logit mask (valid field transitions only)
            mask = self.model._get_logit_mask(curr_seq)
            next_logits = next_logits.masked_fill(~mask, float("-inf"))

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    prev_tokens = set(input_ids[b].tolist())
                    for tok_id in prev_tokens:
                        if next_logits[b, tok_id] > 0:
                            next_logits[b, tok_id] /= repetition_penalty
                        else:
                            next_logits[b, tok_id] *= repetition_penalty

            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_vals, _ = torch.topk(
                    next_logits, min(top_k, next_logits.size(-1)), dim=-1
                )
                next_logits[next_logits < top_k_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above top_p
                # (keep the first token above the threshold)
                sorted_mask = cumulative_probs - sorted_probs > top_p
                sorted_logits[sorted_mask] = float("-inf")

                # Scatter back to original indices
                next_logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

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

            # Note: SEP is the instruction boundary token (every instruction
            # ends with SEP), NOT a program terminator. We must NOT stop
            # generation on SEP — we need multiple instructions per program.
            # Generation stops when max_new_tokens is reached.

        return input_ids

    @torch.no_grad()
    def generate_with_cache(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """Generate tokens with KV-cache for fast autoregressive generation.

        Uses the model's forward_with_cache() to build a KV cache on the
        prompt tokens, then only processes the new token at each step
        instead of re-processing the full sequence. This gives O(1) per-step
        cost instead of O(n), making generation much faster for long sequences.

        Falls back to _generate_with_top_p() if KV-cache generation fails.

        Args:
            input_ids: (batch, seq_len) prompt tokens
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_k: keep only top-k tokens
            top_p: nucleus sampling threshold
            repetition_penalty: penalty for repeating previously generated tokens

        Returns:
            (batch, seq_len + generated_tokens) completed sequences
        """
        self.model.eval()
        batch_size = input_ids.shape[0]

        try:
            # Prefill: run full forward on prompt to populate KV cache
            logits, kv_caches = self.model.forward_with_cache(input_ids)
            next_logits = logits[:, -1, :]

            # Apply logit mask for the prompt
            # _get_logit_mask returns (batch, vocab_size) — already 2D
            mask = self.model._get_logit_mask(input_ids)
            next_logits = next_logits.masked_fill(~mask, float("-inf"))
        except Exception as e:
            logger.warning(f"KV-cache prefill failed, falling back to non-cached generation: {e}")
            return self._generate_with_top_p(
                input_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty
            )

        for _ in range(max_new_tokens):
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    prev_tokens = set(input_ids[b].tolist())
                    for tok_id in prev_tokens:
                        if next_logits[b, tok_id] > 0:
                            next_logits[b, tok_id] /= repetition_penalty
                        else:
                            next_logits[b, tok_id] *= repetition_penalty

            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None and top_k > 0:
                top_k_vals, _ = torch.topk(
                    next_logits, min(top_k, next_logits.size(-1)), dim=-1
                )
                next_logits[next_logits < top_k_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True, dim=-1
                )
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_mask = cumulative_probs - sorted_probs > top_p
                sorted_logits[sorted_mask] = float("-inf")
                next_logits = sorted_logits.scatter(-1, sorted_indices, sorted_logits)

            # Sample
            if temperature == 0.0:
                next_tokens = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                if torch.isnan(probs).any():
                    probs = torch.ones_like(probs) / probs.size(-1)
                next_tokens = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Incremental forward with cache (only processes the new token)
            try:
                logits, kv_caches = self.model.forward_with_cache(
                    next_tokens,  # (B, 1) — just the new token
                    kv_caches=kv_caches,
                )
                next_logits = logits[:, -1, :]

                # Apply logit mask for the new position
                # _get_logit_mask returns (batch, vocab_size) — already 2D
                mask = self.model._get_logit_mask(input_ids)
                next_logits = next_logits.masked_fill(~mask, float("-inf"))
            except Exception as e:
                logger.warning(f"KV-cache step failed at step {_}: {e}")
                # Fall back to non-cached generation for remaining steps
                return self._generate_with_top_p(
                    input_ids, max_new_tokens - _, temperature, top_k, top_p, repetition_penalty
                )

        return input_ids

    def complete(
        self,
        prefix_source: List[str],
        prefix_count: int = 4,
        num_samples: int = 4,
        temperature: Optional[float] = None,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> List[List[str]]:
        """Generate instruction completions from a prefix context.

        Args:
            prefix_source: list of hex instruction strings to use as context
            prefix_count: number of prefix instructions to use (from the end
                         of prefix_source)
            num_samples: number of completions to generate
            temperature: sampling temperature (overrides default)
            top_k: keep only top-k tokens during sampling
            top_p: nucleus sampling threshold
            repetition_penalty: penalty for repeating previously generated tokens

        Returns:
            List of completions, where each completion is a list of hex
            instruction strings. E.g.:
            [["0x02a00093", "0x03a00113", ...], ["0x02a00093", ...], ...]
        """
        if temperature is None:
            temperature = self.temperature

        if not prefix_source or prefix_count <= 0:
            return []

        # Take the last prefix_count instructions as context
        prefix_hex = prefix_source[-prefix_count:]

        # Convert hex strings to instruction dicts, then tokenize
        prefix_insts = []
        for h in prefix_hex:
            inst = self._hex_to_inst_dict(h)
            if inst is not None:
                prefix_insts.append(inst)

        if not prefix_insts:
            return []

        # Tokenize prefix
        prefix_tokens = self.tokenizer.encode_trace(prefix_insts)
        if not prefix_tokens:
            return []

        # Build input: CLS + prefix tokens
        input_ids = torch.tensor(
            [[RV32IVocab.CLS] + prefix_tokens],
            dtype=torch.long,
            device=self.device,
        )

        # Expand for batch generation
        input_ids = input_ids.expand(num_samples, -1)

        # Calculate how many new tokens to generate
        # Each instruction = 9 tokens, generate enough for ~4-8 instructions
        max_new = min(
            self.max_new_tokens,
            max(18, prefix_count * 3) * 9,  # at least 2 full instructions worth
        )
        import sys
        print(f"    [PrefixGenFormula] max_new_tokens={self.max_new_tokens}, prefix_count={prefix_count}, max_new={max_new}", file=sys.stderr)

        # Generate with KV-cache (fast) or fallback to non-cached
        output = self.generate_with_cache(
            input_ids,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Convert generated tokens to hex instruction lists
        completions = []
        for b in range(output.shape[0]):
            tokens = output[b].tolist()
            # Skip CLS token
            if tokens and tokens[0] == RV32IVocab.CLS:
                tokens = tokens[1:]

            hex_list = self._tokens_to_hex_list(tokens)

            # Append EBREAK terminator if not already present
            if hex_list and hex_list[-1].upper() != "0X00100073":
                hex_list.append("0x00100073")

            # Debug: log generation details for first batch item
            if b == 0:
                import sys
                print(f"    [PrefixGenDebug] input_len={input_ids.shape[1]}, "
                      f"max_new={max_new}, output_len={output.shape[1]}, "
                      f"tokens_after_skip={len(tokens)}, hex_insts={len(hex_list)}, "
                      f"hex_list={[h[:10]+'...' if len(h)>10 else h for h in hex_list[:8]]}",
                      file=sys.stderr)

            # Filter: keep only completions with at least 2 instructions
            if len(hex_list) >= 2:
                completions.append(hex_list)

        return completions


# =====================================================================
#  SELF-TEST
# =====================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 64)
    print("PrefixGenerator Self-Test")
    print("=" * 64)

    # Find checkpoint
    base = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(base, "milestone1", "checkpoints", "checkpoint_best.pt")

    if not os.path.isfile(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    print(f"\nLoading checkpoint: {ckpt_path}")
    pg = PrefixGenerator(ckpt_path, temperature=0.9, max_new_tokens=64)

    # Test completions
    test_prefix = [
        "0x02a00093",  # ADDI x1, x0, 42
        "0x03a00113",  # ADDI x2, x0, 58
        "0x002080b3",  # ADD x1, x1, x2
        "0x00000013",  # NOP (ADDI x0, x0, 0)
    ]

    print(f"\nPrefix ({len(test_prefix)} instructions): {test_prefix}")
    print(f"Generating {4} samples with temp=0.9, top_k=40, top_p=0.9...")

    result = pg.complete(
        test_prefix,
        prefix_count=4,
        num_samples=4,
        temperature=0.9,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    print(f"\nGenerated {len(result)} completions:")
    for i, comp in enumerate(result):
        print(f"  [{i}] ({len(comp)} insts): {comp[:5]}{'...' if len(comp) > 5 else ''}")

    print("\n" + "=" * 64)
    print(f"SELF-TEST PASSED: {len(result)} completions generated")
    print("=" * 64)
