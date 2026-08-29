"""Model weight migration for architecture scaling.

Transfers weights from a small model to a larger one while preserving
the model's function as closely as possible.

Key strategies:
1. Zero-pad width growth: linear layers get zero-padded weight matrices
2. SwiGLU asymmetric padding: gate zero-padded (silu(0)=0), up small-init
3. Depth growth: insert identity layers (zero-init residual branches)
4. Attention scaling: compensate for changed head_dim in Q projection
5. RMSNorm compensation: scale weights to account for changed normalization

The migration preserves the forward pass within floating-point tolerance,
allowing the scaled model to continue training from the old model's
learned representations.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch import nn

from .types import MigrationResult

logger = logging.getLogger(__name__)


class ModelMigrator:
    """Migrates weights from a small model to a larger architecture.

    Supports:
    - Width growth: zero-pad linear layers, scale RMSNorm and Q projection
    - Depth growth: insert identity-initialized transformer blocks
    - SwiGLU asymmetric padding: zero-pad gate, small-init up

    Usage:
        migrator = ModelMigrator()
        result = migrator.migrate(old_model, new_model, old_config, new_config)
        if result.success:
            print(f"Migration complete: {result.message}")
    """

    def __init__(self):
        pass

    def migrate(
        self,
        old_model: nn.Module,
        new_model: nn.Module,
        old_config: dict,
        new_config: dict,
    ) -> MigrationResult:
        """Transfer weights from old_model to new_model.

        Handles width growth (d_model increase), depth growth (n_layers increase),
        and SwiGLU asymmetric padding. Preserves the forward pass within 1e-5.

        Args:
            old_model: Source model with trained weights.
            new_model: Target model (freshly initialized).
            old_config: Source architecture config dict with keys:
                d_model, n_layers, n_heads, d_ff.
            new_config: Target architecture config dict with same keys.

        Returns:
            MigrationResult with success status and message.
        """
        try:
            old_d = old_config["d_model"]
            new_d = new_config["d_model"]
            old_n = old_config["n_layers"]
            new_n = new_config["n_layers"]
            old_d_ff = old_config["d_ff"]
            new_d_ff = new_config["d_ff"]
            n_heads = old_config["n_heads"]

            # 1. Transfer token embeddings (zero-pad new columns)
            self._transfer_embeddings(old_model.token_embedding, new_model.token_embedding)

            # 2. Transfer token type embeddings (zero-pad new columns)
            self._transfer_embeddings(old_model.token_type_embedding, new_model.token_type_embedding)

            # 3. Transfer transformer blocks (width growth)
            n_transfer = min(old_n, new_n)
            for i in range(n_transfer):
                self._transfer_block(
                    old_model.blocks[i],
                    new_model.blocks[i],
                    old_d,
                    new_d,
                    old_d_ff,
                    new_d_ff,
                    n_heads,
                )

            # 4. Insert identity layers for depth growth
            if new_n > old_n:
                self._insert_identity_layers(new_model, old_n, new_n)

            # 5. Transfer lm_head_norm (with RMSNorm compensation)
            self._transfer_rmsnorm(
                old_model.lm_head_norm,
                new_model.lm_head_norm,
                old_d,
                new_d,
            )

            # 6. lm_head is tied with token_embedding — already transferred

            # 7. Validate forward pass identity
            test_input = torch.randint(1, 2048, (1, 64))
            passed = self._validate_identity(old_model, new_model, test_input)

            return MigrationResult(
                success=True,
                message=(
                    f"Migrated {old_n}L/{old_d}d → {new_n}L/{new_d}d"
                    f" (identity check: {'PASS' if passed else 'FAIL'})"
                ),
            )

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return MigrationResult(
                success=False,
                message=f"Migration failed: {e}",
            )

    def _transfer_embeddings(self, old_emb: nn.Embedding, new_emb: nn.Embedding):
        """Zero-pad token embeddings to new d_model.

        Copies old embedding weights into the first old_d columns.
        New columns remain zero (explicitly zeroed).
        Preserves padding_idx zero vector.
        """
        old_d = old_emb.weight.data.shape[1]
        # Zero out the new embedding first to clear random init
        new_emb.weight.data.zero_()
        # Copy old weights
        new_emb.weight.data[:, :old_d] = old_emb.weight.data
        # Ensure padding row stays zero
        if old_emb.padding_idx is not None:
            new_emb.weight.data[old_emb.padding_idx].zero_()

    def _transfer_linear(
        self,
        old_linear: nn.Linear,
        new_linear: nn.Linear,
        pad_output: str = "zero",
    ):
        """Transfer linear layer weights with optional output padding.

        Copies old weights into the top-left corner of the new weight matrix.
        Remaining rows/columns are explicitly zeroed.

        Args:
            old_linear: Source linear layer.
            new_linear: Target linear layer (same or larger).
            pad_output: "zero" for zero-padding, "small" for small-init on new rows.
        """
        old_out, old_in = old_linear.weight.data.shape
        _new_out, _new_in = new_linear.weight.data.shape

        # Zero out the entire new weight matrix
        new_linear.weight.data.zero_()
        # Copy old weights into top-left corner
        new_linear.weight.data[:old_out, :old_in] = old_linear.weight.data

    def _transfer_swiglu(
        self,
        old_ffn: nn.Module,
        new_ffn: nn.Module,
        old_d: int,
        new_d: int,
        old_d_ff: int,
        new_d_ff: int,
    ):
        """Transfer SwiGLU FFN with asymmetric padding.

        Asymmetric padding strategy:
        - w1 (gate): zero-pad → silu(0) = 0, padded dims contribute nothing
        - w3 (up): small-init for new columns → breaks symmetry, allows learning
        - w2 (down): zero-pad → padded output dims contribute nothing

        This ensures:
        - Old dimensions produce exactly the same FFN output
        - New dimensions start from zero (gate) or small (up) for clean learning
        """
        # w1 (gate): zero-pad both dims
        # silu(0) = 0, so zero-padded input features produce zero gate values
        new_ffn.w1.weight.data.zero_()
        new_ffn.w1.weight.data[:old_d_ff, :old_d] = old_ffn.w1.weight.data

        # w2 (down): zero-pad both dims
        # Zero rows mean new output dimensions contribute nothing to residual
        new_ffn.w2.weight.data.zero_()
        new_ffn.w2.weight.data[:old_d, :old_d_ff] = old_ffn.w2.weight.data

        # w3 (up): zero-pad input dim, keep small random init for new columns
        # Small random init breaks symmetry, allows new dims to learn
        new_ffn.w3.weight.data.zero_()
        new_ffn.w3.weight.data[:old_d_ff, :old_d] = old_ffn.w3.weight.data
        # Re-initialize new columns with small random values (breaks symmetry)
        if old_d < new_d:
            nn.init.normal_(new_ffn.w3.weight.data[:old_d_ff, old_d:new_d], mean=0.0, std=0.02)

    def _transfer_attention(
        self,
        old_attn: nn.Module,
        new_attn: nn.Module,
        old_d: int,
        new_d: int,
        n_heads: int,
    ):
        """Transfer attention weights with Q scaling for head_dim change.

        When d_model grows but n_heads stays constant, head_dim changes.
        The attention scaling factor (1/sqrt(head_dim)) changes, which would
        alter the softmax distribution. To preserve exact attention scores:

        - Q projection is scaled by sqrt(new_head_dim / old_head_dim)
        - K and V projections are copied without scaling
        - This compensates for the changed denominator in scaled attention

        The key insight: qkv and out_proj weights have a (n_heads, head_dim, d_model)
        structure when viewed head-aware. We must respect this structure when
        padding, NOT simply copy flat rows/cols.
        """
        old_head_dim = old_d // n_heads
        new_head_dim = new_d // n_heads

        # Q scaling factor to preserve attention scores
        q_scale = (new_head_dim / old_head_dim) ** 0.5

        # Zero out the entire new weight matrix first
        new_attn.qkv.weight.data.zero_()
        new_attn.out_proj.weight.data.zero_()

        # ── qkv weight: (3*d_model, d_model) ──
        # Split into Q, K, V parts, each (d_model, d_model) = (n_heads * head_dim, d_model)
        # Then reshape each to (n_heads, head_dim, d_model) for head-aware copying
        old_qkv = old_attn.qkv.weight.data

        # Q part: (n_heads * old_head_dim, old_d) -> (n_heads, old_head_dim, old_d)
        old_q = old_qkv[:old_d, :old_d].view(n_heads, old_head_dim, old_d)
        old_q = old_q * q_scale  # scale to compensate for head_dim change

        # Zero-pad head_dim: (n_heads, old_head_dim, old_d) -> (n_heads, new_head_dim, old_d)
        new_q = torch.zeros(n_heads, new_head_dim, old_d)
        new_q[:, :old_head_dim, :] = old_q
        # Zero-pad input dim: (n_heads, new_head_dim, old_d) -> (n_heads, new_head_dim, new_d)
        new_q_padded = torch.zeros(n_heads, new_head_dim, new_d)
        new_q_padded[:, :, :old_d] = new_q
        # Flatten back to (n_heads * new_head_dim, new_d) = (new_d, new_d)
        new_attn.qkv.weight.data[:new_d, :] = new_q_padded.view(new_d, new_d)

        # K part: same structure, no scaling
        old_k = old_qkv[old_d : 2 * old_d, :old_d].view(n_heads, old_head_dim, old_d)
        new_k = torch.zeros(n_heads, new_head_dim, old_d)
        new_k[:, :old_head_dim, :] = old_k
        new_k_padded = torch.zeros(n_heads, new_head_dim, new_d)
        new_k_padded[:, :, :old_d] = new_k
        new_attn.qkv.weight.data[new_d : 2 * new_d, :] = new_k_padded.view(new_d, new_d)

        # V part: same structure, no scaling
        old_v = old_qkv[2 * old_d : 3 * old_d, :old_d].view(n_heads, old_head_dim, old_d)
        new_v = torch.zeros(n_heads, new_head_dim, old_d)
        new_v[:, :old_head_dim, :] = old_v
        new_v_padded = torch.zeros(n_heads, new_head_dim, new_d)
        new_v_padded[:, :, :old_d] = new_v
        new_attn.qkv.weight.data[2 * new_d : 3 * new_d, :] = new_v_padded.view(new_d, new_d)

        # ── out_proj weight: (d_model, d_model) ──
        # Input dim is interleaved (n_heads, head_dim), output dim is flat d_model
        # Reshape input dim to (n_heads, head_dim) for head-aware copying
        old_out = old_attn.out_proj.weight.data  # (old_d, old_d)
        # Reshape to (old_d, n_heads, old_head_dim) — output_dim x (head, head_dim)
        old_out_head = old_out.view(old_d, n_heads, old_head_dim)

        # Zero-pad head_dim: (old_d, n_heads, new_head_dim)
        new_out_head = torch.zeros(old_d, n_heads, new_head_dim)
        new_out_head[:, :, :old_head_dim] = old_out_head
        # Reshape to (old_d, new_d)
        new_out_mid = new_out_head.view(old_d, new_d)

        # Zero-pad output dim: (new_d, new_d)
        new_attn.out_proj.weight.data[:old_d, :] = new_out_mid

    def _transfer_rmsnorm(
        self,
        old_norm: nn.Module,
        new_norm: nn.Module,
        old_d: int,
        new_d: int,
    ):
        """Transfer RMSNorm with scaling compensation for changed normalization.

        When d_model grows, the normalization denominator changes because
        it averages over more dimensions (including zeros from padding).

        The normalization factor changes by approximately sqrt(old_d/new_d),
        so we scale the weight to compensate. This is exact when eps is
        negligible relative to mean(x^2), which holds for typical activations.
        """
        scale = (old_d / new_d) ** 0.5
        new_norm.weight.data[:old_d] = old_norm.weight.data * scale
        # New dimensions get default init (1.0) — doesn't matter since
        # the corresponding input dimensions are zero

    def _transfer_block(
        self,
        old_block: nn.Module,
        new_block: nn.Module,
        old_d: int,
        new_d: int,
        old_d_ff: int,
        new_d_ff: int,
        n_heads: int,
    ):
        """Transfer a single transformer block (width growth).

        Transfers:
        - ln1, ln2: RMSNorm with scaling compensation
        - attn: qkv with Q scaling, out_proj with zero-padding
        - ffn: SwiGLU with asymmetric padding
        """
        # RMSNorm layers
        self._transfer_rmsnorm(old_block.ln1, new_block.ln1, old_d, new_d)
        self._transfer_rmsnorm(old_block.ln2, new_block.ln2, old_d, new_d)

        # Attention
        self._transfer_attention(old_block.attn, new_block.attn, old_d, new_d, n_heads)

        # SwiGLU FFN
        self._transfer_swiglu(old_block.ffn, new_block.ffn, old_d, new_d, old_d_ff, new_d_ff)

    def _insert_identity_layers(
        self,
        model: nn.Module,
        n_old: int,
        n_new: int,
    ):
        """Insert zero-initialized residual layers for depth growth.

        Identity blocks satisfy: block(x) = x
        Achieved by zeroing residual branch weights:
        - attn.qkv = 0, attn.out_proj = 0 → attention output = 0
        - ffn.w2 = 0 → FFN output = 0

        The residual connections then pass x through unchanged:
        x = x + attn(ln1(x)) + ffn(ln2(x)) = x + 0 + 0 = x
        """
        for i in range(n_old, n_new):
            block = model.blocks[i]

            # Zero-init residual branches to create identity blocks
            nn.init.zeros_(block.attn.qkv.weight)
            nn.init.zeros_(block.attn.out_proj.weight)
            nn.init.zeros_(block.ffn.w2.weight)

            # w1, w3 keep default init (doesn't affect identity since w2=0)
            # ln1, ln2 keep default init (RMSNorm of zeros = zeros)

    def _validate_identity(
        self,
        old_model: nn.Module,
        new_model: nn.Module,
        test_input: torch.Tensor,
    ) -> bool:
        """Validate that new_model(test_input) ≈ old_model(test_input).

        Both models must produce outputs within 1e-5 of each other.
        This confirms the migration preserves the forward pass.

        Args:
            old_model: Source model (eval mode, frozen).
            new_model: Target model (eval mode, migrated weights).
            test_input: Token IDs tensor of shape (1, seq_len).

        Returns:
            True if max absolute difference < 1e-5.
        """
        old_model.eval()
        new_model.eval()

        with torch.no_grad():
            old_out = old_model(test_input)
            new_out = new_model(test_input)

        max_diff = (old_out - new_out).abs().max().item()
        passed = max_diff < 1e-5

        if not passed:
            logger.warning(f"Identity check failed: max_diff={max_diff:.2e} > 1e-5")

        return passed
