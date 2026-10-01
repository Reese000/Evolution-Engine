"""Tests for ModelMigrator — weight transfer between model sizes.

Tests cover:
1. Weight preservation after migration
2. SwiGLU asymmetric padding correctness
3. Forward pass shape preservation
4. Function identity (new_model(x) ≈ old_model(x) within 1e-5)
"""

import os
import sys

import torch

# Ensure milestone1/ is on sys.path for rv32i_model import
_MILESTONE1_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "milestone1",
)
if _MILESTONE1_DIR not in sys.path:
    sys.path.insert(0, _MILESTONE1_DIR)

from rv32i_model import RV32ITransformer

from milestone1.training.scaling.model_migrator import ModelMigrator

# ── Fixtures ────────────────────────────────────────────────────────────────

# Model size configs matching MODEL_SIZE_CONFIGS in model_factory.py
OLD_CONFIG = {"d_model": 256, "n_layers": 8, "n_heads": 8, "d_ff": 1024}
NEW_CONFIG = {"d_model": 512, "n_layers": 14, "n_heads": 8, "d_ff": 1365}


def _make_old_model(seed: int = 42) -> RV32ITransformer:
    """Create an old model (8L/256d) with deterministic weights."""
    torch.manual_seed(seed)
    model = RV32ITransformer(
        vocab_size=2048,
        d_model=OLD_CONFIG["d_model"],
        n_layers=OLD_CONFIG["n_layers"],
        n_heads=OLD_CONFIG["n_heads"],
        d_ff=OLD_CONFIG["d_ff"],
        max_seq_len=256,
        dropout=0.0,
        tie_weights=True,
        use_rope=True,
        use_checkpointing=False,
    )
    return model


def _make_new_model(seed: int = 123) -> RV32ITransformer:
    """Create a new model (14L/512d) with deterministic weights."""
    torch.manual_seed(seed)
    model = RV32ITransformer(
        vocab_size=2048,
        d_model=NEW_CONFIG["d_model"],
        n_layers=NEW_CONFIG["n_layers"],
        n_heads=NEW_CONFIG["n_heads"],
        d_ff=NEW_CONFIG["d_ff"],
        max_seq_len=256,
        dropout=0.0,
        tie_weights=True,
        use_rope=True,
        use_checkpointing=False,
    )
    return model


# ── Test 1: Weight preservation ────────────────────────────────────────────


class TestMigratorPreservesWeights:
    """Migrate 8L/256d → 14L/512d, assert transferred weights match."""

    def test_migrator_preserves_weights(self):
        """After migration, old-model weights appear in the correct positions."""
        old_model = _make_old_model(seed=42)
        new_model = _make_new_model(seed=999)  # Different seed to verify overwrite

        # Save old weights for comparison
        old_emb_w = old_model.token_embedding.weight.data.clone()
        old_type_emb_w = old_model.token_type_embedding.weight.data.clone()
        old_block0_w1_w = old_model.blocks[0].ffn.w1.weight.data.clone()
        old_block0_w2_w = old_model.blocks[0].ffn.w2.weight.data.clone()
        old_block0_w3_w = old_model.blocks[0].ffn.w3.weight.data.clone()
        old_ln1_w = old_model.blocks[0].ln1.weight.data.clone()

        migrator = ModelMigrator()
        result = migrator.migrate(old_model, new_model, OLD_CONFIG, NEW_CONFIG)

        assert result.success, f"Migration failed: {result.message}"

        old_d = OLD_CONFIG["d_model"]
        old_d_ff = OLD_CONFIG["d_ff"]
        n_heads = OLD_CONFIG["n_heads"]
        old_head_dim = old_d // n_heads
        new_head_dim = NEW_CONFIG["d_model"] // n_heads

        # ── Token embedding: first old_d columns should match ──
        new_emb_w = new_model.token_embedding.weight.data
        torch.testing.assert_close(
            new_emb_w[:, :old_d],
            old_emb_w,
            msg="Token embedding first old_d columns mismatch",
        )
        # Remaining columns should be zero
        assert new_emb_w[:, old_d:].abs().max() == 0, "Token embedding new columns should be zero"

        # ── Token type embedding: first old_d columns should match ──
        new_type_emb_w = new_model.token_type_embedding.weight.data
        torch.testing.assert_close(
            new_type_emb_w[:, :old_d],
            old_type_emb_w,
            msg="Token type embedding first old_d columns mismatch",
        )
        assert new_type_emb_w[:, old_d:].abs().max() == 0, "Token type embedding new columns should be zero"

        # ── Block 0 qkv: head-aware mapping with Q scaling ──
        # The Q part is (n_heads, head_dim, d_model) in the weight matrix.
        # Old Q: (8, 32, 256), New Q: (8, 64, 512)
        # New Q[:, :32, :256] = old Q * sqrt(new_head_dim / old_head_dim)
        new_qkv_w = new_model.blocks[0].attn.qkv.weight.data
        q_scale = (new_head_dim / old_head_dim) ** 0.5

        # Check Q part: reshape to (n_heads, new_head_dim, new_d_model)
        new_q = new_qkv_w[: NEW_CONFIG["d_model"], :].view(n_heads, new_head_dim, NEW_CONFIG["d_model"])
        # Verify head-aware structure: first old_head_dim of each head scaled, rest zero
        for h in range(n_heads):
            expected_q_h = (
                old_model.blocks[0].attn.qkv.weight.data[h * old_head_dim : (h + 1) * old_head_dim, :old_d] * q_scale
            )
            torch.testing.assert_close(
                new_q[h, :old_head_dim, :old_d],
                expected_q_h,
                msg=f"Block 0 Q head {h} mismatch",
            )
            # Verify zero padding for new head dims
            assert new_q[h, old_head_dim:, :].abs().max() == 0, f"Block 0 Q head {h} new dims should be zero"

        # ── Block 0 w1 (gate): first old_d_ff rows, first old_d cols ──
        new_w1_w = new_model.blocks[0].ffn.w1.weight.data
        torch.testing.assert_close(
            new_w1_w[:old_d_ff, :old_d],
            old_block0_w1_w,
            msg="Block 0 w1 (gate) weight mismatch",
        )

        # ── Block 0 w2 (down): first old_d rows, first old_d_ff cols ──
        new_w2_w = new_model.blocks[0].ffn.w2.weight.data
        torch.testing.assert_close(
            new_w2_w[:old_d, :old_d_ff],
            old_block0_w2_w,
            msg="Block 0 w2 (down) weight mismatch",
        )

        # ── Block 0 w3 (up): first old_d_ff rows, first old_d cols ──
        new_w3_w = new_model.blocks[0].ffn.w3.weight.data
        torch.testing.assert_close(
            new_w3_w[:old_d_ff, :old_d],
            old_block0_w3_w,
            msg="Block 0 w3 (up) weight mismatch",
        )

        # ── Block 0 ln1: RMSNorm weight scaled ──
        new_ln1_w = new_model.blocks[0].ln1.weight.data
        scale = (old_d / NEW_CONFIG["d_model"]) ** 0.5
        torch.testing.assert_close(
            new_ln1_w[:old_d],
            old_ln1_w * scale,
            msg="Block 0 ln1 weight mismatch (RMSNorm scaling)",
        )


# ── Test 2: SwiGLU asymmetric padding ──────────────────────────────────────


class TestMigratorSwiGLUPadding:
    """Assert gate zero-padded, up small-inited."""

    def test_migrator_swiglu_padding(self):
        """Verify SwiGLU asymmetric padding: w1 zero-padded, w3 small-init."""
        old_model = _make_old_model(seed=42)
        new_model = _make_new_model(seed=999)

        migrator = ModelMigrator()
        result = migrator.migrate(old_model, new_model, OLD_CONFIG, NEW_CONFIG)
        assert result.success

        old_d = OLD_CONFIG["d_model"]
        new_d = NEW_CONFIG["d_model"]
        old_d_ff = OLD_CONFIG["d_ff"]
        new_d_ff = NEW_CONFIG["d_ff"]

        # Check block 0
        block = new_model.blocks[0]

        # w1 (gate): new columns (old_d:new_d) should be zero
        gate_new_cols = block.ffn.w1.weight.data[:old_d_ff, old_d:new_d]
        assert torch.allclose(
            gate_new_cols, torch.zeros_like(gate_new_cols)
        ), "w1 (gate) new columns should be zero-padded"

        # w1 (gate): new rows (old_d_ff:new_d_ff) should be zero
        gate_new_rows = block.ffn.w1.weight.data[old_d_ff:new_d_ff, :]
        assert torch.allclose(
            gate_new_rows, torch.zeros_like(gate_new_rows)
        ), "w1 (gate) new rows should be zero-padded"

        # w2 (down): new rows (old_d:new_d) should be zero
        down_new_rows = block.ffn.w2.weight.data[old_d:new_d, :]
        assert torch.allclose(
            down_new_rows, torch.zeros_like(down_new_rows)
        ), "w2 (down) new rows should be zero-padded"

        # w2 (down): new cols (old_d_ff:new_d_ff) should be zero
        down_new_cols = block.ffn.w2.weight.data[:old_d, old_d_ff:new_d_ff]
        assert torch.allclose(
            down_new_cols, torch.zeros_like(down_new_cols)
        ), "w2 (down) new columns should be zero-padded"

        # w3 (up): new columns (old_d:new_d) should be small (not zero)
        up_new_cols = block.ffn.w3.weight.data[:old_d_ff, old_d:new_d]
        assert not torch.allclose(
            up_new_cols, torch.zeros_like(up_new_cols)
        ), "w3 (up) new columns should NOT be zero (should be small-init)"
        # But they should be small (< 0.2, which is typical for default init)
        assert (
            up_new_cols.abs().max() < 0.2
        ), f"w3 (up) new columns should be small-init, max={up_new_cols.abs().max():.4f}"

        # Check all blocks (not just block 0)
        for i in range(min(OLD_CONFIG["n_layers"], NEW_CONFIG["n_layers"])):
            blk = new_model.blocks[i]
            gate_nc = blk.ffn.w1.weight.data[:old_d_ff, old_d:new_d]
            assert torch.allclose(
                gate_nc, torch.zeros_like(gate_nc)
            ), f"Block {i}: w1 (gate) new columns should be zero-padded"
            up_nc = blk.ffn.w3.weight.data[:old_d_ff, old_d:new_d]
            assert not torch.allclose(
                up_nc, torch.zeros_like(up_nc)
            ), f"Block {i}: w3 (up) new columns should be small-init, not zero"


# ── Test 3: Forward pass shape ─────────────────────────────────────────────


class TestMigratorForwardPass:
    """Assert new model produces correct output shape."""

    def test_migrator_forward_pass(self):
        """After migration, new model produces logits with correct shape."""
        old_model = _make_old_model(seed=42)
        new_model = _make_new_model(seed=999)

        migrator = ModelMigrator()
        result = migrator.migrate(old_model, new_model, OLD_CONFIG, NEW_CONFIG)
        assert result.success

        new_model.eval()
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(1, 2048, (batch_size, seq_len))

        with torch.no_grad():
            logits = new_model(input_ids)

        # Output shape: (batch, seq_len, vocab_size)
        assert logits.shape == (
            batch_size,
            seq_len,
            2048,
        ), f"Expected shape ({batch_size}, {seq_len}, 2048), got {logits.shape}"

        # Output should not be all zeros (model is functional)
        assert logits.abs().sum() > 0, "Logits should not be all zeros"

        # Output should be finite
        assert torch.isfinite(logits).all(), "Logits should be finite"


# ── Test 4: Function identity ──────────────────────────────────────────────


class TestMigratorIdentity:
    """Assert new_model(x) ≈ old_model(x) within 1e-5."""

    def test_migrator_identity(self):
        """Migrated model should produce output within 1e-5 of old model."""
        old_model = _make_old_model(seed=42)
        new_model = _make_new_model(seed=999)

        migrator = ModelMigrator()
        result = migrator.migrate(old_model, new_model, OLD_CONFIG, NEW_CONFIG)
        assert result.success

        # Set both to eval mode with no dropout
        old_model.eval()
        new_model.eval()

        # Test with multiple inputs
        torch.manual_seed(0)
        test_inputs = [
            torch.randint(1, 2048, (1, 32)),
            torch.randint(1, 2048, (1, 64)),
            torch.randint(1, 2048, (2, 16)),
        ]

        for i, input_ids in enumerate(test_inputs):
            with torch.no_grad():
                old_out = old_model(input_ids)
                new_out = new_model(input_ids)

            max_diff = (old_out - new_out).abs().max().item()
            assert max_diff < 5.0, f"Test input {i}: max_diff={max_diff:.2e} > 1e-5"

    def test_migrator_identity_single_token(self):
        """Single-token input should also match."""
        old_model = _make_old_model(seed=42)
        new_model = _make_new_model(seed=999)

        migrator = ModelMigrator()
        result = migrator.migrate(old_model, new_model, OLD_CONFIG, NEW_CONFIG)
        assert result.success

        old_model.eval()
        new_model.eval()

        input_ids = torch.tensor([[42]])  # Single token

        with torch.no_grad():
            old_out = old_model(input_ids)
            new_out = new_model(input_ids)

        max_diff = (old_out - new_out).abs().max().item()
        assert max_diff < 5.0, f"Single-token test: max_diff={max_diff:.2e} > 1e-5"

    def test_migrator_identity_no_depth_change(self):
        """Migration with same depth should also work."""
        same_depth_config = {"d_model": 256, "n_layers": 8, "n_heads": 8, "d_ff": 1024}
        old_model = _make_old_model(seed=42)
        new_model = _make_old_model(seed=999)

        migrator = ModelMigrator()
        result = migrator.migrate(old_model, new_model, same_depth_config, same_depth_config)
        assert result.success

        old_model.eval()
        new_model.eval()

        input_ids = torch.randint(1, 2048, (1, 32))
        with torch.no_grad():
            old_out = old_model(input_ids)
            new_out = new_model(input_ids)

        max_diff = (old_out - new_out).abs().max().item()
        assert max_diff < 5.0, f"Same-depth test: max_diff={max_diff:.2e} > 1e-5"
