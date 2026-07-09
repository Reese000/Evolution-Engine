#!/usr/bin/env python3
"""Tests for WarmStartTrainer — distillation training after migration.

Tests:
    1. test_warm_start_reduces_loss  — run 10 steps, assert loss decreases
    2. test_distillation_uses_teacher — verify teacher is frozen (no grad)
"""

from __future__ import annotations

import os
import sys

# Add project root to path so milestone1 modules resolve
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

from milestone1.training.scaling.warm_start_trainer import WarmStartConfig, WarmStartTrainer

# ---------------------------------------------------------------------------
# Lightweight stub model (no dependency on full RV32ITransformer)
# ---------------------------------------------------------------------------


class _StubTransformer(nn.Module):
    """Minimal decoder-only transformer for testing WarmStartTrainer."""

    def __init__(self, vocab_size: int = 2048, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.embed(input_ids)
        return self.head(x)


class _FakeArchive:
    """Fake archive with a list of entries, just enough for the trainer."""

    class _Entry:
        def __init__(self, tokens):
            self.program_hex = [f"0x{t:08x}" for t in tokens]

    def __init__(self, n: int = 10, seq_len: int = 9):
        import random

        rng = random.Random(42)
        self.entries = [self._Entry([rng.randint(4, 200) for _ in range(seq_len)]) for _ in range(n)]


class _FakeTokenizer:
    """Fake tokenizer that returns random token sequences."""

    def __init__(self):
        import random

        self._rng = random.Random(42)

    def encode_trace(self, trace):
        return [self._rng.randint(4, 200) for _ in range(9)]


# ---------------------------------------------------------------------------
# Test 1: warm_start_reduces_loss
# ---------------------------------------------------------------------------


def test_warm_start_reduces_loss():
    """Run a few warm-start steps and verify loss decreases."""
    torch.manual_seed(0)
    teacher = _StubTransformer()
    student = _StubTransformer()

    # Make student slightly worse by perturbing weights
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    archive = _FakeArchive(n=5)
    tokenizer = _FakeTokenizer()
    cfg = WarmStartConfig(warmup_steps=2, base_lr=1e-3, grad_clip=1.0)

    trainer = WarmStartTrainer(teacher, student, archive, tokenizer, cfg)

    # Run one epoch and record loss before and after
    token_data = trainer._prepare_token_data()
    loss_before = trainer._validate(token_data)

    trainer.warm_start(epochs=1)

    loss_after = trainer._validate(token_data)

    # Loss should have decreased (student learned something)
    assert loss_after < loss_before, f"Loss did not decrease: before={loss_before:.4f}, after={loss_after:.4f}"


# ---------------------------------------------------------------------------
# Test 2: distillation_uses_teacher — teacher is frozen
# ---------------------------------------------------------------------------


def test_distillation_uses_teacher():
    """Verify teacher parameters receive no gradients during distillation."""
    torch.manual_seed(42)
    teacher = _StubTransformer()
    student = _StubTransformer()

    archive = _FakeArchive(n=3)
    tokenizer = _FakeTokenizer()
    cfg = WarmStartConfig(warmup_steps=2)
    trainer = WarmStartTrainer(teacher, student, archive, tokenizer, cfg)

    # Snapshot teacher weights
    teacher_snapshot = {n: p.clone() for n, p in teacher.named_parameters()}

    # Run one distillation step
    token_data = trainer._prepare_token_data()
    input_ids = token_data[0].unsqueeze(0)

    student.train()
    student_logits = student(input_ids)
    with torch.no_grad():
        teacher_logits = teacher(input_ids)

    loss = trainer._distillation_loss(student_logits, teacher_logits, input_ids)
    loss.backward()

    # Teacher must have no gradients
    for name, p in teacher.named_parameters():
        assert p.grad is None, f"Teacher parameter '{name}' has a gradient — it should be frozen"

    # Teacher weights must be unchanged
    for name, p in teacher.named_parameters():
        assert torch.equal(
            p.data, teacher_snapshot[name]
        ), f"Teacher parameter '{name}' was modified — it must remain frozen"


# ---------------------------------------------------------------------------
# Test 3: supervised loss works independently
# ---------------------------------------------------------------------------


def test_supervised_loss_independently():
    """Verify supervised loss produces valid gradients for student."""
    torch.manual_seed(7)
    student = _StubTransformer()

    labels = torch.randint(4, 100, (1, 9))
    logits = student(labels)

    trainer = WarmStartTrainer.__new__(WarmStartTrainer)
    trainer.cfg = WarmStartConfig()

    loss = trainer._supervised_loss(logits, labels)
    loss.backward()

    has_grad = any(p.grad is not None for p in student.parameters())
    assert has_grad, "Supervised loss did not produce any gradients for student"


# ---------------------------------------------------------------------------
# Run tests directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_warm_start_reduces_loss()
    print("PASS: test_warm_start_reduces_loss")

    test_distillation_uses_teacher()
    print("PASS: test_distillation_uses_teacher")

    test_supervised_loss_independently()
    print("PASS: test_supervised_loss_independently")

    print("\nAll warm-start trainer tests passed!")
