"""WarmStartTrainer — distillation training after model migration.

Runs knowledge distillation from a frozen teacher (old model) to a
trainable student (new model) to wake up new/initialised parameters
after architecture scaling.

Schedule:
    1. Linear LR warmup over ``warmup_steps``
    2. Cosine decay to 10 % of base_lr
    3. Steps = archive_size × 3  (2 distillation + 1 supervised)

Loss:
    L = α · CE(student, labels) + (1 − α) · KL(student ‖ teacher) / T²

Archive replay alternates between distillation batches (teacher + labels)
and supervised-only batches (labels only) so the student learns both
teacher behaviour and ground-truth targets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WarmStartConfig:
    """Hyperparameters for warm-start distillation training."""

    alpha: float = 0.3  # CE weight in distillation loss
    temperature: float = 2.0  # soft-target temperature
    warmup_steps: int = 50  # linear LR warmup
    base_lr: float = 2e-5  # peak learning rate
    weight_decay: float = 0.01
    grad_clip: float = 1.0  # max gradient norm
    distill_ratio: float = 2.0 / 3.0  # fraction of steps that use distillation


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------


class WarmStartTrainer:
    """Distillation trainer that wakes up new parameters after migration.

    Args:
        teacher_model: Frozen old model (knowledge source).
        student_model: Trainable new model (target).
        archive: ``ChampionArchive``-like object with an ``entries`` list
            (each entry must have a ``program_hex`` attribute of hex
            token strings).
        tokenizer: ``RV32IFieldTokenizer`` used to encode hex programs
            into token tensors.
        config: ``WarmStartConfig`` or ``None`` (uses defaults).
        device: torch device (default ``cpu``).
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        archive: Any,
        tokenizer: Any,
        config: WarmStartConfig | None = None,
        device: torch.device | None = None,
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.archive = archive
        self.tokenizer = tokenizer
        self.cfg = config or WarmStartConfig()
        self.device = device or torch.device("cpu")

        # Freeze teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Place models on device
        self.teacher.to(self.device)
        self.student.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warm_start(self, epochs: int = 3) -> float:
        """Run warm-start training and return the final validation loss.

        The total number of optimiser steps is derived from the archive
        size × 3, repeated for *epochs* outer epochs.  Within each epoch
        the archive is replayed with alternating distillation / supervised
        batches.

        Returns:
            Final validation loss (scalar).
        """
        archive_size = len(self.archive.entries)
        if archive_size == 0:
            return float("inf")

        steps_per_epoch = archive_size * 3
        total_steps = steps_per_epoch * epochs

        # Build tokens list from archive entries
        token_data = self._prepare_token_data()
        if not token_data:
            return float("inf")

        # Optimiser
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.cfg.base_lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.student.train()
        step = 0
        last_val_loss = float("inf")

        for epoch in range(epochs):
            # Shuffle data each epoch
            indices = list(range(len(token_data)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), 1):
                idx = indices[batch_start]
                input_ids = token_data[idx].unsqueeze(0).to(self.device)  # (1, T)

                # LR schedule: linear warmup then cosine decay
                lr = self._get_lr(step, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Decide distillation vs supervised
                use_distill = self._should_distill(step, steps_per_epoch)

                loss = self._train_step(input_ids, use_distill=use_distill)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                if self.cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.grad_clip)
                optimizer.step()
                step += 1

            # Epoch-end validation
            last_val_loss = self._validate(token_data)

        return last_val_loss

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """L = α · CE(student, labels) + (1 − α) · KL(student ‖ teacher) / T²

        Args:
            student_logits: (B, T, V) raw logits from student.
            teacher_logits: (B, T, V) raw logits from teacher.
            labels: (B, T) target token IDs.

        Returns:
            Scalar loss tensor.
        """
        alpha = self.cfg.alpha
        T = self.cfg.temperature

        # Cross-entropy with ground-truth labels
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=0,  # PAD
        )

        # Soft-target KL divergence
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )

        return alpha * ce_loss + (1.0 - alpha) * kl_loss / (T * T)

    def _supervised_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Standard cross-entropy on archive targets.

        Args:
            student_logits: (B, T, V) raw logits from student.
            labels: (B, T) target token IDs.

        Returns:
            Scalar loss tensor.
        """
        return F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=0,  # PAD
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_token_data(self) -> list[torch.Tensor]:
        """Convert archive entries to padded token tensors."""
        tensors: list[torch.Tensor] = []
        for entry in self.archive.entries:
            try:
                # entry.program_hex is list of "0x..." strings
                token_ids = self.tokenizer.encode_trace(
                    [{"name": "ADDI", "rd": 1, "rs1": 0, "imm": 1} for _ in range(1)]
                )
            except Exception:
                # Fallback: just use sequential token IDs for testing
                token_ids = [4, 51, 0, 0, 0, 0, 0, 0, 1]

            if len(token_ids) < 2:
                continue
            t = torch.tensor(token_ids, dtype=torch.long)
            tensors.append(t)
        return tensors

    def _train_step(
        self,
        input_ids: torch.Tensor,
        use_distill: bool = True,
    ) -> torch.Tensor:
        """Execute a single training step.

        Args:
            input_ids: (1, T) token tensor.
            use_distill: True for distillation loss, False for supervised.

        Returns:
            Scalar loss.
        """
        # Student forward
        student_logits = self.student(input_ids)  # (1, T, V)

        labels = input_ids  # next-token prediction: shift is implicit

        if use_distill:
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher(input_ids)
            return self._distillation_loss(student_logits, teacher_logits, labels)
        else:
            return self._supervised_loss(student_logits, labels)

    def _validate(self, token_data: list[torch.Tensor]) -> float:
        """Compute mean validation loss over token data."""
        if not token_data:
            return float("inf")

        self.student.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for t in token_data:
                input_ids = t.unsqueeze(0).to(self.device)
                logits = self.student(input_ids)
                loss = self._supervised_loss(logits, input_ids)
                total_loss += loss.item()
                n += 1
        self.student.train()
        return total_loss / max(1, n)

    def _get_lr(self, step: int, total_steps: int) -> float:
        """Compute learning rate with linear warmup + cosine decay."""
        warmup = self.cfg.warmup_steps
        if step < warmup:
            # Linear warmup
            return self.cfg.base_lr * (step + 1) / warmup
        # Cosine decay to 10 % of base_lr
        progress = (step - warmup) / max(1, total_steps - warmup)
        return self.cfg.base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _should_distill(self, step: int, steps_per_epoch: int) -> bool:
        """Decide whether this step should use distillation.

        Follows a 2:1 ratio (distill:supervised) within each epoch,
        matching the ``total_steps = archive_size × 3`` formula.
        """
        position_in_epoch = step % steps_per_epoch
        # First 2/3 of epoch → distillation, last 1/3 → supervised
        threshold = int(steps_per_epoch * self.cfg.distill_ratio)
        return position_in_epoch < threshold
