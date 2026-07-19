"""Optimizer, scheduler, SWA, and dropout utilities.

Extracted from the monolithic train_model.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def adjust_dropout(
    model: nn.Module,
    epoch: int,
    total_epochs: int,
    start_drop: float = 0.2,
    end_drop: float = 0.05,
) -> float:
    """Anneal dropout from *start_drop* to *end_drop* over training.

    Returns:
        The current dropout probability applied.
    """
    if epoch >= total_epochs:
        drop = end_drop
    else:
        drop = start_drop + (end_drop - start_drop) * (epoch / total_epochs)
    for mod in model.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = drop
    return drop


def build_optimizer_and_schedulers(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    actual_epochs: int,
    n_train_batches: int,
    *,
    swa_start: int | None = None,
    swa_lr: float | None = None,
) -> dict:
    """Create AdamW optimizer with linear warmup + cosine annealing,
    and optionally SWA components.

    Returns:
        dict with keys: optimizer, scheduler, use_swa, swa_model, swa_scheduler
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)

    warmup_steps = max(1, int(0.05 * actual_epochs * n_train_batches))  # 5% warmup
    total_steps = actual_epochs * n_train_batches

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

    warmup_sched = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    restart_sched = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(1, total_steps // 3),
        T_mult=2,
        eta_min=lr * 0.1,
    )
    scheduler = SequentialLR(optimizer, [warmup_sched, restart_sched], milestones=[warmup_steps])

    # SWA (Stochastic Weight Averaging)
    use_swa = swa_start is not None and swa_start < actual_epochs
    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = torch.optim.swa_utils.AveragedModel(model)
        swa_lr_val = swa_lr or lr * 0.5
        swa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=actual_epochs - swa_start,
            eta_min=swa_lr_val,
        )

    return dict(
        optimizer=optimizer,
        scheduler=scheduler,
        use_swa=use_swa,
        swa_model=swa_model,
        swa_scheduler=swa_scheduler,
    )


def build_swa_components(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr: float,
    swa_start: int | None,
    swa_lr: float | None,
    actual_epochs: int,
) -> tuple[torch.optim.swa_utils.AveragedModel | None, torch.optim.lr_scheduler.LRScheduler | None]:
    """Create SWA (Stochastic Weight Averaging) components.

    Returns:
        (swa_model, swa_scheduler) -- both None if SWA is disabled.
    """
    use_swa = swa_start is not None and swa_start < actual_epochs
    if not use_swa:
        return None, None

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_lr_val = swa_lr or lr * 0.5
    swa_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=actual_epochs - swa_start,
        eta_min=swa_lr_val,
    )
    return swa_model, swa_scheduler
