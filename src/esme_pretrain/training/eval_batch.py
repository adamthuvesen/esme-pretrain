"""Shared cross-entropy evaluation over materialized input/target batches."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

from esme_pretrain.modeling.backbone import language_model_loss
from esme_pretrain.torch import torch


@torch.no_grad()
def mean_ce_loss(
    model: torch.nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str,
    autocast: Any | None = None,
) -> float | None:
    """Mean CE over ``(input_ids, targets)`` pairs. Returns ``None`` when empty."""
    autocast_ctx = nullcontext() if autocast is None else autocast
    was_training = model.training
    model.eval()
    device = torch.device(device)
    weighted_total = 0.0
    total_targets = 0
    for input_ids, targets in batches:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        with autocast_ctx:
            logits = model(input_ids)
            loss, _ = language_model_loss(logits, targets, z_loss_weight=0.0)
        count = int(targets.numel())
        weighted_total += float(loss.detach().cpu()) * count
        total_targets += count
    if was_training:
        model.train()
    if total_targets == 0:
        return None
    return weighted_total / total_targets
