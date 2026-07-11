"""Runtime plumbing for the training loop: seed, device, precision, and WSD schedule."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

from esme_pretrain.torch import torch
from esme_pretrain.training.errors import TrainerError

# bf16 is the training dtype on H100/A100; float32 is the CPU/test fallback. fp16 is
# rejected in resolve_dtype (it needs a GradScaler this loop does not wire).
_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def set_seed(seed: int) -> None:
    """Seed torch's global RNG (checkpoint resume restores the full RNG snapshot)."""
    torch.manual_seed(seed)


def resolve_device(device: str) -> torch.device:
    """Resolve the requested device, failing loudly if CUDA is asked for but absent."""
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainerError("cuda device requested but torch.cuda.is_available() is False")
        # Our matmuls run bf16 under autocast; this covers any residual fp32 matmuls.
        torch.set_float32_matmul_precision("high")
    return resolved


def resolve_dtype(dtype: str) -> torch.dtype:
    """Map the config dtype name to a torch dtype, rejecting unsupported precisions."""
    if dtype == "float16":
        # fp16 needs loss scaling to avoid gradient underflow; this loop is bf16-first
        # (H100/A100) and does not wire a GradScaler. Fail loudly rather than risk NaNs.
        raise TrainerError("float16 is unsupported (no GradScaler); use 'bfloat16' or 'float32'")
    resolved = _DTYPES.get(dtype)
    if resolved is None:
        raise TrainerError(f"unsupported dtype: {dtype}")
    return resolved


def autocast_context(device: torch.device, dtype: torch.dtype) -> Any:
    """bf16 autocast on CUDA; a no-op context otherwise (CPU/tests run fp32)."""
    if device.type == "cuda" and dtype is torch.bfloat16:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def wsd_lr(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
    decay_fraction: float,
) -> float:
    """Warmup -> stable at ``max_lr`` -> cosine-shaped decay to ``min_lr``.

    The decay runs over the final ``decay_fraction`` of ``max_steps`` (WSD).
    """
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_steps = max(1, round(decay_fraction * max_steps))
    decay_start = max_steps - decay_steps
    if step < decay_start:
        return max_lr
    decay_progress = (step - decay_start) / decay_steps
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + coefficient * (max_lr - min_lr)
