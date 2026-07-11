"""GPU pretraining loop for the DenseBackbone 214M model.

Uses bf16 autocast, optional ``torch.compile``, fused AdamW, gradient accumulation,
streaming data loading, periodic eval/checkpointing, and a WSD learning-rate schedule.
Metrics go to local JSONL/CSV with optional W&B mirroring via :class:`RunLogger`.

Plumbing lives in the training spine: seed/device/precision/WSD schedule in
``training/runtime.py``, checkpoint save/load in ``training/checkpointing.py``,
metrics in ``training/metrics.py``. This module holds the config, result, and loop.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from statistics import median
from typing import Any

from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone, language_model_loss
from esme_pretrain.torch import torch
from esme_pretrain.training.checkpointing import (
    capture_rng_state,
    load_pretrain_checkpoint,
    restore_rng_state,
    save_pretrain_checkpoint,
)
from esme_pretrain.training.data_stream import Batch
from esme_pretrain.training.device_profile import peak_tflops_for_device
from esme_pretrain.training.errors import TrainerError
from esme_pretrain.training.eval_batch import mean_ce_loss
from esme_pretrain.training.metrics import RunLogger
from esme_pretrain.training.runtime import (
    autocast_context,
    resolve_device,
    resolve_dtype,
    set_seed,
    wsd_lr,
)


@dataclass(frozen=True)
class PretrainConfig:
    model: BackboneConfig
    max_steps: int
    micro_batch_size: int
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1  # final LR = learning_rate * min_lr_ratio
    decay_fraction: float = 0.2  # WSD: final fraction of steps spent decaying
    warmup_steps: int = 0
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dtype: str = "bfloat16"
    device: str = "cuda"
    use_compile: bool = True
    use_fused_optimizer: bool = True
    seed: int = 0
    log_interval: int = 10
    eval_interval: int = 0  # 0 disables periodic eval
    eval_batches: int = 0
    checkpoint_interval: int = 0  # 0 -> only the final checkpoint
    output_dir: Path = Path("runs/pretrain")
    resume_from: Path | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise TrainerError("max_steps must be at least 1")
        if self.micro_batch_size < 1:
            raise TrainerError("micro_batch_size must be at least 1")
        if self.grad_accum_steps < 1:
            raise TrainerError("grad_accum_steps must be at least 1")
        if self.learning_rate <= 0:
            raise TrainerError("learning_rate must be positive")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise TrainerError("min_lr_ratio must be in [0, 1]")
        if not 0.0 <= self.decay_fraction < 1.0:
            raise TrainerError("decay_fraction must be in [0, 1)")
        if not 0 <= self.warmup_steps <= self.max_steps:
            raise TrainerError("warmup_steps must be in [0, max_steps]")
        if self.grad_clip <= 0:
            raise TrainerError("grad_clip must be positive")
        if self.log_interval < 1:
            raise TrainerError("log_interval must be at least 1")
        for name in ("eval_interval", "eval_batches", "checkpoint_interval"):
            if getattr(self, name) < 0:
                raise TrainerError(f"{name} must be non-negative (0 disables)")

    @property
    def tokens_per_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.model.context_length

    @property
    def stream_tokens_per_step(self) -> int:
        """Tokens pulled from the source per optimizer step (windows are context+1)."""
        return self.micro_batch_size * self.grad_accum_steps * (self.model.context_length + 1)


@dataclass
class PretrainResult:
    steps_completed: int
    start_step: int
    train_loss_first: float
    train_loss_last: float
    train_loss_min: float
    val_loss_first: float | None
    val_loss_last: float | None
    steady_tokens_per_second: float | None
    peak_tokens_per_second: float | None
    steady_step_time_ms: float | None
    mfu: float | None
    peak_memory_gb: float | None
    grad_norm_last: float | None
    final_checkpoint: str
    device: str
    dtype: str
    compiled: bool
    fused_optimizer: bool
    flops_per_token: float
    wandb_status: str
    wandb_run_id: str | None
    wandb_run_url: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Decay 2D matmul weights; skip decay on 1D params (norms) and the embedding.

    The tied token embedding doubles as the output projection. Following the GPT-3
    convention, it is placed in the NO-decay group: it is a lookup table whose rows
    are token representations, not a matmul weight, and decaying it (especially while
    tied to the LM head) pulls every token toward the origin. ``norms`` (1D) are
    already exempt. Everything else 2D (attention/MLP projections) decays.
    """
    base = getattr(model, "_orig_mod", model)  # unwrap torch.compile to find the embedding
    embedding_param = base.token_embedding.weight
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        # Identity check covers the tied LM head too (same tensor as the embedding).
        is_embedding = parameter is embedding_param
        (no_decay if (parameter.dim() < 2 or is_embedding) else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _accumulate_gradients(
    model: torch.nn.Module,
    train_iter: Any,
    config: PretrainConfig,
    autocast: Any,
) -> tuple[float, dict[str, float]] | None:
    micro_losses: list[float] = []
    components: dict[str, float] = {}
    for _ in range(config.grad_accum_steps):
        try:
            batch: Batch = next(train_iter)
        except StopIteration:
            return None
        # Loss is computed inside autocast on purpose: cross_entropy is on autocast's
        # fp32 list, so the 32k-wide softmax runs in fp32 even though the model ran in
        # bf16. backward() is outside autocast, the standard placement.
        with autocast:
            logits = model(batch.input_ids)
            loss, components = language_model_loss(
                logits,
                batch.targets,
                z_loss_weight=config.model.z_loss_weight,
            )
        (loss / config.grad_accum_steps).backward()
        micro_losses.append(components["total_loss"])
    return sum(micro_losses) / len(micro_losses), components


def _data_offset_tokens(
    config: PretrainConfig, *, start_step: int, completed_step: int, resume_data_offset: int
) -> int:
    completed_since_start = completed_step - start_step
    return resume_data_offset + completed_since_start * config.stream_tokens_per_step


def _close_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    *,
    batches: int,
    autocast: Any,
) -> float | None:
    """Mean cross-entropy over a bounded number of eval batches (pure CE, no z-loss)."""
    if batches < 1:
        return None
    iterator = iter(loader)
    try:
        pairs = ((batch.input_ids, batch.targets) for batch in islice(iterator, batches))
        return mean_ce_loss(
            model,
            pairs,
            device=next(model.parameters()).device,
            autocast=autocast,
        )
    finally:
        _close_iterator(iterator)


def run_pretrain(
    config: PretrainConfig,
    train_loader: Any,
    *,
    eval_loader: Any | None = None,
    logger: RunLogger | None = None,
) -> PretrainResult:
    """Train the backbone for ``config.max_steps`` optimizer steps.

    ``train_loader`` / ``eval_loader`` are iterables of device-resident
    :class:`Batch` (typically :class:`StreamingBatchLoader`). The train loader must
    be effectively endless. Returns a :class:`PretrainResult` summary.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    is_cuda = device.type == "cuda"
    dtype = resolve_dtype(config.dtype)
    autocast = autocast_context(device, dtype)
    notes: list[str] = []

    start_step = 0
    resume_optimizer_state: dict[str, Any] | None = None
    resume_data_offset = 0
    if config.resume_from is not None:
        loaded = load_pretrain_checkpoint(config.resume_from, map_location=device)
        if loaded.model.config != config.model:
            raise TrainerError("resume checkpoint config does not match requested pretrain config")
        model: torch.nn.Module = loaded.model.to(device)
        start_step = loaded.step
        resume_optimizer_state = loaded.optimizer_state
        resume_data_offset = loaded.data_offset_tokens
        # Restore RNG so any stochastic op (sampling, future dropout) continues the
        # same sequence instead of restarting from config.seed.
        restore_rng_state(loaded.rng_state)
        # Fast-forward the token stream past the tokens already consumed, so resume
        # continues toward the corpus tail rather than re-reading the head. The loader
        # owns the skip (C-level islice on its deterministic source); counting in tokens
        # keeps it correct even if this run uses a different batch size than the original.
        if resume_data_offset:
            if not hasattr(train_loader, "skip_tokens"):
                raise TrainerError(
                    "resume_from carries a data offset but train_loader has no "
                    "skip_tokens; pass a StreamingBatchLoader to resume the stream"
                )
            train_loader.skip_tokens = resume_data_offset
        notes.append(
            f"resumed from {config.resume_from} at step {start_step} "
            f"(stream offset {resume_data_offset} tokens)"
        )
    else:
        model = DenseBackbone(config.model).to(device)

    model.train()
    compiled = False
    if config.use_compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            compiled = True
        except Exception as error:  # noqa: BLE001 - compile is best-effort headroom
            notes.append(f"torch.compile unavailable, ran eager: {error}")

    fused = config.use_fused_optimizer and is_cuda
    max_lr = config.learning_rate
    min_lr = config.learning_rate * config.min_lr_ratio
    optimizer = torch.optim.AdamW(
        _param_groups(model, config.weight_decay),
        lr=max_lr,
        betas=(config.beta1, config.beta2),
        fused=fused,
    )
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)

    logger = logger or RunLogger(config.output_dir)
    flops_per_token = config.model.flops_per_token(config.model.context_length)
    device_name = torch.cuda.get_device_name(device) if is_cuda else "cpu"
    peak_tflops = peak_tflops_for_device(device_name) if is_cuda else None

    train_iter = iter(train_loader)
    tokens_per_second_samples: list[float] = []
    step_time_samples: list[float] = []
    train_losses: list[float] = []
    val_first: float | None = None
    val_last: float | None = None
    grad_norm_last: float | None = None
    final_step = start_step
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(start_step, config.max_steps):
        learning_rate = wsd_lr(
            step,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            max_lr=max_lr,
            min_lr=min_lr,
            decay_fraction=config.decay_fraction,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        if is_cuda:
            torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        accumulated = _accumulate_gradients(model, train_iter, config, autocast)
        if accumulated is None:
            optimizer.zero_grad(set_to_none=True)
            notes.append(
                f"data exhausted at step {step}; stopped after {final_step} completed steps"
            )
            break

        step_loss, components = accumulated
        if not math.isfinite(step_loss):
            raise RuntimeError(f"non-finite train loss at step {step}: {step_loss}")
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip, error_if_nonfinite=True
        )
        grad_norm_last = float(grad_norm)
        if not math.isfinite(grad_norm_last):
            raise RuntimeError(f"non-finite gradient norm at step {step}: {grad_norm_last}")
        optimizer.step()

        if is_cuda:
            torch.cuda.synchronize(device)
        step_time = time.perf_counter() - step_start

        train_losses.append(step_loss)
        tokens_per_second = config.tokens_per_step / step_time
        # Skip the first compiled step (graph capture) from steady-state stats.
        if not (compiled and step == start_step):
            tokens_per_second_samples.append(tokens_per_second)
            step_time_samples.append(step_time * 1000.0)
        mfu = (tokens_per_second * flops_per_token / 1e12) / peak_tflops if peak_tflops else None
        completed_step = step + 1
        final_step = completed_step

        is_last = completed_step == config.max_steps
        if completed_step % config.log_interval == 0 or is_last:
            metrics: dict[str, Any] = {
                "loss": step_loss,
                "ce_loss": components["ce_loss"],
                "lr": learning_rate,
                "grad_norm": grad_norm_last,
                "tokens": config.tokens_per_step,
                "tokens_per_second": tokens_per_second,
                "step_time_ms": step_time * 1000.0,
                "mfu": mfu,
            }
            if "z_loss" in components:
                metrics["z_loss"] = components["z_loss"]
            if is_cuda:
                metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            logger.log(completed_step, metrics)

        should_eval = (
            eval_loader is not None
            and config.eval_interval > 0
            and config.eval_batches > 0
            and (completed_step % config.eval_interval == 0 or is_last)
        )
        if should_eval:
            val = evaluate(
                model,
                eval_loader,
                batches=config.eval_batches,
                autocast=autocast,
            )
            if val is not None:
                val_first = val if val_first is None else val_first
                val_last = val
                logger.log(completed_step, {"val_loss": val})

        should_checkpoint = (
            config.checkpoint_interval > 0 and completed_step % config.checkpoint_interval == 0
        )
        if should_checkpoint:
            save_pretrain_checkpoint(
                config.output_dir / f"checkpoint-step{completed_step}.pt",
                model=model,
                config=config.model,
                step=completed_step,
                optimizer=optimizer,
                metrics={"loss": step_loss},
                # Total source tokens consumed after this step; resume skips them.
                data_offset_tokens=_data_offset_tokens(
                    config,
                    start_step=start_step,
                    completed_step=completed_step,
                    resume_data_offset=resume_data_offset,
                ),
                rng_state=capture_rng_state(),
            )

    _close_iterator(train_iter)

    final_checkpoint = config.output_dir / "checkpoint.pt"
    save_pretrain_checkpoint(
        final_checkpoint,
        model=model,
        config=config.model,
        step=final_step,
        optimizer=optimizer,
        metrics={"loss": train_losses[-1] if train_losses else float("nan")},
        # Total source tokens consumed at the final checkpoint; resume continues here.
        data_offset_tokens=_data_offset_tokens(
            config,
            start_step=start_step,
            completed_step=final_step,
            resume_data_offset=resume_data_offset,
        ),
        rng_state=capture_rng_state(),
    )

    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if is_cuda else None
    result = PretrainResult(
        steps_completed=final_step - start_step,
        start_step=start_step,
        train_loss_first=train_losses[0] if train_losses else float("nan"),
        train_loss_last=train_losses[-1] if train_losses else float("nan"),
        train_loss_min=min(train_losses) if train_losses else float("nan"),
        val_loss_first=val_first,
        val_loss_last=val_last,
        steady_tokens_per_second=median(tokens_per_second_samples)
        if tokens_per_second_samples
        else None,
        peak_tokens_per_second=max(tokens_per_second_samples)
        if tokens_per_second_samples
        else None,
        steady_step_time_ms=median(step_time_samples) if step_time_samples else None,
        mfu=(
            (median(tokens_per_second_samples) * flops_per_token / 1e12) / peak_tflops
            if tokens_per_second_samples and peak_tflops
            else None
        ),
        peak_memory_gb=peak_memory_gb,
        grad_norm_last=grad_norm_last,
        final_checkpoint=str(final_checkpoint),
        device=str(device),
        dtype=config.dtype,
        compiled=compiled,
        fused_optimizer=fused,
        flops_per_token=flops_per_token,
        wandb_status=logger.wandb_status,
        wandb_run_id=logger.wandb_run_id,
        wandb_run_url=logger.wandb_run_url,
        notes=notes + ([logger.wandb_note] if logger.wandb_note else []),
    )
    logger.summary(result.to_dict())
    return result
