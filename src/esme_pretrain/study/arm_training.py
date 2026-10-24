"""Train one scaling-study arm and score every endpoint on the fixed validation set.

An arm's main run trains first and saves checkpoints at its branches' fork steps;
each branch then continues from its fork under its own schedule. Every run's final
checkpoint is scored in fp32 on the same hash-verified validation batches, and its
result row is written (and committed, on Modal) before the next run starts.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from esme_pretrain.postrun.eval_checkpoints import file_sha256
from esme_pretrain.study.plan import MAIN_RUN, StudyArm, StudyPlan, StudyRun
from esme_pretrain.torch import torch
from esme_pretrain.training.checkpointing import load_pretrain_checkpoint
from esme_pretrain.training.data_stream import Batch
from esme_pretrain.training.eval_batch import hash_eval_batches, mean_ce_loss
from esme_pretrain.training.metrics import RunLogger, WandbSettings
from esme_pretrain.training.pretrain import PretrainConfig, run_pretrain
from esme_pretrain.training.shard_loader import ShardBatchLoader, fixed_validation_batches


def study_validation_batches(
    plan: StudyPlan,
    *,
    shard_dir: Path,
    tokenizer_sha256: str,
    vocab_size: int,
    context_length: int,
    device: str,
    eval_token_budget: int,
    expected_token_batch_sha256: str | None,
) -> list[Batch]:
    return fixed_validation_batches(
        shard_dir / Path(plan.data["validation_manifest"]).name,
        vocab_size=vocab_size,
        tokenizer_sha256=tokenizer_sha256,
        context_length=context_length,
        batch_size=int(plan.eval["batch_size"]),
        eval_token_budget=eval_token_budget,
        expected_token_batch_sha256=expected_token_batch_sha256,
        device=device,
    )


def validation_ce(checkpoint: Path, batches: list[Batch], *, device: str) -> float:
    """Score a checkpoint in true fp32, the path of the accepted eval and the held-out point.

    Training sets TF32 matmuls process-wide (``resolve_device``); left on, study points
    would be scored on a different numeric path than the 214M point they are compared to.
    """
    model = load_pretrain_checkpoint(checkpoint, map_location=device).model.to(device)
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        value = mean_ce_loss(model, ((b.input_ids, b.targets) for b in batches), device=device)
    finally:
        torch.set_float32_matmul_precision(previous_precision)
    if value is None:
        raise ValueError(f"no validation targets for {checkpoint}")
    return value


def train_study_arm(
    arm: StudyArm,
    plan: StudyPlan,
    *,
    shard_dir: Path,
    tokenizer_path: Path,
    work_dir: Path,
    results_dir: Path,
    validation_batches: list[Batch],
    device: str,
    dtype: str = "bfloat16",
    use_compile: bool = True,
    commit_results: Callable[[], None],
    provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    tokenizer_sha256 = file_sha256(tokenizer_path)
    arm_results = results_dir / arm.arm
    arm_results.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in arm.runs:
        output_dir = work_dir / arm.arm / run.run
        started = time.perf_counter()
        config = _pretrain_config(
            run,
            plan,
            arm=arm,
            work_dir=work_dir,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            use_compile=use_compile,
        )
        loader = ShardBatchLoader(
            shard_dir / Path(plan.data["train_manifest"]).name,
            vocab_size=run.model.vocab_size,
            tokenizer_sha256=tokenizer_sha256,
            batch_size=config.micro_batch_size,
            context_length=run.model.context_length,
            device=device,
            shuffle_seed=int(plan.data["shuffle_seed"]),
        )
        logger = RunLogger(output_dir, WandbSettings(enabled=False))
        result = run_pretrain(config, loader, logger=logger)
        logger.finish()
        if result.steps_completed != run.trained_steps:
            raise RuntimeError(
                f"{arm.arm}/{run.run}: trained {result.steps_completed} steps, "
                f"planned {run.trained_steps}; notes: {result.notes}"
            )
        row = {
            "study_id": plan.study_id,
            "arm": arm.arm,
            "stage": arm.stage,
            "run": run.run,
            "arm_runs": [arm_run.run for arm_run in arm.runs],
            "shape": run.model.name,
            "non_embedding_params": run.non_embedding_params,
            "total_params": run.model.parameter_count()["total"],
            "tokens": run.tokens,
            "max_steps": run.max_steps,
            "fork_step": run.fork_step,
            "learning_rate": run.learning_rate,
            "warmup_steps": run.warmup_steps,
            "seed": run.seed,
            "val_ce": validation_ce(
                output_dir / "checkpoint.pt", validation_batches, device=device
            ),
            "val_eval_tokens": sum(int(b.input_ids.numel()) for b in validation_batches),
            "val_token_batch_sha256": hash_eval_batches(validation_batches),
            "train_loss_last": result.train_loss_last,
            "steady_tokens_per_second": result.steady_tokens_per_second,
            "mfu": result.mfu,
            "peak_memory_gb": result.peak_memory_gb,
            "wall_seconds": time.perf_counter() - started,
            "notes": result.notes,
            "torch_version": torch.__version__,
            **(provenance or {}),
        }
        (arm_results / f"{run.run}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
        )
        shutil.copyfile(output_dir / "metrics.jsonl", arm_results / f"{run.run}-metrics.jsonl")
        commit_results()
        rows.append(row)
    return rows


def _pretrain_config(
    run: StudyRun,
    plan: StudyPlan,
    *,
    arm: StudyArm,
    work_dir: Path,
    output_dir: Path,
    device: str,
    dtype: str,
    use_compile: bool,
) -> PretrainConfig:
    optimizer = plan.optimizer
    branch_from = None
    if run.branch_of is not None:
        branch_from = work_dir / arm.arm / MAIN_RUN / f"checkpoint-step{run.fork_step}.pt"
    return PretrainConfig(
        model=run.model,
        max_steps=run.max_steps,
        micro_batch_size=int(optimizer["micro_batch_size"]),
        grad_accum_steps=int(optimizer["grad_accum_steps"]),
        learning_rate=run.learning_rate,
        min_lr_ratio=float(optimizer["min_lr_ratio"]),
        decay_fraction=float(optimizer["decay_fraction"]),
        warmup_steps=run.warmup_steps,
        weight_decay=float(optimizer["weight_decay"]),
        grad_clip=float(optimizer["grad_clip"]),
        dtype=dtype,
        device=device,
        use_compile=use_compile,
        use_fused_optimizer=device == "cuda",
        seed=run.seed,
        log_interval=10,
        checkpoint_steps=run.checkpoint_steps,
        output_dir=output_dir,
        branch_from=branch_from,
    )
