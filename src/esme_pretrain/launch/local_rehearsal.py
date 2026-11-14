"""No-spend local rehearsals of the pretrain launch path: synthetic training and tokenizer smoke."""

from __future__ import annotations

import itertools
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

from esme_pretrain.launch.modal_artifacts import (
    required_artifacts_present,
    write_rehearsal_manifest,
)
from esme_pretrain.launch.modal_tokenizer import (
    load_or_train_tokenizer,
    train_tokenizer,
)
from esme_pretrain.launch.pretrain import (
    build_pretrain_dry_run,
)
from esme_pretrain.modeling.backbone import BackboneConfig
from esme_pretrain.pretrain_run import (
    PretrainLaunchConfig,
)
from esme_pretrain.training.checkpointing import load_pretrain_checkpoint
from esme_pretrain.training.data_stream import (
    StreamingBatchLoader,
    synthetic_token_stream,
)
from esme_pretrain.training.metrics import RunLogger, WandbSettings
from esme_pretrain.training.pretrain import PretrainConfig, run_pretrain

REPO_ROOT = Path(__file__).resolve().parents[3]


# A short repeating sequence, so a working loop visibly learns in 4 steps. Uniform random
# tokens give a loss drop no larger than batch noise, and a broken optimizer still passes.
REHEARSAL_PERIOD_TOKENS = 17


def _rehearsal_tokens(vocab_size: int, *, seed: int) -> Iterator[int]:
    period = synthetic_token_stream(vocab_size, seed=seed, total_tokens=REHEARSAL_PERIOD_TOKENS)
    return itertools.cycle(list(period))


def run_local_dress_rehearsal(config: PretrainLaunchConfig) -> dict[str, Any]:
    """No-spend rehearsal: validate launch config and write the required artifact shape.

    This deliberately uses synthetic token ids and a small CPU model so it never touches
    FineWeb-Edu, Modal, W&B, or paid hardware. The real full run path is
    ``run_pretrain_launch_body``.
    """
    output_dir = _local_evidence_dir(config, "local-dress-rehearsal")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    rehearsal_model = BackboneConfig(
        name="pretrain-dress-rehearsal",
        vocab_size=256,
        context_length=16,
        embedding_dim=64,
        layers=2,
        heads=4,
        feedforward_dim=128,
        z_loss_weight=0.0,
    )
    train_loader = StreamingBatchLoader(
        _rehearsal_tokens(rehearsal_model.vocab_size, seed=0),
        batch_size=4,
        context_length=rehearsal_model.context_length,
        device="cpu",
    )
    eval_loader = StreamingBatchLoader(
        _rehearsal_tokens(rehearsal_model.vocab_size, seed=1),
        batch_size=4,
        context_length=rehearsal_model.context_length,
        device="cpu",
    )
    train_config = PretrainConfig(
        model=rehearsal_model,
        max_steps=4,
        micro_batch_size=4,
        grad_accum_steps=2,
        learning_rate=1e-3,
        warmup_steps=1,
        dtype="float32",
        device="cpu",
        use_compile=False,
        use_fused_optimizer=False,
        log_interval=1,
        eval_interval=2,
        eval_batches=2,
        checkpoint_interval=2,
        output_dir=output_dir,
    )
    logger = RunLogger(output_dir, WandbSettings(enabled=False))
    result = run_pretrain(train_config, train_loader, eval_loader=eval_loader, logger=logger)
    logger.finish()

    # Resume from the mid-run checkpoint under the same schedule, as a preempted run would.
    resumed_config = replace(
        train_config,
        resume_from=output_dir / "checkpoint-step2.pt",
        use_compile=False,
    )
    resume_logger = RunLogger(output_dir, WandbSettings(enabled=False))
    resume_result = run_pretrain(
        resumed_config,
        StreamingBatchLoader(
            _rehearsal_tokens(rehearsal_model.vocab_size, seed=0),
            batch_size=4,
            context_length=rehearsal_model.context_length,
            device="cpu",
        ),
        eval_loader=StreamingBatchLoader(
            _rehearsal_tokens(rehearsal_model.vocab_size, seed=1),
            batch_size=4,
            context_length=rehearsal_model.context_length,
            device="cpu",
        ),
        logger=resume_logger,
    )
    resume_logger.finish()

    checkpoint = load_pretrain_checkpoint(output_dir / "checkpoint.pt")
    write_rehearsal_manifest(
        config,
        output_dir,
        result,
        resume_result,
        checkpoint.data_offset_tokens,
        repo_root=REPO_ROOT,
    )
    return {
        "status": "local_dress_rehearsal_complete",
        "paid_compute": False,
        "output_dir": str(output_dir),
        "config": build_pretrain_dry_run(config),
        "rehearsal": {
            "first_steps": result.steps_completed,
            "resume_start_step": resume_result.start_step,
            "resume_steps": resume_result.steps_completed,
            "checkpoint_step": checkpoint.step,
            "data_offset_tokens": checkpoint.data_offset_tokens,
            "required_artifacts_present": required_artifacts_present(output_dir),
        },
    }


def run_local_tokenizer_smoke(config: PretrainLaunchConfig) -> dict[str, Any]:
    """No-spend smoke for the production byte-level BPE train/load path."""
    output_dir = _local_evidence_dir(config, "local-tokenizer-smoke")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    texts = iter(
        [
            "The quick brown fox jumps over the lazy dog.",
            "LLM pretraining needs boring, durable evidence.",
            "Byte-level BPE should round-trip punctuation, accents, and numbers: café 123.",
        ]
        * 32
    )
    tokenizer, trained_report = train_tokenizer(
        config, output_dir, texts, require_target_vocab=False
    )
    loaded, loaded_report = load_or_train_tokenizer(config, output_dir, require_target_vocab=False)
    return {
        "status": "local_tokenizer_smoke_complete",
        "paid_compute": False,
        "output_dir": str(output_dir),
        "trained_vocab_size": tokenizer.get_vocab_size(),
        "loaded_vocab_size": loaded.get_vocab_size(),
        "trained_source": trained_report["source"],
        "loaded_source": loaded_report["source"],
        "tokenizer_json_exists": (output_dir / "tokenizer.json").exists(),
        "tokenizer_report_exists": (output_dir / "tokenizer-report.json").exists(),
    }


def _local_evidence_dir(config: PretrainLaunchConfig, name: str) -> Path:
    return REPO_ROOT / Path(config.output_dir).parent / name
