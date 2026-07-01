"""Modal pretrain launch body: CLI, local rehearsals, and remote run logic."""

from __future__ import annotations

import argparse
import json
import shutil
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from esme_pretrain.data.corpus_stream import document_text_stream
from esme_pretrain.launch.modal_artifacts import (
    local_git_commit,
    local_git_dirty,
    required_artifacts_present,
    write_cost,
    write_data_report,
    write_environment,
    write_pretrain_report,
    write_rehearsal_manifest,
)
from esme_pretrain.launch.modal_tokenizer import (
    load_or_train_tokenizer,
    require_token_id,
    train_tokenizer,
)
from esme_pretrain.launch.pretrain import (
    LAUNCH_APPROVAL_FLAG,
    build_pretrain_dry_run,
)
from esme_pretrain.modeling.backbone import BackboneConfig
from esme_pretrain.modeling.pretrain_checkpoint import load_pretrain_checkpoint
from esme_pretrain.pretrain_run import (
    PretrainLaunchConfig,
    load_pretrain_config,
    validate_pretrain_payload,
)
from esme_pretrain.training.data_stream import (
    StreamingBatchLoader,
    synthetic_token_stream,
    tokenized_document_stream,
)
from esme_pretrain.training.metrics_logger import RunLogger, WandbSettings
from esme_pretrain.training.pretrain import PretrainConfig, run_pretrain

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_NAME = "esme-pretrain-214m-b200"
VOLUME_MOUNT = Path("/pretrain")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modal_pretrain.py",
        description="Validate and launch the approval-gated 214M B200 pretrain.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Pretrain config JSON path.")
    parser.add_argument(
        LAUNCH_APPROVAL_FLAG,
        action="store_true",
        help="Required after explicit approval of the exact command and spend cap.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; never starts Modal.")
    parser.add_argument(
        "--local-dress-rehearsal",
        action="store_true",
        help="Run a no-spend CPU rehearsal that writes the pretrain artifact manifest shape.",
    )
    parser.add_argument(
        "--local-tokenizer-smoke",
        action="store_true",
        help="Run a no-spend CPU smoke of the byte-level BPE train/load path.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    parser.add_argument(
        "--spawn",
        dest="spawn",
        action="store_true",
        default=True,
        help="Detach the long run: spawn and return the FunctionCall id (the default).",
    )
    parser.add_argument(
        "--no-spawn",
        dest="spawn",
        action="store_false",
        help="Block on the run and return its result inline (use for a short smoke).",
    )
    return parser


def launch(argv: list[str] | None = None, *, run_pretrain_launch: Any | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_pretrain_config(args.config)
    except ValueError as error:
        print(f"pretrain launch failed: {error}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(_format_payload(build_pretrain_dry_run(config), json_output=args.json))
        return 0

    if args.local_dress_rehearsal:
        payload = run_local_dress_rehearsal(config)
        print(_format_payload(payload, json_output=args.json))
        return 0

    if args.local_tokenizer_smoke:
        payload = run_local_tokenizer_smoke(config)
        print(_format_payload(payload, json_output=args.json))
        return 0

    if not args.approved:
        print(
            f"pretrain launch refused: pass {LAUNCH_APPROVAL_FLAG} after chat approval",
            file=sys.stderr,
        )
        return 2
    if run_pretrain_launch is None:
        print("pretrain launch failed: modal is not installed in this environment", file=sys.stderr)
        return 2
    dry_run = build_pretrain_dry_run(config)
    if dry_run["launch_blockers"]:
        print(
            "pretrain launch refused: " + "; ".join(dry_run["launch_blockers"]),
            file=sys.stderr,
        )
        return 2

    function_call = run_pretrain_launch.spawn(
        config.payload, local_git_commit(REPO_ROOT), local_git_dirty(REPO_ROOT)
    )
    if args.spawn:
        print(
            _format_spawn_payload(config, function_call.object_id),
            file=sys.stdout,
        )
        return 0
    result = function_call.get()
    _mirror_volume_report(config, result)
    print(_format_payload(result, json_output=args.json))
    return 0


def run_local_dress_rehearsal(config: PretrainLaunchConfig) -> dict[str, Any]:
    """No-spend rehearsal: validate launch config and write the required artifact shape.

    This deliberately uses synthetic token ids and a tiny CPU model so it never touches
    FineWeb-Edu, Modal, W&B, or paid hardware. The real full run path is
    ``run_pretrain_launch_body``.
    """
    output_dir = _local_evidence_dir(config, "local-dress-rehearsal")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    tiny_model = BackboneConfig(
        name="pretrain-dress-tiny",
        vocab_size=256,
        context_length=16,
        embedding_dim=64,
        layers=2,
        heads=4,
        feedforward_dim=128,
        z_loss_weight=0.0,
    )
    train_loader = StreamingBatchLoader(
        synthetic_token_stream(tiny_model.vocab_size, seed=0),
        batch_size=4,
        context_length=tiny_model.context_length,
        device="cpu",
    )
    eval_loader = StreamingBatchLoader(
        synthetic_token_stream(tiny_model.vocab_size, seed=1),
        batch_size=4,
        context_length=tiny_model.context_length,
        device="cpu",
    )
    train_config = PretrainConfig(
        model=tiny_model,
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
        sample_interval=2,
        sample_new_tokens=3,
        output_dir=output_dir,
    )
    logger = RunLogger(output_dir, WandbSettings(enabled=False))
    result = run_pretrain(train_config, train_loader, eval_loader=eval_loader, logger=logger)
    logger.finish()

    resumed_config = replace(
        train_config,
        max_steps=6,
        resume_from=output_dir / "checkpoint.pt",
        use_compile=False,
    )
    resume_logger = RunLogger(output_dir, WandbSettings(enabled=False))
    resume_result = run_pretrain(
        resumed_config,
        StreamingBatchLoader(
            synthetic_token_stream(tiny_model.vocab_size, seed=0),
            batch_size=4,
            context_length=tiny_model.context_length,
            device="cpu",
        ),
        eval_loader=StreamingBatchLoader(
            synthetic_token_stream(tiny_model.vocab_size, seed=1),
            batch_size=4,
            context_length=tiny_model.context_length,
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


def run_pretrain_launch_body(
    config_payload: dict[str, Any], *, commit: str, dirty: bool
) -> dict[str, Any]:
    config = validate_pretrain_payload(
        config_payload, Path(f"configs/{config_payload.get('run_id', 'pretrain')}.json")
    )
    started = time.perf_counter()
    output_dir = VOLUME_MOUNT / config.payload["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _arm_spend_alarm(config)
    (output_dir / "config.json").write_text(json.dumps(config.payload, indent=2), encoding="utf-8")

    tokenizer, tokenizer_report = load_or_train_tokenizer(config, output_dir)
    train_documents = document_text_stream(config, split="train")
    validation_documents = document_text_stream(config, split="validation")
    eos_id = require_token_id(tokenizer, "<eos>")

    class Encoder:
        def encode(self, text: str) -> list[int]:
            return tokenizer.encode(text).ids

    model = BackboneConfig.from_dict(config.payload["model"])
    training = config.payload["optimizer"]["training"]
    monitoring = config.payload["monitoring"]
    train_config = PretrainConfig(
        model=model,
        max_steps=config.train_steps,
        micro_batch_size=training["micro_batch_size"],
        grad_accum_steps=training["grad_accum_steps"],
        learning_rate=config.payload["optimizer"]["learning_rate"],
        min_lr_ratio=config.payload["optimizer"]["min_lr_ratio"],
        lr_schedule=config.payload["optimizer"]["lr_schedule"],
        decay_fraction=config.payload["optimizer"]["decay_fraction"],
        warmup_steps=config.payload["optimizer"]["warmup_steps"],
        weight_decay=config.payload["optimizer"]["weight_decay"],
        grad_clip=config.payload["optimizer"]["grad_clip"],
        dtype=training["dtype"],
        device="cuda",
        use_compile=training["compile"],
        use_fused_optimizer=training["fused_optimizer"],
        seed=training["seed"],
        log_interval=monitoring["log_interval"],
        eval_interval=monitoring["eval_interval"],
        eval_batches=monitoring["eval_batches"],
        checkpoint_interval=monitoring["checkpoint_interval"],
        sample_interval=monitoring["sample_interval"],
        sample_new_tokens=monitoring["sample_new_tokens"],
        output_dir=output_dir,
        resume_from=_resume_checkpoint(output_dir),
    )
    train_loader = StreamingBatchLoader(
        tokenized_document_stream(train_documents, Encoder(), eos_id=eos_id),
        batch_size=train_config.micro_batch_size,
        context_length=train_config.model.context_length,
        device="cuda",
        pin_memory=True,
        prefetch_batches=4,
    )
    eval_loader = StreamingBatchLoader(
        tokenized_document_stream(validation_documents, Encoder(), eos_id=eos_id),
        batch_size=train_config.micro_batch_size,
        context_length=train_config.model.context_length,
        device="cuda",
        pin_memory=True,
        prefetch_batches=2,
    )
    logger = RunLogger(
        output_dir,
        WandbSettings(
            enabled=True,
            project=monitoring["wandb_project"],
            run_name=config.payload["run_id"],
            run_id=_wandb_resume_run_id(output_dir) if train_config.resume_from else None,
            resume="allow" if train_config.resume_from else None,
            tags=[
                "pretrain",
                config.payload["model"]["name"],
                "deep-thin",
                "gqa",
                config.selected_gpu.lower(),
            ],
            config={
                "commit": commit,
                "dirty": dirty,
                **config.payload,
                "train_steps": config.train_steps,
                "tokens_per_step": config.tokens_per_step,
            },
        ),
    )
    result = run_pretrain(train_config, train_loader, eval_loader=eval_loader, logger=logger)
    logger.finish()

    elapsed = time.perf_counter() - started
    cost = {
        "elapsed_seconds": elapsed,
        "selected_gpu": config.selected_gpu,
        "usd_per_hour": config.selected_gpu_profile["usd_per_hour"],
        "estimated_cost_usd": elapsed * config.selected_gpu_profile["usd_per_hour"] / 3600.0,
        "runtime_spend_stop_usd": config.payload["runtime"]["runtime_spend_stop_usd"],
    }
    write_cost(output_dir, cost)
    write_environment(output_dir)
    write_data_report(config, output_dir, tokenizer_report)
    report = write_pretrain_report(config, output_dir, result.to_dict(), cost, commit, dirty)
    final_step = result.start_step + result.steps_completed
    status_payload = {
        "status": "pretrain_complete",
        "final_step": final_step,
        "expected_final_step": train_config.max_steps,
        "notes": result.notes,
    }
    if final_step < train_config.max_steps:
        status_payload["status"] = "pretrain_incomplete"
        (output_dir / "launch-status.json").write_text(
            json.dumps(status_payload, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"Pretrain incomplete: final step {final_step}/{train_config.max_steps}; "
            f"artifacts preserved in {output_dir}"
        )
    (output_dir / "launch-status.json").write_text(
        json.dumps(status_payload, indent=2), encoding="utf-8"
    )
    return {
        "status": "pretrain_complete",
        "output_dir": str(output_dir),
        "commit": commit,
        "dirty": dirty,
        "final_step": final_step,
        "expected_final_step": train_config.max_steps,
        "result": result.to_dict(),
        "cost": cost,
        "pretrain_report": report,
        "required_artifacts_present": required_artifacts_present(output_dir),
    }


def _arm_spend_alarm(config: PretrainLaunchConfig) -> None:
    if not hasattr(signal, "SIGALRM"):
        return
    seconds_to_spend_stop = int(
        config.payload["runtime"]["runtime_spend_stop_usd"]
        / config.selected_gpu_profile["usd_per_hour"]
        * 3600
    )
    # Leave time for the exception to unwind and the wrapper to commit the Volume.
    soft_stop_seconds = max(60, seconds_to_spend_stop - 10 * 60)

    def abort_for_spend_stop(_signum: int, _frame: Any) -> None:
        raise TimeoutError("Pretrain aborted before the configured runtime spend stop")

    signal.signal(signal.SIGALRM, abort_for_spend_stop)
    signal.alarm(soft_stop_seconds)


def _resume_checkpoint(output_dir: Path) -> Path | None:
    final_checkpoint = output_dir / "checkpoint.pt"
    if final_checkpoint.exists():
        return final_checkpoint
    checkpoints = sorted(
        output_dir.glob("checkpoint-step*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def _wandb_resume_run_id(output_dir: Path) -> str | None:
    run_id_file = output_dir / "wandb-run-id.txt"
    if run_id_file.exists():
        run_id = run_id_file.read_text(encoding="utf-8").strip()
        if run_id:
            return run_id

    latest_run = output_dir / "wandb" / "latest-run"
    if latest_run.exists() or latest_run.is_symlink():
        target = latest_run.resolve().name if latest_run.is_symlink() else latest_run.name
        run_id = _wandb_id_from_run_dir_name(target)
        if run_id:
            return run_id

    runs = sorted((output_dir / "wandb").glob("run-*"), key=lambda path: path.stat().st_mtime)
    for run in reversed(runs):
        run_id = _wandb_id_from_run_dir_name(run.name)
        if run_id:
            return run_id
    return None


def _wandb_id_from_run_dir_name(name: str) -> str | None:
    # W&B local run dirs are shaped like run-20260625_220548-x99drn15.
    if not name.startswith("run-"):
        return None
    parts = name.rsplit("-", maxsplit=1)
    return parts[-1] if len(parts) == 2 and parts[-1] else None


def _local_evidence_dir(config: PretrainLaunchConfig, name: str) -> Path:
    return REPO_ROOT / Path(config.output_dir).parent / name


def _mirror_volume_report(config: PretrainLaunchConfig, result: dict[str, Any]) -> None:
    local_dir = REPO_ROOT / config.output_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "modal-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = result.get("pretrain_report")
    if isinstance(report, str):
        (REPO_ROOT / "docs" / "scaleup-pretrain-report.md").write_text(report, encoding="utf-8")


def _format_spawn_payload(config: PretrainLaunchConfig, function_call_id: str) -> str:
    return "\n".join(
        (
            "pretrain_spawned",
            f"function_call_id: {function_call_id}",
            f"output_dir: {config.output_dir}",
            f"monitor: modal app logs {APP_NAME}",
            f"volume: modal volume ls {APP_NAME}",
        )
    )


def _format_payload(payload: dict[str, Any], json_output: bool) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    return "\n".join(
        (
            str(payload["status"]),
            "output_dir: "
            f"{payload.get('output_dir', payload.get('artifacts', {}).get('output_dir'))}",
            f"launch_command: {payload.get('launch_command', '<not launching>')}",
        )
    )
