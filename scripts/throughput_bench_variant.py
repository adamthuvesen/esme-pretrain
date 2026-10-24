#!/usr/bin/env python3
"""Run one throughput-benchmark variant and print its result as one JSON line.

The benchmark runs this file against two source trees: the code before the
throughput work (``before``) and the current code (``after``). It therefore uses
only API that both trees share, plus ``ShardBatchLoader`` for the shard variant,
which exists only in ``after``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data", required=True, choices=["fixed", "stream", "shards"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--shard-manifest", type=Path)
    parser.add_argument("--warmup-steps", required=True, type=int)
    parser.add_argument("--measure-steps", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help="Tiny model on CPU: checks that a source tree runs this script, no GPU.",
    )
    args = parser.parse_args()
    device = "cpu" if args.rehearsal else "cuda"

    import torch

    import esme_pretrain
    from esme_pretrain.modeling.backbone import BackboneConfig
    from esme_pretrain.pretrain_run import load_pretrain_config
    from esme_pretrain.training.metrics import RunLogger, WandbSettings
    from esme_pretrain.training.pretrain import PretrainConfig, run_pretrain

    config = load_pretrain_config(args.config)
    model = BackboneConfig.from_dict(config.payload["model"])
    optimizer = config.payload["optimizer"]
    training = dict(optimizer["training"])
    if args.rehearsal:
        model = BackboneConfig(
            name="bench-rehearsal",
            vocab_size=model.vocab_size,
            context_length=64,
            embedding_dim=64,
            layers=2,
            heads=4,
            kv_heads=2,
            feedforward_dim=128,
        )
        training.update(micro_batch_size=2, grad_accum_steps=1, dtype="float32", compile=False)
    micro_batch = int(training["micro_batch_size"])
    total_steps = args.warmup_steps + args.measure_steps

    train_loader = _train_loader(args, config, model, micro_batch, device, torch)
    train_config = PretrainConfig(
        model=model,
        max_steps=total_steps,
        micro_batch_size=micro_batch,
        grad_accum_steps=int(training["grad_accum_steps"]),
        learning_rate=float(optimizer["learning_rate"]),
        min_lr_ratio=float(optimizer["min_lr_ratio"]),
        decay_fraction=float(optimizer["decay_fraction"]),
        warmup_steps=min(int(optimizer["warmup_steps"]), args.warmup_steps),
        weight_decay=float(optimizer["weight_decay"]),
        grad_clip=float(optimizer["grad_clip"]),
        dtype=training["dtype"],
        device=device,
        use_compile=bool(training["compile"]),
        use_fused_optimizer=bool(training["fused_optimizer"]),
        seed=int(training["seed"]),
        log_interval=1,
        output_dir=args.output_dir,
    )
    logger = RunLogger(args.output_dir, WandbSettings(enabled=False))
    started = time.perf_counter()
    result = run_pretrain(train_config, train_loader, logger=logger)
    logger.finish()

    records = [
        json.loads(line)
        for line in (args.output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    measured = [
        record
        for record in records
        if "tokens_per_second" in record and int(record["step"]) > args.warmup_steps
    ]
    if len(measured) != args.measure_steps:
        raise SystemExit(
            f"{args.variant}: expected {args.measure_steps} measured steps, got {len(measured)}"
        )
    tokens_per_second = [float(record["tokens_per_second"]) for record in measured]
    step_ms = [float(record["step_time_ms"]) for record in measured]
    flops_per_token = model.flops_per_token(model.context_length)
    payload: dict[str, Any] = {
        "variant": args.variant,
        "data": args.data,
        "steps_completed": result.steps_completed,
        "measured_steps": len(measured),
        "median_tokens_per_second": statistics.median(tokens_per_second),
        "p10_tokens_per_second": _quantile(tokens_per_second, 0.1),
        "p90_tokens_per_second": _quantile(tokens_per_second, 0.9),
        "median_step_time_ms": statistics.median(step_ms),
        "achieved_tflops": statistics.median(tokens_per_second) * flops_per_token / 1e12,
        "peak_memory_gb": result.peak_memory_gb,
        "train_loss_first": result.train_loss_first,
        "train_loss_last": result.train_loss_last,
        "compiled": result.compiled,
        "wall_seconds": time.perf_counter() - started,
        "esme_pretrain_path": esme_pretrain.__file__,
        "torch_version": torch.__version__,
        "gpu_name": None if args.rehearsal else torch.cuda.get_device_name(0),
    }
    print("BENCH_RESULT " + json.dumps(payload, sort_keys=True), flush=True)
    return 0


def _train_loader(
    args: argparse.Namespace, config: Any, model: Any, micro_batch: int, device: str, torch: Any
):
    if args.data == "fixed":
        from esme_pretrain.training.data_stream import Batch

        torch.manual_seed(0)
        window = torch.randint(
            0, model.vocab_size, (micro_batch, model.context_length + 1), device=device
        )
        fixed = Batch(input_ids=window[:, :-1], targets=window[:, 1:])

        class RepeatLoader:
            def __iter__(self):
                while True:
                    yield fixed

        return RepeatLoader()
    if args.data == "stream":
        from tokenizers import Tokenizer

        from esme_pretrain.data.corpus_stream import document_text_stream
        from esme_pretrain.training.data_stream import (
            StreamingBatchLoader,
            tokenized_document_stream,
        )

        tokenizer = Tokenizer.from_file(str(args.tokenizer))

        class Encoder:
            def encode(self, text: str) -> list[int]:
                return tokenizer.encode(text).ids

        return StreamingBatchLoader(
            tokenized_document_stream(
                document_text_stream(config, split="train"),
                Encoder(),
                eos_id=tokenizer.token_to_id("<eos>"),
            ),
            batch_size=micro_batch,
            context_length=model.context_length,
            device=device,
            pin_memory=True,
            prefetch_batches=4,
        )
    from esme_pretrain.training.shard_loader import ShardBatchLoader

    if args.shard_manifest is None:
        raise SystemExit("--shard-manifest is required for the shards variant")
    loader = ShardBatchLoader(
        args.shard_manifest,
        batch_size=micro_batch,
        context_length=model.context_length,
        device=device,
        shuffle_seed=0,
        pin_memory=True,
        prefetch_batches=4,
    )
    if loader.manifest.vocab_size != model.vocab_size:
        raise SystemExit(
            f"shard vocab {loader.manifest.vocab_size} does not match model {model.vocab_size}"
        )
    return loader


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


if __name__ == "__main__":
    raise SystemExit(main())
