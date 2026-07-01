#!/usr/bin/env python3
"""Run the dense-backbone throughput probe locally (CPU/MPS validation or CUDA).

Local CPU/MPS runs are correctness and shape checks; MFU on CPU is meaningless
and reported as null.

    uv run python scripts/throughput_probe.py --model 150M --device cpu \
        --micro-batch 2 --context 64 --warmup 1 --measured 2 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from esme_pretrain.modeling.backbone import PROBE_CONFIGS
from esme_pretrain.training.throughput import ProbeConfig, run_throughput_probe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dense backbone throughput + MFU probe.")
    parser.add_argument("--model", choices=sorted(PROBE_CONFIGS), default="150M")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--dtype", default="float32", help="bfloat16 | float16 | float32")
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--context", type=int, default=None, help="Override context length.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=2)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-fused", action="store_true", help="Disable fused AdamW.")
    parser.add_argument("--peak-tflops", type=float, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model = PROBE_CONFIGS[args.model]
    if args.context is not None:
        model = replace(model, context_length=args.context)
    config = ProbeConfig(
        model=model,
        micro_batch_size=args.micro_batch,
        grad_accum_steps=args.grad_accum,
        warmup_steps=args.warmup,
        measured_steps=args.measured,
        context_length=args.context,
        dtype=args.dtype,
        device=args.device,
        use_compile=args.compile,
        use_fused_optimizer=not args.no_fused,
        device_peak_tflops=args.peak_tflops,
    )
    result = run_throughput_probe(config)
    payload = result.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"model={result.model_name} params={result.parameter_total:,}")
        print(f"device={result.device_name} dtype={result.dtype}")
        print(f"tokens/sec={result.tokens_per_second:,.0f} step_time={result.step_time_ms:.1f}ms")
        print(f"achieved_tflops={result.achieved_tflops:.1f} mfu={result.mfu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
