from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esme_pretrain.postrun.acceptance_report import (
    BaseAcceptanceReportConfig,
    build_base_acceptance_report,
)
from esme_pretrain.postrun.eval_checkpoints import EvalCheckpointConfig, run_eval_checkpoints
from esme_pretrain.postrun.export_bundle import ExportConfig, export_checkpoint


def add_postrun_parsers(subparsers: argparse._SubParsersAction) -> None:
    eval_checkpoints = subparsers.add_parser(
        "eval-checkpoints",
        help="Evaluate checkpoints on one fixed deterministic validation slice.",
    )
    eval_checkpoints.add_argument("--config", required=True, type=Path, help="Run config JSON.")
    eval_checkpoints.add_argument("--tokenizer", required=True, type=Path, help="tokenizer.json.")
    eval_checkpoints.add_argument(
        "--checkpoint",
        required=True,
        action="append",
        type=Path,
        help="Checkpoint to evaluate. May be repeated.",
    )
    eval_checkpoints.add_argument(
        "--eval-token-budget",
        required=True,
        type=int,
        help="Target fixed validation tokens to evaluate.",
    )
    eval_checkpoints.add_argument("--output", required=True, type=Path, help="Output eval JSON.")
    eval_checkpoints.add_argument("--device", default="cpu", help="Torch device. Defaults to cpu.")
    eval_checkpoints.add_argument(
        "--batch-size", default=4, type=int, help="Eval batch size. Defaults to 4."
    )
    eval_checkpoints.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Optional smoke/debug cap on eval batches.",
    )
    eval_checkpoints.add_argument(
        "--json", action="store_true", help="Emit machine-readable result."
    )
    eval_checkpoints.set_defaults(handler=handle_eval_checkpoints)

    acceptance = subparsers.add_parser(
        "base-acceptance-report",
        help="Write the post-pretrain base acceptance Markdown report.",
    )
    acceptance.add_argument("--run-dir", required=True, type=Path, help="Completed run directory.")
    acceptance.add_argument("--eval", required=True, type=Path, help="Fixed eval JSON.")
    acceptance.add_argument("--output", required=True, type=Path, help="Markdown report path.")
    acceptance.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    acceptance.set_defaults(handler=handle_base_acceptance_report)

    export = subparsers.add_parser(
        "export",
        help="Export a checkpoint bundle for llm-infer.",
    )
    export.add_argument("--checkpoint", required=True, type=Path, help="Selected checkpoint.")
    export.add_argument("--tokenizer", required=True, type=Path, help="tokenizer.json.")
    export.add_argument(
        "--format",
        required=True,
        choices=("llm-infer",),
        help="Export format. Only llm-infer is supported.",
    )
    export.add_argument("--output", required=True, type=Path, help="Output bundle directory.")
    export.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    export.set_defaults(handler=handle_export)


def handle_eval_checkpoints(args: argparse.Namespace) -> int:
    try:
        payload = run_eval_checkpoints(
            EvalCheckpointConfig(
                config_path=args.config,
                tokenizer_path=args.tokenizer,
                checkpoint_paths=tuple(args.checkpoint),
                eval_token_budget=args.eval_token_budget,
                output_path=args.output,
                device=args.device,
                batch_size=args.batch_size,
                max_eval_batches=args.max_eval_batches,
            )
        )
    except ValueError as error:
        print(f"eval-checkpoints failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("esme-pretrain eval-checkpoints")
        print(f"output: {args.output}")
        print(f"recommended_checkpoint: {payload['selection']['recommended_checkpoint']}")
        print(f"reason: {payload['selection']['reason']}")
    return 0


def handle_base_acceptance_report(args: argparse.Namespace) -> int:
    try:
        payload = build_base_acceptance_report(
            BaseAcceptanceReportConfig(
                run_dir=args.run_dir,
                eval_path=args.eval,
                output_path=args.output,
            )
        )
    except ValueError as error:
        print(f"base-acceptance-report failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("esme-pretrain base-acceptance-report")
        print(f"output: {args.output}")
        print(f"recommended_checkpoint: {payload['fixed_eval']['recommended_checkpoint']}")
    return 0


def handle_export(args: argparse.Namespace) -> int:
    try:
        payload = export_checkpoint(
            ExportConfig(
                checkpoint_path=args.checkpoint,
                tokenizer_path=args.tokenizer,
                output_dir=args.output,
            )
        )
    except ValueError as error:
        print(f"export failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("esme-pretrain export")
        print(f"output: {args.output}")
        print(f"weights: {args.output / 'weights.pt'}")
    return 0
