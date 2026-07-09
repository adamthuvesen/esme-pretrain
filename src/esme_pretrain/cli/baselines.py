from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esme_pretrain.baselines.config import load_baseline_eval_config
from esme_pretrain.baselines.report import build_comparison
from esme_pretrain.baselines.run import run_baseline_eval, run_gate


def add_baseline_parsers(subparsers: argparse._SubParsersAction) -> None:
    gate = subparsers.add_parser(
        "baseline-gate",
        help="Reproduce the gate model's published downstream numbers before any Esme eval.",
    )
    gate.add_argument("--config", required=True, type=Path, help="Baseline eval config JSON.")
    gate.add_argument("--output", required=True, type=Path, help="Output gate JSON.")
    gate.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    gate.set_defaults(handler=handle_baseline_gate)

    eval_parser = subparsers.add_parser(
        "baseline-eval",
        help="Score one configured model on bits-per-byte slices and downstream tasks.",
    )
    eval_parser.add_argument(
        "--config", required=True, type=Path, help="Baseline eval config JSON."
    )
    eval_parser.add_argument(
        "--model", required=True, help="Model key from the config models section."
    )
    eval_parser.add_argument("--output", required=True, type=Path, help="Output result JSON.")
    eval_parser.add_argument(
        "--gate",
        type=Path,
        default=None,
        help="Passing baseline-gate JSON. Required for bundle models.",
    )
    eval_parser.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    eval_parser.set_defaults(handler=handle_baseline_eval)

    compare = subparsers.add_parser(
        "baseline-compare",
        help="Render the cross-model Markdown comparison from result JSONs.",
    )
    compare.add_argument(
        "--result",
        required=True,
        action="append",
        type=Path,
        help="Per-model baseline-eval result JSON. May be repeated.",
    )
    compare.add_argument("--output", required=True, type=Path, help="Output Markdown path.")
    compare.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    compare.set_defaults(handler=handle_baseline_compare)


def handle_baseline_gate(args: argparse.Namespace) -> int:
    try:
        config = load_baseline_eval_config(args.config)
        payload = run_gate(config, output_path=args.output)
    except ValueError as error:
        print(f"baseline-gate failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("esme-pretrain baseline-gate")
        print(f"output: {args.output}")
        print(f"model: {payload['gate']['model']}")
        print(f"measured_average: {payload['gate']['average']['measured']:.4f}")
        print(f"published_average: {payload['gate']['average']['published']:.4f}")
        print(f"passed: {payload['passed']}")
    return 0 if payload["passed"] else 1


def handle_baseline_eval(args: argparse.Namespace) -> int:
    try:
        config = load_baseline_eval_config(args.config)
        payload = run_baseline_eval(
            config,
            model_key=args.model,
            output_path=args.output,
            gate_path=args.gate,
        )
    except ValueError as error:
        print(f"baseline-eval failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("esme-pretrain baseline-eval")
        print(f"output: {args.output}")
        print(f"model: {args.model}")
        for slice_name, entry in sorted(payload["bpb"].items()):
            print(f"bpb[{slice_name}]: {entry['bits_per_byte']:.4f}")
        print(f"downstream_average: {payload['downstream']['average']:.4f}")
    return 0


def handle_baseline_compare(args: argparse.Namespace) -> int:
    try:
        payload = build_comparison(list(args.result), args.output)
    except ValueError as error:
        print(f"baseline-compare failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("esme-pretrain baseline-compare")
        print(f"output: {args.output}")
        print(f"models: {', '.join(payload['models'])}")
    return 0
