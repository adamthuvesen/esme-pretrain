from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esme_pretrain.data.pipeline import DataPipelineConfig, build_data_report, prepare_data
from esme_pretrain.tokenization.lab import TokenizerLabConfig, parse_vocab_sizes, run_tokenizer_lab


def add_data_parsers(subparsers: argparse._SubParsersAction) -> None:
    data_report = subparsers.add_parser(
        "data-report",
        help="Inspect local raw text and token budget before packing.",
    )
    data_report.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Local text file or directory.",
    )
    data_report.add_argument(
        "--context-length",
        required=True,
        type=int,
        help="Packed row length.",
    )
    data_report.add_argument(
        "--token-budget", required=True, type=int, help="Maximum tokens to pack."
    )
    data_report.add_argument("--json", action="store_true", help="Emit machine-readable report.")
    data_report.set_defaults(handler=handle_data_report)

    prepare = subparsers.add_parser(
        "prepare-data",
        help="Create deterministic train/validation packed-token shards.",
    )
    prepare.add_argument("--input", required=True, type=Path, help="Local text file or directory.")
    prepare.add_argument("--output-dir", required=True, type=Path, help="Empty output directory.")
    prepare.add_argument("--context-length", required=True, type=int, help="Packed row length.")
    prepare.add_argument("--token-budget", required=True, type=int, help="Maximum tokens to pack.")
    prepare.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    prepare.set_defaults(handler=handle_prepare_data)

    tokenizer_lab = subparsers.add_parser(
        "tokenizer-lab",
        help="Compare character and learned tokenizers on compression and small-model loss.",
    )
    tokenizer_lab.add_argument(
        "--input", required=True, type=Path, help="Local text file or directory."
    )
    tokenizer_lab.add_argument(
        "--context-length", required=True, type=int, help="Packed row length."
    )
    tokenizer_lab.add_argument(
        "--token-budget", required=True, type=int, help="Maximum tokens to pack."
    )
    tokenizer_lab.add_argument(
        "--vocab-sizes",
        required=True,
        help="Comma-separated learned tokenizer vocab sizes, such as 48,64.",
    )
    tokenizer_lab.add_argument("--steps", required=True, type=int, help="Small CPU training steps.")
    tokenizer_lab.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    tokenizer_lab.set_defaults(handler=handle_tokenizer_lab)


def handle_data_report(args: argparse.Namespace) -> int:
    config = DataPipelineConfig(
        input_path=args.input,
        context_length=args.context_length,
        token_budget=args.token_budget,
    )
    try:
        report = build_data_report(config)
    except ValueError as error:
        print(f"data-report failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print("esme-pretrain data-report")
        print(f"input: {report.input_path}")
        print(f"files: {len(report.files)}")
        print(
            "tokens: "
            f"{report.budgeted_tokens}/{report.total_tokens} "
            f"(truncated {report.truncated_tokens})"
        )
        print(f"packed_rows: {report.packable_rows}")
        print(
            f"splits: train={report.splits.train_rows} validation={report.splits.validation_rows}"
        )
    return 0


def handle_prepare_data(args: argparse.Namespace) -> int:
    config = DataPipelineConfig(
        input_path=args.input,
        context_length=args.context_length,
        token_budget=args.token_budget,
    )
    try:
        result = prepare_data(config, args.output_dir)
    except ValueError as error:
        print(f"prepare-data failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print("esme-pretrain prepare-data")
        print(f"manifest: {result.manifest_path}")
        print(f"report: {result.report_path}")
        print(
            f"shards: train={len(result.train_shards)} validation={len(result.validation_shards)}"
        )
        print(
            "splits: "
            f"train={result.report.splits.train_rows} "
            f"validation={result.report.splits.validation_rows}"
        )
        print(
            "tokens: "
            f"{result.report.budgeted_tokens}/{result.report.total_tokens} "
            f"(truncated {result.report.truncated_tokens})"
        )
    return 0


def handle_tokenizer_lab(args: argparse.Namespace) -> int:
    try:
        vocab_sizes = parse_vocab_sizes(args.vocab_sizes)
        result = run_tokenizer_lab(
            TokenizerLabConfig(
                input_path=args.input,
                context_length=args.context_length,
                token_budget=args.token_budget,
                vocab_sizes=vocab_sizes,
                steps=args.steps,
            )
        )
    except ValueError as error:
        print(f"tokenizer-lab failed: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print("esme-pretrain tokenizer-lab")
        print(f"input: {result.input_path}")
        print(f"characters: {result.characters}")
        for comparison in result.comparisons:
            print(
                f"{comparison.tokenizer}: vocab={comparison.vocab_size} "
                f"tokens={comparison.tokens} "
                f"chars/token={comparison.chars_per_token:.3f} "
                f"compression={comparison.compression_ratio:.3f} "
                "loss="
                f"{comparison.train_loss_initial:.4f}->{comparison.train_loss_final:.4f}"
            )
    return 0
