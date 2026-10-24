from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from esme_pretrain.data.corpus_stream import document_text_stream
from esme_pretrain.data.token_shards import manifest_path_for, write_token_shards
from esme_pretrain.postrun.eval_checkpoints import file_sha256, load_tokenizer
from esme_pretrain.pretrain_run import load_pretrain_config


def add_pretokenize_parser(subparsers: argparse._SubParsersAction) -> None:
    pretokenize = subparsers.add_parser(
        "pretokenize",
        help="Tokenize one corpus split into uint16 token shards with a manifest.",
    )
    pretokenize.add_argument("--config", required=True, type=Path, help="Run config JSON.")
    pretokenize.add_argument("--tokenizer", required=True, type=Path, help="tokenizer.json.")
    pretokenize.add_argument(
        "--split", required=True, choices=("train", "validation"), help="Corpus split."
    )
    pretokenize.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for shards and manifest."
    )
    pretokenize.add_argument(
        "--shard-tokens",
        default=100_000_000,
        type=int,
        help="Maximum tokens per shard file. Defaults to 100000000.",
    )
    pretokenize.add_argument(
        "--max-tokens",
        default=None,
        type=int,
        help="Stop after exactly this many tokens; the last document may be truncated.",
    )
    pretokenize.add_argument(
        "--corpus-parquet",
        action="append",
        type=Path,
        default=[],
        help="Local corpus parquet file to read instead of streaming the Hub. May be repeated.",
    )
    pretokenize.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    pretokenize.set_defaults(handler=handle_pretokenize)


def handle_pretokenize(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    try:
        config = load_pretrain_config(args.config)
        tokenizer = load_tokenizer(args.tokenizer)
        vocab_size = int(config.payload["model"]["vocab_size"])
        if tokenizer.get_vocab_size() > vocab_size:
            raise ValueError(
                f"tokenizer vocabulary {tokenizer.get_vocab_size()} is larger than the "
                f"model vocab_size {vocab_size}"
            )
        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is None:
            raise ValueError("tokenizer is missing required token '<eos>'")
        dataset_cfg = config.payload["dataset"]
        parquet_files = sorted(args.corpus_parquet)
        missing = [str(path) for path in parquet_files if not path.exists()]
        if missing:
            raise ValueError(f"corpus parquet file does not exist: {', '.join(missing)}")
        provenance = {
            "config_path": str(args.config),
            "tokenizer_path": str(args.tokenizer),
            "dataset": {
                "source": dataset_cfg["source"],
                "subset": dataset_cfg["subset"],
                "revision": dataset_cfg["revision"],
                "split": dataset_cfg["split"],
                "filters": dataset_cfg["filters"],
            },
            "split_config": config.payload["split"],
            "corpus_parquet_files": [
                {"path": str(path), "bytes": path.stat().st_size} for path in parquet_files
            ],
        }
        documents = document_text_stream(config, split=args.split, parquet_files=parquet_files)
        # Close the stream while the interpreter is alive: an abandoned parquet stream
        # leaves pyarrow reader threads that hang interpreter exit on Python 3.14.
        with contextlib.closing(documents):
            manifest = write_token_shards(
                documents,
                tokenizer,
                eos_id=int(eos_id),
                vocab_size=vocab_size,
                tokenizer_sha256=file_sha256(args.tokenizer),
                split=args.split,
                output_dir=args.output_dir,
                shard_tokens=args.shard_tokens,
                max_tokens=args.max_tokens,
                provenance=provenance,
            )
    except (ValueError, OSError) as error:
        print(f"pretokenize failed: {error}", file=sys.stderr)
        return 2
    elapsed_seconds = time.perf_counter() - started
    payload = {
        "manifest_path": str(manifest_path_for(args.output_dir, args.split)),
        "manifest": manifest,
        "elapsed_seconds": elapsed_seconds,
        "tokens_per_second": manifest["total_tokens"] / elapsed_seconds,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("esme-pretrain pretokenize")
        print(f"manifest: {payload['manifest_path']}")
        print(f"documents: {manifest['documents']}")
        print(f"total_tokens: {manifest['total_tokens']}")
        print(f"shards: {len(manifest['shards'])}")
        print(f"elapsed_seconds: {elapsed_seconds:.1f}")
        print(f"tokens_per_second: {payload['tokens_per_second']:.0f}")
    return 0
