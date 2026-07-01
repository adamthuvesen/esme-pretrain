"""Tokenizer train/load helpers for Modal pretrain launches."""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from esme_pretrain.data.corpus_stream import document_text_stream
from esme_pretrain.pretrain_run import PretrainLaunchConfig


def _load_or_train_tokenizer(
    config: PretrainLaunchConfig,
    output_dir: Path,
    *,
    require_target_vocab: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Load the persisted tokenizer on resume, or train it once.

    A resumed container must use the exact tokenizer that produced the checkpoint's
    token stream. Training a fresh BPE can drift across builds, which would make
    resume offsets point into a different stream.
    """
    tokenizers = importlib.import_module("tokenizers")
    tokenizer_path = output_dir / "tokenizer.json"
    report_path = output_dir / "tokenizer-report.json"
    if tokenizer_path.exists():
        tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        report = _read_tokenizer_report(report_path)
        report = {
            **report,
            "source": "loaded_existing_tokenizer",
            "vocab_size": tokenizer.get_vocab_size(),
            "target_vocab_size": config.payload["tokenizer"]["vocab_size"],
        }
        if config.payload["tokenizer"]["require_round_trip_checks"]:
            report["round_trips"] = _round_trip_checks(tokenizer)
        if config.payload["tokenizer"]["require_coverage_report"]:
            report["coverage"] = report.get(
                "coverage", "byte-level BPE; every UTF-8 byte has a fallback path"
            )
        _validate_tokenizer_artifact(
            config, tokenizer, report, require_target_vocab=require_target_vocab
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return tokenizer, report

    return _train_tokenizer(
        config,
        output_dir,
        _bounded_texts_for_tokenizer(config),
        require_target_vocab=require_target_vocab,
    )


def _train_tokenizer(
    config: PretrainLaunchConfig,
    output_dir: Path,
    texts: Iterator[str],
    *,
    require_target_vocab: bool = True,
) -> tuple[Any, dict[str, Any]]:
    tokenizers = importlib.import_module("tokenizers")
    models = importlib.import_module("tokenizers.models")
    trainers = importlib.import_module("tokenizers.trainers")
    pre_tokenizers = importlib.import_module("tokenizers.pre_tokenizers")
    decoders = importlib.import_module("tokenizers.decoders")

    tokenizer = tokenizers.Tokenizer(models.BPE(unk_token="<unk>"))
    # Split digits into single tokens before byte-level pre-tokenization so the
    # BPE never merges across digit boundaries (numbers become digit sequences).
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.payload["tokenizer"]["vocab_size"],
        special_tokens=config.payload["tokenizer"]["special_tokens"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer_path = output_dir / "tokenizer.json"
    report_path = output_dir / "tokenizer-report.json"

    report = {
        "kind": config.payload["tokenizer"]["kind"],
        "trainer": config.payload["tokenizer"]["trainer"],
        "vocab_size": tokenizer.get_vocab_size(),
        "target_vocab_size": config.payload["tokenizer"]["vocab_size"],
        "source": "trained",
        "round_trips": _round_trip_checks(tokenizer),
        "coverage": "byte-level BPE; every UTF-8 byte has a fallback path",
    }
    if (
        require_target_vocab
        and tokenizer.get_vocab_size() != config.payload["tokenizer"]["vocab_size"]
    ):
        raise RuntimeError("tokenizer did not reach the configured vocab size")
    _validate_tokenizer_artifact(
        config,
        tokenizer,
        report,
        require_target_vocab=require_target_vocab,
    )
    tokenizer.save(str(tokenizer_path))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return tokenizer, report


def _validate_tokenizer_artifact(
    config: PretrainLaunchConfig,
    tokenizer: Any,
    report: dict[str, Any],
    *,
    require_target_vocab: bool = True,
) -> None:
    if (
        require_target_vocab
        and tokenizer.get_vocab_size() != config.payload["tokenizer"]["vocab_size"]
    ):
        raise RuntimeError("tokenizer did not reach the configured vocab size")
    if config.payload["tokenizer"]["require_round_trip_checks"]:
        round_trips = report.get("round_trips")
        if not isinstance(round_trips, list) or not round_trips:
            raise RuntimeError("tokenizer round-trip checks are missing")
        if not all(isinstance(item, dict) and item.get("round_trip") for item in round_trips):
            raise RuntimeError("tokenizer round-trip check failed")
    if config.payload["tokenizer"]["require_coverage_report"]:
        coverage = report.get("coverage")
        if not isinstance(coverage, str) or not coverage.strip():
            raise RuntimeError("tokenizer coverage report is missing")


def _read_tokenizer_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"tokenizer-report.json is not valid JSON: {error.msg}") from error
    if not isinstance(report, dict):
        raise RuntimeError("tokenizer-report.json must contain a JSON object")
    return report


def _round_trip_checks(tokenizer: Any) -> list[dict[str, Any]]:
    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "LLM pretraining needs boring, durable evidence.",
    ]
    round_trips = []
    for text in examples:
        ids = tokenizer.encode(text).ids
        decoded = tokenizer.decode(ids)
        round_trips.append({"text": text, "tokens": len(ids), "round_trip": decoded == text})
    if not all(item["round_trip"] for item in round_trips):
        raise RuntimeError("tokenizer round-trip check failed")
    return round_trips


def _bounded_texts_for_tokenizer(config: PretrainLaunchConfig) -> Iterator[str]:
    emitted_bytes = 0
    # Tokenizer budget is in tokens in the config; bytes are a conservative streaming
    # proxy before the tokenizer exists. The report records this explicitly.
    max_bytes = config.payload["budgets"]["tokenizer_training_token_budget"] * 4
    for text in document_text_stream(config, split="train"):
        encoded = text.encode("utf-8", errors="strict")
        if emitted_bytes + len(encoded) > max_bytes:
            return
        emitted_bytes += len(encoded)
        yield text


def _token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise RuntimeError(f"tokenizer is missing required token {token!r}")
    return int(token_id)
