from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from esme_pretrain.data.dataset import PackedTokens, pack_token_ids
from esme_pretrain.tokenization.tokenizer import CharTokenizer
from esme_pretrain.torch import torch

DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_SPLIT_SEED = 17
DEFAULT_SHARD_ROWS = 1024


@dataclass(frozen=True)
class DataPipelineConfig:
    input_path: Path
    context_length: int
    token_budget: int
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    seed: int = DEFAULT_SPLIT_SEED
    shard_rows: int = DEFAULT_SHARD_ROWS


@dataclass(frozen=True)
class SourceFile:
    path: str
    bytes: int
    characters: int
    lines: int
    sha256: str


@dataclass(frozen=True)
class IngestedCorpus:
    input_path: Path
    input_root: Path
    files: tuple[SourceFile, ...]
    text: str

    @property
    def characters(self) -> int:
        return len(self.text)

    @property
    def lines(self) -> int:
        return self.text.count("\n") + (1 if self.text else 0)


@dataclass(frozen=True)
class SplitCounts:
    train_rows: int
    validation_rows: int


@dataclass(frozen=True)
class BudgetedTokenIds:
    token_ids: list[int]
    truncated_tokens: int


@dataclass(frozen=True)
class DataReport:
    input_path: Path
    input_root: Path
    files: tuple[SourceFile, ...]
    context_length: int
    token_budget: int
    total_tokens: int
    budgeted_tokens: int
    truncated_tokens: int
    packable_rows: int
    split_seed: int
    validation_fraction: float
    splits: SplitCounts
    vocab_size: int
    corpus_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["input_path"] = str(self.input_path)
        payload["input_root"] = str(self.input_root)
        return payload


@dataclass(frozen=True)
class ShardInfo:
    path: str
    rows: int
    row_start: int
    row_end: int


@dataclass(frozen=True)
class PrepareDataResult:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    report: DataReport
    train_shards: tuple[ShardInfo, ...]
    validation_shards: tuple[ShardInfo, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "output_dir": str(self.output_dir),
            "manifest_path": str(self.manifest_path),
            "report_path": str(self.report_path),
            "report": self.report.to_dict(),
            "shards": {
                "train": [asdict(shard) for shard in self.train_shards],
                "validation": [asdict(shard) for shard in self.validation_shards],
            },
            "split_sizes": {
                "train_rows": self.report.splits.train_rows,
                "validation_rows": self.report.splits.validation_rows,
            },
            "token_budget": {
                "requested": self.report.token_budget,
                "total_tokens": self.report.total_tokens,
                "budgeted_tokens": self.report.budgeted_tokens,
                "truncated_tokens": self.report.truncated_tokens,
            },
        }
        return payload


def _validate_config(config: DataPipelineConfig) -> None:
    if config.context_length < 2:
        raise ValueError("context length must be at least 2")
    if config.token_budget < 1:
        raise ValueError("token budget must be at least 1")
    if config.token_budget <= config.context_length:
        raise ValueError(
            "token budget must be larger than context length to build next-token windows"
        )
    if not 0 < config.validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    if config.shard_rows < 1:
        raise ValueError("shard rows must be at least 1")


def _source_file(path: Path, root: Path, text: str, raw: bytes) -> SourceFile:
    relative_path = path.relative_to(root).as_posix()
    return SourceFile(
        path=relative_path,
        bytes=len(raw),
        characters=len(text),
        lines=text.count("\n") + (1 if text else 0),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _read_text_file(path: Path, root: Path) -> tuple[SourceFile, str]:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"input file does not exist: {path}") from error

    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"input file escapes input root: {path}") from error

    raw = resolved.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"input file is not valid UTF-8 text: {path}") from error
    return _source_file(resolved, root, text, raw), text


def ingest_local_text(input_path: Path) -> IngestedCorpus:
    try:
        resolved_input = input_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"input path does not exist: {input_path}") from error

    if resolved_input.is_file():
        input_root = resolved_input.parent
        source_file, text = _read_text_file(resolved_input, input_root)
        files = (source_file,)
        texts = [text]
    elif resolved_input.is_dir():
        input_root = resolved_input
        paths = sorted(path for path in resolved_input.rglob("*") if path.is_file())
        if not paths:
            raise ValueError(f"input directory contains no files: {input_path}")
        files_list: list[SourceFile] = []
        texts = []
        for path in paths:
            source_file, text = _read_text_file(path, input_root)
            files_list.append(source_file)
            texts.append(text)
        files = tuple(files_list)
    else:
        raise ValueError(f"input path is neither a file nor a directory: {input_path}")

    corpus_text = "\n".join(texts)
    if not corpus_text:
        raise ValueError(f"input corpus is empty: {input_path}")

    return IngestedCorpus(
        input_path=resolved_input,
        input_root=input_root,
        files=files,
        text=corpus_text,
    )


def _split_indices(
    row_count: int, validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    if row_count < 2:
        raise ValueError("need at least 2 packed rows for train/validation split")

    validation_rows = max(1, round(row_count * validation_fraction))
    validation_rows = min(validation_rows, row_count - 1)
    indices = list(range(row_count))
    random.Random(seed).shuffle(indices)
    validation = sorted(indices[:validation_rows])
    train = sorted(indices[validation_rows:])
    return train, validation


def apply_token_budget(token_ids: list[int], token_budget: int) -> BudgetedTokenIds:
    if token_budget < 1:
        raise ValueError("token budget must be at least 1")
    budgeted_token_ids = token_ids[:token_budget]
    return BudgetedTokenIds(
        token_ids=budgeted_token_ids,
        truncated_tokens=max(0, len(token_ids) - len(budgeted_token_ids)),
    )


def _select_rows(packed: PackedTokens, indices: list[int]) -> PackedTokens:
    row_index = torch.tensor(indices, dtype=torch.long)
    return PackedTokens(
        inputs=packed.inputs.index_select(0, row_index),
        targets=packed.targets.index_select(0, row_index),
    )


def split_packed_tokens(
    packed: PackedTokens, validation_fraction: float, seed: int
) -> tuple[PackedTokens, PackedTokens]:
    train_indices, validation_indices = _split_indices(
        packed.rows,
        validation_fraction,
        seed,
    )
    return _select_rows(packed, train_indices), _select_rows(packed, validation_indices)


def _build_report_and_packed(
    config: DataPipelineConfig,
) -> tuple[DataReport, CharTokenizer, PackedTokens, PackedTokens]:
    _validate_config(config)
    corpus = ingest_local_text(config.input_path)
    tokenizer = CharTokenizer.from_text(corpus.text)
    token_ids = tokenizer.encode(corpus.text)
    total_tokens = len(token_ids)
    if total_tokens <= config.context_length:
        raise ValueError(
            f"input has {total_tokens} tokens; need more than {config.context_length} to pack"
        )

    budgeted = apply_token_budget(token_ids, config.token_budget)
    packed = pack_token_ids(budgeted.token_ids, config.context_length)
    train, validation = split_packed_tokens(packed, config.validation_fraction, config.seed)

    report = DataReport(
        input_path=corpus.input_path,
        input_root=corpus.input_root,
        files=corpus.files,
        context_length=config.context_length,
        token_budget=config.token_budget,
        total_tokens=total_tokens,
        budgeted_tokens=len(budgeted.token_ids),
        truncated_tokens=budgeted.truncated_tokens,
        packable_rows=packed.rows,
        split_seed=config.seed,
        validation_fraction=config.validation_fraction,
        splits=SplitCounts(train_rows=train.rows, validation_rows=validation.rows),
        vocab_size=tokenizer.vocab_size,
        corpus_sha256=hashlib.sha256(corpus.text.encode("utf-8")).hexdigest(),
    )
    return report, tokenizer, train, validation


def build_data_report(config: DataPipelineConfig) -> DataReport:
    report, _tokenizer, _train, _validation = _build_report_and_packed(config)
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_shards(
    output_dir: Path, split: str, packed: PackedTokens, shard_rows: int
) -> tuple[ShardInfo, ...]:
    shards: list[ShardInfo] = []
    for shard_index, row_start in enumerate(range(0, packed.rows, shard_rows)):
        row_end = min(row_start + shard_rows, packed.rows)
        shard_path = output_dir / f"{split}-{shard_index:05d}.pt"
        torch.save(
            {
                "split": split,
                "row_start": row_start,
                "row_end": row_end,
                "inputs": packed.inputs[row_start:row_end],
                "targets": packed.targets[row_start:row_end],
            },
            shard_path,
        )
        shards.append(
            ShardInfo(
                path=shard_path.name,
                rows=row_end - row_start,
                row_start=row_start,
                row_end=row_end,
            )
        )
    return tuple(shards)


def prepare_data(config: DataPipelineConfig, output_dir: Path) -> PrepareDataResult:
    report, tokenizer, train, validation = _build_report_and_packed(config)
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")

    train_shards = _write_shards(output_dir, "train", train, config.shard_rows)
    validation_shards = _write_shards(output_dir, "validation", validation, config.shard_rows)
    report_path = output_dir / "data-report.json"
    manifest_path = output_dir / "manifest.json"

    _write_json(report_path, report.to_dict())
    manifest = {
        "schema_version": 1,
        "format": "packed-token-shards-v1",
        "config": {
            "context_length": config.context_length,
            "token_budget": config.token_budget,
            "validation_fraction": config.validation_fraction,
            "split_seed": config.seed,
            "shard_rows": config.shard_rows,
        },
        "tokenizer": tokenizer.to_dict(),
        "report_path": report_path.name,
        "report": report.to_dict(),
        "splits": {
            "train": {
                "rows": train.rows,
                "shards": [asdict(shard) for shard in train_shards],
            },
            "validation": {
                "rows": validation.rows,
                "shards": [asdict(shard) for shard in validation_shards],
            },
        },
    }
    _write_json(manifest_path, manifest)

    return PrepareDataResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        report=report,
        train_shards=train_shards,
        validation_shards=validation_shards,
    )
