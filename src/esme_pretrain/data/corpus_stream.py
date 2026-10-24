"""Shared corpus streaming helpers for train, tokenizer, and fixed eval flows."""

from __future__ import annotations

import importlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from esme_pretrain.data.document_split import select_documents


class SupportsCorpusConfig(Protocol):
    payload: dict[str, Any]


def document_text_stream(
    config: SupportsCorpusConfig, *, split: str, parquet_files: Sequence[Path] = ()
) -> Iterator[str]:
    """Load the configured dataset stream and yield texts for the requested split.

    ``parquet_files`` reads local copies of the configured subset's parquet files
    instead of streaming them from the Hub. They are read in sorted path order, the
    order the Hub stream uses, so the selected documents are the same. Nothing here
    checks that the files are the pinned revision; callers verify by content hash.
    """
    try:
        datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as error:
        raise ValueError(
            "datasets is required to build the FineWeb-Edu corpus stream; "
            "tests use synthetic fixtures and do not need it"
        ) from error

    dataset_cfg = config.payload["dataset"]
    if parquet_files:
        missing = [str(path) for path in parquet_files if not path.exists()]
        if missing:
            raise ValueError(f"corpus parquet file does not exist: {', '.join(missing)}")
        stream = datasets.load_dataset(
            "parquet",
            data_files=[str(path) for path in sorted(parquet_files)],
            split="train",
            streaming=True,
        )
    else:
        stream = datasets.load_dataset(
            dataset_cfg["source"],
            name=dataset_cfg["subset"],
            split=dataset_cfg["split"],
            streaming=dataset_cfg["streaming"],
            revision=dataset_cfg["revision"],
        )
    yield from select_documents(
        stream, dataset_cfg=dataset_cfg, split_cfg=config.payload["split"], split=split
    )
