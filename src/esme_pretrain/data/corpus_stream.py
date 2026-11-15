"""Shared corpus streaming helpers for train, tokenizer, and fixed eval flows."""

from __future__ import annotations

import importlib
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Protocol

from esme_pretrain.data.corpus_config import DatasetConfig, DocumentSplitConfig
from esme_pretrain.data.document_split import select_documents


class SupportsCorpusSettings(Protocol):
    @property
    def dataset(self) -> DatasetConfig: ...

    @property
    def split(self) -> DocumentSplitConfig: ...


class SupportsCorpusConfig(Protocol):
    @property
    def config(self) -> SupportsCorpusSettings: ...


def document_text_stream(
    config: SupportsCorpusConfig, *, split: str, parquet_files: Sequence[Path] = ()
) -> Generator[str]:
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

    dataset_cfg = config.config.dataset
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
            dataset_cfg.source,
            name=dataset_cfg.subset,
            split=dataset_cfg.split,
            streaming=dataset_cfg.streaming,
            revision=dataset_cfg.revision,
        )
    yield from select_documents(
        stream,
        dataset_cfg=dataset_cfg.model_dump(mode="json"),
        split_cfg=config.config.split.model_dump(mode="json"),
        split=split,
    )
