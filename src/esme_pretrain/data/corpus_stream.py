"""Shared corpus streaming helpers for train, tokenizer, and fixed eval flows."""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Any, Protocol

from esme_pretrain.data.document_split import select_documents


class SupportsCorpusConfig(Protocol):
    payload: dict[str, Any]


def document_text_stream(config: SupportsCorpusConfig, *, split: str) -> Iterator[str]:
    """Load the configured dataset stream and yield texts for the requested split."""
    try:
        datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as error:
        raise ValueError(
            "datasets is required to build the FineWeb-Edu corpus stream; "
            "tests use synthetic fixtures and do not need it"
        ) from error

    dataset_cfg = config.payload["dataset"]
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
