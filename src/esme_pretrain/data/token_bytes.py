from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from esme_pretrain.torch import torch


class TokenOffsets(Protocol):
    @property
    def offsets(self) -> Sequence[tuple[int, int]]: ...


@dataclass(frozen=True)
class FixedEvalBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    target_byte_counts: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.input_ids.numel())

    @property
    def byte_count(self) -> int:
        return int(self.target_byte_counts.sum().item())


def token_byte_counts(text: str, encoding: TokenOffsets, *, token_count: int) -> list[int]:
    offsets = getattr(encoding, "offsets", None)
    if offsets is None or len(offsets) != token_count:
        raise ValueError("tokenizer encoding must expose offsets for bits-per-byte eval")
    byte_counts = partitioned_byte_counts(text, offsets)
    text_bytes = len(text.encode("utf-8"))
    if sum(byte_counts) != text_bytes:
        raise ValueError(
            f"byte accounting mismatch: tokens cover {sum(byte_counts)} bytes, "
            f"document has {text_bytes}"
        )
    return byte_counts


def partitioned_byte_counts(text: str, offsets: Sequence[tuple[int, int]]) -> list[int]:
    """Per-token UTF-8 byte counts that partition the full text exactly.

    Token offsets from byte-level BPE tokenizers can trim whitespace or overlap,
    so raw offset spans under- or over-count bytes. Partitioning by token start
    boundaries assigns every byte to exactly one token: the counts always sum to
    ``len(text.encode("utf-8"))``.
    """
    if not offsets:
        if text:
            raise ValueError("tokenizer produced no tokens for non-empty text")
        return []
    starts = [int(start) for start, _end in offsets]
    boundaries = [0]
    for start in starts[1:]:
        boundaries.append(max(start, boundaries[-1]))
    boundaries.append(len(text))
    counts = []
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        counts.append(len(text[begin:end].encode("utf-8")))
    return counts


def batch_token_byte_windows(
    token_bytes: Iterable[tuple[int, int]], *, window: int, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if window < 2:
        raise ValueError("window must be at least 2 (context_length + 1)")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    iterator = iter(token_bytes)
    needed = window * batch_size
    while True:
        chunk = list(itertools.islice(iterator, needed))
        if len(chunk) < needed:
            return
        token_ids = [token_id for token_id, _byte_count in chunk]
        byte_counts = [byte_count for _token_id, byte_count in chunk]
        yield (
            torch.tensor(token_ids, dtype=torch.long).view(batch_size, window),
            torch.tensor(byte_counts, dtype=torch.long).view(batch_size, window),
        )
