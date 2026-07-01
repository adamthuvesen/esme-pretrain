"""Streaming, prefetched token loader for GPU pretraining.

Streams the corpus without a full download, tokenizes on the fly, packs fixed
context windows, and prefetches pinned CPU batches for non-blocking GPU copies.
"""

from __future__ import annotations

import itertools
import queue
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from esme_pretrain.torch import torch


class SupportsEncode(Protocol):
    def encode(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class Batch:
    """One training batch already on the target device."""

    input_ids: torch.Tensor  # [batch, context_length]
    targets: torch.Tensor  # [batch, context_length]

    @property
    def token_count(self) -> int:
        return int(self.input_ids.numel())


def batch_token_windows(
    tokens: Iterable[int], *, window: int, batch_size: int
) -> Iterator[torch.Tensor]:
    """Pack a flat token stream into ``[batch_size, window]`` CPU LongTensors.

    ``window`` is ``context_length + 1`` so each row yields a length-``context``
    input and its shifted target. Tokens are consumed in C-level ``islice`` chunks
    of ``batch_size * window`` — no per-token Python loop — so a single producer
    thread can keep the GPU fed. The trailing partial batch is dropped; the training
    stream is effectively endless.
    """
    if window < 2:
        raise ValueError("window must be at least 2 (context_length + 1)")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    iterator = iter(tokens)
    needed = window * batch_size
    while True:
        chunk = list(itertools.islice(iterator, needed))
        if len(chunk) < needed:
            return
        yield torch.tensor(chunk, dtype=torch.long).view(batch_size, window)


class StreamingBatchLoader:
    """Yields device-resident :class:`Batch` objects from a token-id iterable.

    A background thread builds pinned CPU batch tensors and puts them on a bounded
    queue; the consuming iterator pops them and copies to ``device`` with
    ``non_blocking=True`` so the host→device transfer overlaps GPU compute. Pinning
    and non-blocking copies only engage on CUDA (they are no-ops elsewhere).

    **Resumable streaming.** ``skip_tokens`` drops that many leading tokens from the
    source before packing, so a run preempted after consuming N tokens resumes by
    re-creating the same deterministic token stream and skipping N — it continues from
    where it stopped rather than re-reading the corpus head (which would silently
    re-train on seen tokens and never reach the corpus tail). The offset is counted in
    **tokens**, not batches, so resume works with a different ``batch_size`` (the
    window is fixed, so a token offset that is a whole number of windows lands on a
    window boundary at any batch size). ``tokens_yielded`` counts tokens actually
    handed to the consumer (post-skip). The skip is exact because the corpus stream and
    the :func:`batch_token_windows` packing are both deterministic given the same source.
    """

    def __init__(
        self,
        tokens: Iterable[int],
        *,
        batch_size: int,
        context_length: int,
        device: str | torch.device = "cpu",
        pin_memory: bool = True,
        prefetch_batches: int = 4,
        skip_tokens: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if context_length < 1:
            raise ValueError("context_length must be at least 1")
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be at least 1")
        if skip_tokens < 0:
            raise ValueError("skip_tokens must be non-negative")
        self._tokens = tokens
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = torch.device(device)
        # Pinning requires CUDA; silently fall back to unpinned on CPU/MPS.
        self.pin_memory = pin_memory and self.device.type == "cuda"
        self.prefetch_batches = prefetch_batches
        self.skip_tokens = skip_tokens
        # Tokens handed to the consumer so far (post-skip); the loop reads this to
        # persist the next resume offset (skip_tokens + tokens_yielded).
        self.tokens_yielded = 0

    def __iter__(self) -> Iterator[Batch]:
        window = self.context_length + 1
        tokens_per_batch = window * self.batch_size
        batch_queue: queue.Queue[torch.Tensor | object] = queue.Queue(maxsize=self.prefetch_batches)
        sentinel = object()
        error_box: list[BaseException] = []
        stop = threading.Event()
        skip_tokens = self.skip_tokens

        def producer() -> None:
            try:
                source = self._tokens
                # Skip already-consumed tokens on the producer side (C-level islice),
                # so resume fast-forwards the deterministic stream without touching the
                # GPU. Counting in tokens keeps the skip correct across batch sizes.
                if skip_tokens:
                    source = itertools.islice(iter(source), skip_tokens, None)
                batches = batch_token_windows(source, window=window, batch_size=self.batch_size)
                for cpu_batch in batches:
                    if stop.is_set():
                        break
                    if self.pin_memory:
                        cpu_batch = cpu_batch.pin_memory()
                    batch_queue.put(cpu_batch)
            except BaseException as error:  # noqa: BLE001 - surfaced to the consumer below
                error_box.append(error)
            finally:
                batch_queue.put(sentinel)

        worker = threading.Thread(target=producer, name="token-prefetch", daemon=True)
        worker.start()
        try:
            while True:
                item = batch_queue.get()
                if item is sentinel:
                    break
                cpu_batch = item  # type: ignore[assignment]
                batch = cpu_batch.to(self.device, non_blocking=self.pin_memory)
                self.tokens_yielded += tokens_per_batch
                yield Batch(input_ids=batch[:, :-1], targets=batch[:, 1:])
            if error_box:
                raise error_box[0]
        finally:
            stop.set()
            # Drain so a blocked producer can put its sentinel and exit cleanly.
            try:
                while True:
                    batch_queue.get_nowait()
            except queue.Empty:
                pass


def synthetic_token_stream(
    vocab_size: int, *, seed: int = 0, total_tokens: int | None = None
) -> Iterator[int]:
    """Deterministic random token ids for tests / network-free local dry-runs.

    Endless when ``total_tokens`` is None. Token *values* do not change train-step
    FLOPs, so this is a valid stand-in for measuring loop mechanics and throughput.
    """
    generator = torch.Generator().manual_seed(seed)
    emitted = 0
    chunk = 8192
    while total_tokens is None or emitted < total_tokens:
        size = chunk if total_tokens is None else min(chunk, total_tokens - emitted)
        block = torch.randint(0, vocab_size, (size,), generator=generator, dtype=torch.long)
        yield from block.tolist()
        emitted += size


def tokenized_document_stream(
    documents: Iterable[str], tokenizer: SupportsEncode, *, eos_id: int | None = None
) -> Iterator[int]:
    """Encode a document stream to a flat token-id stream, EOS-separated.

    ``eos_id`` (when given) is appended after each document so windows that span a
    document boundary still carry the separator the model learns to stop on.
    """
    for text in documents:
        yield from tokenizer.encode(text)
        if eos_id is not None:
            yield eos_id
