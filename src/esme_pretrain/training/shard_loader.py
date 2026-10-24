"""Shuffled, seekable batch loader over pre-tokenized uint16 token shards."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from esme_pretrain.data.token_shards import load_token_shard_manifest
from esme_pretrain.torch import torch
from esme_pretrain.training.data_stream import Batch, prefetch_device_batches
from esme_pretrain.training.eval_batch import hash_eval_batches


class ShardBatchLoader:
    """Yields device-resident :class:`Batch` objects from token shards, one pass.

    Each shard is cut into non-overlapping ``context_length + 1`` windows. A window
    never crosses a shard boundary, and each shard's tail shorter than a window is
    dropped (``dropped_tail_tokens``). The window order is sequential when
    ``shuffle_seed`` is None, else a seeded permutation of all windows. A trailing
    partial batch is dropped, as in :class:`StreamingBatchLoader`.

    ``skip_tokens`` is the resume offset in tokens. It must be a whole number of
    windows, and iteration starts at window ``skip_tokens // window`` of the same
    order, so resume reads no consumed tokens. ``tokens_yielded`` counts tokens
    handed to the consumer after the skip.
    """

    def __init__(
        self,
        manifest_path: Path,
        *,
        vocab_size: int,
        tokenizer_sha256: str,
        batch_size: int,
        context_length: int,
        device: str | torch.device = "cpu",
        shuffle_seed: int | None = None,
        skip_tokens: int = 0,
        pin_memory: bool = True,
        prefetch_batches: int = 4,
        verify_hashes: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if context_length < 1:
            raise ValueError("context_length must be at least 1")
        if prefetch_batches < 1:
            raise ValueError("prefetch_batches must be at least 1")
        self.manifest = load_token_shard_manifest(manifest_path, verify_hashes=verify_hashes)
        if self.manifest.vocab_size != vocab_size:
            raise ValueError(
                f"token shards in {manifest_path} use vocab {self.manifest.vocab_size}, "
                f"the model uses {vocab_size}"
            )
        if self.manifest.tokenizer_sha256 != tokenizer_sha256:
            raise ValueError(
                f"token shards in {manifest_path} were written with tokenizer "
                f"{self.manifest.tokenizer_sha256[:12]}..., not {tokenizer_sha256[:12]}..."
            )
        self.batch_size = batch_size
        self.context_length = context_length
        self.window = context_length + 1
        self.device = torch.device(device)
        self.pin_memory = pin_memory and self.device.type == "cuda"
        self.prefetch_batches = prefetch_batches

        self._shards = [
            np.memmap(shard.path, dtype="<u2", mode="r", shape=(shard.tokens,))
            for shard in self.manifest.shards
        ]
        windows_per_shard = np.array(
            [shard.tokens // self.window for shard in self.manifest.shards], dtype=np.int64
        )
        self.window_count = int(windows_per_shard.sum())
        if self.window_count == 0:
            raise ValueError(
                f"token shards in {manifest_path} hold no full {self.window}-token window"
            )
        self.dropped_tail_tokens = self.manifest.total_tokens - self.window_count * self.window
        self._shard_window_ends = np.cumsum(windows_per_shard)
        self._shard_window_starts = self._shard_window_ends - windows_per_shard
        if shuffle_seed is None:
            self._order = np.arange(self.window_count, dtype=np.int64)
        else:
            self._order = np.random.default_rng(shuffle_seed).permutation(self.window_count)
        self.skip_tokens = skip_tokens
        self.tokens_yielded = 0

    @property
    def skip_tokens(self) -> int:
        return self._skip_tokens

    @skip_tokens.setter
    def skip_tokens(self, value: int) -> None:
        if value < 0:
            raise ValueError("skip_tokens must be non-negative")
        if value % self.window:
            raise ValueError(
                f"skip_tokens {value} is not a whole number of {self.window}-token windows"
            )
        if value // self.window > self.window_count:
            raise ValueError(
                f"skip_tokens {value} is past the end of the token shards "
                f"({self.window_count} windows of {self.window} tokens)"
            )
        self._skip_tokens = value

    def __iter__(self) -> Iterator[Batch]:
        first_window = self._skip_tokens // self.window
        tokens_per_batch = self.window * self.batch_size

        def cpu_batches() -> Iterator[torch.Tensor]:
            last_start = self.window_count - self.batch_size
            for start in range(first_window, last_start + 1, self.batch_size):
                yield torch.from_numpy(
                    self._read_windows(self._order[start : start + self.batch_size])
                )

        with contextlib.closing(
            prefetch_device_batches(
                cpu_batches,
                device=self.device,
                pin_memory=self.pin_memory,
                prefetch_batches=self.prefetch_batches,
                thread_name="shard-prefetch",
            )
        ) as batches:
            for batch in batches:
                self.tokens_yielded += tokens_per_batch
                yield batch

    def _read_windows(self, window_ids: np.ndarray) -> np.ndarray:
        shard_ids = np.searchsorted(self._shard_window_ends, window_ids, side="right")
        shard_starts = self._shard_window_starts[shard_ids]
        rows = np.empty((window_ids.size, self.window), dtype=np.int64)
        for row, (shard_id, window_id, shard_start) in enumerate(
            zip(shard_ids, window_ids, shard_starts, strict=True)
        ):
            offset = (int(window_id) - int(shard_start)) * self.window
            rows[row] = self._shards[shard_id][offset : offset + self.window]
        return rows


def fixed_validation_batches(
    manifest_path: Path,
    *,
    vocab_size: int,
    tokenizer_sha256: str,
    context_length: int,
    batch_size: int,
    eval_token_budget: int,
    expected_token_batch_sha256: str | None,
    device: str | torch.device = "cpu",
) -> list[Batch]:
    """Read the first validation windows, in order, until ``eval_token_budget`` input tokens.

    Packing matches ``eval-checkpoints``: sequential ``context_length + 1`` windows,
    ``batch_size`` per batch, stopping at the first batch that reaches the budget. With
    ``expected_token_batch_sha256`` the batches must hash to a reference eval's value,
    which proves they are the same tokens.
    """
    loader = ShardBatchLoader(
        manifest_path,
        vocab_size=vocab_size,
        tokenizer_sha256=tokenizer_sha256,
        batch_size=batch_size,
        context_length=context_length,
        device=device,
        pin_memory=False,
        prefetch_batches=2,
    )
    batches: list[Batch] = []
    eval_tokens = 0
    iterator = iter(loader)
    try:
        for batch in iterator:
            batches.append(batch)
            eval_tokens += int(batch.input_ids.numel())
            if eval_tokens >= eval_token_budget:
                break
    finally:
        iterator.close()
    if eval_tokens < eval_token_budget:
        raise ValueError(
            f"validation shards hold {eval_tokens} eval tokens, {eval_token_budget} required"
        )
    if expected_token_batch_sha256 is not None:
        actual = hash_eval_batches(batches)
        if actual != expected_token_batch_sha256:
            raise ValueError(
                f"validation batches hash to {actual}, expected {expected_token_batch_sha256}"
            )
    return batches
