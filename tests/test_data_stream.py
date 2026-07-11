from __future__ import annotations

import pytest

from esme_pretrain.torch import torch
from esme_pretrain.training.data_stream import (
    Batch,
    StreamingBatchLoader,
    batch_token_windows,
    synthetic_token_stream,
    tokenized_document_stream,
)


def test_batch_token_windows_packs_and_drops_partial_tail() -> None:
    # 10 tokens, window 3, batch 1 -> 3 batches of [1, 3] (token 9 dropped).
    batches = list(batch_token_windows(range(10), window=3, batch_size=1))
    assert [b.tolist() for b in batches] == [[[0, 1, 2]], [[3, 4, 5]], [[6, 7, 8]]]
    # batch_size groups consecutive windows: 12 tokens, window 3, batch 2 -> one [2, 3].
    grouped = list(batch_token_windows(range(12), window=3, batch_size=2))
    assert len(grouped) == 2
    assert grouped[0].tolist() == [[0, 1, 2], [3, 4, 5]]


def test_batch_token_windows_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="window must be at least 2"):
        list(batch_token_windows(range(10), window=1, batch_size=1))
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        list(batch_token_windows(range(10), window=3, batch_size=0))


def test_loader_shapes_and_next_token_shift() -> None:
    # 36 tokens -> 7 windows of 5 -> 3 batches of 2 (one window dropped).
    stream = synthetic_token_stream(50, seed=0, total_tokens=36)
    loader = StreamingBatchLoader(stream, batch_size=2, context_length=4, device="cpu")
    batches = list(loader)
    assert len(batches) == 3
    first = batches[0]
    assert isinstance(first, Batch)
    assert first.input_ids.shape == (2, 4)
    assert first.targets.shape == (2, 4)
    # targets are inputs shifted one position within each packed window.
    assert torch.equal(first.input_ids[:, 1:], first.targets[:, :-1])


def test_synthetic_stream_is_deterministic() -> None:
    a = list(synthetic_token_stream(50, seed=1, total_tokens=20))
    b = list(synthetic_token_stream(50, seed=1, total_tokens=20))
    assert a == b
    assert all(0 <= token < 50 for token in a)


def test_tokenized_document_stream_inserts_eos() -> None:
    class ByteEncoder:
        def encode(self, text: str) -> list[int]:
            return [ord(character) for character in text]

    tokens = list(tokenized_document_stream(["ab", "c"], ByteEncoder(), eos_id=0))
    assert tokens == [97, 98, 0, 99, 0]


def test_loader_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        StreamingBatchLoader(iter([]), batch_size=0, context_length=4)
    with pytest.raises(ValueError, match="prefetch_batches"):
        StreamingBatchLoader(iter([]), batch_size=1, context_length=4, prefetch_batches=0)
    with pytest.raises(ValueError, match="skip_tokens"):
        StreamingBatchLoader(iter([]), batch_size=1, context_length=4, skip_tokens=-1)


def test_skip_tokens_resumes_stream_without_rereading_head() -> None:
    # Monotonic token ids so each token's value is its global stream position; a
    # resumed loader must yield positions *after* the consumed prefix, never re-read
    # the head. window=5 (context 4), batch 2 -> 10 tokens per batch.
    def positions():
        n = 0
        while True:
            yield n
            n += 1

    full = StreamingBatchLoader(positions(), batch_size=2, context_length=4, device="cpu")
    first_three = []
    it = iter(full)
    for _ in range(3):
        first_three.append(next(it))
    it.close()
    consumed = {int(t) for b in first_three for t in b.input_ids.flatten().tolist()}
    consumed |= {int(t) for b in first_three for t in b.targets.flatten().tolist()}
    # 3 batches * 2 rows * 5 window = 30 tokens consumed.
    assert full.tokens_yielded == 30

    # A fresh loader on the same deterministic source, skipping the 30 consumed tokens.
    resumed = StreamingBatchLoader(
        positions(), batch_size=2, context_length=4, device="cpu", skip_tokens=30
    )
    resumed_batches = []
    rit = iter(resumed)
    for _ in range(3):
        resumed_batches.append(next(rit))
    rit.close()
    resumed_tokens = {int(t) for b in resumed_batches for t in b.input_ids.flatten().tolist()}

    # The resumed stream must start exactly where the first run stopped: token 30, and
    # nothing from the head.
    assert min(resumed_tokens) == 30
    assert consumed.isdisjoint(resumed_tokens)


def test_skip_tokens_robust_to_different_batch_size() -> None:
    # The same 30-token prefix, re-packed at a DIFFERENT batch size on resume, must
    # still continue at token 30 (token offset + fixed window => window boundary).
    def positions():
        n = 0
        while True:
            yield n
            n += 1

    resumed = StreamingBatchLoader(
        positions(), batch_size=3, context_length=4, device="cpu", skip_tokens=30
    )
    rit = iter(resumed)
    first = next(rit)
    rit.close()
    assert int(first.input_ids.flatten()[0]) == 30


def test_tokens_yielded_counts_consumer_side() -> None:
    loader = StreamingBatchLoader(
        synthetic_token_stream(50, seed=0, total_tokens=36),
        batch_size=2,
        context_length=4,
        device="cpu",
    )
    assert loader.tokens_yielded == 0
    batches = list(loader)
    # 3 batches * 2 rows * 5 window = 30 tokens.
    assert loader.tokens_yielded == 30
    assert len(batches) == 3
