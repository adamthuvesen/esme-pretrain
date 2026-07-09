"""Bits-per-byte scoring of one model over one fixed text slice."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from esme_pretrain.postrun.eval_checkpoints import (
    FixedEvalBatch,
    batch_token_byte_windows,
    hash_eval_batches,
)
from esme_pretrain.training.eval_batch import mean_ce_loss

if TYPE_CHECKING:
    from esme_pretrain.baselines.models import EvalModel


@dataclass(frozen=True)
class SliceBpbResult:
    slice_name: str
    document_count: int
    text_sha256: str
    raw_bytes: int
    token_batch_sha256: str
    eval_tokens: int
    eval_bytes: int
    eval_batches: int
    ce_loss: float
    perplexity: float
    bits_per_byte: float


def hash_texts(texts: list[str]) -> str:
    """Order-sensitive hash of the exact slice documents."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def slice_bpb(
    model: EvalModel,
    texts: list[str],
    *,
    slice_name: str,
    batch_size: int,
    device: str,
) -> SliceBpbResult:
    if not texts:
        raise ValueError(f"text slice {slice_name} has no documents")

    token_bytes: list[tuple[int, int]] = []
    raw_bytes = 0
    for text in texts:
        encoded = model.encode(text)
        text_bytes = len(text.encode("utf-8"))
        counted = sum(encoded.byte_counts)
        if counted != text_bytes:
            raise ValueError(
                f"byte accounting mismatch for {model.name} on slice {slice_name}: "
                f"tokens cover {counted} bytes, document has {text_bytes}"
            )
        raw_bytes += text_bytes
        token_bytes.extend(zip(encoded.ids, encoded.byte_counts, strict=True))
        if model.eos_id is not None:
            token_bytes.append((int(model.eos_id), 0))

    batches = [
        FixedEvalBatch(
            input_ids=token_window[:, :-1].clone(),
            targets=token_window[:, 1:].clone(),
            target_byte_counts=byte_window[:, 1:].clone(),
        )
        for token_window, byte_window in batch_token_byte_windows(
            iter(token_bytes),
            window=model.context_length + 1,
            batch_size=batch_size,
        )
    ]
    if not batches:
        raise ValueError(
            f"slice {slice_name} did not produce any full eval batches for {model.name}; "
            "increase document_budget or reduce batch size"
        )

    pairs = ((batch.input_ids, batch.targets) for batch in batches)
    ce_loss = mean_ce_loss(model.module(), pairs, device=device)
    if ce_loss is None:
        raise ValueError(f"slice {slice_name} produced no eval targets for {model.name}")
    eval_tokens = sum(batch.token_count for batch in batches)
    eval_bytes = sum(batch.byte_count for batch in batches)
    if eval_bytes <= 0:
        raise ValueError(f"slice {slice_name} produced no eval bytes for {model.name}")

    return SliceBpbResult(
        slice_name=slice_name,
        document_count=len(texts),
        text_sha256=hash_texts(texts),
        raw_bytes=raw_bytes,
        token_batch_sha256=hash_eval_batches(batches),
        eval_tokens=eval_tokens,
        eval_bytes=eval_bytes,
        eval_batches=len(batches),
        ce_loss=ce_loss,
        perplexity=math.exp(ce_loss),
        bits_per_byte=ce_loss * eval_tokens / math.log(2) / eval_bytes,
    )
