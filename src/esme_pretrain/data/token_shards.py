"""Pre-tokenized token shards: raw little-endian uint16 files plus a JSON manifest.

A split is written as ``<split>-00000.bin``, ``<split>-00001.bin``, ... and a
``<split>-manifest.json`` written last, so a directory without a manifest is an
unfinished write. Documents are encoded in the same order and with the same
tokenizer call as the streaming path, with ``eos_id`` appended after each one.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

import numpy as np

TOKEN_SHARDS_FORMAT = "esme_token_shards_v1"
TOKEN_SHARD_DTYPE = "uint16"
TOKEN_SHARD_BYTES_PER_TOKEN = 2
UINT16_TOKEN_LIMIT = 65536
ENCODE_BATCH_DOCUMENTS = 1024


class SupportsIds(Protocol):
    ids: list[int]


class SupportsEncodeBatch(Protocol):
    def encode_batch(self, texts: list[str]) -> Sequence[SupportsIds]: ...


@dataclass(frozen=True)
class TokenShard:
    path: Path
    tokens: int
    sha256: str


@dataclass(frozen=True)
class TokenShardManifest:
    path: Path
    split: str
    eos_id: int
    vocab_size: int
    tokenizer_sha256: str
    documents: int
    total_tokens: int
    shards: tuple[TokenShard, ...]
    payload: dict[str, Any]


def manifest_path_for(output_dir: Path, split: str) -> Path:
    return output_dir / f"{split}-manifest.json"


def write_token_shards(
    documents: Iterable[str],
    tokenizer: SupportsEncodeBatch,
    *,
    eos_id: int,
    vocab_size: int,
    tokenizer_sha256: str,
    split: str,
    output_dir: Path,
    shard_tokens: int,
    max_tokens: int | None = None,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Encode ``documents`` into uint16 token shards and return the written manifest.

    ``max_tokens`` truncates the token stream at exactly that many tokens, so the
    last document may be cut mid-way and lose its ``eos_id``. ``documents`` counts
    every document with at least one written token, including a truncated one.
    """
    if not 0 < vocab_size <= UINT16_TOKEN_LIMIT:
        raise ValueError(f"vocab_size must be in [1, {UINT16_TOKEN_LIMIT}] for uint16 shards")
    if not 0 <= eos_id < vocab_size:
        raise ValueError(f"eos_id {eos_id} is outside the vocabulary [0, {vocab_size})")
    if shard_tokens < 1:
        raise ValueError("shard_tokens must be at least 1")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be at least 1 when set")
    manifest_path = manifest_path_for(output_dir, split)
    if manifest_path.exists():
        raise ValueError(f"token shard manifest already exists: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = _ShardWriter(output_dir, split=split, shard_tokens=shard_tokens)
    document_count = 0
    total_tokens = 0
    document_iter = iter(documents)
    while max_tokens is None or total_tokens < max_tokens:
        texts = list(itertools.islice(document_iter, ENCODE_BATCH_DOCUMENTS))
        if not texts:
            break
        encodings = tokenizer.encode_batch(texts)
        lengths = np.fromiter((len(encoding.ids) + 1 for encoding in encodings), dtype=np.int64)
        flat: list[int] = []
        for encoding in encodings:
            flat.extend(encoding.ids)
            flat.append(eos_id)
        token_ids = np.asarray(flat, dtype=np.int64)
        if max_tokens is not None and total_tokens + token_ids.size > max_tokens:
            keep = max_tokens - total_tokens
            token_ids = token_ids[:keep]
            document_count += int(np.searchsorted(np.cumsum(lengths), keep, side="left")) + 1
        else:
            document_count += len(texts)
        _check_token_range(token_ids, vocab_size=vocab_size, first_position=total_tokens)
        writer.write(token_ids.astype("<u2"))
        total_tokens += token_ids.size
    shards = writer.finish()
    if total_tokens == 0:
        raise ValueError(f"document stream for split {split!r} produced no tokens")

    manifest = {
        "format": TOKEN_SHARDS_FORMAT,
        "dtype": TOKEN_SHARD_DTYPE,
        "split": split,
        "eos_id": eos_id,
        "vocab_size": vocab_size,
        "tokenizer_sha256": tokenizer_sha256,
        "documents": document_count,
        "total_tokens": total_tokens,
        "max_tokens": max_tokens,
        "shard_tokens": shard_tokens,
        "shards": shards,
        "provenance": provenance,
    }
    _write_atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load_token_shard_manifest(
    manifest_path: Path, *, verify_hashes: bool = False
) -> TokenShardManifest:
    """Read a shard manifest and check every shard file against it.

    File sizes are always checked. ``verify_hashes`` also re-hashes every shard,
    which reads the full split from disk.
    """
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"token shard manifest does not exist: {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"token shard manifest is not valid JSON: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"token shard manifest must be a JSON object: {manifest_path}")
    if payload.get("format") != TOKEN_SHARDS_FORMAT:
        raise ValueError(
            f"token shard manifest format is {payload.get('format')!r}; "
            f"expected {TOKEN_SHARDS_FORMAT!r}"
        )
    if payload.get("dtype") != TOKEN_SHARD_DTYPE:
        raise ValueError(f"token shard dtype is {payload.get('dtype')!r}; expected uint16")
    vocab_size = payload.get("vocab_size")
    if not isinstance(vocab_size, int) or not 0 < vocab_size <= UINT16_TOKEN_LIMIT:
        raise ValueError(f"token shard vocab_size is invalid: {vocab_size!r}")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError(f"token shard manifest lists no shards: {manifest_path}")

    root = manifest_path.parent.resolve()
    shards: list[TokenShard] = []
    for entry in raw_shards:
        relative = Path(entry["path"])
        shard_path = (root / relative).resolve()
        if relative.is_absolute() or not shard_path.is_relative_to(root):
            raise ValueError(f"token shard path escapes the manifest directory: {relative}")
        tokens = int(entry["tokens"])
        if tokens < 1:
            raise ValueError(f"token shard {relative} records {tokens} tokens")
        if not shard_path.exists():
            raise ValueError(f"token shard does not exist: {shard_path}")
        expected_bytes = tokens * TOKEN_SHARD_BYTES_PER_TOKEN
        actual_bytes = shard_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"token shard {shard_path} has {actual_bytes} bytes; "
                f"manifest records {tokens} tokens ({expected_bytes} bytes)"
            )
        if verify_hashes and _file_sha256(shard_path) != entry["sha256"]:
            raise ValueError(f"token shard sha256 does not match the manifest: {shard_path}")
        shards.append(TokenShard(path=shard_path, tokens=tokens, sha256=entry["sha256"]))

    total_tokens = sum(shard.tokens for shard in shards)
    if total_tokens != payload.get("total_tokens"):
        raise ValueError(
            f"token shards hold {total_tokens} tokens; "
            f"manifest total_tokens is {payload.get('total_tokens')}"
        )
    return TokenShardManifest(
        path=manifest_path,
        split=str(payload["split"]),
        eos_id=int(payload["eos_id"]),
        vocab_size=vocab_size,
        tokenizer_sha256=str(payload["tokenizer_sha256"]),
        documents=int(payload["documents"]),
        total_tokens=total_tokens,
        shards=tuple(shards),
        payload=payload,
    )


def _check_token_range(token_ids: np.ndarray, *, vocab_size: int, first_position: int) -> None:
    out_of_range = np.flatnonzero((token_ids < 0) | (token_ids >= vocab_size))
    if out_of_range.size:
        position = int(out_of_range[0])
        raise ValueError(
            f"token id {int(token_ids[position])} at stream position "
            f"{first_position + position} is outside the vocabulary [0, {vocab_size})"
        )


class _ShardWriter:
    def __init__(self, output_dir: Path, *, split: str, shard_tokens: int) -> None:
        self._output_dir = output_dir
        self._split = split
        self._shard_tokens = shard_tokens
        self._shards: list[dict[str, Any]] = []
        self._file: BinaryIO | None = None
        self._final_path: Path | None = None
        self._tmp_path: Path | None = None
        self._digest = hashlib.sha256()
        self._tokens_in_shard = 0

    def write(self, token_ids: np.ndarray) -> None:
        offset = 0
        while offset < token_ids.size:
            if self._file is None:
                self._open_next()
            take = min(self._shard_tokens - self._tokens_in_shard, token_ids.size - offset)
            chunk = token_ids[offset : offset + take].tobytes()
            self._file.write(chunk)
            self._digest.update(chunk)
            self._tokens_in_shard += take
            offset += take
            if self._tokens_in_shard == self._shard_tokens:
                self._close_current()

    def finish(self) -> list[dict[str, Any]]:
        if self._file is not None:
            self._close_current()
        return self._shards

    def _open_next(self) -> None:
        name = f"{self._split}-{len(self._shards):05d}.bin"
        self._final_path = self._output_dir / name
        self._tmp_path = self._output_dir / f"{name}.tmp"
        self._file = self._tmp_path.open("wb")
        self._digest = hashlib.sha256()
        self._tokens_in_shard = 0

    def _close_current(self) -> None:
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        os.replace(self._tmp_path, self._final_path)
        self._shards.append(
            {
                "path": self._final_path.name,
                "tokens": self._tokens_in_shard,
                "sha256": self._digest.hexdigest(),
            }
        )
        self._file = None


def _write_atomic_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
