"""Deterministic FineWeb-Edu document train/validation split."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from typing import Any


def is_validation_document(
    document_id: str,
    *,
    seed: int,
    validation_modulo: int,
    validation_remainder: int,
) -> bool:
    digest = hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()
    bucket = int(digest[:16], 16) % validation_modulo
    return bucket == validation_remainder


def passes_int_score_filter(
    row: dict[str, Any], *, min_int_score: int, score_field: str = "int_score"
) -> bool:
    return int(row.get(score_field, 0)) >= min_int_score


def document_id_from_row(row: dict[str, Any], *, id_field: str, text_field: str) -> str:
    return str(row.get(id_field) or row.get(text_field, ""))


def select_documents(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_cfg: dict[str, Any],
    split_cfg: dict[str, Any],
    split: str,
) -> Iterator[str]:
    """Yield each row's text that passes the int_score filter and belongs to ``split``.

    ``split`` is ``"train"``, ``"validation"``, or any other value to take both sides.
    Empty-text rows are skipped. The same filter and deterministic split are applied for
    both the training stream and the post-run eval slice, so they cannot drift apart.
    """
    for row in rows:
        if not passes_int_score_filter(row, min_int_score=dataset_cfg["filters"]["min_int_score"]):
            continue
        row_id = document_id_from_row(
            row, id_field=dataset_cfg["id_field"], text_field=dataset_cfg["text_field"]
        )
        is_validation = is_validation_document(
            row_id,
            seed=int(split_cfg["seed"]),
            validation_modulo=int(split_cfg["validation_modulo"]),
            validation_remainder=int(split_cfg["validation_remainder"]),
        )
        if split == "validation" and not is_validation:
            continue
        if split == "train" and is_validation:
            continue
        text = row.get(dataset_cfg["text_field"])
        if text:
            yield str(text)
