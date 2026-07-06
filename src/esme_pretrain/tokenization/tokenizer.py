from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self


def _token_to_id(id_to_token: tuple[str, ...]) -> dict[str, int]:
    return {token: token_id for token_id, token in enumerate(id_to_token)}


def _decode_token_ids(id_to_token: tuple[str, ...], token_ids: list[int], label: str) -> str:
    pieces: list[str] = []
    vocab_size = len(id_to_token)
    for token_id in token_ids:
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"token id {token_id} is outside the {label} vocabulary")
        pieces.append(id_to_token[token_id])
    return "".join(pieces)


def _id_to_token_from_dict(payload: Mapping[str, object]) -> tuple[str, ...]:
    id_to_token = payload.get("id_to_token")
    if not isinstance(id_to_token, list) or not all(
        isinstance(token, str) for token in id_to_token
    ):
        raise ValueError("checkpoint tokenizer payload is malformed")
    if len(set(id_to_token)) != len(id_to_token):
        raise ValueError("checkpoint tokenizer vocabulary contains duplicates")
    return tuple(id_to_token)


@dataclass(frozen=True)
class CharTokenizer:
    id_to_token: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> Self:
        if not text:
            raise ValueError("cannot build tokenizer from empty corpus")
        return cls(tuple(sorted(set(text))))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def token_to_id(self) -> dict[str, int]:
        return _token_to_id(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        token_to_id = self.token_to_id
        unknown = sorted({token for token in text if token not in token_to_id})
        if unknown:
            raise ValueError(f"text contains tokens outside the character vocabulary: {unknown!r}")
        return [token_to_id[token] for token in text]

    def decode(self, token_ids: list[int]) -> str:
        return _decode_token_ids(self.id_to_token, token_ids, "character")

    def to_dict(self) -> dict[str, list[str]]:
        return {"id_to_token": list(self.id_to_token)}

    @classmethod
    def from_dict(cls, payload: dict[str, list[str]]) -> Self:
        return cls(_id_to_token_from_dict(payload))


@dataclass(frozen=True)
class PairMergeTokenizer:
    id_to_token: tuple[str, ...]
    merges: tuple[tuple[str, str], ...]

    @classmethod
    def from_text(cls, text: str, target_vocab_size: int) -> Self:
        if not text:
            raise ValueError("cannot build tokenizer from empty corpus")
        if target_vocab_size < 1:
            raise ValueError("target vocab size must be at least 1")

        id_to_token = list(sorted(set(text)))
        if target_vocab_size < len(id_to_token):
            raise ValueError(
                "target vocab size "
                f"{target_vocab_size} is smaller than required character coverage "
                f"{len(id_to_token)}"
            )

        tokenized = list(text)
        merges: list[tuple[str, str]] = []
        token_set = set(id_to_token)
        while len(id_to_token) < target_vocab_size:
            pair_counts = Counter(zip(tokenized, tokenized[1:], strict=False))
            candidates = [
                (count, left, right)
                for (left, right), count in pair_counts.items()
                if count >= 2 and left + right not in token_set
            ]
            if not candidates:
                break

            _count, left, right = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
            merged = left + right
            id_to_token.append(merged)
            token_set.add(merged)
            merges.append((left, right))
            tokenized = _apply_pair_merge(tokenized, left, right)

        return cls(tuple(id_to_token), tuple(merges))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def token_to_id(self) -> dict[str, int]:
        return _token_to_id(self.id_to_token)

    def encode(self, text: str) -> list[int]:
        character_vocab_size = self.vocab_size - len(self.merges)
        character_vocab = set(self.id_to_token[:character_vocab_size])
        unknown = sorted({character for character in text if character not in character_vocab})
        if unknown:
            raise ValueError(
                f"text contains characters outside the tokenizer vocabulary: {unknown!r}"
            )

        tokenized = list(text)
        for left, right in self.merges:
            tokenized = _apply_pair_merge(tokenized, left, right)

        token_to_id = self.token_to_id
        return [token_to_id[token] for token in tokenized]

    def decode(self, token_ids: list[int]) -> str:
        return _decode_token_ids(self.id_to_token, token_ids, "tokenizer")

    def to_dict(self) -> dict[str, list[str] | list[list[str]]]:
        return {
            "id_to_token": list(self.id_to_token),
            "merges": [[left, right] for left, right in self.merges],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[str] | list[list[str]]]) -> Self:
        id_to_token = _id_to_token_from_dict(payload)
        merges = payload.get("merges")
        if not isinstance(merges, list) or not all(
            isinstance(merge, list)
            and len(merge) == 2
            and all(isinstance(token, str) for token in merge)
            for merge in merges
        ):
            raise ValueError("checkpoint tokenizer merges payload is malformed")
        return cls(
            id_to_token,
            tuple((left, right) for left, right in merges),
        )


def _apply_pair_merge(tokens: list[str], left: str, right: str) -> list[str]:
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        if index + 1 < len(tokens) and tokens[index] == left and tokens[index + 1] == right:
            merged.append(left + right)
            index += 2
        else:
            merged.append(tokens[index])
            index += 1
    return merged
