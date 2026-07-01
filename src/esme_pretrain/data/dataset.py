from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

from esme_pretrain.tokenization.tokenizer import CharTokenizer
from esme_pretrain.torch import torch


@dataclass(frozen=True)
class PackedTokens:
    inputs: torch.Tensor
    targets: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.inputs.shape[0])


def read_pilot_corpus() -> str:
    return files("esme_pretrain").joinpath("data/pilot_corpus.txt").read_text(encoding="utf-8")


def pack_token_ids(token_ids: list[int], context_length: int) -> PackedTokens:
    if context_length < 2:
        raise ValueError("context length must be at least 2")
    if len(token_ids) <= context_length:
        raise ValueError(
            f"need more than {context_length} token ids to build next-token training windows"
        )

    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    for start in range(0, len(token_ids) - context_length, context_length):
        window = token_ids[start : start + context_length + 1]
        if len(window) == context_length + 1:
            input_rows.append(window[:-1])
            target_rows.append(window[1:])

    if not input_rows:
        raise ValueError("pilot corpus did not produce any packed token rows")

    return PackedTokens(
        inputs=torch.tensor(input_rows, dtype=torch.long),
        targets=torch.tensor(target_rows, dtype=torch.long),
    )


def build_pilot_datasets(
    text: str,
    tokenizer: CharTokenizer,
    context_length: int,
    validation_fraction: float = 0.2,
) -> tuple[PackedTokens, PackedTokens]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")

    token_ids = tokenizer.encode(text)
    split_at = int(len(token_ids) * (1 - validation_fraction))
    split_at -= split_at % context_length
    if split_at <= context_length or len(token_ids) - split_at <= context_length:
        raise ValueError("pilot corpus is too small for the requested split and context length")

    train = pack_token_ids(token_ids[:split_at], context_length)
    validation = pack_token_ids(token_ids[split_at:], context_length)
    return train, validation
