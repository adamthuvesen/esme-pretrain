from __future__ import annotations

from dataclasses import dataclass

from esme_pretrain.torch import torch


@dataclass(frozen=True)
class PackedTokens:
    inputs: torch.Tensor
    targets: torch.Tensor

    @property
    def rows(self) -> int:
        return int(self.inputs.shape[0])


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
        raise ValueError("token ids did not produce any packed rows")

    return PackedTokens(
        inputs=torch.tensor(input_rows, dtype=torch.long),
        targets=torch.tensor(target_rows, dtype=torch.long),
    )
