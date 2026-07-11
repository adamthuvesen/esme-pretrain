from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from esme_pretrain.modeling.backbone import DenseBackbone
from esme_pretrain.postrun.eval_checkpoints import load_tokenizer
from esme_pretrain.torch import torch
from esme_pretrain.training.checkpointing import load_pretrain_checkpoint
from esme_pretrain.training.runtime import resolve_device


class SupportsTokenizer(Protocol):
    def encode(self, text: str) -> Any: ...

    def decode(self, ids: list[int]) -> str: ...

    def token_to_id(self, token: str) -> int | None: ...

    def get_vocab_size(self) -> int: ...


@dataclass(frozen=True)
class SampleCheckpointConfig:
    checkpoint_path: Path
    tokenizer_path: Path
    output_path: Path
    prompts: tuple[str, ...]
    max_new_tokens: int = 64
    device: str = "cpu"


def sample_checkpoint(config: SampleCheckpointConfig) -> dict[str, Any]:
    if not config.prompts:
        raise ValueError("at least one --prompt is required")
    if any(not prompt for prompt in config.prompts):
        raise ValueError("--prompt must not be empty")
    if config.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if config.output_path in (config.checkpoint_path, config.tokenizer_path):
        raise ValueError("--output must differ from the checkpoint and tokenizer paths")

    device = resolve_device(config.device)
    loaded = load_pretrain_checkpoint(config.checkpoint_path, map_location=device)
    tokenizer = cast(SupportsTokenizer, load_tokenizer(config.tokenizer_path))
    if tokenizer.get_vocab_size() != loaded.model.config.vocab_size:
        raise ValueError("tokenizer vocab size does not match checkpoint config")
    model = loaded.model.to(device)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("tokenizer is missing required token '<eos>'")
    samples = [
        generate_completion(
            model,
            tokenizer,
            prompt,
            max_new_tokens=config.max_new_tokens,
            eos_id=eos_id,
            device=device,
        )
        for prompt in config.prompts
    ]
    payload = {
        "schema_version": 1,
        "checkpoint": str(config.checkpoint_path),
        "checkpoint_step": loaded.step,
        "tokenizer": str(config.tokenizer_path),
        "device": str(device),
        "decoding": "greedy",
        "max_new_tokens": config.max_new_tokens,
        "samples": samples,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(_render_samples(payload), encoding="utf-8")
    return payload


@torch.no_grad()
def generate_completion(
    model: DenseBackbone,
    tokenizer: SupportsTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    eos_id: int | None,
    device: torch.device,
) -> dict[str, Any]:
    encoding = tokenizer.encode(prompt)
    prompt_ids = [int(token_id) for token_id in getattr(encoding, "ids", encoding)]
    if not prompt_ids:
        raise ValueError("--prompt must encode to at least one token")

    context_length = model.config.context_length
    if len(prompt_ids) > context_length:
        raise ValueError(
            f"prompt encodes to {len(prompt_ids)} tokens, above checkpoint context "
            f"length {context_length}"
        )

    model.eval()
    generated_ids = list(prompt_ids)
    continuation_ids: list[int] = []
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([generated_ids[-context_length:]], dtype=torch.long, device=device)
        next_id = int(model(input_ids)[:, -1, :].argmax(dim=-1).item())
        generated_ids.append(next_id)
        continuation_ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

    return {
        "prompt": prompt,
        "continuation": tokenizer.decode(continuation_ids),
        "text": tokenizer.decode(generated_ids),
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(continuation_ids),
        "stopped_on_eos": bool(continuation_ids and continuation_ids[-1] == eos_id),
    }


def _render_samples(payload: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint Samples",
        "",
        f"Checkpoint: `{payload['checkpoint']}` (step {payload['checkpoint_step']})",
        "",
        f"Decoding: greedy, up to {payload['max_new_tokens']} new tokens",
    ]
    for index, sample in enumerate(payload["samples"], start=1):
        lines.extend(
            [
                "",
                f"## Sample {index}",
                "",
                "Prompt:",
                "",
                _indent(str(sample["prompt"])),
                "",
                "Continuation:",
                "",
                _indent(str(sample["continuation"])),
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.split("\n"))
