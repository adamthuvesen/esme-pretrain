from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from esme_pretrain.data.corpus_stream import document_text_stream
from esme_pretrain.launch.pretrain import PretrainLaunchConfig, load_pretrain_config
from esme_pretrain.modeling.backbone import BackboneConfig
from esme_pretrain.modeling.pretrain_checkpoint import load_pretrain_checkpoint
from esme_pretrain.torch import torch
from esme_pretrain.training.eval_batch import mean_ce_loss

FINAL_WITHIN_BEST_MARGIN = 0.02


class SupportsEncode(Protocol):
    def encode(self, text: str) -> Any: ...

    def token_to_id(self, token: str) -> int | None: ...


@dataclass(frozen=True)
class FixedEvalBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    target_byte_counts: torch.Tensor

    @property
    def token_count(self) -> int:
        return int(self.input_ids.numel())

    @property
    def byte_count(self) -> int:
        return int(self.target_byte_counts.sum().item())


@dataclass(frozen=True)
class EvalCheckpointConfig:
    config_path: Path
    tokenizer_path: Path
    checkpoint_paths: tuple[Path, ...]
    eval_token_budget: int
    output_path: Path
    device: str = "cpu"
    batch_size: int = 4
    max_eval_batches: int | None = None


@dataclass(frozen=True)
class CheckpointEvalResult:
    path: str
    checkpoint_step: int
    checkpoint_sha256: str
    ce_loss: float
    perplexity: float
    eval_tokens: int
    eval_bytes: int
    bits_per_byte: float
    eval_batches: int
    runtime_seconds: float


def run_eval_checkpoints(config: EvalCheckpointConfig) -> dict[str, Any]:
    if not config.checkpoint_paths:
        raise ValueError("at least one --checkpoint is required")
    if config.eval_token_budget < 1:
        raise ValueError("--eval-token-budget must be positive")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if config.max_eval_batches is not None and config.max_eval_batches < 1:
        raise ValueError("--max-eval-batches must be positive when set")

    launch_config = load_pretrain_config(config.config_path)
    tokenizer = load_tokenizer(config.tokenizer_path)
    model_config = BackboneConfig.from_dict(launch_config.payload["model"])
    batches = build_fixed_validation_batches(
        launch_config,
        tokenizer,
        model_config=model_config,
        eval_token_budget=config.eval_token_budget,
        batch_size=config.batch_size,
        max_eval_batches=config.max_eval_batches,
    )
    if not batches:
        raise ValueError("validation slice did not produce any full eval batches")

    token_batch_sha256 = hash_eval_batches(batches)
    results = [
        evaluate_checkpoint(
            path,
            batches,
            device=config.device,
            expected_config=model_config,
        )
        for path in config.checkpoint_paths
    ]
    selection = select_checkpoint(results)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": str(config.config_path),
        "tokenizer": str(config.tokenizer_path),
        "eval_token_budget": config.eval_token_budget,
        "device": config.device,
        "batch_size": config.batch_size,
        "max_eval_batches": config.max_eval_batches,
        "fixed_validation": {
            "source": "fineweb-edu deterministic validation document split",
            "eval_tokens": sum(batch.token_count for batch in batches),
            "eval_bytes": sum(batch.byte_count for batch in batches),
            "eval_batches": len(batches),
            "token_batch_sha256": token_batch_sha256,
        },
        "checkpoints": [asdict(result) for result in results],
        "selection": selection,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_tokenizer(path: Path) -> SupportsEncode:
    if not path.exists():
        raise ValueError(f"tokenizer does not exist: {path}")
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as error:
        raise ValueError(
            "tokenizers is required to load tokenizer.json; install the optional "
            "runtime package before real checkpoint eval"
        ) from error
    return Tokenizer.from_file(str(path))


def build_fixed_validation_batches(
    config: PretrainLaunchConfig,
    tokenizer: SupportsEncode,
    *,
    model_config: BackboneConfig,
    eval_token_budget: int,
    batch_size: int,
    max_eval_batches: int | None = None,
) -> list[FixedEvalBatch]:
    eos_id = tokenizer.token_to_id("<eos>")
    token_bytes = tokenized_validation_stream(config, tokenizer, eos_id=eos_id)
    window_batches = batch_token_byte_windows(
        token_bytes,
        window=model_config.context_length + 1,
        batch_size=batch_size,
    )
    fixed: list[FixedEvalBatch] = []
    eval_tokens = 0
    for token_window, byte_window in window_batches:
        batch = FixedEvalBatch(
            input_ids=token_window[:, :-1].clone(),
            targets=token_window[:, 1:].clone(),
            target_byte_counts=byte_window[:, 1:].clone(),
        )
        fixed.append(batch)
        eval_tokens += batch.token_count
        if eval_tokens >= eval_token_budget:
            break
        if max_eval_batches is not None and len(fixed) >= max_eval_batches:
            break
    return fixed


def tokenized_validation_stream(
    config: PretrainLaunchConfig, tokenizer: SupportsEncode, *, eos_id: int | None
) -> Iterator[tuple[int, int]]:
    for text in document_text_stream(config, split="validation"):
        encoding = tokenizer.encode(text)
        ids = [int(token_id) for token_id in _encoding_ids(encoding)]
        byte_counts = token_byte_counts(text, encoding, token_count=len(ids))
        yield from zip(ids, byte_counts, strict=True)
        if eos_id is not None:
            yield int(eos_id), 0


def _encoding_ids(encoding: Any) -> list[int]:
    if hasattr(encoding, "ids"):
        return list(encoding.ids)
    return list(encoding)


def token_byte_counts(text: str, encoding: Any, *, token_count: int) -> list[int]:
    offsets = getattr(encoding, "offsets", None)
    if offsets is None or len(offsets) != token_count:
        raise ValueError("tokenizer encoding must expose offsets for bits-per-byte eval")
    byte_counts: list[int] = []
    for start, end in offsets:
        if start == end:
            byte_counts.append(0)
            continue
        byte_counts.append(len(text[int(start) : int(end)].encode("utf-8")))
    return byte_counts


def batch_token_byte_windows(
    token_bytes: Iterable[tuple[int, int]], *, window: int, batch_size: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if window < 2:
        raise ValueError("window must be at least 2 (context_length + 1)")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    iterator = iter(token_bytes)
    needed = window * batch_size
    while True:
        chunk = list(itertools.islice(iterator, needed))
        if len(chunk) < needed:
            return
        token_ids = [token_id for token_id, _byte_count in chunk]
        byte_counts = [byte_count for _token_id, byte_count in chunk]
        yield (
            torch.tensor(token_ids, dtype=torch.long).view(batch_size, window),
            torch.tensor(byte_counts, dtype=torch.long).view(batch_size, window),
        )


def hash_eval_batches(batches: Iterable[FixedEvalBatch]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        for tensor in (batch.input_ids, batch.targets):
            cpu = tensor.detach().to(device="cpu", dtype=torch.long).contiguous()
            values = cpu.view(-1).tolist()
            digest.update(struct.pack(f"<{len(values)}q", *values))
    return digest.hexdigest()


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    fixed_batches: Sequence[FixedEvalBatch],
    *,
    device: str,
    expected_config: BackboneConfig,
) -> CheckpointEvalResult:
    start = time.perf_counter()
    loaded = load_pretrain_checkpoint(checkpoint_path, map_location=device)
    if loaded.config != expected_config:
        raise ValueError(f"checkpoint config does not match eval config: {checkpoint_path}")

    model = loaded.model.to(torch.device(device))
    pairs = ((batch.input_ids, batch.targets) for batch in fixed_batches)
    ce_loss = mean_ce_loss(
        model,
        pairs,
        device=device,
        logit_soft_cap=loaded.config.logit_soft_cap,
    )
    if ce_loss is None:
        raise ValueError(f"no eval targets in fixed batches for {checkpoint_path}")
    total_targets = sum(batch.token_count for batch in fixed_batches)
    total_bytes = sum(batch.byte_count for batch in fixed_batches)
    bits_per_byte = ce_loss / math.log(2) / total_bytes
    return CheckpointEvalResult(
        path=str(checkpoint_path),
        checkpoint_step=loaded.step,
        checkpoint_sha256=file_sha256(checkpoint_path),
        ce_loss=ce_loss,
        perplexity=math.exp(ce_loss),
        eval_tokens=total_targets,
        eval_bytes=total_bytes,
        bits_per_byte=bits_per_byte,
        eval_batches=len(fixed_batches),
        runtime_seconds=round(time.perf_counter() - start, 6),
    )


def file_sha256(path: Path) -> str:
    if not path.exists():
        raise ValueError(f"required file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_checkpoint(results: Sequence[CheckpointEvalResult | dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one checkpoint result is required")
    rows = [asdict(row) if isinstance(row, CheckpointEvalResult) else dict(row) for row in results]
    best = min(rows, key=lambda row: float(row["ce_loss"]))
    finals = [row for row in rows if Path(str(row["path"])).name == "checkpoint.pt"]
    final = finals[0] if finals else None
    margin = float("inf") if final is None else float(final["ce_loss"]) - float(best["ce_loss"])
    if final is not None and margin <= FINAL_WITHIN_BEST_MARGIN:
        recommended = final
        reason = "final checkpoint is within 0.02 CE of the best fixed-val checkpoint"
    else:
        recommended = best
        reason = (
            "no final checkpoint named checkpoint.pt was evaluated"
            if final is None
            else "final checkpoint is more than 0.02 CE worse than the best fixed-val checkpoint"
        )
    return {
        "recommended_checkpoint": recommended["path"],
        "recommended_step": recommended["checkpoint_step"],
        "best_checkpoint": best["path"],
        "best_ce_loss": best["ce_loss"],
        "best_bits_per_byte": best.get("bits_per_byte"),
        "final_checkpoint": final["path"] if final else None,
        "final_ce_loss": final["ce_loss"] if final else None,
        "final_bits_per_byte": final.get("bits_per_byte") if final else None,
        "margin_ce_loss": None if final is None else margin,
        "within_final_margin": bool(final is not None and margin <= FINAL_WITHIN_BEST_MARGIN),
        "reason": reason,
    }
