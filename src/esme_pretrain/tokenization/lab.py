from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from esme_pretrain.data.dataset import PackedTokens, pack_token_ids
from esme_pretrain.data.pipeline import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    DataPipelineConfig,
    apply_token_budget,
    ingest_local_text,
    split_packed_tokens,
)
from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone, language_model_loss
from esme_pretrain.tokenization.tokenizer import CharTokenizer, PairMergeTokenizer
from esme_pretrain.torch import torch

LAB_MODEL_CONFIG = BackboneConfig(
    name="tokenizer-lab",
    vocab_size=256,
    context_length=32,
    embedding_dim=32,
    layers=2,
    heads=4,
    feedforward_dim=64,
    attention_kind="mha",
    qk_norm=False,
    z_loss_weight=0.0,
    logit_soft_cap=0.0,
)


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: list[int]) -> str: ...


@dataclass(frozen=True)
class LossComparison:
    train_loss_initial: float
    train_loss_final: float
    validation_loss_initial: float
    validation_loss_final: float
    train_rows: int
    validation_rows: int


@dataclass(frozen=True)
class TokenizerComparison:
    tokenizer: str
    vocab_size: int
    requested_vocab_size: int | None
    tokens: int
    budgeted_tokens: int
    truncated_tokens: int
    chars_per_token: float
    compression_ratio: float
    train_loss_initial: float
    train_loss_final: float
    validation_loss_initial: float
    validation_loss_final: float
    train_rows: int
    validation_rows: int
    round_trip_examples: tuple[dict[str, str | int], ...]
    unknown_tokens: int
    coverage: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TokenizerLabConfig:
    input_path: Path
    context_length: int
    token_budget: int
    vocab_sizes: tuple[int, ...]
    steps: int
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    seed: int = DEFAULT_SPLIT_SEED
    learning_rate: float = 0.03


@dataclass(frozen=True)
class TokenizerLabResult:
    input_path: Path
    context_length: int
    token_budget: int
    steps: int
    characters: int
    corpus_tokens_baseline: int
    comparisons: tuple[TokenizerComparison, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": str(self.input_path),
            "context_length": self.context_length,
            "token_budget": self.token_budget,
            "steps": self.steps,
            "characters": self.characters,
            "corpus_tokens_baseline": self.corpus_tokens_baseline,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }


def parse_vocab_sizes(raw: str) -> tuple[int, ...]:
    parts = raw.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError("vocab sizes must be a comma-separated list of integers")
    try:
        vocab_sizes = tuple(int(part.strip()) for part in parts)
    except ValueError as error:
        raise ValueError("vocab sizes must be a comma-separated list of integers") from error
    if any(vocab_size < 1 for vocab_size in vocab_sizes):
        raise ValueError("vocab sizes must all be positive integers")
    if len(set(vocab_sizes)) != len(vocab_sizes):
        raise ValueError("vocab sizes must not contain duplicates")
    return vocab_sizes


def run_tokenizer_lab(config: TokenizerLabConfig) -> TokenizerLabResult:
    _validate_lab_config(config)
    corpus = ingest_local_text(config.input_path)
    char_tokenizer = CharTokenizer.from_text(corpus.text)
    baseline_token_ids = char_tokenizer.encode(corpus.text)
    comparisons = [
        _compare_tokenizer(
            name="char",
            tokenizer=char_tokenizer,
            requested_vocab_size=None,
            text=corpus.text,
            baseline_tokens=len(baseline_token_ids),
            config=config,
        )
    ]
    for vocab_size in config.vocab_sizes:
        learned = PairMergeTokenizer.from_text(corpus.text, vocab_size)
        comparisons.append(
            _compare_tokenizer(
                name="pair_merge",
                tokenizer=learned,
                requested_vocab_size=vocab_size,
                text=corpus.text,
                baseline_tokens=len(baseline_token_ids),
                config=config,
            )
        )

    return TokenizerLabResult(
        input_path=corpus.input_path,
        context_length=config.context_length,
        token_budget=config.token_budget,
        steps=config.steps,
        characters=corpus.characters,
        corpus_tokens_baseline=len(baseline_token_ids),
        comparisons=tuple(comparisons),
    )


def _validate_lab_config(config: TokenizerLabConfig) -> None:
    data_config = DataPipelineConfig(
        input_path=config.input_path,
        context_length=config.context_length,
        token_budget=config.token_budget,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    if data_config.context_length < 2:
        raise ValueError("context length must be at least 2")
    if data_config.token_budget <= data_config.context_length:
        raise ValueError(
            "token budget must be larger than context length to build next-token windows"
        )
    if not 0 < data_config.validation_fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    if config.steps < 1:
        raise ValueError("tokenizer lab training requires at least one step")
    if not config.vocab_sizes:
        raise ValueError("at least one learned vocab size is required")


def _compare_tokenizer(
    name: str,
    tokenizer: Tokenizer,
    requested_vocab_size: int | None,
    text: str,
    baseline_tokens: int,
    config: TokenizerLabConfig,
) -> TokenizerComparison:
    token_ids = tokenizer.encode(text)
    budgeted = apply_token_budget(token_ids, config.token_budget)
    if len(budgeted.token_ids) <= config.context_length:
        raise ValueError(
            f"{name} tokenizer produced {len(budgeted.token_ids)} budgeted tokens; "
            f"need more than {config.context_length} to pack"
        )

    packed = pack_token_ids(budgeted.token_ids, config.context_length)
    train, validation = split_packed_tokens(packed, config.validation_fraction, config.seed)
    losses = _run_downstream_loss(
        tokenizer=tokenizer,
        train=train,
        validation=validation,
        context_length=config.context_length,
        steps=config.steps,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    examples = _round_trip_examples(tokenizer, text)

    return TokenizerComparison(
        tokenizer=name,
        vocab_size=tokenizer.vocab_size,
        requested_vocab_size=requested_vocab_size,
        tokens=len(token_ids),
        budgeted_tokens=len(budgeted.token_ids),
        truncated_tokens=budgeted.truncated_tokens,
        chars_per_token=len(text) / len(token_ids),
        compression_ratio=baseline_tokens / len(token_ids),
        train_loss_initial=losses.train_loss_initial,
        train_loss_final=losses.train_loss_final,
        validation_loss_initial=losses.validation_loss_initial,
        validation_loss_final=losses.validation_loss_final,
        train_rows=losses.train_rows,
        validation_rows=losses.validation_rows,
        round_trip_examples=examples,
        unknown_tokens=0,
        coverage="full corpus coverage; unseen characters raise ValueError",
    )


def _run_downstream_loss(
    tokenizer: Tokenizer,
    train: PackedTokens,
    validation: PackedTokens,
    context_length: int,
    steps: int,
    learning_rate: float,
    seed: int,
) -> LossComparison:
    _set_determinism(seed)
    model_config = replace(
        LAB_MODEL_CONFIG,
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
    )
    model = DenseBackbone(model_config)
    train_initial = _evaluate(model, train)
    validation_initial = _evaluate(model, validation)
    _train(model, train, steps, learning_rate)
    train_final = _evaluate(model, train)
    validation_final = _evaluate(model, validation)
    return LossComparison(
        train_loss_initial=train_initial,
        train_loss_final=train_final,
        validation_loss_initial=validation_initial,
        validation_loss_final=validation_final,
        train_rows=train.rows,
        validation_rows=validation.rows,
    )


def _set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _loss(model: DenseBackbone, batch: PackedTokens) -> torch.Tensor:
    logits = model(batch.inputs)
    loss, _ = language_model_loss(logits, batch.targets, z_loss_weight=0.0, logit_soft_cap=0.0)
    return loss


@torch.no_grad()
def _evaluate(model: DenseBackbone, batch: PackedTokens) -> float:
    model.eval()
    return float(_loss(model, batch).item())


def _train(
    model: DenseBackbone,
    train: PackedTokens,
    steps: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, train)
        loss.backward()
        optimizer.step()


def _round_trip_examples(tokenizer: Tokenizer, text: str) -> tuple[dict[str, str | int], ...]:
    examples: list[dict[str, str | int]] = []
    for sample in (text[:32], text[:80]):
        if not sample:
            continue
        token_ids = tokenizer.encode(sample)
        decoded = tokenizer.decode(token_ids)
        examples.append(
            {
                "characters": len(sample),
                "tokens": len(token_ids),
                "input": sample,
                "decoded": decoded,
            }
        )
    return tuple(examples)
