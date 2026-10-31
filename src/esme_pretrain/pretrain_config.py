"""Pretrain run card schema and the spend limits every run card obeys.

The accepted run's exact values live in pretrain_run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from esme_pretrain.data.corpus_config import DatasetConfig, DocumentSplitConfig
from esme_pretrain.modeling.backbone import BackboneConfig

# The absolute pretrain cap from docs/run-cards/pretrain-214m-b200.md (abort rules).
PRETRAIN_SPEND_CAP_USD = 100.0
MODAL_MAX_TIMEOUT_HOURS = 24
MODAL_GPUS = frozenset({"H100", "H100!", "H200", "B200"})


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class TokenBudgets(RunSettings):
    train_token_budget: int
    validation_token_budget: int
    tokenizer_training_token_budget: int
    hard_read_token_budget: int


class TokenizerConfig(RunSettings):
    kind: str
    trainer: str
    vocab_size: int
    special_tokens: tuple[str, ...]
    train_on: str
    output_file: str
    require_round_trip_checks: bool
    require_coverage_report: bool
    split_digits: bool


class TrainingSettings(RunSettings):
    micro_batch_size: int
    grad_accum_steps: int
    dtype: str
    use_compile: bool = Field(alias="compile")
    fused_optimizer: bool
    seed: int


class OptimizerConfig(RunSettings):
    name: str
    learning_rate: float
    min_lr_ratio: float
    warmup_steps: int
    weight_decay: float
    grad_clip: float
    lr_schedule: str
    decay_fraction: float
    training: TrainingSettings


class GpuProfile(RunSettings):
    modal_gpu: str
    usd_per_hour: float = Field(gt=0)
    projected_tokens_per_second: int = Field(gt=0)
    expected_duration_hours: float = Field(gt=0)
    projection_source: str = Field(pattern=r"\S")
    measured: bool


class RuntimeConfig(RunSettings):
    provider: str
    selected_gpu: str
    precision: str
    gpu_profiles: Mapping[str, GpuProfile]
    max_cost_usd: float
    runtime_spend_stop_usd: float
    allow_retries: bool
    modal_volume: str
    timeout_hours: int

    @field_validator("gpu_profiles", mode="after")
    @classmethod
    def freeze_profiles(cls, value: Mapping[str, GpuProfile]) -> Mapping[str, GpuProfile]:
        return MappingProxyType(dict(value))

    @field_serializer("gpu_profiles")
    def serialize_profiles(self, value: Mapping[str, GpuProfile]) -> dict[str, GpuProfile]:
        return dict(value)

    @model_validator(mode="after")
    def check_spend_and_hardware(self) -> Self:
        if not 0 < self.max_cost_usd <= PRETRAIN_SPEND_CAP_USD:
            raise ValueError(
                f"max_cost_usd {self.max_cost_usd:g} must be in (0, {PRETRAIN_SPEND_CAP_USD:g}], "
                "the absolute pretrain spend cap. A larger cap needs a new run card and approval"
            )
        if self.runtime_spend_stop_usd != self.max_cost_usd:
            raise ValueError("runtime_spend_stop_usd must equal max_cost_usd")
        if self.allow_retries:
            raise ValueError("allow_retries must be false: a retry rents the GPU again")
        if not 0 < self.timeout_hours <= MODAL_MAX_TIMEOUT_HOURS:
            raise ValueError(
                f"timeout_hours must be in 1..{MODAL_MAX_TIMEOUT_HOURS} (Modal maximum)"
            )
        unknown = sorted(set(self.gpu_profiles) - MODAL_GPUS)
        if unknown:
            raise ValueError(f"unsupported GPU profiles {unknown}; use {sorted(MODAL_GPUS)}")
        if self.selected_gpu not in self.gpu_profiles:
            raise ValueError("selected_gpu must name a gpu_profiles entry")
        worst_case_usd = self.timeout_hours * self.gpu_profiles[self.selected_gpu].usd_per_hour
        if worst_case_usd > self.max_cost_usd:
            raise ValueError(
                f"timeout_hours {self.timeout_hours} on {self.selected_gpu} can spend "
                f"${worst_case_usd:.2f}, over max_cost_usd ${self.max_cost_usd:g}"
            )
        for name, profile in self.gpu_profiles.items():
            if profile.modal_gpu != name:
                raise ValueError(f"gpu_profiles.{name}.modal_gpu must be {name}")
        return self


class MonitoringConfig(RunSettings):
    log_interval: int
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    wandb_project: str


class ArtifactsConfig(RunSettings):
    output_dir: str
    required_files: tuple[str, ...]


class PretrainRunConfig(RunSettings):
    run_id: str
    run_card: str
    requires_approval: bool
    dataset: DatasetConfig
    split: DocumentSplitConfig
    budgets: TokenBudgets
    tokenizer: TokenizerConfig
    model: BackboneConfig
    optimizer: OptimizerConfig
    runtime: RuntimeConfig
    monitoring: MonitoringConfig
    artifacts: ArtifactsConfig
    abort_rules: tuple[str, ...]

    @field_validator("model", mode="before")
    @classmethod
    def complete_model_config(cls, value: object) -> object:
        if isinstance(value, dict) and set(value) != {
            field.name for field in fields(BackboneConfig)
        }:
            raise ValueError("model must contain exactly the BackboneConfig fields")
        return value

    @model_validator(mode="after")
    def check_estimated_cost(self) -> Self:
        estimated = self.estimated_cost_usd()
        if estimated > self.runtime.max_cost_usd:
            raise ValueError(
                f"estimated cost ${estimated:.2f} exceeds runtime.max_cost_usd "
                f"${self.runtime.max_cost_usd:g} on {self.runtime.selected_gpu}"
            )
        return self

    def estimated_cost_usd(self) -> float:
        profile = self.runtime.gpu_profiles[self.runtime.selected_gpu]
        seconds = self.budgets.train_token_budget / profile.projected_tokens_per_second
        return seconds * profile.usd_per_hour / 3600.0


def parse_run_config(data: str, source: Path) -> PretrainRunConfig:
    try:
        return PretrainRunConfig.model_validate_json(data)
    except ValidationError as error:
        detail = error.errors(include_url=False)[0]
        field = ".".join(str(part) for part in detail["loc"]) or "JSON"
        message = detail["msg"].removeprefix("Value error, ")
        raise ValueError(
            f"invalid run config {source}: {field}: {message}. Fix the run card JSON."
        ) from error
