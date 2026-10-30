"""JSON structure for pretrain run cards. Accepted run policy lives in pretrain_run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_serializer,
    field_validator,
)

from esme_pretrain.data.corpus_config import DatasetConfig, DocumentSplitConfig
from esme_pretrain.modeling.backbone import BackboneConfig


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
    usd_per_hour: float
    projected_tokens_per_second: int
    expected_duration_hours: float
    projection_source: str
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


def parse_run_config(data: str, source: Path) -> PretrainRunConfig:
    try:
        return PretrainRunConfig.model_validate_json(data)
    except ValidationError as error:
        detail = error.errors(include_url=False)[0]
        field = ".".join(str(part) for part in detail["loc"]) or "JSON"
        raise ValueError(
            f"invalid run config {source}: {field}: {detail['msg']}. Fix the run card JSON."
        ) from error
