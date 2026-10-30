from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from esme_pretrain.modeling.backbone import BackboneConfig, baseline_config
from esme_pretrain.pretrain_config import (
    ArtifactsConfig,
    DatasetConfig,
    DocumentSplitConfig,
    GpuProfile,
    MonitoringConfig,
    OptimizerConfig,
    PretrainRunConfig,
    RuntimeConfig,
    TokenBudgets,
    TokenizerConfig,
    parse_run_config,
)

EXPECTED_DATASET_SOURCE = "HuggingFaceFW/fineweb-edu"
EXPECTED_DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
EXPECTED_TOKENIZER_VOCAB = 32_768
EXPECTED_ARTIFACTS: tuple[str, ...] = (
    "config.json",
    "tokenizer.json",
    "tokenizer-report.json",
    "data-report.json",
    "metrics.jsonl",
    "throughput.csv",
    "checkpoint.pt",
    "environment.txt",
    "cost.json",
    "run-summary.json",
    "scaleup-pretrain-report.md",
)


@dataclass(frozen=True)
class AcceptedRun:
    model_name: str = "214M"
    run_card: str = "docs/run-cards/pretrain-214m-b200.md"
    train_token_budget: int = 10_229_514_240
    expected_params: int = 213_960_192
    micro_batch_size: int = 24
    grad_accum_steps: int = 16
    output_prefix: tuple[str, str] = ("runs", "pretrain-214m-b200")
    modal_volume: str = "esme-pretrain-214m-b200"


PRETRAIN_PROFILES = {"pretrain_214m_b200": AcceptedRun()}
ALLOWED_GPU_PROFILES = {"H100", "H100!", "H200", "B200"}


@dataclass(frozen=True)
class PretrainLaunchConfig:
    config: PretrainRunConfig
    config_path: Path
    artifact_manifest: dict[str, str]
    train_steps: int
    tokens_per_step: int
    estimated_cost_usd: float

    @property
    def model_config(self) -> BackboneConfig:
        return self.config.model

    @property
    def output_dir(self) -> str:
        return self.config.artifacts.output_dir

    @property
    def train_token_budget(self) -> int:
        return self.config.budgets.train_token_budget

    @property
    def selected_gpu(self) -> str:
        return self.config.runtime.selected_gpu

    @property
    def selected_gpu_profile(self) -> GpuProfile:
        return self.config.runtime.gpu_profiles[self.selected_gpu]


def load_pretrain_config(config_path: Path) -> PretrainLaunchConfig:
    try:
        config = parse_run_config(config_path.read_text(encoding="utf-8"), config_path)
    except OSError as error:
        raise ValueError(f"cannot read config {config_path}: {error}") from error
    return validate_pretrain_config(config, config_path)


def validate_pretrain_payload(payload: object, config_path: Path) -> PretrainLaunchConfig:
    return validate_pretrain_config(parse_run_config(json.dumps(payload), config_path), config_path)


def validate_pretrain_config(config: PretrainRunConfig, config_path: Path) -> PretrainLaunchConfig:
    if config.run_id not in PRETRAIN_PROFILES:
        raise ValueError(f"run_id must be one of {sorted(PRETRAIN_PROFILES)}")
    profile = PRETRAIN_PROFILES[config.run_id]
    if config.run_card != profile.run_card:
        raise ValueError(f"run_card must point to {profile.run_card}")
    if config.requires_approval is not True:
        raise ValueError("requires_approval must be true")
    _validate_dataset(config.dataset)
    _validate_split(config.split)
    _validate_budgets(config.budgets, profile)
    _validate_model(config.model, profile)
    _validate_tokenizer(config.tokenizer)
    _validate_optimizer(config.optimizer, profile)
    _validate_runtime(config.runtime, profile)
    _validate_monitoring(config.monitoring)
    artifact_manifest = _validate_artifacts(config.artifacts, profile)
    _validate_abort_rules(config.abort_rules)
    training = config.optimizer.training
    selected_profile = config.runtime.gpu_profiles[config.runtime.selected_gpu]
    tokens_per_step = (
        training.micro_batch_size * training.grad_accum_steps * config.model.context_length
    )
    train_steps = math.ceil(config.budgets.train_token_budget / tokens_per_step)
    estimated_cost = _estimate_cost_usd(
        train_tokens=config.budgets.train_token_budget,
        projected_tokens_per_second=selected_profile.projected_tokens_per_second,
        usd_per_hour=selected_profile.usd_per_hour,
    )
    if estimated_cost > config.runtime.max_cost_usd:
        raise ValueError("estimated cost exceeds runtime.max_cost_usd")
    return PretrainLaunchConfig(
        config=config,
        config_path=config_path,
        artifact_manifest=artifact_manifest,
        train_steps=train_steps,
        tokens_per_step=tokens_per_step,
        estimated_cost_usd=estimated_cost,
    )


def _estimate_cost_usd(
    *, train_tokens: int, projected_tokens_per_second: float, usd_per_hour: float
) -> float:
    if projected_tokens_per_second <= 0:
        raise ValueError("projected_tokens_per_second must be positive")
    seconds = train_tokens / projected_tokens_per_second
    return seconds * usd_per_hour / 3600.0


def _require_values(payload: BaseModel, expected: Mapping[str, object], label: str) -> None:
    for key, value in expected.items():
        if getattr(payload, key) != value:
            raise ValueError(
                f"{label}.{key}: got {getattr(payload, key)!r}, expected {value!r}. "
                "Restore the accepted run card value."
            )


def _validate_dataset(dataset: DatasetConfig) -> None:
    expected = {
        "name": "fineweb-edu",
        "source": EXPECTED_DATASET_SOURCE,
        "revision": EXPECTED_DATASET_REVISION,
        "subset": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "id_field": "id",
        "license": "odc-by",
        "language": "en",
        "streaming": True,
    }
    _require_values(dataset, expected, "dataset")
    filters = dataset.filters
    if filters.min_int_score != 3:
        raise ValueError("dataset.filters.min_int_score must be 3")


def _validate_split(split: DocumentSplitConfig) -> None:
    expected = {
        "unit": "source_document",
        "rule": "deterministic_hash",
        "hash_field": "id",
        "validation_modulo": 100,
        "validation_remainder": 0,
        "seed": 0,
    }
    _require_values(split, expected, "split")


def _validate_budgets(budgets: TokenBudgets, profile: AcceptedRun) -> None:
    if budgets.train_token_budget != profile.train_token_budget:
        raise ValueError(f"train_token_budget must be {profile.train_token_budget}")
    if budgets.validation_token_budget < 20_000_000:
        raise ValueError("validation_token_budget must be at least 20000000")
    if budgets.tokenizer_training_token_budget < 20_000_000:
        raise ValueError("tokenizer_training_token_budget must be at least 20000000")
    expected_hard = (
        budgets.train_token_budget
        + budgets.validation_token_budget
        + budgets.tokenizer_training_token_budget
    )
    if budgets.hard_read_token_budget != expected_hard:
        raise ValueError("hard_read_token_budget must equal train + validation + tokenizer budgets")


def _validate_tokenizer(tokenizer: TokenizerConfig) -> None:
    if tokenizer.kind != "byte_level_bpe":
        raise ValueError("tokenizer.kind must be byte_level_bpe")
    if tokenizer.trainer != "huggingface-tokenizers":
        raise ValueError("tokenizer.trainer must be huggingface-tokenizers")
    if tokenizer.vocab_size != EXPECTED_TOKENIZER_VOCAB:
        raise ValueError(f"tokenizer.vocab_size must be {EXPECTED_TOKENIZER_VOCAB}")
    if tokenizer.special_tokens != ("<pad>", "<bos>", "<eos>", "<unk>"):
        raise ValueError("tokenizer.special_tokens must be the locked four-token list")
    if tokenizer.train_on != "deterministic_training_split_prefix":
        raise ValueError("tokenizer.train_on must be deterministic_training_split_prefix")
    if tokenizer.output_file != "tokenizer.json":
        raise ValueError("tokenizer.output_file must be tokenizer.json")
    if tokenizer.require_round_trip_checks is not True:
        raise ValueError("tokenizer.require_round_trip_checks must be true")
    if tokenizer.require_coverage_report is not True:
        raise ValueError("tokenizer.require_coverage_report must be true")
    if tokenizer.split_digits is not True:
        raise ValueError("tokenizer.split_digits must be true")


def _validate_model(model: BackboneConfig, profile: AcceptedRun) -> BackboneConfig:
    locked = baseline_config(profile.model_name)
    if model != locked:
        raise ValueError(f"model must match baseline_config({profile.model_name!r}) exactly")
    params = locked.parameter_count()["total"]
    if params != profile.expected_params:
        raise ValueError("model parameter count drifted outside the 214M B200 pretrain target")
    return locked


def _validate_optimizer(optimizer: OptimizerConfig, profile: AcceptedRun) -> None:
    expected = {
        "name": "AdamW",
        "learning_rate": 0.0006,
        "min_lr_ratio": 0.1,
        "warmup_steps": 450,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "lr_schedule": "wsd",
        "decay_fraction": 0.2,
    }
    _require_values(optimizer, expected, "optimizer")
    training = optimizer.training
    expected_training = {
        "micro_batch_size": profile.micro_batch_size,
        "grad_accum_steps": profile.grad_accum_steps,
        "dtype": "bfloat16",
        "use_compile": True,
        "fused_optimizer": True,
        "seed": 0,
    }
    _require_values(training, expected_training, "optimizer.training")


def _validate_runtime(runtime: RuntimeConfig, profile: AcceptedRun) -> None:
    expected = {
        "provider": "modal",
        "precision": "bfloat16",
        "allow_retries": False,
        "modal_volume": profile.modal_volume,
    }
    _require_values(runtime, expected, "runtime")
    if runtime.runtime_spend_stop_usd != runtime.max_cost_usd:
        raise ValueError("runtime.runtime_spend_stop_usd must equal runtime.max_cost_usd")
    if float(runtime.max_cost_usd) <= 0:
        raise ValueError("runtime.max_cost_usd must be positive")
    if int(runtime.timeout_hours) <= 0:
        raise ValueError("runtime.timeout_hours must be positive")
    if int(runtime.timeout_hours) > 24:
        raise ValueError("runtime.timeout_hours must not exceed Modal's 24h function maximum")
    gpu_profiles = runtime.gpu_profiles
    selected_gpu = runtime.selected_gpu
    if selected_gpu not in gpu_profiles:
        raise ValueError("runtime.selected_gpu must name a gpu_profiles entry")
    if selected_gpu not in ALLOWED_GPU_PROFILES:
        raise ValueError(f"runtime.selected_gpu must be one of {sorted(ALLOWED_GPU_PROFILES)}")
    if profile.model_name == "214M" and set(gpu_profiles) != {"H100!", "H200", "B200"}:
        raise ValueError("214M gpu_profiles must cover H100!, H200, and B200")
    for name, gpu_profile in gpu_profiles.items():
        if name not in ALLOWED_GPU_PROFILES:
            raise ValueError(f"unsupported GPU profile: {name}")
        _validate_gpu_profile(name, gpu_profile)


def _validate_gpu_profile(name: str, gpu_profile: GpuProfile) -> None:
    if gpu_profile.modal_gpu != name:
        raise ValueError(f"runtime.gpu_profiles.{name}.modal_gpu must be {name}")
    if float(gpu_profile.usd_per_hour) <= 0:
        raise ValueError(f"runtime.gpu_profiles.{name}.usd_per_hour must be positive")
    if int(gpu_profile.projected_tokens_per_second) <= 0:
        raise ValueError(
            f"runtime.gpu_profiles.{name}.projected_tokens_per_second must be positive"
        )
    if float(gpu_profile.expected_duration_hours) <= 0:
        raise ValueError(f"runtime.gpu_profiles.{name}.expected_duration_hours must be positive")
    if not gpu_profile.projection_source.strip():
        raise ValueError(f"runtime.gpu_profiles.{name}.projection_source must be non-empty")


def _validate_monitoring(monitoring: MonitoringConfig) -> None:
    expected = {
        "log_interval": 10,
        "eval_interval": 500,
        "eval_batches": 20,
        "checkpoint_interval": 500,
        "wandb_project": "esme-pretrain",
    }
    _require_values(monitoring, expected, "monitoring")


def _validate_artifacts(artifacts: ArtifactsConfig, profile: AcceptedRun) -> dict[str, str]:
    output_dir = artifacts.output_dir
    if not output_dir:
        raise ValueError("artifacts.output_dir must be a non-empty relative path")
    output_path = Path(output_dir)
    if output_path.is_absolute() or ".." in output_path.parts:
        raise ValueError("artifacts.output_dir must stay inside the repository")
    if output_path.parts[:2] != profile.output_prefix:
        raise ValueError(f"artifacts.output_dir must be under {'/'.join(profile.output_prefix)}")
    required_files = artifacts.required_files
    if required_files != EXPECTED_ARTIFACTS:
        raise ValueError("artifacts.required_files must match the pretrain evidence manifest")
    return {file_name: str(output_path / file_name) for file_name in EXPECTED_ARTIFACTS}


def _validate_abort_rules(abort_rules: tuple[str, ...]) -> None:
    if not abort_rules:
        raise ValueError("abort_rules must be a non-empty list of rule strings")
    for rule in abort_rules:
        if not rule.strip():
            raise ValueError("abort_rules must contain non-empty strings")
