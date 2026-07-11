from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esme_pretrain.launch.validators import expect_object as _expect_object
from esme_pretrain.launch.validators import require_keys as _require_keys
from esme_pretrain.modeling.backbone import BackboneConfig, baseline_config

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
PRETRAIN_PROFILES: dict[str, dict[str, Any]] = {
    "pretrain_214m_b200": {
        "model_name": "214M",
        "run_card": "docs/run-cards/pretrain-214m-b200.md",
        "train_token_budget": 10_229_514_240,
        "expected_params": 213_960_192,
        "micro_batch_size": 24,
        "grad_accum_steps": 16,
        "output_prefix": ("runs", "pretrain-214m-b200"),
        "modal_volume": "esme-pretrain-214m-b200",
    },
}
ALLOWED_GPU_PROFILES = {"H100", "H100!", "H200", "B200"}


@dataclass(frozen=True)
class PretrainLaunchConfig:
    payload: dict[str, Any]
    config_path: Path
    artifact_manifest: dict[str, str]
    train_steps: int
    tokens_per_step: int
    estimated_cost_usd: float
    model_config: BackboneConfig

    @property
    def output_dir(self) -> str:
        return str(self.payload["artifacts"]["output_dir"])

    @property
    def train_token_budget(self) -> int:
        return int(self.payload["budgets"]["train_token_budget"])

    @property
    def selected_gpu(self) -> str:
        return str(self.payload["runtime"]["selected_gpu"])

    @property
    def selected_gpu_profile(self) -> dict[str, Any]:
        profiles = self.payload["runtime"]["gpu_profiles"]
        return dict(profiles[self.selected_gpu])


def load_pretrain_config(config_path: Path) -> PretrainLaunchConfig:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"config path does not exist: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"config is not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    return validate_pretrain_payload(payload, config_path)


def validate_pretrain_payload(payload: dict[str, Any], config_path: Path) -> PretrainLaunchConfig:
    _require_keys(
        payload,
        {
            "run_id",
            "run_card",
            "requires_approval",
            "dataset",
            "split",
            "budgets",
            "tokenizer",
            "model",
            "optimizer",
            "runtime",
            "monitoring",
            "artifacts",
            "abort_rules",
        },
        "config",
    )
    run_id = payload["run_id"]
    if run_id not in PRETRAIN_PROFILES:
        raise ValueError(f"run_id must be one of {sorted(PRETRAIN_PROFILES)}")
    profile = PRETRAIN_PROFILES[run_id]
    if payload["run_card"] != profile["run_card"]:
        raise ValueError(f"run_card must point to {profile['run_card']}")
    if payload["requires_approval"] is not True:
        raise ValueError("requires_approval must be true")

    _validate_dataset(_expect_object(payload["dataset"], "dataset"))
    _validate_split(_expect_object(payload["split"], "split"))
    _validate_budgets(_expect_object(payload["budgets"], "budgets"), profile)
    model_config = _validate_model(_expect_object(payload["model"], "model"), profile)
    _validate_tokenizer(_expect_object(payload["tokenizer"], "tokenizer"))
    _validate_optimizer(_expect_object(payload["optimizer"], "optimizer"), profile)
    _validate_runtime(_expect_object(payload["runtime"], "runtime"), profile)
    _validate_monitoring(_expect_object(payload["monitoring"], "monitoring"))
    artifact_manifest = _validate_artifacts(
        _expect_object(payload["artifacts"], "artifacts"), profile
    )
    _validate_abort_rules(payload["abort_rules"])

    training = payload["optimizer"]["training"]
    runtime = payload["runtime"]
    selected_profile = runtime["gpu_profiles"][runtime["selected_gpu"]]
    tokens_per_step = (
        int(training["micro_batch_size"])
        * int(training["grad_accum_steps"])
        * int(payload["model"]["context_length"])
    )
    train_steps = math.ceil(int(payload["budgets"]["train_token_budget"]) / tokens_per_step)
    estimated_cost = _estimate_cost_usd(
        train_tokens=int(payload["budgets"]["train_token_budget"]),
        projected_tokens_per_second=float(selected_profile["projected_tokens_per_second"]),
        usd_per_hour=float(selected_profile["usd_per_hour"]),
    )
    if estimated_cost > float(payload["runtime"]["max_cost_usd"]):
        raise ValueError("estimated cost exceeds runtime.max_cost_usd")

    return PretrainLaunchConfig(
        payload=payload,
        config_path=config_path,
        artifact_manifest=artifact_manifest,
        train_steps=train_steps,
        tokens_per_step=tokens_per_step,
        estimated_cost_usd=estimated_cost,
        model_config=model_config,
    )


def _estimate_cost_usd(
    *, train_tokens: int, projected_tokens_per_second: float, usd_per_hour: float
) -> float:
    if projected_tokens_per_second <= 0:
        raise ValueError("projected_tokens_per_second must be positive")
    seconds = train_tokens / projected_tokens_per_second
    return seconds * usd_per_hour / 3600.0


def _require_values(payload: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, value in expected.items():
        if payload[key] != value:
            raise ValueError(f"{label}.{key} must be {value}")


def _validate_dataset(dataset: dict[str, Any]) -> None:
    _require_keys(
        dataset,
        {
            "name",
            "source",
            "revision",
            "subset",
            "split",
            "text_field",
            "id_field",
            "license",
            "language",
            "streaming",
            "filters",
        },
        "dataset",
    )
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
    filters = _expect_object(dataset["filters"], "dataset.filters")
    _require_keys(filters, {"min_int_score"}, "dataset.filters")
    if filters["min_int_score"] != 3:
        raise ValueError("dataset.filters.min_int_score must be 3")


def _validate_split(split: dict[str, Any]) -> None:
    _require_keys(
        split,
        {"unit", "rule", "hash_field", "validation_modulo", "validation_remainder", "seed"},
        "split",
    )
    expected = {
        "unit": "source_document",
        "rule": "deterministic_hash",
        "hash_field": "id",
        "validation_modulo": 100,
        "validation_remainder": 0,
        "seed": 0,
    }
    _require_values(split, expected, "split")


def _validate_budgets(budgets: dict[str, Any], profile: dict[str, Any]) -> None:
    _require_keys(
        budgets,
        {
            "train_token_budget",
            "validation_token_budget",
            "tokenizer_training_token_budget",
            "hard_read_token_budget",
        },
        "budgets",
    )
    if budgets["train_token_budget"] != profile["train_token_budget"]:
        raise ValueError(f"train_token_budget must be {profile['train_token_budget']}")
    if not isinstance(budgets["validation_token_budget"], int) or (
        budgets["validation_token_budget"] < 20_000_000
    ):
        raise ValueError("validation_token_budget must be at least 20000000")
    if not isinstance(budgets["tokenizer_training_token_budget"], int) or (
        budgets["tokenizer_training_token_budget"] < 20_000_000
    ):
        raise ValueError("tokenizer_training_token_budget must be at least 20000000")
    expected_hard = (
        budgets["train_token_budget"]
        + budgets["validation_token_budget"]
        + budgets["tokenizer_training_token_budget"]
    )
    if budgets["hard_read_token_budget"] != expected_hard:
        raise ValueError("hard_read_token_budget must equal train + validation + tokenizer budgets")


def _validate_tokenizer(tokenizer: dict[str, Any]) -> None:
    _require_keys(
        tokenizer,
        {
            "kind",
            "trainer",
            "vocab_size",
            "special_tokens",
            "train_on",
            "output_file",
            "require_round_trip_checks",
            "require_coverage_report",
            "split_digits",
        },
        "tokenizer",
    )
    if tokenizer["kind"] != "byte_level_bpe":
        raise ValueError("tokenizer.kind must be byte_level_bpe")
    if tokenizer["trainer"] != "huggingface-tokenizers":
        raise ValueError("tokenizer.trainer must be huggingface-tokenizers")
    if tokenizer["vocab_size"] != EXPECTED_TOKENIZER_VOCAB:
        raise ValueError(f"tokenizer.vocab_size must be {EXPECTED_TOKENIZER_VOCAB}")
    if tokenizer["special_tokens"] != ["<pad>", "<bos>", "<eos>", "<unk>"]:
        raise ValueError("tokenizer.special_tokens must be the locked four-token list")
    if tokenizer["train_on"] != "deterministic_training_split_prefix":
        raise ValueError("tokenizer.train_on must be deterministic_training_split_prefix")
    if tokenizer["output_file"] != "tokenizer.json":
        raise ValueError("tokenizer.output_file must be tokenizer.json")
    if tokenizer["require_round_trip_checks"] is not True:
        raise ValueError("tokenizer.require_round_trip_checks must be true")
    if tokenizer["require_coverage_report"] is not True:
        raise ValueError("tokenizer.require_coverage_report must be true")
    if tokenizer["split_digits"] is not True:
        raise ValueError("tokenizer.split_digits must be true")


def _validate_model(model: dict[str, Any], profile: dict[str, Any]) -> BackboneConfig:
    locked = baseline_config(profile["model_name"])
    expected = locked.to_dict()
    if model != expected:
        raise ValueError(f"model must match baseline_config({profile['model_name']!r}) exactly")
    params = locked.parameter_count()["total"]
    if params != profile["expected_params"]:
        raise ValueError("model parameter count drifted outside the 214M B200 pretrain target")
    return locked


def _validate_optimizer(optimizer: dict[str, Any], profile: dict[str, Any]) -> None:
    _require_keys(
        optimizer,
        {
            "name",
            "learning_rate",
            "min_lr_ratio",
            "warmup_steps",
            "weight_decay",
            "grad_clip",
            "lr_schedule",
            "decay_fraction",
            "training",
        },
        "optimizer",
    )
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
    training = _expect_object(optimizer["training"], "optimizer.training")
    _require_keys(
        training,
        {
            "micro_batch_size",
            "grad_accum_steps",
            "dtype",
            "compile",
            "fused_optimizer",
            "seed",
        },
        "optimizer.training",
    )
    expected_training = {
        "micro_batch_size": profile["micro_batch_size"],
        "grad_accum_steps": profile["grad_accum_steps"],
        "dtype": "bfloat16",
        "compile": True,
        "fused_optimizer": True,
        "seed": 0,
    }
    _require_values(training, expected_training, "optimizer.training")


def _validate_runtime(runtime: dict[str, Any], profile: dict[str, Any]) -> None:
    _require_keys(
        runtime,
        {
            "provider",
            "selected_gpu",
            "precision",
            "gpu_profiles",
            "max_cost_usd",
            "runtime_spend_stop_usd",
            "allow_retries",
            "modal_volume",
            "timeout_hours",
        },
        "runtime",
    )
    expected = {
        "provider": "modal",
        "precision": "bfloat16",
        "allow_retries": False,
        "modal_volume": profile["modal_volume"],
    }
    _require_values(runtime, expected, "runtime")
    if runtime["runtime_spend_stop_usd"] != runtime["max_cost_usd"]:
        raise ValueError("runtime.runtime_spend_stop_usd must equal runtime.max_cost_usd")
    if float(runtime["max_cost_usd"]) <= 0:
        raise ValueError("runtime.max_cost_usd must be positive")
    if int(runtime["timeout_hours"]) <= 0:
        raise ValueError("runtime.timeout_hours must be positive")
    if int(runtime["timeout_hours"]) > 24:
        raise ValueError("runtime.timeout_hours must not exceed Modal's 24h function maximum")
    gpu_profiles = _expect_object(runtime["gpu_profiles"], "runtime.gpu_profiles")
    selected_gpu = runtime["selected_gpu"]
    if selected_gpu not in gpu_profiles:
        raise ValueError("runtime.selected_gpu must name a gpu_profiles entry")
    if selected_gpu not in ALLOWED_GPU_PROFILES:
        raise ValueError(f"runtime.selected_gpu must be one of {sorted(ALLOWED_GPU_PROFILES)}")
    if profile["model_name"] == "214M" and set(gpu_profiles) != {"H100!", "H200", "B200"}:
        raise ValueError("214M gpu_profiles must cover H100!, H200, and B200")
    for name, gpu_profile in gpu_profiles.items():
        if name not in ALLOWED_GPU_PROFILES:
            raise ValueError(f"unsupported GPU profile: {name}")
        _validate_gpu_profile(name, _expect_object(gpu_profile, f"runtime.gpu_profiles.{name}"))


def _validate_gpu_profile(name: str, gpu_profile: dict[str, Any]) -> None:
    _require_keys(
        gpu_profile,
        {
            "modal_gpu",
            "usd_per_hour",
            "projected_tokens_per_second",
            "expected_duration_hours",
            "projection_source",
            "measured",
        },
        f"runtime.gpu_profiles.{name}",
    )
    if gpu_profile["modal_gpu"] != name:
        raise ValueError(f"runtime.gpu_profiles.{name}.modal_gpu must be {name}")
    if float(gpu_profile["usd_per_hour"]) <= 0:
        raise ValueError(f"runtime.gpu_profiles.{name}.usd_per_hour must be positive")
    if int(gpu_profile["projected_tokens_per_second"]) <= 0:
        raise ValueError(
            f"runtime.gpu_profiles.{name}.projected_tokens_per_second must be positive"
        )
    if float(gpu_profile["expected_duration_hours"]) <= 0:
        raise ValueError(f"runtime.gpu_profiles.{name}.expected_duration_hours must be positive")
    if (
        not isinstance(gpu_profile["projection_source"], str)
        or not gpu_profile["projection_source"].strip()
    ):
        raise ValueError(f"runtime.gpu_profiles.{name}.projection_source must be non-empty")
    if not isinstance(gpu_profile["measured"], bool):
        raise ValueError(f"runtime.gpu_profiles.{name}.measured must be boolean")


def _validate_monitoring(monitoring: dict[str, Any]) -> None:
    _require_keys(
        monitoring,
        {
            "log_interval",
            "eval_interval",
            "eval_batches",
            "checkpoint_interval",
            "wandb_project",
        },
        "monitoring",
    )
    expected = {
        "log_interval": 10,
        "eval_interval": 500,
        "eval_batches": 20,
        "checkpoint_interval": 500,
        "wandb_project": "esme-pretrain",
    }
    _require_values(monitoring, expected, "monitoring")


def _validate_artifacts(artifacts: dict[str, Any], profile: dict[str, Any]) -> dict[str, str]:
    _require_keys(artifacts, {"output_dir", "required_files"}, "artifacts")
    output_dir = artifacts["output_dir"]
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("artifacts.output_dir must be a non-empty relative path")
    output_path = Path(output_dir)
    if output_path.is_absolute() or ".." in output_path.parts:
        raise ValueError("artifacts.output_dir must stay inside the repository")
    if output_path.parts[:2] != profile["output_prefix"]:
        raise ValueError(f"artifacts.output_dir must be under {'/'.join(profile['output_prefix'])}")
    required_files = artifacts["required_files"]
    if not isinstance(required_files, list) or tuple(required_files) != EXPECTED_ARTIFACTS:
        raise ValueError("artifacts.required_files must match the pretrain evidence manifest")
    return {file_name: str(output_path / file_name) for file_name in EXPECTED_ARTIFACTS}


def _validate_abort_rules(abort_rules: Any) -> None:
    if not isinstance(abort_rules, list) or not abort_rules:
        raise ValueError("abort_rules must be a non-empty list of rule strings")
    for rule in abort_rules:
        if not isinstance(rule, str) or not rule.strip():
            raise ValueError("abort_rules must contain non-empty strings")
