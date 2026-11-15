from __future__ import annotations

from pathlib import Path
from typing import Any

from esme_pretrain.launch.budget import LAUNCH_APPROVAL_FLAG
from esme_pretrain.launch.modal_image import image_package_pins, locked_version
from esme_pretrain.pretrain_config import RuntimeConfig
from esme_pretrain.pretrain_run import PretrainLaunchConfig

MODAL_CLIENT_VERSION = locked_version("modal")


def build_pretrain_dry_run(config: PretrainLaunchConfig) -> dict[str, Any]:
    payload = config.config.model_dump(mode="json", by_alias=True)
    runtime = config.config.runtime
    selected_profile = config.selected_gpu_profile
    launch_blockers = _launch_safety_blockers(config)
    status = "ready_for_pretrain_launch" if not launch_blockers else "blocked_by_launch_safety"
    return {
        "status": status,
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "launch_blockers": launch_blockers,
        "launch_command": _launch_command(config.config_path, runtime),
        "run_id": payload["run_id"],
        "dataset": payload["dataset"],
        "split": payload["split"],
        "budgets": payload["budgets"],
        "tokenizer": payload["tokenizer"],
        "model": payload["model"],
        "parameter_count": config.model_config.parameter_count(),
        "optimizer": payload["optimizer"],
        "runtime": {
            **payload["runtime"],
            "selected_gpu_profile": selected_profile.model_dump(mode="json"),
            "train_steps": config.train_steps,
            "tokens_per_step": config.tokens_per_step,
            "estimated_cost_usd": round(config.estimated_cost_usd, 2),
            "estimated_usd_per_1b_tokens": round(
                config.estimated_cost_usd * 1_000_000_000 / config.train_token_budget, 2
            ),
        },
        "monitoring": payload["monitoring"],
        "dependency_pins": {
            "modal": MODAL_CLIENT_VERSION,
            **image_package_pins(),
        },
        "artifacts": {
            "output_dir": config.output_dir,
            "manifest": config.artifact_manifest,
            "required_files": list(config.config.artifacts.required_files),
        },
        "abort_rules": payload["abort_rules"],
        "will_download_data": False,
        "will_start_modal_job": False,
    }


def _launch_command(config_path: Path, runtime: RuntimeConfig) -> str:
    selected = runtime.selected_gpu
    modal_gpu = runtime.gpu_profiles[selected].modal_gpu
    timeout_hours = runtime.timeout_hours
    return (
        f"PRETRAIN_GPU='{modal_gpu}' PRETRAIN_TIMEOUT_HOURS={timeout_hours} "
        f"uv run --with modal=={MODAL_CLIENT_VERSION} "
        "python scripts/modal_pretrain.py "
        f"--config {config_path.as_posix()} {LAUNCH_APPROVAL_FLAG} --json"
    )


def _launch_safety_blockers(config: PretrainLaunchConfig) -> list[str]:
    # Over-cost and over-timeout configs never reach this point: the RuntimeConfig type
    # rejects them at load. The only condition that validates but is unsafe to launch is a
    # projected duration that would outlive the Modal function timeout.
    runtime = config.config.runtime
    selected_profile = config.selected_gpu_profile
    blockers: list[str] = []
    if float(selected_profile.expected_duration_hours) > float(runtime.timeout_hours):
        blockers.append(
            "selected GPU projected duration exceeds the configured Modal function timeout"
        )
    return blockers


def validate_modal_runtime(config: PretrainLaunchConfig, *, gpu: str, timeout_hours: int) -> None:
    expected_gpu = config.selected_gpu_profile.modal_gpu
    expected_timeout = config.config.runtime.timeout_hours
    if (gpu, timeout_hours) != (expected_gpu, expected_timeout):
        raise ValueError(
            f"Modal runtime differs from the run plan: GPU {gpu!r}, timeout {timeout_hours}h, "
            f"expected GPU {expected_gpu!r}, timeout {expected_timeout}h. "
            f"Use {_launch_command(config.config_path, config.config.runtime)}"
        )
