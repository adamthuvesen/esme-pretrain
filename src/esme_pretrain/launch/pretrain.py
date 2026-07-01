from __future__ import annotations

from pathlib import Path
from typing import Any

from esme_pretrain.pretrain_run import PretrainLaunchConfig

LAUNCH_APPROVAL_FLAG = "--approved"
MODAL_CLIENT_VERSION = "1.5.0"
IMAGE_PACKAGE_PINS: dict[str, str] = {
    "torch": "2.12.1",
    "datasets": "5.0.0",
    "tokenizers": "0.23.1",
    "wandb": "0.27.2",
}


def build_pretrain_dry_run(config: PretrainLaunchConfig) -> dict[str, Any]:
    runtime = config.payload["runtime"]
    selected_profile = config.selected_gpu_profile
    launch_blockers = _launch_safety_blockers(config)
    status = "ready_for_pretrain_launch" if not launch_blockers else "blocked_by_launch_safety"
    return {
        "status": status,
        "requires_approval": True,
        "approval_flag": LAUNCH_APPROVAL_FLAG,
        "launch_blockers": launch_blockers,
        "launch_command": _launch_command(config.config_path, runtime),
        "run_id": config.payload["run_id"],
        "dataset": config.payload["dataset"],
        "split": config.payload["split"],
        "budgets": config.payload["budgets"],
        "tokenizer": config.payload["tokenizer"],
        "model": config.payload["model"],
        "parameter_count": config.model_config.parameter_count(),
        "optimizer": config.payload["optimizer"],
        "runtime": {
            **runtime,
            "selected_gpu_profile": selected_profile,
            "train_steps": config.train_steps,
            "tokens_per_step": config.tokens_per_step,
            "estimated_cost_usd": round(config.estimated_cost_usd, 2),
            "estimated_usd_per_1b_tokens": round(
                config.estimated_cost_usd * 1_000_000_000 / config.train_token_budget, 2
            ),
        },
        "monitoring": config.payload["monitoring"],
        "dependency_pins": {
            "modal": MODAL_CLIENT_VERSION,
            **IMAGE_PACKAGE_PINS,
        },
        "artifacts": {
            "output_dir": config.output_dir,
            "manifest": config.artifact_manifest,
            "required_files": config.payload["artifacts"]["required_files"],
        },
        "abort_rules": config.payload["abort_rules"],
        "will_download_data": False,
        "will_start_modal_job": False,
    }


def _launch_command(config_path: Path, runtime: dict[str, Any]) -> str:
    selected = runtime["selected_gpu"]
    modal_gpu = runtime["gpu_profiles"][selected]["modal_gpu"]
    timeout_hours = runtime["timeout_hours"]
    return (
        f"PRETRAIN_GPU='{modal_gpu}' PRETRAIN_TIMEOUT_HOURS={timeout_hours} "
        f"uv run --with modal=={MODAL_CLIENT_VERSION} "
        "modal run --detach scripts/modal_pretrain.py "
        f"--config {config_path.as_posix()} {LAUNCH_APPROVAL_FLAG} --json"
    )


def _launch_safety_blockers(config: PretrainLaunchConfig) -> list[str]:
    # Over-cost and over-timeout configs never reach this point: validate_pretrain_payload
    # rejects them at load. The only condition that validates but is unsafe to launch is a
    # projected duration that would outlive the Modal function timeout.
    runtime = config.payload["runtime"]
    selected_profile = config.selected_gpu_profile
    blockers: list[str] = []
    if float(selected_profile["expected_duration_hours"]) > float(runtime["timeout_hours"]):
        blockers.append(
            "selected GPU projected duration exceeds the configured Modal function timeout"
        )
    return blockers
