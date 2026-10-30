"""Launch scaling-study stages on Modal B200s.

Stages run separately (``smoke``, ``lr-gate``, ``study``, ``heldout``), so each
spend is gated on its own. Every arm is spawned at launch, one container each, and Modal
runs at most ``MAX_CONCURRENT_ARMS`` at once, so a client disconnect neither
stops nor drops an arm. Each container stops itself when its FLOPs-sized budget
runs out. Every finished run's row is committed to the study volume as it lands.
The ``heldout`` stage scores the accepted 214M checkpoint on the same validation
batches, in the same fp32 path, and refuses to run before the fit's prediction is
written: the run card pre-registers that the actual value is read only after it.
The run card is ``docs/run-cards/scaling-study-b200.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from esme_pretrain.data.token_shards import load_token_shard_manifest
from esme_pretrain.launch.budget import (
    LaunchBudget,
    approved_command,
    check_spend_cap,
    require_approval,
)
from esme_pretrain.launch.modal_artifacts import local_git_commit, local_git_dirty
from esme_pretrain.launch.modal_image import build_pretrain_modal_image
from esme_pretrain.pretrain_config import PretrainRunConfig
from esme_pretrain.pretrain_run import load_pretrain_config
from esme_pretrain.study.data_paths import StudyDataPaths, resolve_study_data
from esme_pretrain.study.lr_gate import check_lr_gate
from esme_pretrain.study.plan import (
    StudyArm,
    StudyConfig,
    StudyPlan,
    load_study_config,
    study_plan,
)
from esme_pretrain.study.prediction import (
    PredictionReference,
    written_prediction,
)
from esme_pretrain.study.results import StudyProvenance

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_GPU = "B200"
STUDY_VOLUME = "esme-pretrain-study"
HELDOUT_VOLUME = "llm-pretrain-214m-b200"
MAX_ARM_TIMEOUT_SECONDS = 5 * 3600
ARM_TIMEOUT_SLACK_SECONDS = 20 * 60
# Leaves room under the shared account's GPU quota for other work.
MAX_CONCURRENT_ARMS = 4
ARM_BUDGET_EXCEEDED_EXIT = 3
HELDOUT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_OUTPUT_DIR = Path("runs/scaling-study")


class PlannedStudyArm(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    arm: StudyArm
    budget: LaunchBudget


class StudyLaunchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    study: StudyPlan
    data_paths: StudyDataPaths
    base: PretrainRunConfig
    stage: str
    arms: tuple[PlannedStudyArm, ...]
    heldout_budget: LaunchBudget | None
    prediction: PredictionReference | None
    assumed_tflops: float
    spend_cap_usd: float

    @property
    def worst_case_cost_usd(self) -> float:
        return sum(work.budget.worst_case_cost_usd for work in self.arms) + (
            self.heldout_budget.worst_case_cost_usd if self.heldout_budget is not None else 0
        )


def build_study_launch_plan(
    config: StudyConfig,
    base: PretrainRunConfig,
    *,
    stage: str,
    names: tuple[str, ...],
    assumed_tflops: float,
    spend_cap_usd: float,
    prediction_path: Path | None = None,
) -> StudyLaunchPlan:
    if not math.isfinite(assumed_tflops) or assumed_tflops <= 0:
        raise ValueError("--assumed-tflops must be positive and finite")
    if config.data.volume != STUDY_VOLUME:
        raise ValueError(f"data.volume must be {STUDY_VOLUME}")
    if config.heldout.volume != HELDOUT_VOLUME:
        raise ValueError(f"heldout.volume must be {HELDOUT_VOLUME}")
    plan = study_plan(config, base.model)
    with_heldout = stage == "heldout"
    if with_heldout and names:
        raise ValueError("--arm cannot select runs in the heldout stage")
    arms = () if with_heldout else plan.arms_for(stage, names)
    if not with_heldout and not arms:
        raise ValueError(f"stage {stage!r} has no arms")
    if stage == "study":
        check_lr_gate(arms, plan)
    rate = base.runtime.gpu_profiles[STUDY_GPU].usd_per_hour
    prediction = (
        written_prediction(
            prediction_path,
            non_embedding_params=base.model.parameter_count()["non_embedding"],
            tokens=plan.heldout.tokens,
            plan=plan,
        )
        if with_heldout
        else None
    )
    launch_plan = StudyLaunchPlan(
        study=plan,
        data_paths=resolve_study_data(plan.data, Path("/study")),
        base=base,
        stage=stage,
        arms=tuple(
            PlannedStudyArm(
                arm=arm,
                budget=LaunchBudget(
                    STUDY_GPU,
                    arm_timeout_seconds(arm, assumed_tflops=assumed_tflops),
                    rate,
                    spend_cap_usd,
                ),
            )
            for arm in arms
        ),
        heldout_budget=LaunchBudget(STUDY_GPU, HELDOUT_TIMEOUT_SECONDS, rate, spend_cap_usd)
        if with_heldout
        else None,
        prediction=prediction,
        assumed_tflops=assumed_tflops,
        spend_cap_usd=spend_cap_usd,
    )
    check_spend_cap(launch_plan.worst_case_cost_usd, spend_cap_usd)
    return launch_plan


try:
    import modal
except ImportError:  # pragma: no cover - Modal is supplied by the dev extra.
    modal = None

if modal is not None:  # pragma: no cover - exercised on Modal, not in unit tests.
    image = build_pretrain_modal_image(repo_root_src=str(REPO_ROOT / "src"), modal_module=modal)
    app = modal.App("esme-pretrain-scaling-study", image=image)
    study_volume = modal.Volume.from_name(STUDY_VOLUME)
    heldout_volume = modal.Volume.from_name(HELDOUT_VOLUME)

    @app.function(
        gpu=STUDY_GPU,
        timeout=MAX_ARM_TIMEOUT_SECONDS,
        volumes={"/study": study_volume},
        max_containers=MAX_CONCURRENT_ARMS,
        # One arm per container: a reused container would carry the previous arm's
        # budget timer into the next arm.
        single_use_containers=True,
    )
    def train_arm(plan_payload: dict[str, Any], arm: str, params: dict[str, Any]) -> dict[str, Any]:
        return train_arm_body(plan_payload, arm, params, commit_results=study_volume.commit)

    @app.function(
        gpu=STUDY_GPU,
        timeout=HELDOUT_TIMEOUT_SECONDS,
        volumes={"/study": study_volume, "/accepted": heldout_volume},
    )
    def eval_heldout(plan_payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return eval_heldout_body(plan_payload, params, commit_results=study_volume.commit)


def arm_timeout_seconds(arm: StudyArm, *, assumed_tflops: float) -> int:
    estimate = arm.flops / (assumed_tflops * 1e12)
    return min(MAX_ARM_TIMEOUT_SECONDS, int(2 * estimate + ARM_TIMEOUT_SLACK_SECONDS))


def launch(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one scaling-study stage on Modal.")
    parser.add_argument("--config", type=Path, default=Path("configs/scaling_study.json"))
    parser.add_argument("--stage", required=True, choices=["smoke", "lr-gate", "study", "heldout"])
    parser.add_argument(
        "--prediction",
        type=Path,
        help="The written scaling-fit output; the heldout stage refuses to run without it.",
    )
    parser.add_argument("--arm", action="append", default=[], help="Run only these arms.")
    parser.add_argument("--assumed-tflops", type=float, default=430.0)
    parser.add_argument("--spend-cap-usd", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--approved", action="store_true", help="Required to rent GPUs.")
    parser.add_argument("--check-only", action="store_true", help="Plan and price, no spend.")
    args = parser.parse_args(argv)

    try:
        study_config = load_study_config(args.config)
        base = load_pretrain_config(Path(study_config.base_config))
        launch_plan = build_study_launch_plan(
            study_config,
            base.config,
            stage=args.stage,
            names=tuple(args.arm),
            assumed_tflops=args.assumed_tflops,
            spend_cap_usd=args.spend_cap_usd,
            prediction_path=args.prediction,
        )
    except (ValueError, KeyError) as error:
        print(f"scaling study refused: {error}", file=sys.stderr)
        return 2

    plan = launch_plan.study
    arms = tuple(work.arm for work in launch_plan.arms)
    timeouts = {work.arm.arm: work.budget.timeout_seconds for work in launch_plan.arms}
    with_heldout = launch_plan.heldout_budget is not None
    prediction = launch_plan.prediction.model_dump(mode="json") if launch_plan.prediction else None
    usd_per_hour = launch_plan.base.runtime.gpu_profiles[STUDY_GPU].usd_per_hour
    estimated_hours = sum(arm.flops for arm in arms) / (launch_plan.assumed_tflops * 1e12) / 3600
    report: dict[str, Any] = {
        "stage": args.stage,
        "gpu": STUDY_GPU,
        "arms": [
            {
                "arm": arm.arm,
                "runs": [run.run for run in arm.runs],
                "flops": arm.flops,
                "timeout_hours": round(timeouts[arm.arm] / 3600, 2),
            }
            for arm in arms
        ],
        "heldout_eval": with_heldout,
        "prediction": prediction,
        "assumed_tflops": args.assumed_tflops,
        "estimated_gpu_hours": round(estimated_hours, 2),
        "estimated_cost_usd": round(estimated_hours * usd_per_hour, 2),
        "worst_case_cost_usd": round(launch_plan.worst_case_cost_usd, 2),
        "spend_cap_usd": args.spend_cap_usd,
        "max_concurrent": MAX_CONCURRENT_ARMS,
        "commit": local_git_commit(REPO_ROOT),
        "dirty": local_git_dirty(REPO_ROOT),
    }
    if args.check_only:
        print(json.dumps({"status": "check_ok", **report}, indent=2, sort_keys=True))
        return 0
    command = approved_command("scripts/modal_scaling_study.py", argv)
    if not require_approval(args.approved, run="scaling study", command=command):
        return 2
    if with_heldout and not _committed_in_head(args.prediction):
        print(
            f"scaling study refused: commit {args.prediction} before the held-out eval "
            "reveals the 214M loss",
            file=sys.stderr,
        )
        return 2
    if report["dirty"]:
        print("scaling study refused: commit the working tree first", file=sys.stderr)
        return 2
    if modal is None:
        print("scaling study failed: modal is not installed", file=sys.stderr)
        return 2

    if with_heldout:
        checkpoint = plan.heldout.checkpoint
        listed = {
            entry.path
            for entry in modal.Volume.from_name(HELDOUT_VOLUME).listdir(
                str(Path(checkpoint).parent)
            )
        }
        if checkpoint not in listed:
            print(
                f"scaling study failed: {HELDOUT_VOLUME}:{checkpoint} does not exist",
                file=sys.stderr,
            )
            return 2

    study_run_id = f"{args.stage}-{report['commit'][:7]}-{time.strftime('%Y%m%dT%H%M%S')}"
    params = {"study_run_id": study_run_id, "commit": report["commit"]}
    print(
        f"scaling study: run {study_run_id}; rows are committed to "
        f"{STUDY_VOLUME}:/results/{study_run_id}/ as they finish",
        flush=True,
    )
    started = time.perf_counter()
    outcomes: dict[str, Any] = {}
    # Detached, and every arm spawned before waiting: if this client disconnects, the
    # queued arms still start and finished rows are already on the volume.
    with modal.enable_output(), app.run(detach=True):
        calls = {
            arm.arm: train_arm.spawn(
                launch_plan.model_dump(mode="json", by_alias=True),
                arm.arm,
                params,
            )
            for arm in sorted(arms, key=lambda arm: arm.flops, reverse=True)
        }
        if with_heldout:
            calls["heldout"] = eval_heldout.spawn(
                launch_plan.model_dump(mode="json", by_alias=True), params
            )
        print(
            "scaling study: spawned "
            + ", ".join(f"{name}={call.object_id}" for name, call in calls.items()),
            flush=True,
        )
        for name, call in calls.items():
            try:
                outcomes[name] = {"status": "ok", **call.get()}
            except Exception as error:  # one arm's failure must not hide the rest
                outcomes[name] = {"status": "failed", "error": repr(error)[-2000:]}

    summary = {
        **report,
        "study_run_id": study_run_id,
        "outcomes": outcomes,
        "client_wall_seconds": time.perf_counter() - started,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{study_run_id}.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**summary, "local_output_path": str(output_path)}, indent=2, sort_keys=True))
    failed = [name for name, outcome in outcomes.items() if outcome["status"] != "ok"]
    if failed:
        print(f"scaling study: arms did not complete: {failed}", file=sys.stderr)
        return 1
    return 0


def _committed_in_head(path: Path) -> bool:
    """True when ``path`` is a tracked file whose HEAD blob matches its contents."""
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        return False
    head_blob = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{resolved.relative_to(REPO_ROOT)}"],
        capture_output=True,
        check=False,
    )
    return head_blob.returncode == 0 and head_blob.stdout == resolved.read_bytes()


def _stop_after(budget_seconds: int, arm_name: str) -> threading.Timer:
    """End this container once the arm's budget is spent; finished rows are already saved.

    All arms share one Modal function (so ``max_containers`` caps them together),
    which means one function timeout; this bounds each arm's spend separately.
    """

    def stop() -> None:
        print(f"scaling study: {arm_name} exceeded its {budget_seconds} s budget", flush=True)
        os._exit(ARM_BUDGET_EXCEEDED_EXIT)

    timer = threading.Timer(budget_seconds, stop)
    timer.daemon = True
    timer.start()
    return timer


def stage_study_data(paths: StudyDataPaths, local_root: Path) -> StudyDataPaths:
    local_root = local_root.resolve()
    files = {paths.tokenizer, paths.train_manifest, paths.validation_manifest}
    for manifest_path in (paths.train_manifest, paths.validation_manifest):
        manifest = load_token_shard_manifest(manifest_path)
        files.update(shard.path for shard in manifest.shards)
    for source in sorted(files):
        resolved = source.resolve()
        if not resolved.is_relative_to(paths.root):
            raise ValueError(
                f"study input {source} resolves outside {paths.root}: {resolved}. "
                "Correct the configured path or shard manifest."
            )
        if not resolved.is_file():
            raise ValueError(
                f"study input {source} is not a file. "
                "Restore the configured tokenizer, manifest, or shard."
            )
        relative = resolved.relative_to(paths.root)
        destination = local_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return StudyDataPaths(
        root=local_root,
        tokenizer=local_root / paths.tokenizer.relative_to(paths.root),
        train_manifest=local_root / paths.train_manifest.relative_to(paths.root),
        validation_manifest=local_root / paths.validation_manifest.relative_to(paths.root),
    )


def _study_validation(
    plan: StudyPlan, paths: StudyDataPaths, *, vocab_size: int, context_length: int
):
    from esme_pretrain.study.arm_training import study_validation_batches

    return study_validation_batches(
        plan,
        data_paths=paths,
        vocab_size=vocab_size,
        context_length=context_length,
        device="cuda",
    )


def train_arm_body(
    plan_payload: dict[str, Any],
    arm_name: str,
    params: dict[str, Any],
    *,
    commit_results: Callable[[], None],
) -> dict[str, Any]:
    import torch

    from esme_pretrain.study.arm_training import train_study_arm

    launch_plan = StudyLaunchPlan.model_validate_json(json.dumps(plan_payload))
    plan = launch_plan.study
    work = next(work for work in launch_plan.arms if work.arm.arm == arm_name)
    arm = work.arm
    budget_timer = _stop_after(work.budget.timeout_seconds, arm_name)
    model = arm.runs[0].model
    data_paths = stage_study_data(launch_plan.data_paths, Path("/tmp/study-inputs"))
    rows = train_study_arm(
        arm,
        plan,
        data_paths=data_paths,
        work_dir=Path("/tmp/work"),
        results_dir=Path("/study/results") / params["study_run_id"],
        validation_batches=_study_validation(
            plan, data_paths, vocab_size=model.vocab_size, context_length=model.context_length
        ),
        device="cuda",
        commit_results=commit_results,
        provenance=StudyProvenance(
            study_run_id=params["study_run_id"],
            commit=params["commit"],
            gpu_name=torch.cuda.get_device_name(0),
        ),
    )
    budget_timer.cancel()
    return {
        "arm": arm_name,
        "rows": [row.model_dump(mode="json", exclude_unset=True) for row in rows],
    }


def eval_heldout_body(
    plan_payload: dict[str, Any],
    params: dict[str, Any],
    *,
    commit_results: Callable[[], None],
) -> dict[str, Any]:
    from esme_pretrain.artifacts.hashes import file_sha256
    from esme_pretrain.study.arm_training import validation_ce
    from esme_pretrain.training.checkpointing import load_pretrain_checkpoint
    from esme_pretrain.training.eval_batch import hash_eval_batches

    launch_plan = StudyLaunchPlan.model_validate_json(json.dumps(plan_payload))
    plan = launch_plan.study
    checkpoint = Path("/accepted") / plan.heldout.checkpoint
    loaded = load_pretrain_checkpoint(checkpoint)
    model = loaded.model.config
    if model != launch_plan.base.model:
        raise ValueError(f"{checkpoint} is not the base config's model: {model}")
    training = launch_plan.base.optimizer.training
    tokens_per_step = training.micro_batch_size * training.grad_accum_steps
    trained_tokens = loaded.step * tokens_per_step * model.context_length
    if trained_tokens != int(plan.heldout.tokens):
        raise ValueError(
            f"{checkpoint} stopped at step {loaded.step} ({trained_tokens} tokens); "
            f"the held-out point is {plan.heldout.tokens} tokens"
        )
    batches = _study_validation(
        plan,
        stage_study_data(launch_plan.data_paths, Path("/tmp/study-inputs")),
        vocab_size=model.vocab_size,
        context_length=model.context_length,
    )
    row = {
        "study_id": plan.study_id,
        "arm": "heldout-214m",
        "checkpoint": plan.heldout.checkpoint,
        "checkpoint_sha256": file_sha256(checkpoint),
        "non_embedding_params": model.parameter_count()["non_embedding"],
        "tokens": int(plan.heldout.tokens),
        "val_ce": validation_ce(checkpoint, batches, device="cuda"),
        "val_token_batch_sha256": hash_eval_batches(batches),
        "val_eval_tokens": sum(int(batch.input_ids.numel()) for batch in batches),
        "val_eval_batches": len(batches),
        "study_run_id": params["study_run_id"],
        "commit": params["commit"],
        "prediction": launch_plan.prediction.model_dump(mode="json")
        if launch_plan.prediction
        else None,
    }
    results_dir = Path("/study/results") / params["study_run_id"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "heldout-214m.json").write_text(
        json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
    )
    commit_results()
    return row
