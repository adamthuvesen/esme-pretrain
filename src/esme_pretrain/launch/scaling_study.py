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
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

from esme_pretrain.launch.gpu_smoke import _check_spend_cap
from esme_pretrain.launch.modal_artifacts import local_git_commit, local_git_dirty
from esme_pretrain.launch.modal_image import build_pretrain_modal_image
from esme_pretrain.modeling.backbone import BackboneConfig
from esme_pretrain.study.plan import (
    StudyArm,
    StudyPlan,
    load_study_plan,
    study_plan_from_payloads,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_GPU = "B200"
STUDY_VOLUME = "esme-pretrain-study"
HELDOUT_VOLUME = "llm-pretrain-214m-b200"
MAX_ARM_TIMEOUT_SECONDS = 5 * 3600
ARM_TIMEOUT_SLACK_SECONDS = 20 * 60
# Leaves room under the shared account's GPU quota for other work.
MAX_CONCURRENT_ARMS = 4
LR_GATE_RESULT = REPO_ROOT / "docs/results/scaling-study/lr-gate.json"
# The run card's pre-registered stop rule for the LR gate.
LR_GATE_MAX_GAP_NATS = 0.02
ARM_BUDGET_EXCEEDED_EXIT = 3
HELDOUT_TIMEOUT_SECONDS = 30 * 60
DEFAULT_OUTPUT_DIR = Path("runs/scaling-study")

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
    def train_arm(
        study_payload: dict[str, Any], base_payload: dict[str, Any], arm: str, params: dict
    ) -> dict[str, Any]:
        return train_arm_body(
            study_payload, base_payload, arm, params, commit_results=study_volume.commit
        )

    @app.function(
        gpu=STUDY_GPU,
        timeout=HELDOUT_TIMEOUT_SECONDS,
        volumes={"/study": study_volume, "/accepted": heldout_volume},
    )
    def eval_heldout(
        study_payload: dict[str, Any], base_payload: dict[str, Any], params: dict
    ) -> dict[str, Any]:
        return eval_heldout_body(
            study_payload, base_payload, params, commit_results=study_volume.commit
        )


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
        plan = load_study_plan(args.config)
        if plan.data["volume"] != STUDY_VOLUME:
            raise ValueError(f"config volume {plan.data['volume']!r} is not {STUDY_VOLUME!r}")
        study_payload = json.loads(args.config.read_text(encoding="utf-8"))
        base_payload = json.loads(Path(study_payload["base_config"]).read_text(encoding="utf-8"))
        with_heldout = args.stage == "heldout"
        if plan.heldout["volume"] != HELDOUT_VOLUME:
            raise ValueError(f"heldout volume {plan.heldout['volume']!r} is not {HELDOUT_VOLUME!r}")
        prediction = (
            _written_prediction(
                args.prediction,
                non_embedding_params=BackboneConfig.from_dict(
                    base_payload["model"]
                ).parameter_count()["non_embedding"],
                tokens=int(plan.heldout["tokens"]),
                fit_points=_planned_fit_points(plan),
            )
            if with_heldout
            else None
        )
        arms = () if with_heldout else plan.arms_for(args.stage, tuple(args.arm))
        if args.stage == "study":
            _check_lr_gate(arms, LR_GATE_RESULT)
        usd_per_hour = float(base_payload["runtime"]["gpu_profiles"][STUDY_GPU]["usd_per_hour"])
        timeouts = {
            arm.arm: arm_timeout_seconds(arm, assumed_tflops=args.assumed_tflops) for arm in arms
        }
        reserved_seconds = sum(timeouts.values()) + (HELDOUT_TIMEOUT_SECONDS if with_heldout else 0)
        worst_case_cost = round(reserved_seconds / 3600 * usd_per_hour, 2)
        _check_spend_cap(worst_case_cost, args.spend_cap_usd)
    except (ValueError, KeyError) as error:
        print(f"scaling study refused: {error}", file=sys.stderr)
        return 2

    estimated_hours = sum(arm.flops for arm in arms) / (args.assumed_tflops * 1e12) / 3600
    report = {
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
        "worst_case_cost_usd": worst_case_cost,
        "spend_cap_usd": args.spend_cap_usd,
        "max_concurrent": MAX_CONCURRENT_ARMS,
        "commit": local_git_commit(REPO_ROOT),
        "dirty": local_git_dirty(REPO_ROOT),
    }
    if args.check_only:
        print(json.dumps({"status": "check_ok", **report}, indent=2, sort_keys=True))
        return 0
    if not args.approved:
        print("scaling study refused: pass --approved to rent GPUs", file=sys.stderr)
        return 2
    if report["dirty"]:
        print("scaling study refused: commit the working tree first", file=sys.stderr)
        return 2
    if modal is None:
        print("scaling study failed: modal is not installed", file=sys.stderr)
        return 2

    if with_heldout:
        checkpoint = plan.heldout["checkpoint"]
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
    params = {"study_run_id": study_run_id, "commit": report["commit"], "prediction": prediction}
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
                study_payload,
                base_payload,
                arm.arm,
                {**params, "budget_seconds": timeouts[arm.arm]},
            )
            for arm in sorted(arms, key=lambda arm: arm.flops, reverse=True)
        }
        if with_heldout:
            calls["heldout"] = eval_heldout.spawn(study_payload, base_payload, params)
        print(
            "scaling study: spawned "
            + ", ".join(f"{name}={call.object_id}" for name, call in calls.items()),
            flush=True,
        )
        for name, call in calls.items():
            try:
                outcomes[name] = {"status": "ok", **call.get()}
            except Exception as error:  # noqa: BLE001 - one arm's failure must not hide the rest
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


def _check_lr_gate(arms: tuple[StudyArm, ...], gate_result: Path) -> None:
    """Refuse study arms whose LR the LR-gate rows put past the stop rule."""
    if not gate_result.is_file():
        raise ValueError(f"the study stage needs the LR-gate result: {gate_result}")
    gate = json.loads(gate_result.read_text(encoding="utf-8"))
    val_ce_by_lr = {
        float(row["learning_rate"]): float(row["val_ce"])
        for outcome in gate["outcomes"].values()
        for row in outcome["rows"]
    }
    best = min(val_ce_by_lr.values())
    for arm in arms:
        for run in arm.runs:
            gap = val_ce_by_lr.get(run.learning_rate, float("inf")) - best
            if gap > LR_GATE_MAX_GAP_NATS:
                raise ValueError(
                    f"{arm.arm}/{run.run}: LR {run.learning_rate:g} is {gap:.3f} nats worse "
                    f"than the best LR-gate row in {gate_result}; the stop rule is "
                    f"{LR_GATE_MAX_GAP_NATS} nats, so the LR decision goes back to Adam"
                )


def _planned_fit_points(plan: StudyPlan) -> set[tuple[str, int, int]]:
    """``(arm/run, N, D)`` of every run the pre-registered fit uses."""
    fit_arms = plan.arms_for("study", tuple(plan.heldout["fit_arms"]))
    return {
        (f"{arm.arm}/{run.run}", run.non_embedding_params, run.tokens)
        for arm in fit_arms
        for run in arm.runs
    }


def _written_prediction(
    path: Path | None,
    *,
    non_embedding_params: int,
    tokens: int,
    fit_points: set[tuple[str, int, int]],
) -> dict[str, Any]:
    """The fit's prediction, refused unless it is the pre-registered fit of the 214M point."""
    if path is None or not path.is_file():
        raise ValueError("the heldout stage needs --prediction, the written scaling-fit output")
    from esme_pretrain.postrun.eval_checkpoints import file_sha256

    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload["prediction"]
    if (target["non_embedding_params"], target["tokens"]) != (non_embedding_params, tokens):
        raise ValueError(
            f"{path} predicts N={target['non_embedding_params']}, D={target['tokens']}; "
            f"the held-out point is N={non_embedding_params}, D={tokens}"
        )
    written = [
        (point["arm"], point["non_embedding_params"], point["tokens"])
        for point in payload["points"]
    ]
    if len(written) != len(fit_points) or set(written) != fit_points:
        missing = sorted(label for label, _, _ in fit_points - set(written))
        raise ValueError(
            f"{path} is not fit on the pre-registered {len(fit_points)} points; "
            f"missing or changed: {missing}"
        )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "val_ce": float(payload["prediction"]["val_ce"]),
    }


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


def _stage_shards() -> Path:
    shard_dir = Path("/tmp/shards")
    if not shard_dir.exists():
        # Shuffled windows are random reads; serve them from local disk, not the volume.
        shutil.copytree("/study/shards", shard_dir)
    return shard_dir


def _study_validation(plan: Any, shard_dir: Path, *, vocab_size: int, context_length: int):
    from esme_pretrain.postrun.eval_checkpoints import file_sha256
    from esme_pretrain.study.arm_training import study_validation_batches

    return study_validation_batches(
        plan,
        shard_dir=shard_dir,
        tokenizer_sha256=file_sha256(Path("/study/tokenizer.json")),
        vocab_size=vocab_size,
        context_length=context_length,
        device="cuda",
        eval_token_budget=int(plan.eval["eval_token_budget"]),
        expected_token_batch_sha256=plan.eval["expected_token_batch_sha256"],
    )


def train_arm_body(
    study_payload: dict[str, Any],
    base_payload: dict[str, Any],
    arm_name: str,
    params: dict[str, Any],
    *,
    commit_results: Any,
) -> dict[str, Any]:
    import torch

    from esme_pretrain.study.arm_training import train_study_arm

    budget_timer = _stop_after(params["budget_seconds"], arm_name)
    plan = study_plan_from_payloads(study_payload, base_payload)
    arm = next(arm for arm in plan.arms if arm.arm == arm_name)
    model = arm.runs[0].model
    shard_dir = _stage_shards()
    rows = train_study_arm(
        arm,
        plan,
        shard_dir=shard_dir,
        tokenizer_path=Path("/study/tokenizer.json"),
        work_dir=Path("/tmp/work"),
        results_dir=Path("/study/results") / params["study_run_id"],
        validation_batches=_study_validation(
            plan, shard_dir, vocab_size=model.vocab_size, context_length=model.context_length
        ),
        device="cuda",
        commit_results=commit_results,
        provenance={
            "study_run_id": params["study_run_id"],
            "commit": params["commit"],
            "gpu_name": torch.cuda.get_device_name(0),
        },
    )
    budget_timer.cancel()
    return {"arm": arm_name, "rows": rows}


def eval_heldout_body(
    study_payload: dict[str, Any],
    base_payload: dict[str, Any],
    params: dict[str, Any],
    *,
    commit_results: Any,
) -> dict[str, Any]:
    from esme_pretrain.postrun.eval_checkpoints import file_sha256
    from esme_pretrain.study.arm_training import validation_ce
    from esme_pretrain.training.checkpointing import load_pretrain_checkpoint
    from esme_pretrain.training.eval_batch import hash_eval_batches

    plan = study_plan_from_payloads(study_payload, base_payload)
    checkpoint = Path("/accepted") / plan.heldout["checkpoint"]
    loaded = load_pretrain_checkpoint(checkpoint)
    model = loaded.model.config
    if model != BackboneConfig.from_dict(base_payload["model"]):
        raise ValueError(f"{checkpoint} is not the base config's model: {model}")
    training = base_payload["optimizer"]["training"]
    tokens_per_step = training["micro_batch_size"] * training["grad_accum_steps"]
    trained_tokens = loaded.step * tokens_per_step * model.context_length
    if trained_tokens != int(plan.heldout["tokens"]):
        raise ValueError(
            f"{checkpoint} stopped at step {loaded.step} ({trained_tokens} tokens); "
            f"the held-out point is {plan.heldout['tokens']} tokens"
        )
    batches = _study_validation(
        plan, _stage_shards(), vocab_size=model.vocab_size, context_length=model.context_length
    )
    row = {
        "study_id": plan.study_id,
        "arm": "heldout-214m",
        "checkpoint": plan.heldout["checkpoint"],
        "checkpoint_sha256": file_sha256(checkpoint),
        "non_embedding_params": model.parameter_count()["non_embedding"],
        "tokens": int(plan.heldout["tokens"]),
        "val_ce": validation_ce(checkpoint, batches, device="cuda"),
        "val_token_batch_sha256": hash_eval_batches(batches),
        "study_run_id": params["study_run_id"],
        "commit": params["commit"],
        "prediction": params["prediction"],
    }
    results_dir = Path("/study/results") / params["study_run_id"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "heldout-214m.json").write_text(
        json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
    )
    commit_results()
    return row
