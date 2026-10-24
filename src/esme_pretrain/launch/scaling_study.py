"""Launch scaling-study stages on Modal B200s.

Stages run separately (``smoke``, ``lr-gate``, ``study``), so each spend is
gated on its own. Each arm runs in its own container with a timeout sized from
its FLOPs, at most ``--max-concurrent`` at a time. Every finished run's row is
committed to the study volume as it lands. The study stage also scores the
accepted 214M checkpoint on the same validation batches, in the same fp32 path.
The run card is ``docs/run-cards/scaling-study-b200.md``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from esme_pretrain.launch.gpu_smoke import _check_spend_cap
from esme_pretrain.launch.modal_artifacts import local_git_commit, local_git_dirty
from esme_pretrain.launch.modal_image import build_pretrain_modal_image
from esme_pretrain.study.plan import StudyArm, load_study_plan, study_plan_from_payloads

REPO_ROOT = Path(__file__).resolve().parents[3]
STUDY_GPU = "B200"
STUDY_VOLUME = "esme-pretrain-study"
HELDOUT_VOLUME = "llm-pretrain-214m-b200"
MAX_ARM_TIMEOUT_SECONDS = 5 * 3600
ARM_TIMEOUT_SLACK_SECONDS = 20 * 60
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

    @app.function(gpu=STUDY_GPU, timeout=MAX_ARM_TIMEOUT_SECONDS, volumes={"/study": study_volume})
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
    parser.add_argument("--stage", required=True, choices=["smoke", "lr-gate", "study"])
    parser.add_argument("--arm", action="append", default=[], help="Run only these arms.")
    parser.add_argument("--assumed-tflops", type=float, default=430.0)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--spend-cap-usd", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--approved", action="store_true", help="Required to rent GPUs.")
    parser.add_argument("--check-only", action="store_true", help="Plan and price, no spend.")
    args = parser.parse_args(argv)

    try:
        if not 1 <= args.max_concurrent <= 8:
            raise ValueError("--max-concurrent must be in [1, 8]")
        plan = load_study_plan(args.config)
        if plan.data["volume"] != STUDY_VOLUME:
            raise ValueError(f"config volume {plan.data['volume']!r} is not {STUDY_VOLUME!r}")
        arms = plan.arms_for(args.stage, tuple(args.arm))
        study_payload = json.loads(args.config.read_text(encoding="utf-8"))
        base_payload = json.loads(Path(study_payload["base_config"]).read_text(encoding="utf-8"))
        usd_per_hour = float(base_payload["runtime"]["gpu_profiles"][STUDY_GPU]["usd_per_hour"])
        timeouts = {
            arm.arm: arm_timeout_seconds(arm, assumed_tflops=args.assumed_tflops) for arm in arms
        }
        with_heldout = args.stage == "study" and not args.arm
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
        "assumed_tflops": args.assumed_tflops,
        "estimated_gpu_hours": round(estimated_hours, 2),
        "estimated_cost_usd": round(estimated_hours * usd_per_hour, 2),
        "worst_case_cost_usd": worst_case_cost,
        "spend_cap_usd": args.spend_cap_usd,
        "max_concurrent": args.max_concurrent,
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

    study_run_id = f"{args.stage}-{report['commit'][:7]}-{time.strftime('%Y%m%dT%H%M%S')}"
    params = {"study_run_id": study_run_id, "commit": report["commit"]}
    print(
        f"scaling study: run {study_run_id}; rows are committed to "
        f"{STUDY_VOLUME}:/results/{study_run_id}/ as they finish",
        flush=True,
    )
    started = time.perf_counter()
    outcomes: dict[str, Any] = {}
    with (
        modal.enable_output(),
        app.run(detach=True),
        ThreadPoolExecutor(max_workers=args.max_concurrent) as pool,
    ):
        futures = {
            arm.arm: pool.submit(
                train_arm.with_options(timeout=timeouts[arm.arm]).remote,
                study_payload,
                base_payload,
                arm.arm,
                params,
            )
            for arm in sorted(arms, key=lambda arm: arm.flops, reverse=True)
        }
        if with_heldout:
            futures["heldout"] = pool.submit(
                eval_heldout.remote, study_payload, base_payload, params
            )
        for name, future in futures.items():
            try:
                outcomes[name] = {"status": "ok", **future.result()}
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
    model = load_pretrain_checkpoint(checkpoint).model.config
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
    }
    results_dir = Path("/study/results") / params["study_run_id"]
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "heldout-214m.json").write_text(
        json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
    )
    commit_results()
    return row
