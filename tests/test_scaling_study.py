from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from esme_pretrain.data.token_shards import write_token_shards
from esme_pretrain.launch import scaling_study
from esme_pretrain.pretrain_config import PretrainRunConfig
from esme_pretrain.study.arm_training import study_validation_batches, train_study_arm
from esme_pretrain.study.data_paths import resolve_study_data
from esme_pretrain.study.lr_gate import LR_GATE_RESULTS, check_lr_gate
from esme_pretrain.study.plan import StudyConfig, load_study_plan, study_plan
from esme_pretrain.study.prediction import fit_settings
from esme_pretrain.study.results import load_study_fit_input
from esme_pretrain.torch import torch
from esme_pretrain.training.eval_batch import hash_eval_batches
from esme_pretrain.training.shard_loader import (
    fixed_validation_batches,
    fixed_validation_token_count,
)

CONFIG = Path("configs/scaling_study.json")
VOCAB_SIZE = 128


def test_real_study_plan_branches_fork_inside_the_main_runs_peak_phase() -> None:
    plan = load_study_plan(CONFIG)
    warmup = int(plan.optimizer.warmup_steps)
    for arm in plan.arms_for("study"):
        main, *branches = arm.runs
        assert main.checkpoint_steps == tuple(sorted(branch.fork_step for branch in branches))
        for branch in branches:
            assert warmup <= branch.fork_step < branch.max_steps <= main.max_steps
    by_arm = {arm.arm: arm for arm in plan.arms}
    # The instrument check compares runs that end on the same step.
    l12_10x = next(run for run in by_arm["L12"].runs if run.run == "branch-10x")
    assert l12_10x.max_steps == by_arm["L12-scratch-10x"].runs[0].max_steps


def test_study_plan_refuses_a_branch_that_forks_inside_warmup() -> None:
    payload = json.loads(CONFIG.read_text())
    base = json.loads(Path(payload["base_config"]).read_text())
    payload["arms"] = [
        {
            "arm": "bad",
            "stage": "study",
            "shape": "d768-L6",
            "steps": 300,
            "branch_steps": [100],
            "seed": 0,
        }
    ]
    with pytest.raises(ValueError, match="outside the main run's peak-LR phase"):
        study_plan(
            StudyConfig.model_validate_json(json.dumps(payload)),
            PretrainRunConfig.model_validate_json(json.dumps(base)).model,
        )


def _tiny_study(tmp_path: Path, *, added_arm: bool = False):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"fixture": true}', encoding="utf-8")
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()

    class Tokenizer:
        def encode_batch(self, texts):
            return [type("Encoding", (), {"ids": [int(t) for t in text.split()]}) for text in texts]

    generator = torch.Generator().manual_seed(0)
    documents = [
        " ".join(str(int(t)) for t in torch.randint(1, VOCAB_SIZE, (40,), generator=generator))
        for _ in range(200)
    ]
    shard_dir = tmp_path / "shards"
    for split in ("train", "validation"):
        write_token_shards(
            documents,
            Tokenizer(),
            eos_id=0,
            vocab_size=VOCAB_SIZE,
            tokenizer_sha256=tokenizer_sha256,
            split=split,
            output_dir=shard_dir,
            shard_tokens=10_000,
            provenance={"source": "fixture"},
        )
    payload = json.loads(CONFIG.read_text())
    base = json.loads(Path(payload["base_config"]).read_text())
    base = copy.deepcopy(base)
    base["model"].update(vocab_size=VOCAB_SIZE, context_length=16)
    payload["optimizer"].update(micro_batch_size=2, grad_accum_steps=1)
    payload["shapes"] = {
        "tiny": {
            "embedding_dim": 64,
            "layers": 2,
            "heads": 4,
            "kv_heads": 2,
            "feedforward_dim": 128,
        }
    }
    payload["arms"] = [
        {
            "arm": "tiny",
            "stage": "smoke",
            "shape": "tiny",
            "steps": 8,
            "warmup_steps": 2,
            "branch_steps": [5],
            "seed": 0,
        }
    ]
    if added_arm:
        payload["arms"].append(
            {
                "arm": "tiny-added",
                "stage": "smoke",
                "shape": "tiny",
                "steps": 4,
                "warmup_steps": 1,
                "seed": 0,
            }
        )
    plan = study_plan(
        StudyConfig.model_validate_json(json.dumps(payload)),
        PretrainRunConfig.model_validate_json(json.dumps(base)).model,
    )
    batches = fixed_validation_batches(
        shard_dir / "validation-manifest.json",
        vocab_size=VOCAB_SIZE,
        tokenizer_sha256=tokenizer_sha256,
        context_length=16,
        batch_size=plan.eval.batch_size,
        eval_token_budget=64,
        expected_token_batch_sha256=None,
    )
    plan = replace(
        plan,
        eval=plan.eval.model_copy(
            update={
                "eval_token_budget": 64,
                "expected_token_batch_sha256": hash_eval_batches(batches),
            }
        ),
    )
    return plan, shard_dir, tokenizer, tokenizer_sha256


def test_study_arm_trains_main_and_branch_and_writes_scored_rows(tmp_path: Path) -> None:
    plan, shard_dir, tokenizer, tokenizer_sha256 = _tiny_study(tmp_path)
    arm = plan.arms_for("smoke")[0]
    validation = study_validation_batches(
        plan,
        data_paths=resolve_study_data(plan.data, shard_dir.parent),
        vocab_size=VOCAB_SIZE,
        context_length=16,
        device="cpu",
    )
    commits = []

    rows = train_study_arm(
        arm,
        plan,
        data_paths=resolve_study_data(plan.data, shard_dir.parent),
        work_dir=tmp_path / "work",
        results_dir=tmp_path / "results",
        validation_batches=validation,
        device="cpu",
        dtype="float32",
        use_compile=False,
        commit_results=lambda: commits.append(True),
    )

    assert [row.run for row in rows] == ["main", "branch-5steps"]
    assert all(row.arm_runs == ("main", "branch-5steps") for row in rows)
    assert rows[1].fork_step == 4 and rows[1].tokens == 5 * 32
    assert all(math.isfinite(row.val_ce) for row in rows)
    assert len(commits) == 2
    for row in rows:
        on_disk = json.loads((tmp_path / "results" / "tiny" / f"{row.run}.json").read_text())
        assert on_disk["val_ce"] == row.val_ce
        assert (tmp_path / "results" / "tiny" / f"{row.run}-metrics.jsonl").exists()


def test_check_only_prices_a_stage_without_spending(capsys) -> None:
    gate_arms = ["lr-3e-4", "lr-6e-4", "lr-1.2e-3"]
    selection = [option for arm in gate_arms for option in ("--arm", arm)]
    code = scaling_study.launch(
        ["--stage", "lr-gate", *selection, "--spend-cap-usd", "20", "--check-only"]
    )

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert [arm["arm"] for arm in report["arms"]] == gate_arms
    assert report["worst_case_cost_usd"] <= 20


def test_lr_gate_refuses_the_lr_that_failed_it() -> None:
    # The recorded full sweep rejects the former L6 learning rate under the same stop rule.
    plan = load_study_plan(CONFIG)
    failed = plan.arms_for("lr-gate", ("lr-6e-4",))
    with pytest.raises(ValueError, match="LR 0.0006 is 0.139 nats worse"):
        check_lr_gate(failed, plan, LR_GATE_RESULTS)


def test_study_stage_passes_the_recorded_lr_sweep(capsys) -> None:
    # The study arms carry the LRs the pre-registered selection rule picked.
    code = scaling_study.launch(["--stage", "study", "--spend-cap-usd", "1000", "--check-only"])

    assert code == 0, capsys.readouterr().err


def test_study_stage_refuses_a_spend_cap_below_its_worst_case(capsys) -> None:
    code = scaling_study.launch(["--stage", "study", "--spend-cap-usd", "20", "--check-only"])

    assert code == 2
    assert "exceeds cap" in capsys.readouterr().err


def test_launch_without_approval_refuses_before_modal(capsys) -> None:
    code = scaling_study.launch(["--stage", "smoke", "--spend-cap-usd", "20"])

    assert code == 2
    assert "--approved" in capsys.readouterr().err


def test_validation_scores_in_true_fp32_after_training_enabled_tf32(
    tmp_path: Path, monkeypatch
) -> None:
    # Training sets TF32 matmuls process-wide; study points must still be scored on the
    # fp32 path of the held-out 214M point, then leave the setting as it was.
    from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone
    from esme_pretrain.study import arm_training
    from esme_pretrain.training.checkpointing import save_pretrain_checkpoint

    config = BackboneConfig(
        name="tiny",
        vocab_size=VOCAB_SIZE,
        context_length=16,
        embedding_dim=64,
        layers=2,
        heads=4,
        feedforward_dim=128,
    )
    checkpoint = tmp_path / "checkpoint.pt"
    save_pretrain_checkpoint(checkpoint, model=DenseBackbone(config), config=config, step=1)
    seen = []

    def recording_mean_ce_loss(*args, **kwargs):
        seen.append(torch.get_float32_matmul_precision())
        return 1.0

    monkeypatch.setattr(arm_training, "mean_ce_loss", recording_mean_ce_loss)
    torch.set_float32_matmul_precision("high")
    try:
        arm_training.validation_ce(checkpoint, [], device="cpu")
        assert seen == ["highest"]
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision("highest")


def test_arm_budget_watchdog_ends_the_container_process() -> None:
    import subprocess
    import sys

    script = (
        "import time\n"
        "from esme_pretrain.launch.scaling_study import _stop_after\n"
        "_stop_after(0.2, 'probe')\n"
        "time.sleep(30)\n"
    )
    started = __import__("time").perf_counter()
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == scaling_study.ARM_BUDGET_EXCEEDED_EXIT
    assert "exceeded its 0.2 s budget" in result.stdout
    assert __import__("time").perf_counter() - started < 20


def test_a_cancelled_arm_budget_timer_never_fires() -> None:
    # An arm that finishes inside its budget cancels its timer, so the deadline can
    # never end a later process.
    import subprocess
    import sys

    script = (
        "import time\n"
        "from esme_pretrain.launch.scaling_study import _stop_after\n"
        "_stop_after(0.3, 'probe').cancel()\n"
        "time.sleep(1.0)\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0
    assert "exceeded" not in result.stdout


def test_heldout_stage_refuses_to_run_before_a_prediction_is_written(
    tmp_path: Path, capsys
) -> None:
    # The run card pre-registers that the 214M actual is read only after the fit's
    # prediction exists; the launcher enforces that order.
    code = scaling_study.launch(["--stage", "heldout", "--spend-cap-usd", "20", "--check-only"])
    assert code == 2
    assert "needs --prediction" in capsys.readouterr().err

    # A fit written for any other (N, D) than the held-out 214M point is refused.
    prediction = tmp_path / "fit.json"
    target = {
        "non_embedding_params": 188794368,
        "tokens": 10229514240,
        "val_ce": 2.8,
        "interval": 0.9,
        "lower": 2.7,
        "upper": 2.9,
    }
    prediction.write_text(
        json.dumps(
            {
                "prediction": {**target, "tokens": 5_000_000_000},
                "points": [],
                "settings": fit_settings(resamples=1000, seed=0).model_dump(mode="json"),
                "points_path": "None",
                "point_count": 0,
                "fit": {"E": 1.0, "A": 1.0, "alpha": 0.1, "B": 1.0, "beta": 0.1, "objective": 0.0},
            }
        )
    )
    args = ["--stage", "heldout", "--prediction", str(prediction), "--spend-cap-usd", "20"]
    code = scaling_study.launch([*args, "--check-only"])
    assert code == 2
    assert "the held-out point is N=188794368, D=10229514240" in capsys.readouterr().err

    # A fit that dropped a pre-registered point (here all of L18) is refused too.
    plan = load_study_plan(CONFIG)
    points = [
        {
            "arm": f"{arm.arm}/{run.run}",
            "non_embedding_params": run.non_embedding_params,
            "tokens": run.tokens,
            "val_ce": 3.0,
        }
        for arm in plan.arms_for("study", ("L6", "L12", "L18"))
        for run in arm.runs
    ]
    assert len(points) == 12
    short_fit = [point for point in points if not point["arm"].startswith("L18/")]
    results_dir = tmp_path / "study-results"
    for arm in plan.arms_for("study", plan.heldout.fit_arms):
        directory = results_dir / arm.arm
        directory.mkdir(parents=True)
        for run in arm.runs:
            (directory / f"{run.run}.json").write_text(
                json.dumps(
                    {
                        "arm": arm.arm,
                        "run": run.run,
                        "non_embedding_params": run.non_embedding_params,
                        "study_id": plan.study_id,
                        "stage": arm.stage,
                        "shape": run.model.name,
                        "max_steps": run.max_steps,
                        "fork_step": run.fork_step,
                        "learning_rate": run.learning_rate,
                        "warmup_steps": run.warmup_steps,
                        "seed": run.seed,
                        "tokens": run.tokens,
                        "val_ce": 3.0,
                        "val_eval_tokens": fixed_validation_token_count(
                            context_length=run.model.context_length,
                            batch_size=plan.eval.batch_size,
                            eval_token_budget=plan.eval.eval_token_budget,
                        ),
                        "val_token_batch_sha256": plan.eval.expected_token_batch_sha256,
                    }
                )
            )
    evidence = load_study_fit_input(results_dir, plan.heldout.fit_arms, plan).evidence
    settings = fit_settings(resamples=1000, seed=0).model_dump(mode="json")
    fit = {
        "prediction": target,
        "points": points,
        "settings": settings,
        "study_evidence": evidence.model_dump(mode="json"),
        "points_path": "None",
        "point_count": len(points),
        "fit": {"E": 1.0, "A": 1.0, "alpha": 0.1, "B": 1.0, "beta": 0.1, "objective": 0.0},
    }
    prediction.write_text(json.dumps({**fit, "points": short_fit, "point_count": len(short_fit)}))
    code = scaling_study.launch([*args, "--check-only"])
    assert code == 2
    assert "not fit on the pre-registered 12 points" in capsys.readouterr().err

    # So is a rehearsal fit with fewer bootstrap resamples than pre-registered.
    prediction.write_text(json.dumps({**fit, "settings": {**settings, "resamples": 1}}))
    code = scaling_study.launch([*args, "--check-only"])
    assert code == 2
    assert "not fit with the pre-registered settings" in capsys.readouterr().err

    prediction.write_text(json.dumps(fit))
    code = scaling_study.launch([*args, "--check-only"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["arms"] == [] and report["heldout_eval"] is True
    assert report["prediction"]["val_ce"] == 2.8

    # A valid fit still reveals nothing until it is committed: this one is outside
    # the repository, so the real launch refuses before renting a GPU.
    code = scaling_study.launch([*args, "--approved"])
    assert code == 2
    assert "before the held-out eval reveals the 214M loss" in capsys.readouterr().err


def test_committed_in_head_matches_the_tracked_blob(tmp_path: Path) -> None:
    tracked = scaling_study.REPO_ROOT / ".python-version"
    assert scaling_study._committed_in_head(tracked)
    copy = tmp_path / ".python-version"
    copy.write_bytes(tracked.read_bytes())
    assert not scaling_study._committed_in_head(copy)


def test_study_stage_no_longer_scores_the_heldout_checkpoint(capsys) -> None:
    code = scaling_study.launch(["--stage", "study", "--spend-cap-usd", "1000", "--check-only"])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["heldout_eval"] is False


def test_changed_data_paths_and_added_arm_reach_training_and_result_parser(tmp_path: Path) -> None:
    import shutil

    plan, shard_dir, tokenizer, tokenizer_sha256 = _tiny_study(tmp_path, added_arm=True)
    volume_root = tmp_path / "volume"
    configured = {
        "tokenizer": "tokenizers/bpe-v2.json",
        "train_manifest": "corpus/train/v2.json",
        "validation_manifest": "validation/fixed/v2.json",
    }
    selected_tokenizer = volume_root / configured["tokenizer"]
    selected_tokenizer.parent.mkdir(parents=True)
    shutil.copyfile(tokenizer, selected_tokenizer)
    (volume_root / "tokenizer.json").write_text("wrong tokenizer")
    for split, field in (("train", "train_manifest"), ("validation", "validation_manifest")):
        manifest = json.loads((shard_dir / f"{split}-manifest.json").read_text())
        target = volume_root / configured[field]
        target.parent.mkdir(parents=True)
        for shard in manifest["shards"]:
            old_path = shard_dir / shard["path"]
            shard["path"] = f"chunks/{shard['path']}"
            destination = target.parent / shard["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old_path, destination)
        target.write_text(json.dumps(manifest))
    plan = replace(plan, data=plan.data.model_copy(update=configured))
    source_paths = resolve_study_data(plan.data, volume_root)
    paths = scaling_study.stage_study_data(source_paths, tmp_path / "local-stage")
    assert paths.tokenizer.read_bytes() == tokenizer.read_bytes()
    assert paths.train_manifest == paths.root / configured["train_manifest"]
    assert paths.validation_manifest == paths.root / configured["validation_manifest"]
    validation = study_validation_batches(
        plan,
        data_paths=paths,
        vocab_size=VOCAB_SIZE,
        context_length=16,
        device="cpu",
    )
    arm = plan.arms_for("smoke", ("tiny-added",))[0]
    rows = train_study_arm(
        arm,
        plan,
        data_paths=paths,
        work_dir=tmp_path / "work-new",
        results_dir=tmp_path / "new-results",
        validation_batches=validation,
        device="cpu",
        dtype="float32",
        use_compile=False,
        commit_results=lambda: None,
    )
    parsed = load_study_fit_input(tmp_path / "new-results", ("tiny-added",), plan)
    assert parsed.points[0].val_ce == rows[0].val_ce
    assert rows[0].val_eval_batches == 1
    path = tmp_path / "new-results/tiny-added/main.json"
    saved = json.loads(path.read_text())
    assert saved["val_eval_batches"] == 1
    path.write_text(json.dumps({**saved, "val_eval_batches": 2}))
    with pytest.raises(ValueError, match="val_eval_batches"):
        load_study_fit_input(tmp_path / "new-results", ("tiny-added",), plan)


@pytest.mark.parametrize(
    "value", ["../tokenizer.json", "/tokenizer.json", "nested/../../file.json", ""]
)
def test_study_data_paths_reject_volume_escape(value: str) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["data"]["tokenizer"] = value
    with pytest.raises(ValueError, match="relative file path inside the study volume"):
        StudyConfig.model_validate_json(json.dumps(payload))
