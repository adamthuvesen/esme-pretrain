from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from esme_pretrain.data.token_shards import write_token_shards
from esme_pretrain.launch import scaling_study
from esme_pretrain.study.arm_training import study_validation_batches, train_study_arm
from esme_pretrain.study.plan import load_study_plan, study_plan_from_payloads
from esme_pretrain.torch import torch

CONFIG = Path("configs/scaling_study.json")
VOCAB_SIZE = 128


def test_real_study_plan_branches_fork_inside_the_main_runs_peak_phase() -> None:
    plan = load_study_plan(CONFIG)
    warmup = int(plan.optimizer["warmup_steps"])
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
        study_plan_from_payloads(payload, base)


def _tiny_study(tmp_path: Path):
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
    return study_plan_from_payloads(payload, base), shard_dir, tokenizer, tokenizer_sha256


def test_study_arm_trains_main_and_branch_and_writes_scored_rows(tmp_path: Path) -> None:
    plan, shard_dir, tokenizer, tokenizer_sha256 = _tiny_study(tmp_path)
    arm = plan.arms_for("smoke")[0]
    validation = study_validation_batches(
        plan,
        shard_dir=shard_dir,
        tokenizer_sha256=tokenizer_sha256,
        vocab_size=VOCAB_SIZE,
        context_length=16,
        device="cpu",
        eval_token_budget=64,
        expected_token_batch_sha256=None,
    )
    commits = []

    rows = train_study_arm(
        arm,
        plan,
        shard_dir=shard_dir,
        tokenizer_path=tokenizer,
        work_dir=tmp_path / "work",
        results_dir=tmp_path / "results",
        validation_batches=validation,
        device="cpu",
        dtype="float32",
        use_compile=False,
        commit_results=lambda: commits.append(True),
    )

    assert [row["run"] for row in rows] == ["main", "branch-5steps"]
    assert rows[1]["fork_step"] == 4 and rows[1]["tokens"] == 5 * 32
    assert all(math.isfinite(row["val_ce"]) for row in rows)
    assert len(commits) == 2
    for row in rows:
        on_disk = json.loads((tmp_path / "results" / "tiny" / f"{row['run']}.json").read_text())
        assert on_disk["val_ce"] == row["val_ce"]
        assert (tmp_path / "results" / "tiny" / f"{row['run']}-metrics.jsonl").exists()


def test_check_only_prices_a_stage_without_spending(capsys) -> None:
    code = scaling_study.launch(["--stage", "lr-gate", "--spend-cap-usd", "20", "--check-only"])

    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert [arm["arm"] for arm in report["arms"]] == ["lr-3e-4", "lr-6e-4", "lr-1.2e-3"]
    assert report["worst_case_cost_usd"] <= 20


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

    prediction = tmp_path / "fit.json"
    prediction.write_text(json.dumps({"prediction": {"val_ce": 2.8}}))
    code = scaling_study.launch(
        [
            "--stage",
            "heldout",
            "--prediction",
            str(prediction),
            "--spend-cap-usd",
            "20",
            "--check-only",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert report["arms"] == [] and report["heldout_eval"] is True
    assert report["prediction"]["val_ce"] == 2.8


def test_study_stage_no_longer_scores_the_heldout_checkpoint(capsys) -> None:
    code = scaling_study.launch(["--stage", "study", "--spend-cap-usd", "1000", "--check-only"])

    assert code == 0
    assert json.loads(capsys.readouterr().out)["heldout_eval"] is False
