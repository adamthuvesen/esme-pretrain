from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from esme_pretrain.pretrain_run import load_pretrain_config
from esme_pretrain.study.plan import StudyConfig, load_study_plan, study_plan
from esme_pretrain.study.results import load_study_fit_input
from esme_pretrain.training.shard_loader import fixed_validation_token_count

CONFIG = Path("configs/scaling_study.json")


def _write_rows(tmp_path: Path):
    plan = load_study_plan(CONFIG)
    arm = plan.arms_for("study", ("L6",))[0]
    directory = tmp_path / arm.arm
    directory.mkdir()
    paths = []
    for run in arm.runs:
        path = directory / f"{run.run}.json"
        path.write_text(
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
        paths.append(path)
    return plan, paths


def test_early_rows_without_arm_runs_use_the_authoritative_plan(tmp_path: Path) -> None:
    plan, _ = _write_rows(tmp_path)
    points = load_study_fit_input(tmp_path, ("L6",), plan).points
    assert len(points) == 4
    assert {point.arm for point in points} == {
        f"L6/{run.run}" for run in plan.arms_for("study", ("L6",))[0].runs
    }


def test_partial_rows_cannot_declare_themselves_complete(tmp_path: Path) -> None:
    plan, paths = _write_rows(tmp_path)
    for path in paths[2:]:
        path.unlink()
    claimed = [json.loads(path.read_text())["run"] for path in paths[:2]]
    for path in paths[:2]:
        row = json.loads(path.read_text())
        path.write_text(json.dumps({**row, "arm_runs": claimed}))
    with pytest.raises(ValueError, match="arm_runs conflicts with the study plan"):
        load_study_fit_input(tmp_path, ("L6",), plan)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "tokens", "conflicting_runs", "nan"])
def test_result_boundaries_reject_incomplete_or_conflicting_rows(
    tmp_path: Path, defect: str
) -> None:
    plan, paths = _write_rows(tmp_path)
    path = paths[0]
    row = json.loads(path.read_text())
    if defect == "missing":
        path.unlink()
    elif defect == "duplicate":
        (path.parent / "duplicate.json").write_text(json.dumps(row))
    elif defect == "tokens":
        path.write_text(json.dumps({**row, "tokens": row["tokens"] + 1}))
    elif defect == "conflicting_runs":
        path.write_text(json.dumps({**row, "arm_runs": ["main"]}))
    else:
        path.write_text(json.dumps({**row, "val_ce": float("nan")}))
    with pytest.raises(ValueError):
        load_study_fit_input(tmp_path, ("L6",), plan)


def test_study_json_rejects_unknown_settings_and_coerced_numbers() -> None:
    payload = json.loads(CONFIG.read_text())
    payload["optimizer"]["micro_batch_size"] = "24"
    payload["eval"]["unknown_setting"] = True
    with pytest.raises(ValidationError) as caught:
        StudyConfig.model_validate_json(json.dumps(payload))
    assert {error["type"] for error in caught.value.errors()} == {"int_type", "extra_forbidden"}


def test_study_preserves_fractional_multipliers_and_zero_decay() -> None:
    payload = json.loads(CONFIG.read_text())
    payload["optimizer"]["decay_fraction"] = 0.0
    payload["arms"] = [
        {
            "arm": "fractional",
            "stage": "study",
            "shape": "d768-L6",
            "seed": 0,
            "multiple": 40.5,
            "branch_multiples": [5.5],
        }
    ]
    config = StudyConfig.model_validate_json(json.dumps(payload))
    base = load_pretrain_config(Path(config.base_config))
    plan = study_plan(config, base.model_config)
    main, branch = plan.arms[0].runs
    assert main.max_steps == round(40.5 * main.non_embedding_params / plan.tokens_per_step)
    assert branch.run == "branch-5.5x" and branch.fork_step == branch.max_steps


def test_validated_run_profiles_cannot_change_after_cost_validation() -> None:
    config = load_pretrain_config(Path("configs/pretrain_214m_b200.json"))
    with pytest.raises(TypeError):
        config.config.runtime.gpu_profiles["B200"] = config.selected_gpu_profile  # pyright: ignore[reportIndexIssue]  # the test proves the mapping refuses this write
    with pytest.raises(ValidationError):
        config.selected_gpu_profile.usd_per_hour = 0.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("val_eval_tokens", 1),
        ("val_eval_tokens", 10_000_000),
        ("val_token_batch_sha256", "0" * 64),
        ("val_eval_batches", 1),
    ],
)
def test_fit_refuses_incomparable_evaluation_identity(tmp_path: Path, field: str, value) -> None:
    plan, paths = _write_rows(tmp_path)
    row = json.loads(paths[0].read_text())
    paths[0].write_text(json.dumps({**row, field: value}))
    with pytest.raises(ValueError) as caught:
        load_study_fit_input(tmp_path, ("L6",), plan)
    message = str(caught.value)
    assert str(paths[0]) in message and field in message
    assert "got" in message and "expected" in message
    assert "fixed validation batches" in message


@pytest.mark.parametrize("field", ["val_eval_tokens", "val_token_batch_sha256"])
def test_fit_requires_the_eval_fields_present_in_every_recorded_result(
    tmp_path: Path, field: str
) -> None:
    plan, paths = _write_rows(tmp_path)
    row = json.loads(paths[0].read_text())
    del row[field]
    paths[0].write_text(json.dumps(row))
    with pytest.raises(ValueError, match=field):
        load_study_fit_input(tmp_path, ("L6",), plan)
