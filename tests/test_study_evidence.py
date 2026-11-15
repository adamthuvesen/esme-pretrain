from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from esme_pretrain.cli import main
from esme_pretrain.cli import study as study_cli
from esme_pretrain.launch import scaling_study
from esme_pretrain.study.lr_gate import LR_GATE_RESULTS, check_lr_gate, load_lr_gate_evidence
from esme_pretrain.study.plan import load_study_plan
from esme_pretrain.study.prediction import written_prediction
from esme_pretrain.study.results import load_study_fit_input, study_fit_sha256
from esme_pretrain.study.scaling_law import BootstrapPrediction, ScalingLawFit
from esme_pretrain.training.shard_loader import fixed_validation_token_count

CONFIG = Path("configs/scaling_study.json")


def _gate_copies(tmp_path: Path) -> tuple[Path, ...]:
    paths = []
    for original in LR_GATE_RESULTS:
        path = tmp_path / original.name
        path.write_bytes(original.read_bytes())
        paths.append(path)
    return tuple(paths)


def test_recorded_lr_evidence_covers_the_configured_sweep() -> None:
    plan = load_study_plan(CONFIG)
    rows = load_lr_gate_evidence(plan)
    assert {(row.arm, row.run) for row in rows} == {
        (arm.arm, run.run) for arm in plan.arms_for("lr-gate") for run in arm.runs
    }
    check_lr_gate(plan.arms_for("study"), plan)


@pytest.mark.parametrize(
    "defect",
    [
        "nan",
        "infinite",
        "duplicate",
        "missing",
        "wrong_count",
        "wrong_hash",
        "conflicting_lr",
        "malformed",
    ],
)
def test_lr_gate_rejects_invalid_evidence(tmp_path: Path, defect: str) -> None:
    paths = _gate_copies(tmp_path)
    payload = json.loads(paths[0].read_text())
    name = next(iter(payload["outcomes"]))
    outcome = payload["outcomes"][name]
    row = outcome["rows"][0]
    if defect == "nan":
        row["val_ce"] = float("nan")
    elif defect == "infinite":
        row["val_ce"] = float("inf")
    elif defect == "duplicate":
        outcome["rows"].append(dict(row))
    elif defect == "missing":
        del payload["outcomes"][name]
    elif defect == "wrong_count":
        row["val_eval_tokens"] = 1
    elif defect == "wrong_hash":
        row["val_token_batch_sha256"] = "0" * 64
    elif defect == "conflicting_lr":
        row["learning_rate"] *= 2
    else:
        del row["val_eval_tokens"]
    paths[0].write_text(json.dumps(payload))
    plan = load_study_plan(CONFIG)
    with pytest.raises(ValueError) as caught:
        check_lr_gate(plan.arms_for("study"), plan, paths)
    message = str(caught.value)
    assert "Recover" in message or "Use" in message or "Keep" in message
    if defect != "missing":
        assert str(paths[0]) in message


def test_nonfinite_lr_evidence_refuses_before_launch_planning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    paths = _gate_copies(tmp_path)
    for path in paths:
        payload = json.loads(path.read_text())
        for outcome in payload["outcomes"].values():
            for row in outcome["rows"]:
                row["val_ce"] = float("nan")
        path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        scaling_study, "check_lr_gate", lambda arms, plan: check_lr_gate(arms, plan, paths)
    )
    assert (
        scaling_study.launch(["--stage", "study", "--spend-cap-usd", "1000", "--check-only"]) == 2
    )
    result = capsys.readouterr()
    assert result.out == ""
    assert "val_ce" in result.err and "finite" in result.err


def _write_fit_sources(tmp_path: Path):
    plan = load_study_plan(CONFIG)
    results_dir = tmp_path / "result-rows"
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
                    },
                    indent=3,
                )
            )
    return plan, results_dir


def _fit_fixture(tmp_path: Path, monkeypatch, source: str):
    plan, results_dir = _write_fit_sources(tmp_path)
    study_input = load_study_fit_input(results_dir, plan.heldout.fit_arms, plan)
    monkeypatch.setattr(
        study_cli,
        "bootstrap_prediction",
        lambda *_args, **_kwargs: BootstrapPrediction(
            fit=ScalingLawFit(E=1.0, A=1.0, alpha=0.1, B=1.0, beta=0.1, objective=0.0),
            prediction=2.8,
            lower=2.7,
            upper=2.9,
            interval=0.9,
            resamples=1000,
        ),
    )
    output = tmp_path / "fit.json"
    if source == "study":
        source_args = [
            "--results-dir",
            str(results_dir),
            *[option for arm in plan.heldout.fit_arms for option in ("--arm", arm)],
        ]
    else:
        points = tmp_path / "points.json"
        points.write_text(json.dumps([vars(point) for point in study_input.points]))
        source_args = ["--points", str(points)]
    assert (
        main(
            [
                "scaling-fit",
                *source_args,
                "--predict-non-embedding-params",
                "188794368",
                "--predict-tokens",
                str(plan.heldout.tokens),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    return plan, results_dir, output


def test_exploratory_points_cannot_approve_heldout_eval(tmp_path: Path, monkeypatch) -> None:
    plan, _, path = _fit_fixture(tmp_path, monkeypatch, "exploratory")
    assert json.loads(path.read_text())["study_evidence"] is None
    with pytest.raises(ValueError, match="Exploratory --points fits cannot approve heldout"):
        written_prediction(
            path, non_embedding_params=188794368, tokens=plan.heldout.tokens, plan=plan
        )


def test_fit_evidence_binds_values_and_exact_source_bytes(tmp_path: Path, monkeypatch) -> None:
    plan, results_dir, path = _fit_fixture(tmp_path, monkeypatch, "study")
    payload = json.loads(path.read_text())
    for source in payload["study_evidence"]["points"]:
        assert (
            source["result_sha256"]
            == hashlib.sha256(Path(source["result_path"]).read_bytes()).hexdigest()
        )
    results_dir.rename(tmp_path / "source-archive")
    assert (
        written_prediction(
            path, non_embedding_params=188794368, tokens=plan.heldout.tokens, plan=plan
        ).val_ce
        == 2.8
    )
    payload["points"][0]["val_ce"] += 0.1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fit points changed after evidence validation"):
        written_prediction(
            path, non_embedding_params=188794368, tokens=plan.heldout.tokens, plan=plan
        )


@pytest.mark.parametrize(
    "defect", ["missing_source", "duplicate_source", "wrong_hash", "wrong_count", "changed_plan"]
)
def test_heldout_requires_matching_evidence(tmp_path: Path, monkeypatch, defect: str) -> None:
    plan, _, path = _fit_fixture(tmp_path, monkeypatch, "study")
    payload = json.loads(path.read_text())
    evidence = payload["study_evidence"]
    if defect == "missing_source":
        evidence["points"].pop()
    elif defect == "duplicate_source":
        evidence["points"].append(dict(evidence["points"][0]))
    elif defect == "wrong_hash":
        evidence["points"][0]["val_token_batch_sha256"] = "0" * 64
    elif defect == "wrong_count":
        evidence["points"][0]["val_eval_tokens"] = 1
    else:
        plan = replace(plan, data=plan.data.model_copy(update={"shuffle_seed": 1}))
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError) as caught:
        written_prediction(
            path, non_embedding_params=188794368, tokens=plan.heldout.tokens, plan=plan
        )
    assert str(path) in str(caught.value)


def test_fit_fingerprint_ignores_unselected_arms_and_heldout_output_location() -> None:
    plan = load_study_plan(CONFIG)
    expected = study_fit_sha256(plan, plan.heldout.fit_arms)
    smoke = plan.arms_for("smoke")[0]
    extra = replace(smoke, arm="extra-smoke")
    changed = replace(
        plan,
        arms=(*plan.arms, extra),
        heldout=plan.heldout.model_copy(update={"checkpoint": "other/checkpoint.pt"}),
    )
    assert study_fit_sha256(changed, changed.heldout.fit_arms) == expected


@pytest.mark.parametrize(
    "field",
    [
        "study_id",
        "stage",
        "shape",
        "max_steps",
        "fork_step",
        "learning_rate",
        "warmup_steps",
        "seed",
    ],
)
def test_lr_gate_requires_recorded_semantic_identity(tmp_path: Path, field: str) -> None:
    paths = _gate_copies(tmp_path)
    payload = json.loads(paths[0].read_text())
    outcome = next(iter(payload["outcomes"].values()))
    del outcome["rows"][0][field]
    paths[0].write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=field):
        load_lr_gate_evidence(load_study_plan(CONFIG), paths)
