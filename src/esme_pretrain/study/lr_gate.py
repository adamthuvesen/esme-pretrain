"""Validate the recorded LR sweep and apply the run card's pre-registered stop rule."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from esme_pretrain.study.plan import StudyArm, StudyPlan
from esme_pretrain.study.results import StudyResultRow, validate_study_result

REPO_ROOT = Path(__file__).resolve().parents[3]
LR_GATE_RESULTS = (
    REPO_ROOT / "docs/results/scaling-study/lr-gate.json",
    REPO_ROOT / "docs/results/scaling-study/lr-sweep.json",
    REPO_ROOT / "docs/results/scaling-study/lr-sweep-extension.json",
)
# Source: docs/run-cards/scaling-study-b200.md, learning-rate selection rule 6.
LR_GATE_MAX_GAP_NATS = 0.02


class LrGateOutcome(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    arm: str
    status: Literal["ok"]
    rows: tuple[StudyResultRow, ...]


class LrGateArtifact(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True)
    stage: Literal["lr-gate"]
    outcomes: dict[str, LrGateOutcome]


def load_lr_gate_evidence(
    plan: StudyPlan, paths: tuple[Path, ...] = LR_GATE_RESULTS
) -> tuple[StudyResultRow, ...]:
    expected = {
        (arm.arm, run.run): (arm, run) for arm in plan.arms_for("lr-gate") for run in arm.runs
    }
    if not expected:
        raise ValueError(
            "study config has no LR-gate arms. Define the sweep before selecting study LRs."
        )
    rows = {}
    shape_rates = {}
    for path in paths:
        try:
            artifact = LrGateArtifact.model_validate_json(path.read_bytes())
        except ValidationError as error:
            detail = error.errors(include_url=False)[0]
            raise ValueError(
                f"invalid LR-gate evidence {path}: {detail['loc']}: {detail['msg']}. "
                "Recover the complete finite LR-gate results before launching the study."
            ) from error
        except OSError as error:
            raise ValueError(
                f"cannot read LR-gate evidence {path}: {error}. Recover this recorded result."
            ) from error
        for name, outcome in artifact.outcomes.items():
            if name != outcome.arm:
                raise ValueError(
                    f"{path}: outcome {name!r} declares arm {outcome.arm!r}. "
                    "Recover the matching LR-gate result."
                )
            for row in outcome.rows:
                key = (row.arm, row.run)
                source = f"{path}:{name}/{row.run}"
                if row.arm != name or key not in expected:
                    raise ValueError(
                        f"{source}: unplanned LR-gate row {key}. "
                        "Use evidence from this study config."
                    )
                if key in rows:
                    raise ValueError(
                        f"{source}: duplicate LR-gate row {key}. "
                        "Keep exactly one recorded result per planned run."
                    )
                arm, run = expected[key]
                validate_study_result(row, source=source, plan=plan, arm=arm, run=run)
                # The run card registers one measurement per shape and LR, without aggregation.
                shape_rate = (run.model.name, run.learning_rate)
                if shape_rate in shape_rates:
                    raise ValueError(
                        f"{source}: conflicting measurements for {shape_rate}, "
                        f"also in {shape_rates[shape_rate]}. "
                        "Resolve the LR evidence before launching."
                    )
                shape_rates[shape_rate] = source
                rows[key] = row
    if rows.keys() != expected.keys():
        raise ValueError(
            f"LR-gate evidence is incomplete: missing {sorted(expected.keys() - rows.keys())}. "
            "Recover every planned LR-gate run before launching the study."
        )
    return tuple(rows.values())


def check_lr_gate(
    arms: tuple[StudyArm, ...], plan: StudyPlan, paths: tuple[Path, ...] = LR_GATE_RESULTS
) -> None:
    rows = load_lr_gate_evidence(plan, paths)
    by_arm = {arm.arm: arm for arm in plan.arms_for("lr-gate")}
    val_ce_by_shape: dict[str, dict[float, float]] = {}
    params_by_shape: dict[str, int] = {}
    for row in rows:
        run = next(run for run in by_arm[row.arm].runs if run.run == row.run)
        val_ce_by_shape.setdefault(run.model.name, {})[run.learning_rate] = row.val_ce
        params_by_shape[run.model.name] = run.non_embedding_params
    for arm in arms:
        for run in arm.runs:
            shape = min(
                params_by_shape,
                key=lambda swept: abs(params_by_shape[swept] - run.non_embedding_params),
            )
            val_ce_by_lr = val_ce_by_shape[shape]
            if run.learning_rate not in val_ce_by_lr:
                raise ValueError(
                    f"{arm.arm}/{run.run}: LR {run.learning_rate:g} has no "
                    f"{shape} LR-gate measurement. "
                    f"Select a measured LR from {sorted(val_ce_by_lr)}."
                )
            gap = val_ce_by_lr[run.learning_rate] - min(val_ce_by_lr.values())
            if gap > LR_GATE_MAX_GAP_NATS:
                raise ValueError(
                    f"{arm.arm}/{run.run}: LR {run.learning_rate:g} is {gap:.3f} nats worse "
                    f"than the best {shape} LR-gate row. The run card's stop rule is "
                    f"{LR_GATE_MAX_GAP_NATS} nats. Return the LR decision to Adam."
                )
