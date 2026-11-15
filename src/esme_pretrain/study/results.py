"""Validate planned study results and preserve the evidence used by a scaling fit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from esme_pretrain.study.plan import StudyArm, StudyPlan, StudyRun
from esme_pretrain.study.scaling_law import LossPoint
from esme_pretrain.training.shard_loader import fixed_validation_token_count

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StudyResultRow(BaseModel):
    # Early recorded results omit arm_runs. Unknown telemetry does not change the fit contract.
    model_config = ConfigDict(extra="ignore", strict=True, frozen=True, allow_inf_nan=False)
    arm: str
    run: str
    non_embedding_params: Annotated[int, Field(gt=0)]
    tokens: Annotated[int, Field(gt=0)]
    val_ce: Annotated[float, Field(gt=0)]
    val_eval_tokens: Annotated[int, Field(gt=0)]
    val_token_batch_sha256: Sha256
    val_eval_batches: Annotated[int, Field(gt=0)] | None = None
    arm_runs: tuple[str, ...] | None = None
    study_id: str
    stage: str
    shape: str
    max_steps: Annotated[int, Field(gt=0)]
    fork_step: Annotated[int, Field(ge=0)] | None
    learning_rate: Annotated[float, Field(gt=0)]
    warmup_steps: Annotated[int, Field(ge=0)]
    seed: int
    total_params: int | None = None
    train_loss_last: float | None = None
    steady_tokens_per_second: float | None = None
    mfu: float | None = None
    peak_memory_gb: float | None = None
    wall_seconds: float | None = None
    notes: tuple[str, ...] = ()
    torch_version: str | None = None
    study_run_id: str | None = None
    commit: str | None = None
    gpu_name: str | None = None


class StudyProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    study_run_id: str
    commit: str
    gpu_name: str


class LossPointRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    arm: Annotated[str, Field(min_length=1)]
    non_embedding_params: Annotated[int, Field(gt=0)]
    tokens: Annotated[int, Field(gt=0)]
    val_ce: Annotated[float, Field(gt=0)]

    def loss_point(self) -> LossPoint:
        return LossPoint(self.arm, self.non_embedding_params, self.tokens, self.val_ce)


class StudyPointEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    arm: str
    point_sha256: Sha256
    result_path: str
    result_sha256: Sha256
    val_eval_tokens: Annotated[int, Field(gt=0)]
    val_token_batch_sha256: Sha256


class StudyFitEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    study_id: str
    plan_sha256: Sha256
    fit_arms: tuple[str, ...]
    points: tuple[StudyPointEvidence, ...]


@dataclass(frozen=True)
class StudyFitInput:
    points: tuple[LossPoint, ...]
    evidence: StudyFitEvidence


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def point_sha256(point: LossPointRecord) -> str:
    return _json_sha256(point.model_dump(mode="json"))


def study_fit_sha256(plan: StudyPlan, names: tuple[str, ...]) -> str:
    selected = tuple(
        sorted((arm for arm in plan.arms if arm.arm in names), key=lambda arm: arm.arm)
    )
    return _json_sha256(
        {
            "study_id": plan.study_id,
            "optimizer": plan.optimizer.model_dump(mode="json"),
            "data": plan.data.model_dump(mode="json"),
            "eval": plan.eval.model_dump(mode="json"),
            "arms": TypeAdapter(tuple[StudyArm, ...]).dump_python(selected, mode="json"),
            "heldout_tokens": plan.heldout.tokens,
        }
    )


def validate_evaluation_identity(
    *,
    source: str,
    plan: StudyPlan,
    context_length: int,
    token_count: int,
    token_batch_sha256: str,
    batch_count: int | None = None,
) -> None:
    expected_tokens = fixed_validation_token_count(
        context_length=context_length,
        batch_size=plan.eval.batch_size,
        eval_token_budget=plan.eval.eval_token_budget,
    )
    expected = {
        "val_eval_tokens": (token_count, expected_tokens),
        "val_token_batch_sha256": (token_batch_sha256, plan.eval.expected_token_batch_sha256),
    }
    if batch_count is not None:
        expected["val_eval_batches"] = (
            batch_count,
            expected_tokens // (context_length * plan.eval.batch_size),
        )
    for field, (actual, required) in expected.items():
        if actual != required:
            raise ValueError(
                f"{source}: {field} conflicts with the study plan: "
                f"got {actual!r}, expected {required!r}. "
                "Use this study's fixed validation batches."
            )


def validate_study_result(
    row: StudyResultRow, *, source: str, plan: StudyPlan, arm: StudyArm, run: StudyRun
) -> None:
    identity = {
        "arm": arm.arm,
        "run": run.run,
        "non_embedding_params": run.non_embedding_params,
        "tokens": run.tokens,
        "study_id": plan.study_id,
        "stage": arm.stage,
        "shape": run.model.name,
        "max_steps": run.max_steps,
        "fork_step": run.fork_step,
        "learning_rate": run.learning_rate,
        "warmup_steps": run.warmup_steps,
        "seed": run.seed,
    }
    for field, value in identity.items():
        if getattr(row, field) != value:
            raise ValueError(
                f"{source}: {field} conflicts with the study plan: "
                f"got {getattr(row, field)!r}, expected {value!r}. "
                "Use results from this study config and its fixed validation batches."
            )
    validate_evaluation_identity(
        source=source,
        plan=plan,
        context_length=run.model.context_length,
        token_count=row.val_eval_tokens,
        token_batch_sha256=row.val_token_batch_sha256,
        batch_count=row.val_eval_batches,
    )
    expected = {planned.run for planned in arm.runs}
    if row.arm_runs is not None and (
        len(row.arm_runs) != len(expected) or set(row.arm_runs) != expected
    ):
        raise ValueError(
            f"{source}: arm_runs conflicts with the study plan: "
            f"got {sorted(row.arm_runs)}, expected {sorted(expected)}. "
            "Recover the missing planned runs before fitting."
        )


def load_loss_points(path: Path) -> list[LossPoint]:
    try:
        records = TypeAdapter(list[LossPointRecord]).validate_json(path.read_bytes())
    except ValidationError as error:
        detail = error.errors(include_url=False)[0]
        raise ValueError(
            f"cannot read loss points {path}: {detail['loc']}: {detail['msg']}. "
            "Fix the exploratory point JSON."
        ) from error
    except OSError as error:
        raise ValueError(f"cannot read loss points {path}: {error}") from error
    if len({record.arm for record in records}) != len(records):
        raise ValueError(f"duplicate loss point names in {path}. Give each point one unique name.")
    return [record.loss_point() for record in records]


def load_study_fit_input(
    results_dir: Path, arms: tuple[str, ...], plan: StudyPlan
) -> StudyFitInput:
    if not arms:
        raise ValueError("--results-dir needs at least one --arm")
    if len(set(arms)) != len(arms):
        raise ValueError("duplicate --arm selection. Select each arm once.")
    selected = {arm.arm: arm for arm in plan.arms}
    unknown = set(arms) - selected.keys()
    if unknown:
        raise ValueError(
            f"unknown study arms: {sorted(unknown)}. Select arms from the study config."
        )
    points = []
    evidence = []
    for name in arms:
        arm = selected[name]
        expected = {run.run: run for run in arm.runs}
        rows: dict[str, StudyResultRow] = {}
        for path in sorted((results_dir / name).glob("*.json")):
            try:
                source_bytes = path.read_bytes()
                row = StudyResultRow.model_validate_json(source_bytes)
            except ValidationError as error:
                detail = error.errors(include_url=False)[0]
                raise ValueError(
                    f"cannot read study result {path}: {detail['loc']}: {detail['msg']}. "
                    "Recover the complete result row from this study run."
                ) from error
            except OSError as error:
                raise ValueError(f"cannot read study result {path}: {error}") from error
            if row.run in rows:
                raise ValueError(
                    f"duplicate result for {name}/{row.run}: {path}. "
                    "Keep one result per planned run."
                )
            if row.run not in expected:
                raise ValueError(
                    f"unplanned result {name}/{row.run}: {path}. Use the selected study config."
                )
            validate_study_result(row, source=str(path), plan=plan, arm=arm, run=expected[row.run])
            rows[row.run] = row
            point = LossPointRecord(
                arm=f"{name}/{row.run}",
                non_embedding_params=row.non_embedding_params,
                tokens=row.tokens,
                val_ce=row.val_ce,
            )
            points.append(point.loss_point())
            evidence.append(
                StudyPointEvidence(
                    arm=point.arm,
                    point_sha256=point_sha256(point),
                    result_path=str(path),
                    result_sha256=hashlib.sha256(source_bytes).hexdigest(),
                    val_eval_tokens=row.val_eval_tokens,
                    val_token_batch_sha256=row.val_token_batch_sha256,
                )
            )
        if rows.keys() != expected.keys():
            raise ValueError(
                f"arm {name!r} is incomplete in {results_dir / name}: "
                f"missing {sorted(expected.keys() - rows.keys())}. "
                "Complete these planned runs before fitting."
            )
    return StudyFitInput(
        points=tuple(points),
        evidence=StudyFitEvidence(
            study_id=plan.study_id,
            plan_sha256=study_fit_sha256(plan, arms),
            fit_arms=tuple(sorted(arms)),
            points=tuple(evidence),
        ),
    )
