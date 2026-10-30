"""Expand the scaling-study config into concrete training runs.

An arm is one model shape trained once. A ``multiple`` arm trains to
``multiple x N`` tokens (N = non-embedding params). Its ``branch_multiples``
fork cooldown branches from the arm's main run: each branch shares the main
run's past up to its own decay start, then decays over its last
``decay_fraction``. See ``docs/run-cards/scaling-study-b200.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from esme_pretrain.modeling.backbone import BackboneConfig
from esme_pretrain.pretrain_config import parse_run_config
from esme_pretrain.study.data_paths import relative_volume_path
from esme_pretrain.training.runtime import wsd_decay_start

STAGES = ("smoke", "lr-gate", "study")
MAIN_RUN = "main"


@dataclass(frozen=True)
class StudyRun:
    arm: str
    run: str
    model: BackboneConfig
    max_steps: int
    learning_rate: float
    warmup_steps: int
    seed: int
    tokens_per_step: int
    branch_of: str | None = None
    fork_step: int | None = None
    checkpoint_steps: tuple[int, ...] = ()

    @property
    def tokens(self) -> int:
        return self.max_steps * self.tokens_per_step

    @property
    def non_embedding_params(self) -> int:
        return self.model.parameter_count()["non_embedding"]

    @property
    def trained_steps(self) -> int:
        """Steps this run trains itself; a branch inherits its first ``fork_step``."""
        return self.max_steps - (self.fork_step or 0)

    @property
    def flops(self) -> float:
        return self.model.flops_per_token() * self.trained_steps * self.tokens_per_step


@dataclass(frozen=True)
class StudyArm:
    arm: str
    stage: str
    runs: tuple[StudyRun, ...]

    @property
    def flops(self) -> float:
        return sum(run.flops for run in self.runs)


class StudySettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
Stage = Literal["smoke", "lr-gate", "study"]


class StudyOptimizer(StudySettings):
    learning_rate: PositiveFloat
    min_lr_ratio: Annotated[float, Field(ge=0, le=1)]
    warmup_steps: Annotated[int, Field(ge=0)]
    decay_fraction: Annotated[float, Field(ge=0, lt=1)]
    weight_decay: Annotated[float, Field(ge=0)]
    grad_clip: PositiveFloat
    micro_batch_size: PositiveInt
    grad_accum_steps: PositiveInt


class StudyData(StudySettings):
    volume: str
    train_manifest: str
    validation_manifest: str
    tokenizer: str
    shuffle_seed: int

    @field_validator("train_manifest", "validation_manifest", "tokenizer")
    @classmethod
    def safe_volume_path(cls, value: str) -> str:
        return relative_volume_path(value)


class StudyEval(StudySettings):
    eval_token_budget: PositiveInt
    batch_size: PositiveInt
    expected_token_batch_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class StudyHeldout(StudySettings):
    volume: str
    checkpoint: str
    tokens: PositiveInt
    fit_arms: tuple[str, ...]

    @field_validator("checkpoint")
    @classmethod
    def safe_checkpoint_path(cls, value: str) -> str:
        return relative_volume_path(value)


class StudyShape(StudySettings):
    embedding_dim: PositiveInt
    layers: PositiveInt
    heads: PositiveInt
    kv_heads: PositiveInt
    feedforward_dim: PositiveInt


class ArmSpec(StudySettings):
    arm: Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
    stage: Stage
    shape: str
    seed: int
    steps: PositiveInt | None = None
    tokens: PositiveInt | None = None
    multiple: PositiveInt | PositiveFloat | None = None
    learning_rate: PositiveFloat | None = None
    warmup_steps: Annotated[int, Field(ge=0)] | None = None
    branch_multiples: tuple[PositiveInt | PositiveFloat, ...] = ()
    branch_steps: tuple[PositiveInt, ...] = ()

    @model_validator(mode="after")
    def one_budget(self) -> ArmSpec:
        if sum(value is not None for value in (self.steps, self.tokens, self.multiple)) != 1:
            raise ValueError(f"arm {self.arm}: set exactly one of steps, tokens, multiple")
        return self


class StudyConfig(StudySettings):
    study_id: str
    run_card: str
    base_config: str
    optimizer: StudyOptimizer
    data: StudyData
    eval: StudyEval
    heldout: StudyHeldout
    shapes: dict[str, StudyShape]
    arms: tuple[ArmSpec, ...]


@dataclass(frozen=True)
class StudyPlan:
    study_id: str
    optimizer: StudyOptimizer
    data: StudyData
    eval: StudyEval
    heldout: StudyHeldout
    tokens_per_step: int
    arms: tuple[StudyArm, ...]

    def arms_for(self, stage: str, names: tuple[str, ...] = ()) -> tuple[StudyArm, ...]:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; known: {STAGES}")
        selected = tuple(arm for arm in self.arms if arm.stage == stage)
        if names:
            unknown = set(names) - {arm.arm for arm in selected}
            if unknown:
                raise ValueError(f"no {stage} arms named {sorted(unknown)}")
            selected = tuple(arm for arm in selected if arm.arm in names)
        return selected


def load_study_config(path: Path) -> StudyConfig:
    try:
        return StudyConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        detail = error.errors(include_url=False)[0]
        field = ".".join(str(part) for part in detail["loc"]) or "JSON"
        raise ValueError(
            f"invalid study config {path}: {field}: {detail['msg']}. Fix the study JSON."
        ) from error
    except OSError as error:
        raise ValueError(f"cannot read study config {path}: {error}") from error


def load_study_plan(path: Path) -> StudyPlan:
    config = load_study_config(path)
    try:
        base = parse_run_config(
            Path(config.base_config).read_text(encoding="utf-8"), Path(config.base_config)
        )
    except OSError as error:
        raise ValueError(f"cannot read base config {config.base_config}: {error}") from error
    return study_plan(config, base.model)


def study_plan(config: StudyConfig, base_model: BackboneConfig) -> StudyPlan:
    optimizer = config.optimizer
    tokens_per_step = (
        optimizer.micro_batch_size * optimizer.grad_accum_steps * base_model.context_length
    )
    arms = tuple(
        _expand_arm(
            entry,
            shapes=config.shapes,
            base_model=base_model,
            optimizer=optimizer,
            tokens_per_step=tokens_per_step,
        )
        for entry in config.arms
    )
    names = [arm.arm for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    return StudyPlan(
        study_id=config.study_id,
        optimizer=optimizer,
        data=config.data,
        eval=config.eval,
        heldout=config.heldout,
        tokens_per_step=tokens_per_step,
        arms=arms,
    )


def _expand_arm(
    entry: ArmSpec,
    *,
    shapes: dict[str, StudyShape],
    base_model: BackboneConfig,
    optimizer: StudyOptimizer,
    tokens_per_step: int,
) -> StudyArm:
    arm, stage = entry.arm, entry.stage
    if stage not in STAGES:
        raise ValueError(f"arm {arm}: unknown stage {stage!r}")
    if entry.shape not in shapes:
        raise ValueError(f"arm {arm}: unknown shape {entry.shape!r}")
    model = replace(base_model, name=entry.shape, **shapes[entry.shape].model_dump())
    non_embedding = model.parameter_count()["non_embedding"]
    if entry.steps is not None:
        max_steps = entry.steps
    elif entry.tokens is not None:
        max_steps = round(entry.tokens / tokens_per_step)
    else:
        assert entry.multiple is not None
        max_steps = round(entry.multiple * non_embedding / tokens_per_step)
    learning_rate = (
        entry.learning_rate if entry.learning_rate is not None else optimizer.learning_rate
    )
    seed = entry.seed
    warmup_steps = entry.warmup_steps if entry.warmup_steps is not None else optimizer.warmup_steps
    decay_fraction = optimizer.decay_fraction
    if max_steps <= warmup_steps:
        raise ValueError(f"arm {arm}: {max_steps} steps do not exceed {warmup_steps} warmup steps")

    branch_specs = [
        (f"branch-{multiple}x", round(float(multiple) * non_embedding / tokens_per_step))
        for multiple in entry.branch_multiples
    ] + [(f"branch-{steps}steps", int(steps)) for steps in entry.branch_steps]
    if len({name for name, _ in branch_specs}) != len(branch_specs):
        raise ValueError(f"arm {arm}: branch names must be unique")
    branches = []
    for name, branch_steps in branch_specs:
        fork_step = wsd_decay_start(max_steps=branch_steps, decay_fraction=decay_fraction)
        main_decay_start = wsd_decay_start(max_steps=max_steps, decay_fraction=decay_fraction)
        if not warmup_steps <= fork_step <= main_decay_start or branch_steps >= max_steps:
            raise ValueError(
                f"arm {arm}: {name} forks at step {fork_step}, outside the main "
                f"run's peak-LR phase [{warmup_steps}, {main_decay_start}]"
            )
        branches.append(
            StudyRun(
                arm=arm,
                run=name,
                model=model,
                max_steps=branch_steps,
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                seed=seed,
                tokens_per_step=tokens_per_step,
                branch_of=MAIN_RUN,
                fork_step=fork_step,
            )
        )
    main = StudyRun(
        arm=arm,
        run=MAIN_RUN,
        model=model,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        seed=seed,
        tokens_per_step=tokens_per_step,
        checkpoint_steps=tuple(
            sorted({branch.fork_step for branch in branches if branch.fork_step is not None})
        ),
    )
    return StudyArm(arm=arm, stage=stage, runs=(main, *branches))
