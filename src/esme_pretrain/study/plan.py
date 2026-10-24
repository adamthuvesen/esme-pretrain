"""Expand the scaling-study config into concrete training runs.

An arm is one model shape trained once. A ``multiple`` arm trains to
``multiple x N`` tokens (N = non-embedding params). Its ``branch_multiples``
fork cooldown branches from the arm's main run: each branch shares the main
run's past up to its own decay start, then decays over its last
``decay_fraction``. See ``docs/run-cards/scaling-study-b200.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from esme_pretrain.modeling.backbone import BackboneConfig
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


@dataclass(frozen=True)
class StudyPlan:
    study_id: str
    optimizer: dict[str, Any]
    data: dict[str, Any]
    eval: dict[str, Any]
    heldout: dict[str, Any]
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


def load_study_plan(path: Path) -> StudyPlan:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base = json.loads(Path(payload["base_config"]).read_text(encoding="utf-8"))
    return study_plan_from_payloads(payload, base)


def study_plan_from_payloads(payload: dict[str, Any], base: dict[str, Any]) -> StudyPlan:
    """Build the plan from the study config and its base run config, already parsed."""
    optimizer = payload["optimizer"]
    base_model = BackboneConfig.from_dict(base["model"])
    tokens_per_step = (
        int(optimizer["micro_batch_size"])
        * int(optimizer["grad_accum_steps"])
        * base_model.context_length
    )
    shapes = payload["shapes"]
    arms = tuple(
        _expand_arm(
            entry,
            shapes=shapes,
            base_model=base_model,
            optimizer=optimizer,
            tokens_per_step=tokens_per_step,
        )
        for entry in payload["arms"]
    )
    names = [arm.arm for arm in arms]
    if len(set(names)) != len(names):
        raise ValueError("arm names must be unique")
    return StudyPlan(
        study_id=payload["study_id"],
        optimizer=optimizer,
        data=payload["data"],
        eval=payload["eval"],
        heldout=payload["heldout"],
        tokens_per_step=tokens_per_step,
        arms=arms,
    )


def _expand_arm(
    entry: dict[str, Any],
    *,
    shapes: dict[str, dict[str, int]],
    base_model: BackboneConfig,
    optimizer: dict[str, Any],
    tokens_per_step: int,
) -> StudyArm:
    arm, stage = entry["arm"], entry["stage"]
    if stage not in STAGES:
        raise ValueError(f"arm {arm}: unknown stage {stage!r}")
    if entry["shape"] not in shapes:
        raise ValueError(f"arm {arm}: unknown shape {entry['shape']!r}")
    model = replace(base_model, name=entry["shape"], **shapes[entry["shape"]])
    non_embedding = model.parameter_count()["non_embedding"]
    budgets = [key for key in ("steps", "tokens", "multiple") if key in entry]
    if len(budgets) != 1:
        raise ValueError(f"arm {arm}: set exactly one of steps, tokens, multiple")
    if "steps" in entry:
        max_steps = int(entry["steps"])
    elif "tokens" in entry:
        max_steps = round(int(entry["tokens"]) / tokens_per_step)
    else:
        max_steps = round(float(entry["multiple"]) * non_embedding / tokens_per_step)
    learning_rate = float(entry.get("learning_rate", optimizer["learning_rate"]))
    seed = int(entry["seed"])
    warmup_steps = int(entry.get("warmup_steps", optimizer["warmup_steps"]))
    decay_fraction = float(optimizer["decay_fraction"])
    if max_steps <= warmup_steps:
        raise ValueError(f"arm {arm}: {max_steps} steps do not exceed {warmup_steps} warmup steps")

    branch_specs = [
        (f"branch-{multiple}x", round(float(multiple) * non_embedding / tokens_per_step))
        for multiple in entry.get("branch_multiples", [])
    ] + [(f"branch-{steps}steps", int(steps)) for steps in entry.get("branch_steps", [])]
    branches = []
    for name, branch_steps in branch_specs:
        fork_step = wsd_decay_start(max_steps=branch_steps, decay_fraction=decay_fraction)
        main_decay_start = wsd_decay_start(max_steps=max_steps, decay_fraction=decay_fraction)
        if not warmup_steps <= fork_step <= main_decay_start:
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
        checkpoint_steps=tuple(sorted(branch.fork_step for branch in branches)),
    )
    return StudyArm(arm=arm, stage=stage, runs=(main, *branches))
