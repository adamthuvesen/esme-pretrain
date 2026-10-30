"""Scaling-fit settings, output artifacts, and the pre-registered held-out prediction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from esme_pretrain.artifacts.hashes import file_sha256
from esme_pretrain.study.plan import StudyPlan
from esme_pretrain.study.results import (
    LossPointRecord,
    StudyFitEvidence,
    point_sha256,
    study_fit_sha256,
    validate_evaluation_identity,
)
from esme_pretrain.study.scaling_law import (
    DEFAULT_HUBER_DELTA,
    POLISH_ITERATIONS,
    POLISHED_CANDIDATES,
    SCREEN_ITERATIONS,
    BootstrapPrediction,
    LossPoint,
    initialization_grid,
)

BOOTSTRAP_INTERVAL = 0.9
PREREGISTERED_RESAMPLES = 1000
PREREGISTERED_SEED = 0


class PredictionFields(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class InitializationGrid(PredictionFields):
    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    log_E: tuple[float, ...]
    log_A: tuple[float, ...]
    log_B: tuple[float, ...]


class FitSettings(PredictionFields):
    form: str
    huber_delta: float
    initialization_grid: InitializationGrid
    screen_iterations: int
    polished_candidates: int
    polish_iterations: int
    resamples: int
    seed: int


def fit_settings(*, resamples: int, seed: int) -> FitSettings:
    grid = initialization_grid()
    return FitSettings(
        form="L(N, D) = E + A / N^alpha + B / D^beta",
        huber_delta=DEFAULT_HUBER_DELTA,
        initialization_grid=InitializationGrid(
            alpha=tuple(grid["alpha"]),
            beta=tuple(grid["beta"]),
            log_E=tuple(grid["log_E"]),
            log_A=tuple(grid["log_A"]),
            log_B=tuple(grid["log_B"]),
        ),
        screen_iterations=SCREEN_ITERATIONS,
        polished_candidates=POLISHED_CANDIDATES,
        polish_iterations=POLISH_ITERATIONS,
        resamples=resamples,
        seed=seed,
    )


class FitParameters(PredictionFields):
    E: float
    A: float
    alpha: float
    B: float
    beta: float
    objective: float


class PredictionTarget(PredictionFields):
    non_embedding_params: Annotated[int, Field(gt=0)]
    tokens: Annotated[int, Field(gt=0)]
    val_ce: Annotated[float, Field(gt=0)]
    interval: Annotated[float, Field(gt=0, lt=1)]
    lower: float
    upper: float

    @model_validator(mode="after")
    def ordered_interval(self) -> PredictionTarget:
        if self.lower > self.upper:
            raise ValueError("prediction lower bound exceeds upper bound")
        return self


class FitArtifact(PredictionFields):
    points: tuple[LossPointRecord, ...]
    settings: FitSettings
    prediction: PredictionTarget
    points_path: str
    point_count: int
    fit: FitParameters
    study_evidence: StudyFitEvidence | None = None

    @model_validator(mode="after")
    def consistent_points(self) -> FitArtifact:
        if self.point_count != len(self.points):
            raise ValueError("point_count differs from the written points")
        if len({point.arm for point in self.points}) != len(self.points):
            raise ValueError("duplicate prediction fit points")
        return self


class PredictionReference(PredictionFields):
    path: str
    sha256: str
    val_ce: float


def fit_artifact(
    points: Sequence[LossPoint],
    bootstrap: BootstrapPrediction,
    *,
    non_embedding_params: int,
    tokens: int,
    points_path: str,
    seed: int,
    study_evidence: StudyFitEvidence | None = None,
) -> FitArtifact:
    fit = bootstrap.fit
    return FitArtifact(
        points_path=points_path,
        study_evidence=study_evidence,
        point_count=len(points),
        points=tuple(
            LossPointRecord(
                arm=p.arm,
                non_embedding_params=p.non_embedding_params,
                tokens=p.tokens,
                val_ce=p.val_ce,
            )
            for p in points
        ),
        fit=FitParameters(
            E=fit.E, A=fit.A, alpha=fit.alpha, B=fit.B, beta=fit.beta, objective=fit.objective
        ),
        prediction=PredictionTarget(
            non_embedding_params=non_embedding_params,
            tokens=tokens,
            val_ce=bootstrap.prediction,
            interval=bootstrap.interval,
            lower=bootstrap.lower,
            upper=bootstrap.upper,
        ),
        settings=fit_settings(resamples=bootstrap.resamples, seed=seed),
    )


def planned_fit_points(plan: StudyPlan) -> set[tuple[str, int, int]]:
    return {
        (f"{arm.arm}/{run.run}", run.non_embedding_params, run.tokens)
        for arm in plan.arms_for("study", plan.heldout.fit_arms)
        for run in arm.runs
    }


def written_prediction(
    path: Path | None,
    *,
    non_embedding_params: int,
    tokens: int,
    plan: StudyPlan,
) -> PredictionReference:
    if path is None or not path.is_file():
        raise ValueError("the heldout stage needs --prediction, the written scaling-fit output")
    try:
        payload = FitArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        detail = error.errors(include_url=False)[0]
        raise ValueError(
            f"cannot read prediction {path}: {detail['loc']}: {detail['msg']}"
        ) from error
    except OSError as error:
        raise ValueError(f"cannot read prediction {path}: {error}") from error
    target = payload.prediction
    if (target.non_embedding_params, target.tokens) != (non_embedding_params, tokens):
        raise ValueError(
            f"{path} predicts N={target.non_embedding_params}, D={target.tokens}; "
            f"the held-out point is N={non_embedding_params}, D={tokens}"
        )
    preregistered = fit_settings(resamples=PREREGISTERED_RESAMPLES, seed=PREREGISTERED_SEED)
    if payload.settings != preregistered or target.interval != BOOTSTRAP_INTERVAL:
        changed = [
            f"{field}: got {getattr(payload.settings, field)!r}, "
            f"expected {getattr(preregistered, field)!r}"
            for field in FitSettings.model_fields
            if field != "initialization_grid"
            and getattr(payload.settings, field) != getattr(preregistered, field)
        ]
        changed.extend(
            f"initialization_grid.{field} differs"
            for field in InitializationGrid.model_fields
            if getattr(payload.settings.initialization_grid, field)
            != getattr(preregistered.initialization_grid, field)
        )
        if target.interval != BOOTSTRAP_INTERVAL:
            changed.append(f"interval: got {target.interval}, expected {BOOTSTRAP_INTERVAL}")
        raise ValueError(
            f"{path} was not fit with the pre-registered settings: {', '.join(changed)}. "
            "Run scaling-fit with the run card settings."
        )
    fit_points = planned_fit_points(plan)
    written = [(p.arm, p.non_embedding_params, p.tokens) for p in payload.points]
    if len(written) != len(fit_points) or set(written) != fit_points:
        missing = sorted(fit_points - set(written))
        unexpected = sorted(set(written) - fit_points)
        raise ValueError(
            f"{path} is not fit on the pre-registered {len(fit_points)} points: "
            f"missing: {missing}, unexpected: {unexpected}. "
            "Fit the planned arms with the selected study config."
        )
    evidence = payload.study_evidence
    if evidence is None:
        raise ValueError(
            f"{path}: missing study_evidence. "
            "Exploratory --points fits cannot approve heldout evaluation. "
            "Regenerate with scaling-fit --results-dir and the planned --arm selections."
        )
    expected_arms = tuple(sorted(plan.heldout.fit_arms))
    expected_contract = {
        "study_id": plan.study_id,
        "fit_arms": expected_arms,
        "plan_sha256": study_fit_sha256(plan, expected_arms),
    }
    for field, expected in expected_contract.items():
        actual = getattr(evidence, field)
        if actual != expected:
            raise ValueError(
                f"{path}: study_evidence.{field}: got {actual!r}, expected {expected!r}. "
                "Regenerate the fit from this study config's result rows."
            )
    sources = {point.arm: point for point in evidence.points}
    written_labels = {point.arm for point in payload.points}
    if len(sources) != len(evidence.points) or sources.keys() != written_labels:
        raise ValueError(
            f"{path}: study_evidence points must match the fit exactly once: "
            f"missing {sorted(written_labels - sources.keys())}, "
            f"unexpected {sorted(sources.keys() - written_labels)}. "
            "Regenerate from complete study result rows."
        )
    runs = {
        f"{arm.arm}/{run.run}": run
        for arm in plan.arms_for("study", plan.heldout.fit_arms)
        for run in arm.runs
    }
    for point in payload.points:
        source = sources[point.arm]
        expected_hash = point_sha256(point)
        if source.point_sha256 != expected_hash:
            raise ValueError(
                f"{path}: study_evidence for {point.arm} has point hash "
                f"{source.point_sha256}, expected {expected_hash}. "
                "The fit points changed after evidence validation. "
                "Regenerate from study result rows."
            )
        validate_evaluation_identity(
            source=f"{path}:study_evidence:{point.arm}",
            plan=plan,
            context_length=runs[point.arm].model.context_length,
            token_count=source.val_eval_tokens,
            token_batch_sha256=source.val_token_batch_sha256,
        )
    return PredictionReference(path=str(path), sha256=file_sha256(path), val_ce=target.val_ce)
