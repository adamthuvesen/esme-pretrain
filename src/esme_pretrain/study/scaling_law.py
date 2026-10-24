"""Hoffmann et al. (2022) approach-3 scaling-law fit for the scaling study.

L(N, D) = E + A / N^alpha + B / D^beta, fit on log L with a Huber loss and L-BFGS
from a grid of initializations. The method is pre-registered in
docs/run-cards/scaling-study-b200.md, so every setting here is part of that record.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from esme_pretrain.torch import torch

MIN_FIT_POINTS = 5
DEFAULT_HUBER_DELTA = 1e-3
ALPHA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
BETA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)
LOG_E_GRID = (-1.0, -0.5, 0.0, 0.5, 1.0)
LOG_A_GRID = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
LOG_B_GRID = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
SCREEN_ITERATIONS = 200
POLISHED_CANDIDATES = 4
POLISH_ITERATIONS = 1000


@dataclass(frozen=True)
class LossPoint:
    arm: str
    non_embedding_params: int
    tokens: int
    val_ce: float

    def __post_init__(self) -> None:
        if not isinstance(self.arm, str) or not self.arm:
            raise ValueError(f"loss point arm must be a non-empty string, got {self.arm!r}")
        for name in ("non_embedding_params", "tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"loss point {self.arm}: {name} must be a positive int, got {value!r}"
                )
        if (
            isinstance(self.val_ce, bool)
            or not isinstance(self.val_ce, (int, float))
            or not math.isfinite(self.val_ce)
            or self.val_ce <= 0
        ):
            raise ValueError(
                f"loss point {self.arm}: val_ce must be a positive finite number, "
                f"got {self.val_ce!r}"
            )


@dataclass(frozen=True)
class ScalingLawFit:
    E: float
    A: float
    alpha: float
    B: float
    beta: float
    objective: float

    def predict(self, non_embedding_params: float, tokens: float) -> float:
        _require_positive_target(non_embedding_params, tokens)
        return self.E + self.A / non_embedding_params**self.alpha + self.B / tokens**self.beta


@dataclass(frozen=True)
class BootstrapPrediction:
    fit: ScalingLawFit
    prediction: float
    lower: float
    upper: float
    interval: float
    resamples: int


def fit_scaling_law(
    points: Sequence[LossPoint], *, huber_delta: float = DEFAULT_HUBER_DELTA
) -> ScalingLawFit:
    """Fit L(N, D) from the full Hoffmann initialization grid.

    Screen: all 4,500 grid initializations run `SCREEN_ITERATIONS` steps as one batched
    L-BFGS problem. The objective is a sum of independent per-init objectives, so each
    row gets its own gradient, but the rows share one line search and curvature history.
    Polish: the `POLISHED_CANDIDATES` best screened rows then run to convergence one at a
    time with their own L-BFGS state, and the lowest objective wins.
    """
    log_n, log_d, log_l = _fit_tensors(points, huber_delta)
    params, objective = _minimize(
        _initialization_tensor(), log_n, log_d, log_l, huber_delta, screen=True
    )
    return _fit_from_params(params[0], float(objective[0]))


def bootstrap_prediction(
    points: Sequence[LossPoint],
    *,
    non_embedding_params: int,
    tokens: int,
    resamples: int = 1000,
    seed: int = 0,
    interval: float = 0.9,
    huber_delta: float = DEFAULT_HUBER_DELTA,
) -> BootstrapPrediction:
    """Percentile bootstrap interval for one prediction, resampling points with replacement.

    Warm start: the full-data fit runs the whole grid once. Each resample refits only
    from the `POLISHED_CANDIDATES` polished full-data results (the first is the full-data
    optimum), each with its own L-BFGS run to convergence. A full grid per resample would
    cost 1,000 full fits. With 12 points, 1,000 resamples take about one minute on a
    laptop CPU. The random state is numpy's default_rng(seed), so the interval is
    reproducible.
    """
    _require_positive_target(non_embedding_params, tokens)
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not 0 < interval < 1:
        raise ValueError(f"interval must be in (0, 1), got {interval}")
    log_n, log_d, log_l = _fit_tensors(points, huber_delta)
    warm_starts, full_objective = _minimize(
        _initialization_tensor(), log_n, log_d, log_l, huber_delta, screen=True
    )
    full_fit = _fit_from_params(warm_starts[0], float(full_objective[0]))

    rng = np.random.default_rng(seed)
    predictions = np.empty(resamples, dtype=np.float64)
    for resample in range(resamples):
        index = torch.from_numpy(rng.integers(0, len(points), size=len(points)))
        params, objective = _minimize(
            warm_starts, log_n[index], log_d[index], log_l[index], huber_delta, screen=False
        )
        predictions[resample] = _fit_from_params(params[0], float(objective[0])).predict(
            non_embedding_params, tokens
        )

    tail = (1 - interval) / 2 * 100
    lower, upper = np.percentile(predictions, [tail, 100 - tail])
    return BootstrapPrediction(
        fit=full_fit,
        prediction=full_fit.predict(non_embedding_params, tokens),
        lower=float(lower),
        upper=float(upper),
        interval=interval,
        resamples=resamples,
    )


def initialization_grid() -> dict[str, list[float]]:
    return {
        "alpha": list(ALPHA_GRID),
        "beta": list(BETA_GRID),
        "log_E": list(LOG_E_GRID),
        "log_A": list(LOG_A_GRID),
        "log_B": list(LOG_B_GRID),
    }


def _require_positive_target(non_embedding_params: float, tokens: float) -> None:
    if non_embedding_params <= 0 or tokens <= 0:
        raise ValueError(
            "prediction target must have positive non_embedding_params and tokens, "
            f"got {non_embedding_params!r} and {tokens!r}"
        )


def _fit_tensors(
    points: Sequence[LossPoint], huber_delta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(points) < MIN_FIT_POINTS:
        raise ValueError(
            f"scaling-law fit needs at least {MIN_FIT_POINTS} points, got {len(points)}"
        )
    if not huber_delta > 0:
        raise ValueError(f"huber_delta must be positive, got {huber_delta}")
    for point in points:
        if not isinstance(point, LossPoint):
            raise ValueError(f"expected LossPoint, got {type(point).__name__}")
    log_n = torch.tensor([math.log(p.non_embedding_params) for p in points], dtype=torch.float64)
    log_d = torch.tensor([math.log(p.tokens) for p in points], dtype=torch.float64)
    log_l = torch.tensor([math.log(p.val_ce) for p in points], dtype=torch.float64)
    return log_n, log_d, log_l


def _objective(
    params: torch.Tensor,
    log_n: torch.Tensor,
    log_d: torch.Tensor,
    log_l: torch.Tensor,
    huber_delta: float,
) -> torch.Tensor:
    """Per-row objective for params of shape (rows, 5): a, alpha, b, beta, e.

    The log tensors have shape (points,) shared by every row, or (rows, points).
    """
    a, alpha, b, beta, e = (column[:, None] for column in params.unbind(dim=1))
    params_term = a - alpha * log_n
    tokens_term = b - beta * log_d
    terms = torch.stack(
        torch.broadcast_tensors(params_term, tokens_term, e.expand_as(params_term)), dim=0
    )
    predicted = torch.logsumexp(terms, dim=0)
    residual = predicted - log_l
    return torch.nn.functional.huber_loss(
        residual, torch.zeros_like(residual), reduction="none", delta=huber_delta
    ).sum(dim=1)


def _run_lbfgs(
    params: torch.Tensor,
    log_n: torch.Tensor,
    log_d: torch.Tensor,
    log_l: torch.Tensor,
    huber_delta: float,
    max_iter: int,
) -> torch.Tensor:
    params = params.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [params],
        lr=1.0,
        max_iter=max_iter,
        history_size=20,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = _objective(params, log_n, log_d, log_l, huber_delta).sum() / huber_delta**2
        loss.backward()
        return loss

    optimizer.step(closure)
    return params.detach()


def _minimize(
    starts: torch.Tensor,
    log_n: torch.Tensor,
    log_d: torch.Tensor,
    log_l: torch.Tensor,
    huber_delta: float,
    *,
    screen: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return refined params (raw a, alpha, b, beta, e) and objectives, best first.

    L-BFGS runs on log N and log D centered at their means, with a and b shifted to
    match (a' = a - alpha * mean log N). The model and objective are unchanged; the
    shift removes the a/alpha correlation that otherwise stalls L-BFGS in a flat valley.
    The loss L-BFGS sees is divided by huber_delta**2 so that residuals near delta give
    O(1) values; unscaled, the absolute tolerances in torch's L-BFGS stop it early.
    """
    center_n, center_d = log_n.mean(), log_d.mean()
    centered_n, centered_d = log_n - center_n, log_d - center_d
    candidates = starts.clone()
    candidates[:, 0] -= candidates[:, 1] * center_n
    candidates[:, 2] -= candidates[:, 3] * center_d
    if screen:
        candidates = _run_lbfgs(
            candidates, centered_n, centered_d, log_l, huber_delta, SCREEN_ITERATIONS
        )
        with torch.no_grad():
            screened_objective = _objective(candidates, centered_n, centered_d, log_l, huber_delta)
        order = torch.argsort(torch.nan_to_num(screened_objective, nan=math.inf), stable=True)
        candidates = candidates[order[:POLISHED_CANDIDATES]]
    refined = torch.cat(
        [
            _run_lbfgs(
                candidate[None, :], centered_n, centered_d, log_l, huber_delta, POLISH_ITERATIONS
            )
            for candidate in candidates
        ]
    )
    with torch.no_grad():
        objective = torch.nan_to_num(
            _objective(refined, centered_n, centered_d, log_l, huber_delta), nan=math.inf
        )
    refined[:, 0] += refined[:, 1] * center_n
    refined[:, 2] += refined[:, 3] * center_d
    ranking = torch.argsort(objective, stable=True)
    if not math.isfinite(float(objective[ranking[0]])):
        raise ValueError("scaling-law fit did not reach a finite objective from any start")
    return refined[ranking], objective[ranking]


def _initialization_tensor() -> torch.Tensor:
    return torch.tensor(
        [
            (log_a, alpha, log_b, beta, log_e)
            for alpha, beta, log_e, log_a, log_b in itertools.product(
                ALPHA_GRID, BETA_GRID, LOG_E_GRID, LOG_A_GRID, LOG_B_GRID
            )
        ],
        dtype=torch.float64,
    )


def _fit_from_params(params: torch.Tensor, objective: float) -> ScalingLawFit:
    log_a, alpha, log_b, beta, log_e = (float(value) for value in params)
    return ScalingLawFit(
        E=math.exp(log_e),
        A=math.exp(log_a),
        alpha=alpha,
        B=math.exp(log_b),
        beta=beta,
        objective=objective,
    )
