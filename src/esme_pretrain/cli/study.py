from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from esme_pretrain.study.scaling_law import (
    DEFAULT_HUBER_DELTA,
    POLISH_ITERATIONS,
    POLISHED_CANDIDATES,
    SCREEN_ITERATIONS,
    LossPoint,
    bootstrap_prediction,
    initialization_grid,
)

BOOTSTRAP_INTERVAL = 0.9
LOSS_POINT_FIELDS = ("arm", "non_embedding_params", "tokens", "val_ce")


def add_study_parsers(subparsers: argparse._SubParsersAction) -> None:
    scaling_fit = subparsers.add_parser(
        "scaling-fit",
        help="Fit the pre-registered scaling law and bootstrap one held-out prediction.",
    )
    scaling_fit.add_argument(
        "--points", required=True, type=Path, help="JSON list of loss points to fit."
    )
    scaling_fit.add_argument(
        "--predict-non-embedding-params",
        required=True,
        type=int,
        help="Non-embedding parameter count of the held-out prediction.",
    )
    scaling_fit.add_argument(
        "--predict-tokens",
        required=True,
        type=int,
        help="Training tokens of the held-out prediction.",
    )
    scaling_fit.add_argument(
        "--resamples", default=1000, type=int, help="Bootstrap resamples. Defaults to 1000."
    )
    scaling_fit.add_argument(
        "--seed", default=0, type=int, help="Bootstrap random seed. Defaults to 0."
    )
    scaling_fit.add_argument("--output", required=True, type=Path, help="Output fit JSON.")
    scaling_fit.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    scaling_fit.set_defaults(handler=handle_scaling_fit)


def handle_scaling_fit(args: argparse.Namespace) -> int:
    try:
        points = load_loss_points(args.points)
        bootstrap = bootstrap_prediction(
            points,
            non_embedding_params=args.predict_non_embedding_params,
            tokens=args.predict_tokens,
            resamples=args.resamples,
            seed=args.seed,
            interval=BOOTSTRAP_INTERVAL,
        )
    except ValueError as error:
        print(f"scaling-fit failed: {error}", file=sys.stderr)
        return 2
    fit = bootstrap.fit
    payload: dict[str, Any] = {
        "points_path": str(args.points),
        "point_count": len(points),
        "fit": {
            "E": fit.E,
            "A": fit.A,
            "alpha": fit.alpha,
            "B": fit.B,
            "beta": fit.beta,
            "objective": fit.objective,
        },
        "prediction": {
            "non_embedding_params": args.predict_non_embedding_params,
            "tokens": args.predict_tokens,
            "val_ce": bootstrap.prediction,
            "interval": bootstrap.interval,
            "lower": bootstrap.lower,
            "upper": bootstrap.upper,
        },
        "settings": {
            "form": "L(N, D) = E + A / N^alpha + B / D^beta",
            "huber_delta": DEFAULT_HUBER_DELTA,
            "initialization_grid": initialization_grid(),
            "screen_iterations": SCREEN_ITERATIONS,
            "polished_candidates": POLISHED_CANDIDATES,
            "polish_iterations": POLISH_ITERATIONS,
            "resamples": bootstrap.resamples,
            "seed": args.seed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("esme-pretrain scaling-fit")
        print(f"output: {args.output}")
        print(f"alpha: {fit.alpha:.4f}  beta: {fit.beta:.4f}  E: {fit.E:.4f}")
        print(
            f"prediction: {bootstrap.prediction:.4f} "
            f"({bootstrap.interval:.0%} interval {bootstrap.lower:.4f} to {bootstrap.upper:.4f})"
        )
    return 0


def load_loss_points(path: Path) -> list[LossPoint]:
    if not path.is_file():
        raise ValueError(f"points file does not exist: {path}")
    try:
        records = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"points file is not valid JSON: {path}: {error}") from error
    if not isinstance(records, list):
        raise ValueError(f"points file must hold a JSON list: {path}")
    points = []
    for position, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != set(LOSS_POINT_FIELDS):
            raise ValueError(
                f"point {position} must be an object with exactly the fields "
                f"{', '.join(LOSS_POINT_FIELDS)}"
            )
        points.append(LossPoint(**record))
    return points
