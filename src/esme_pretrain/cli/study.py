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
    source = scaling_fit.add_mutually_exclusive_group(required=True)
    source.add_argument("--points", type=Path, help="JSON list of loss points to fit.")
    source.add_argument(
        "--results-dir",
        type=Path,
        help="Study results directory (<arm>/<run>.json rows); fit the --arm rows.",
    )
    scaling_fit.add_argument(
        "--arm", action="append", default=[], help="Arm to fit from --results-dir. Repeat."
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
        if args.points is not None:
            points = load_loss_points(args.points)
        else:
            points = load_study_loss_points(args.results_dir, tuple(args.arm))
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
        "points": [vars(point) for point in points],
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


def load_study_loss_points(results_dir: Path, arms: tuple[str, ...]) -> list[LossPoint]:
    """One loss point per run row the study wrote under ``<results_dir>/<arm>/``.

    An arm whose container stopped part way has some rows but not all; each row
    names the arm's full run set, so a partial arm is refused, not fit.
    """
    if not arms:
        raise ValueError("--results-dir needs at least one --arm")
    points = []
    for arm in arms:
        rows = sorted((results_dir / arm).glob("*.json"))
        if not rows:
            raise ValueError(f"no result rows for arm {arm!r} in {results_dir / arm}")
        arm_rows = [json.loads(row_path.read_text()) for row_path in rows]
        for row_path, row in zip(rows, arm_rows, strict=True):
            if row.get("arm") != arm:
                raise ValueError(f"{row_path} belongs to arm {row.get('arm')!r}, not {arm!r}")
            if "arm_runs" not in row:
                raise ValueError(f"{row_path} has no arm_runs field to check the arm is complete")
        written = sorted(row["run"] for row in arm_rows)
        expected = sorted(arm_rows[0]["arm_runs"])
        if written != expected:
            raise ValueError(f"arm {arm!r} is incomplete: wrote runs {written}, planned {expected}")
        for row in arm_rows:
            points.append(
                LossPoint(
                    arm=f"{arm}/{row['run']}",
                    non_embedding_params=int(row["non_embedding_params"]),
                    tokens=int(row["tokens"]),
                    val_ce=float(row["val_ce"]),
                )
            )
    return points
