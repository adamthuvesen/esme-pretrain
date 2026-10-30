from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esme_pretrain.study.plan import load_study_plan
from esme_pretrain.study.prediction import (
    BOOTSTRAP_INTERVAL,
    PREREGISTERED_RESAMPLES,
    PREREGISTERED_SEED,
    fit_artifact,
)
from esme_pretrain.study.results import load_loss_points, load_study_fit_input
from esme_pretrain.study.scaling_law import bootstrap_prediction


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
        "--config",
        type=Path,
        default=Path("configs/scaling_study.json"),
        help="Study config that defines the planned result rows.",
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
        "--resamples",
        default=PREREGISTERED_RESAMPLES,
        type=int,
        help=f"Bootstrap resamples. Defaults to {PREREGISTERED_RESAMPLES}.",
    )
    scaling_fit.add_argument(
        "--seed",
        default=PREREGISTERED_SEED,
        type=int,
        help=f"Bootstrap random seed. Defaults to {PREREGISTERED_SEED}.",
    )
    scaling_fit.add_argument("--output", required=True, type=Path, help="Output fit JSON.")
    scaling_fit.add_argument("--json", action="store_true", help="Emit machine-readable result.")
    scaling_fit.set_defaults(handler=handle_scaling_fit)


def handle_scaling_fit(args: argparse.Namespace) -> int:
    try:
        study_evidence = None
        if args.points is not None:
            points = load_loss_points(args.points)
        else:
            study_input = load_study_fit_input(
                args.results_dir, tuple(args.arm), load_study_plan(args.config)
            )
            points = list(study_input.points)
            study_evidence = study_input.evidence
        bootstrap = bootstrap_prediction(
            points,
            non_embedding_params=args.predict_non_embedding_params,
            tokens=args.predict_tokens,
            resamples=args.resamples,
            seed=args.seed,
            interval=BOOTSTRAP_INTERVAL,
        )
    except ValueError as error:
        print(f"scaling-fit failed: {str(error).replace(chr(10), ' ')}", file=sys.stderr)
        return 2
    fit = bootstrap.fit
    payload = fit_artifact(
        points,
        bootstrap,
        non_embedding_params=args.predict_non_embedding_params,
        tokens=args.predict_tokens,
        points_path=str(args.points),
        seed=args.seed,
        study_evidence=study_evidence,
    ).model_dump(mode="json")
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
