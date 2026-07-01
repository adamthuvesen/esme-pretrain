from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from esme_pretrain.launch.pretrain import build_pretrain_dry_run, load_pretrain_config


def add_pretrain_parser(subparsers: argparse._SubParsersAction) -> None:
    pretrain_214m_b200 = subparsers.add_parser(
        "pretrain-214m-b200",
        help=("Validate the 214M B200 pretrain config without launching external work."),
    )
    pretrain_214m_b200.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain_214m_b200.json"),
        help="214M B200 pretrain config JSON path.",
    )
    pretrain_214m_b200.add_argument(
        "--dry-run",
        action="store_true",
        help="Required. Validate launch readiness without downloading data or starting Modal.",
    )
    pretrain_214m_b200.add_argument(
        "--json", action="store_true", help="Emit machine-readable result."
    )
    pretrain_214m_b200.set_defaults(handler=handle_pretrain_214m_b200)


def handle_pretrain_214m_b200(args: argparse.Namespace) -> int:
    if not args.dry_run:
        print(
            f"{args.command} failed: --dry-run is required for the local CLI",
            file=sys.stderr,
        )
        return 2
    try:
        config = load_pretrain_config(args.config)
    except ValueError as error:
        print(f"{args.command} failed: {error}", file=sys.stderr)
        return 2
    payload = build_pretrain_dry_run(config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"esme-pretrain {args.command} dry-run")
        print(f"status: {payload['status']}")
        print(f"train_token_budget: {payload['budgets']['train_token_budget']}")
        print(f"estimated_cost_usd: {payload['runtime']['estimated_cost_usd']}")
        print(f"launch_command: {payload['launch_command']}")
        print(f"will_download_data: {payload['will_download_data']}")
        print(f"will_start_modal_job: {payload['will_start_modal_job']}")
    return 0
