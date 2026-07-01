from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from esme_pretrain import __version__
from esme_pretrain.status import current_status


def add_status_parser(subparsers: argparse._SubParsersAction) -> None:
    status = subparsers.add_parser("status", help="Show current scaffold status.")
    status.add_argument("--json", action="store_true", help="Emit machine-readable status.")
    status.set_defaults(handler=handle_status)


def handle_status(args: argparse.Namespace) -> int:
    if args.json:
        status = current_status()
        payload = asdict(status)
        payload["pipeline"] = [asdict(stage) for stage in status.pipeline]
        payload["version"] = __version__
        print(json.dumps(payload, indent=2))
    else:
        status = current_status()
        print(f"esme-pretrain {__version__}")
        print(f"state: {status.state}")
        print(status.summary)
        print("")
        print("pipeline:")
        for stage in status.pipeline:
            print(f"  {stage.order}. {stage.name} ({stage.status})")
        print("")
        print(f"next: {status.next_milestone}")
        print(f"run_card: {status.run_card_path}")
        print(f"policy: {status.spend_policy}")
    return 0
