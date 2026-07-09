from __future__ import annotations

import argparse

from esme_pretrain import __version__
from esme_pretrain.cli.baselines import add_baseline_parsers
from esme_pretrain.cli.data import add_data_parsers
from esme_pretrain.cli.doctor import add_doctor_parser
from esme_pretrain.cli.postrun import add_postrun_parsers
from esme_pretrain.cli.pretrain import add_pretrain_parser
from esme_pretrain.cli.status import add_status_parser, handle_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="esme-pretrain")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(handler=handle_status, json=False)

    add_status_parser(subparsers)
    add_doctor_parser(subparsers)
    add_data_parsers(subparsers)
    add_pretrain_parser(subparsers)
    add_postrun_parsers(subparsers)
    add_baseline_parsers(subparsers)

    return parser
