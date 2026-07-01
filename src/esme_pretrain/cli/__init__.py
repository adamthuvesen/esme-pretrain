from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable

from esme_pretrain.cli.doctor import DoctorCheck, run_doctor
from esme_pretrain.cli.parser import build_parser

__all__ = ("DoctorCheck", "build_parser", "main", "run_doctor")

CommandHandler = Callable[[Namespace], int]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: CommandHandler = args.handler
    return handler(args)
