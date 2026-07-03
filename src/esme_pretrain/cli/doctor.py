from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_PROJECT_FILES = ("AGENTS.md", "README.md", "pyproject.toml")
DEFAULT_EXPECTED_ORIGIN = "adamthuvesen/esme-pretrain"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str


def _git_remote_urls(repo_root: Path) -> dict[str, str]:
    if not (repo_root / ".git").exists():
        return {}
    result = subprocess.run(
        ["git", "-C", str(repo_root), "remote", "-v"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"<error>": result.stderr.strip()}
    remotes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            remotes.setdefault(parts[0], parts[1])
    return remotes


def run_doctor(
    repo_root: Path, expected_origin: str = DEFAULT_EXPECTED_ORIGIN
) -> tuple[bool, list[DoctorCheck]]:
    repo_root = repo_root.resolve()
    checks: list[DoctorCheck] = []

    checks.append(
        DoctorCheck(
            "python",
            sys.version_info >= (3, 11),
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    for file_name in REQUIRED_PROJECT_FILES:
        path = repo_root / file_name
        checks.append(DoctorCheck(file_name, path.exists(), str(path)))

    remotes = _git_remote_urls(repo_root)
    origin_url = remotes.get("origin")
    checks.append(
        DoctorCheck(
            "git remote",
            origin_url is None or expected_origin in origin_url,
            "no origin configured" if origin_url is None else origin_url,
        )
    )

    return all(check.ok for check in checks), checks


def add_doctor_parser(subparsers: argparse._SubParsersAction) -> None:
    doctor = subparsers.add_parser("doctor", help="Check local scaffold assumptions.")
    doctor.add_argument(
        "--repo-root",
        default=".",
        type=Path,
        help="Repository root to check. Defaults to the current directory.",
    )
    doctor.add_argument(
        "--expected-origin",
        default=DEFAULT_EXPECTED_ORIGIN,
        help=(
            "Owner/repo substring expected in the origin URL. "
            f"Defaults to {DEFAULT_EXPECTED_ORIGIN}."
        ),
    )
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable checks.")
    doctor.set_defaults(handler=handle_doctor)


def handle_doctor(args: argparse.Namespace) -> int:
    repo_root = args.repo_root
    ok, checks = run_doctor(repo_root, expected_origin=args.expected_origin)
    if args.json:
        print(json.dumps({"ok": ok, "checks": [asdict(check) for check in checks]}, indent=2))
    else:
        print("esme-pretrain doctor")
        for check in checks:
            prefix = "ok" if check.ok else "fail"
            print(f"[{prefix}] {check.name}: {check.detail}")
    return 0 if ok else 1
