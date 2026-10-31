"""No comment turns strict type checking off for a file or silences an import ban."""

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNGRADE = re.compile(r"^\s*#\s*pyright:\s*(?!strict\s*$|ignore\b)\S")
BAN_SUPPRESSION = re.compile(r"\bnoqa\b[^#\n]*\bTID251\b", re.IGNORECASE)


def is_python_file(path: Path) -> bool:
    if path.suffix:
        return path.suffix == ".py"
    with path.open("rb") as handle:
        first_line = handle.readline()
    return first_line.startswith(b"#!") and b"python" in first_line


def candidate_python_files() -> list[Path]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
    ).stdout.decode()
    paths = (ROOT / name for name in listing.split("\0") if name)
    return [path for path in paths if path.is_file() and is_python_file(path)]


def test_no_file_downgrades_type_checking_mode() -> None:
    downgrades = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in candidate_python_files()
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if DOWNGRADE.match(line) or BAN_SUPPRESSION.search(line)
    ]
    assert not downgrades, (
        "a `# pyright:` file comment weakens strict mode, or a `noqa` silences an import ban "
        "(add the file to the owner per-file-ignores in pyproject.toml instead): "
        + ", ".join(downgrades)
    )


def test_the_patterns_find_each_downgrade_and_spare_rule_ignores() -> None:
    for line in ("# pyright: basic", "#pyright: strict=false", "# pyright: reportX=false"):
        assert DOWNGRADE.match(line), line
    ban = "TID251"
    for line in (f"# ruff: noqa: {ban}", f"import socket  # noqa: F401, {ban}"):
        assert BAN_SUPPRESSION.search(line), line
    assert not DOWNGRADE.match("# pyright: strict")
    assert not DOWNGRADE.match("x = 1  # pyright: ignore[reportUnknownMemberType]")
    assert not BAN_SUPPRESSION.search("import torch  # noqa: E402")
