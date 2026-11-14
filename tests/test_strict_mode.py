"""No comment weakens strict mode for a file, silences an import ban, or ignores without a reason."""

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNGRADE = re.compile(r"^\s*#\s*pyright:\s*(?!strict\s*$|ignore\b)\S")
BAN_SUPPRESSION = re.compile(r"\bnoqa\b[^#\n]*\bTID251\b", re.IGNORECASE)
INLINE_IGNORE = re.compile(r"#\s*(?:pyright:\s*ignore|noqa)\b", re.IGNORECASE)
EXPLAINED_IGNORE = re.compile(
    r"#\s*(?:pyright:\s*ignore\[\w+(?:,\s*\w+)*\]|noqa:\s*[A-Z]+\d+(?:,\s*[A-Z]+\d+)*)\s+#\s*\S"
)


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
    unexplained = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in candidate_python_files()
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1)
        if INLINE_IGNORE.search(line) and not EXPLAINED_IGNORE.search(line)
    ]
    assert not unexplained, (
        "each inline ignore names its rule and ends with `  # <reason>`: " + ", ".join(unexplained)
    )
    assert not downgrades, (
        "a `# pyright:` file comment weakens strict mode, or a `noqa` silences an import ban "
        "(add the file to the owner per-file-ignores in pyproject.toml instead): "
        + ", ".join(downgrades)
    )


def test_the_patterns_find_each_downgrade_and_spare_rule_ignores() -> None:
    for line in ("# pyright: basic", "#pyright: strict=false", "# pyright: reportX=false"):
        assert DOWNGRADE.match(line), line
    ban = "TID251"
    ignore, noqa = "pyright: ignore", "noqa"
    for line in (f"# ruff: {noqa}: {ban}", f"import socket  # {noqa}: F401, {ban}"):
        assert BAN_SUPPRESSION.search(line), line
    assert not DOWNGRADE.match("# pyright: strict")
    explained = f"x = 1  # {ignore}[reportAny]  # untyped JSON"
    assert not DOWNGRADE.match(explained)
    assert not BAN_SUPPRESSION.search(f"import torch  # {noqa}: E402")
    assert EXPLAINED_IGNORE.search(explained)
    assert EXPLAINED_IGNORE.search(f"import torch  # {noqa}: E402  # loads after the filter")
    upper = f"x  # {noqa.upper()}: F401"
    for bare in (
        f"x = 1  # {ignore}[reportAny]",
        f"x = 1  # {ignore}",
        f"x  # {noqa}: E402",
        upper,
    ):
        assert INLINE_IGNORE.search(bare) and not EXPLAINED_IGNORE.search(bare), bare
