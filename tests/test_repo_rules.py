"""Repository rules outside the code: tracked files and the AGENTS.md rule table."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# An allowlist, so a new artifact type (.pkl, .onnx, .h5, .xlsx) fails until someone decides.
TRACKED_SUFFIXES = {".py", ".md", ".json", ".toml", ".yml", ".lock", ".svg"}
TRACKED_EXTENSIONLESS = {
    *("Makefile", "LICENSE", ".gitignore", ".python-version", ".claude/skills"),
    ".agents/skills/verify-esme-pretrain/verify",
}
ARTIFACT_DIRS = ("data/", "runs/", "checkpoints/", "exports/", "wandb/", "logs/")
MAX_TRACKED_BYTES = 512 * 1024
SIZE_EXEMPT = {"uv.lock"}


def _artifacts(names: list[str], root: Path = ROOT) -> list[str]:
    return [
        name
        for name in filter(None, names)
        if (root / name).exists()
        and (
            name.startswith(ARTIFACT_DIRS)
            or Path(name).name.startswith(".env")
            or (Path(name).suffix not in TRACKED_SUFFIXES and name not in TRACKED_EXTENSIONLESS)
            or (name not in SIZE_EXEMPT and (root / name).stat().st_size > MAX_TRACKED_BYTES)
        )
    ]


def test_no_run_outputs_or_data_are_tracked() -> None:
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout.split("\0")
    bad = _artifacts(tracked)
    assert not bad, (
        f"run outputs, data, secrets, or files over {MAX_TRACKED_BYTES} bytes are tracked: "
        f"{bad}. Keep them on the Modal volume or under gitignored runs/, then `git rm --cached`. "
        "A new source file type goes in TRACKED_SUFFIXES."
    )


def _rule_rows(agents: str) -> list[tuple[str, list[str]]]:
    assert "\n## Enforced Rules\n" in agents, "AGENTS.md needs an `## Enforced Rules` section"
    section = agents.split("\n## Enforced Rules\n", 1)[1].split("\n## ", 1)[0]
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    start = next((i for i, line in enumerate(lines) if line.startswith("|")), len(lines))
    table = lines[start:]
    assert all(line.startswith("|") for line in table), (
        "AGENTS.md `## Enforced Rules`: prose may come only before the table"
    )
    cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in table]
    assert cells and cells[0][:2] == ["Rule", "Enforced by"], (
        "AGENTS.md `## Enforced Rules` table must start with `| Rule | Enforced by |`"
    )
    assert len(cells) > 1 and all(set(cell) <= set("-: ") and cell for cell in cells[1]), (
        "AGENTS.md `## Enforced Rules` table needs a `| --- | --- |` row under its header"
    )
    rows = cells[2:]
    assert rows, "AGENTS.md `## Enforced Rules` has no rows"
    return [(row[0], re.findall(r"`([^`]+)`", row[1] if len(row) > 1 else "")) for row in rows]


def _defines(path: str, name: str) -> bool:
    text = (ROOT / path).read_text(encoding="utf-8")
    if path.endswith(".py") or text.startswith("#!/usr/bin/env python"):
        pattern = rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b"
    elif path == "Makefile":
        pattern = rf"^{re.escape(name)}:"
    elif name.startswith("["):
        pattern = rf"^{re.escape(name)}\s*$"
    else:
        pattern = rf'^name\s*=\s*"{re.escape(name)}"\s*$'
    return re.search(pattern, text, re.M) is not None


def _row_problems(rule: str, citations: list[str]) -> list[str]:
    problems = []
    if not any("::" in cite or cite.startswith("make ") for cite in citations):
        problems.append(f"{rule!r} cites no `path::name` check or make target")
    for cite in citations:
        path, _, name = cite.partition("::")
        if cite.startswith("make "):
            path, name = "Makefile", cite.removeprefix("make ")
        if not (ROOT / path).exists():
            problems.append(f"{rule!r} cites missing path {path!r}")
        elif name and not _defines(path, name):
            problems.append(f"{rule!r} cites {name!r}, which {path} does not define")
    return problems


def _table_problems(agents: str) -> list[str]:
    rows = _rule_rows(agents)
    return [problem for rule, cites in rows for problem in _row_problems(rule, cites)]


def test_enforced_rules_cite_real_checks() -> None:
    problems = _table_problems((ROOT / "AGENTS.md").read_text(encoding="utf-8"))

    assert not problems, f"fix AGENTS.md Enforced Rules: {problems}"


def test_every_audited_corpus_recipe_cites_an_existing_run_card() -> None:
    from esme_pretrain.data.corpus_config import AUDITED_CORPUS_RECIPES

    missing = [card for card in AUDITED_CORPUS_RECIPES.values() if not (ROOT / card).is_file()]

    assert AUDITED_CORPUS_RECIPES
    assert not missing, f"AUDITED_CORPUS_RECIPES cites missing run cards {missing}"


def test_the_checks_find_known_violations(tmp_path: Path) -> None:
    names = ["runs/x/metrics.txt", ".env.local", "shard.parquet", "big.md", "model.pkl"]
    names += ["uv.lock", "ok.md", "Makefile"]
    for name in names:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x" * (MAX_TRACKED_BYTES + 1 if "big" in name else 1))
    (tmp_path / "uv.lock").write_bytes(b"x" * (MAX_TRACKED_BYTES + 1))
    assert _artifacts(names, tmp_path) == names[:5]
    table = (
        "\n## Enforced Rules\n\n| Rule | Enforced by |\n| --- | --- |\n"
        "| ok | `make check`, `tests/test_repo_rules.py::test_enforced_rules_cite_real_checks` |\n"
        "| bare | `src/esme_pretrain/data/corpus_config.py` |\n"
        "| target | `make check-foo` |\n"
        "| def | `tests/test_repo_rules.py::no_such_test` |\n"
        "|unspaced|`make check-foo`|\n"
    )
    assert _table_problems(table) == [
        "'bare' cites no `path::name` check or make target",
        "'target' cites 'check-foo', which Makefile does not define",
        "'def' cites 'no_such_test', which tests/test_repo_rules.py does not define",
        "'unspaced' cites 'check-foo', which Makefile does not define",
    ]
    for broken in (
        "\n## Enforced Rules\n\n| Check | Enforced by |\n| --- | --- |\n| a | `make check` |\n",
        "\n## Enforced Rules\n\n| Rule | Enforced by |\n| --- | --- |\nprose\n| a | `make check` |\n",
        "\n## Enforced Rules\n\n| Rule | Enforced by |\n| bad | `make nope` |\n| a | `make check` |\n",
    ):
        with pytest.raises(AssertionError):
            _rule_rows(broken)


# Each owner file lifts every TID251 ban in pyproject.toml, so this map holds what each may use.
BAN_OWNERS = {
    "src/esme_pretrain/launch/gpu_smoke.py": {"subprocess", "modal"},
    "src/esme_pretrain/launch/modal_artifacts.py": {"subprocess"},
    "src/esme_pretrain/launch/scaling_study.py": {"subprocess", "modal"},
    "src/esme_pretrain/launch/throughput_bench.py": {"subprocess", "modal"},
    "scripts/modal_pretrain.py": {"modal"},
    "src/esme_pretrain/training/metrics.py": {"wandb"},
    "tests/test_repo_rules.py": {"subprocess"},
    "tests/test_scaling_study.py": {"subprocess"},
    "tests/test_strict_mode.py": {"subprocess"},
    ".agents/skills/verify-esme-pretrain/verify": {"subprocess", "socket"},
}


def _banned_uses(paths: list[str]) -> dict[str, set[str]]:
    ruff = str(Path(sys.executable).with_name("ruff"))
    command = [ruff, "check", "--no-cache", "--exit-zero", "--config", "pyproject.toml"]
    command += ["--select", "TID251"]
    command += ["--config", "lint.per-file-ignores={}"]
    command += ["--config", "lint.extend-per-file-ignores={}", "--output-format", "json"]
    report = subprocess.run(
        [*command, *paths],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    uses: dict[str, set[str]] = {}
    for item in json.loads(report):
        name = re.search(r"`([^`]+)` is banned", item["message"])
        assert name, item["message"]
        path = Path(item["filename"]).resolve().relative_to(ROOT).as_posix()
        uses.setdefault(path, set()).add(name.group(1))
    return uses


def _lifts_bans(code: str) -> bool:
    return code == "ALL" or "TID251".startswith(code)


def test_ban_owners_use_only_the_bans_they_own() -> None:
    lint = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["ruff"][
        "lint"
    ]
    ignores = [*lint.get("per-file-ignores", {}).items()]
    ignores += lint.get("extend-per-file-ignores", {}).items()
    owners = {path for path, codes in ignores if any(map(_lifts_bans, codes))}
    assert "TID251" in lint["select"]
    global_ignores = [*lint.get("ignore", []), *lint.get("extend-ignore", [])]
    assert not [code for code in global_ignores if _lifts_bans(code)], "an ignore lifts the bans"
    assert owners == set(BAN_OWNERS)
    extra = {
        path: sorted(used - BAN_OWNERS[path])
        for path, used in _banned_uses(sorted(BAN_OWNERS)).items()
        if used - BAN_OWNERS[path]
    }
    assert not extra, f"owner files use bans they do not own: {extra}. Move the call to its owner."
