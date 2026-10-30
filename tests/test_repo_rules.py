"""Repository rules outside the code: tracked files and the AGENTS.md rule table."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SUFFIXES = {
    *(".parquet", ".arrow", ".jsonl", ".csv", ".npy", ".npz", ".log"),
    *(".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".gguf"),
}
ARTIFACT_DIRS = ("data/", "runs/", "checkpoints/", "exports/", "wandb/", "logs/")
MAX_TRACKED_BYTES = 512 * 1024
SIZE_EXEMPT = {"uv.lock"}


def test_no_run_outputs_or_data_are_tracked() -> None:
    tracked = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, check=True
    ).stdout.split("\0")
    bad = [
        name
        for name in filter(None, tracked)
        if (ROOT / name).exists()
        and (
            name.startswith(ARTIFACT_DIRS)
            or Path(name).name.startswith(".env")
            or Path(name).suffix in ARTIFACT_SUFFIXES
            or (name not in SIZE_EXEMPT and (ROOT / name).stat().st_size > MAX_TRACKED_BYTES)
        )
    ]
    assert not bad, (
        f"run outputs, data, secrets, or files over {MAX_TRACKED_BYTES} bytes are tracked: "
        f"{bad}. Keep them on the Modal volume or under gitignored runs/, then `git rm --cached`."
    )


def _rule_rows(agents: str) -> list[tuple[str, list[str]]]:
    assert "\n## Enforced Rules\n" in agents, "AGENTS.md needs an `## Enforced Rules` section"
    table = agents.split("\n## Enforced Rules\n", 1)[1].split("\n## ", 1)[0]
    rows = [line.split("|") for line in table.splitlines() if line.startswith("| ")][2:]
    assert rows, "AGENTS.md `## Enforced Rules` has no rows"
    return [(row[1].strip(), re.findall(r"`([^`]+)`", row[2])) for row in rows]


def _defines(path: str, name: str) -> bool:
    text = (ROOT / path).read_text(encoding="utf-8")
    if path.endswith(".py"):
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
