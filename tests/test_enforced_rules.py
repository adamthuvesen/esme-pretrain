from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _enforced_by_citations() -> list[str]:
    section = (ROOT / "AGENTS.md").read_text(encoding="utf-8").split("## Enforced Rules", 1)[1]
    rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line]
    cells = [row.split("|")[2] for row in rows[1:]]
    return [cite for cell in cells for cite in re.findall(r"`([^`]+)`", cell)]


def _missing(citation: str) -> bool:
    if citation.startswith("make "):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        return re.search(rf"^{citation.removeprefix('make ')}:", makefile, re.M) is None
    path, _, test_name = citation.partition("::")
    if not (ROOT / path).exists():
        return True
    return bool(test_name) and f"def {test_name}(" not in (ROOT / path).read_text(encoding="utf-8")


def test_enforced_rules_cite_real_checks() -> None:
    citations = _enforced_by_citations()
    missing = [citation for citation in citations if _missing(citation)]

    assert citations
    assert not missing, f"AGENTS.md Enforced Rules cites missing checks {missing}; fix the row"
