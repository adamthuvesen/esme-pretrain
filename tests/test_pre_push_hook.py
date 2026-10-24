from __future__ import annotations

import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / ".githooks" / "pre-push"
SHA = "0" * 40


def _push(remote_ref: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(HOOK), "origin", "git@github.com:adamthuvesen/esme-pretrain.git"],
        input=f"refs/heads/main {SHA} {remote_ref} {SHA}\n",
        capture_output=True,
        text=True,
        check=False,
    )


def test_pre_push_refuses_main_and_names_the_supported_path() -> None:
    result = _push("refs/heads/main")

    assert result.returncode == 1
    assert "feature branch and PR" in result.stderr


def test_pre_push_allows_feature_branches() -> None:
    assert _push("refs/heads/adam/feat-scaling-study").returncode == 0
