from __future__ import annotations

import json

from esme_pretrain.launch import throughput_bench

CONFIG = "configs/pretrain_214m_b200.json"


def test_check_only_reports_the_plan_without_spending(capsys) -> None:
    code = throughput_bench.launch(
        ["--config", CONFIG, "--spend-cap-usd", "10", "--before-commit", "HEAD", "--check-only"]
    )

    assert code == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "check_ok"
    assert plan["gpu"] == "B200"
    assert plan["variants"][0] == "before-fixed"
    assert plan["variants"][-1] == "before-fixed-repeat"


def test_spend_cap_below_the_worst_case_refuses(capsys) -> None:
    code = throughput_bench.launch(
        ["--config", CONFIG, "--spend-cap-usd", "1", "--before-commit", "HEAD", "--check-only"]
    )

    assert code == 2
    assert "exceeds cap" in capsys.readouterr().err


def test_launch_without_approval_refuses_before_modal(capsys) -> None:
    code = throughput_bench.launch(
        ["--config", CONFIG, "--spend-cap-usd", "10", "--before-commit", "HEAD"]
    )

    assert code == 2
    assert "--approved" in capsys.readouterr().err


def test_unknown_before_commit_refuses(capsys) -> None:
    code = throughput_bench.launch(
        [
            "--config",
            CONFIG,
            "--spend-cap-usd",
            "10",
            "--before-commit",
            "not-a-ref",
            "--check-only",
        ]
    )

    assert code == 2
    assert "not a commit" in capsys.readouterr().err
