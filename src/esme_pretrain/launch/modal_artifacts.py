"""Artifact writers and completeness checks for Modal pretrain launches."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from esme_pretrain.pretrain_run import EXPECTED_ARTIFACTS, PretrainLaunchConfig
from esme_pretrain.torch import torch


def _required_artifacts_present(output_dir: Path) -> bool:
    return all((output_dir / file_name).exists() for file_name in EXPECTED_ARTIFACTS)


def _write_cost(output_dir: Path, cost: dict[str, Any]) -> None:
    (output_dir / "cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")


def _write_data_report(
    config: PretrainLaunchConfig, output_dir: Path, tokenizer_report: dict[str, Any]
) -> None:
    report = {
        "dataset": config.payload["dataset"],
        "split": config.payload["split"],
        "budgets": config.payload["budgets"],
        "tokenizer": tokenizer_report,
        "note": "Full run streams rows lazily; local dress rehearsal uses synthetic data.",
    }
    (output_dir / "data-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _write_environment(output_dir: Path) -> None:
    lines = [
        f"python={sys.version.split()[0]}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.extend(
            [
                f"cuda={torch.version.cuda}",
                f"gpu={torch.cuda.get_device_name(0)}",
            ]
        )
    (output_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pretrain_report(
    config: PretrainLaunchConfig,
    output_dir: Path,
    result: dict[str, Any],
    cost: dict[str, Any],
    commit: str,
    dirty: bool,
) -> str:
    report = f"""# 214M B200 Pretrain Report

- run: `{config.payload["run_id"]}`
- commit: `{commit}`
- dirty: `{dirty}`
- output: `{output_dir}`
- train tokens target: `{config.payload["budgets"]["train_token_budget"]}`
- max cost: `${config.payload["runtime"]["max_cost_usd"]}`
- estimated/actual cost payload: `{json.dumps(cost, sort_keys=True)}`

## Result

```json
{json.dumps(result, indent=2, sort_keys=True)}
```
"""
    (output_dir / "scaleup-pretrain-report.md").write_text(report, encoding="utf-8")
    return report


def _write_rehearsal_manifest(
    config: PretrainLaunchConfig,
    output_dir: Path,
    result: Any,
    resume_result: Any,
    data_offset_tokens: int,
    *,
    repo_root: Path,
) -> None:
    (output_dir / "config.json").write_text(json.dumps(config.payload, indent=2), encoding="utf-8")
    (output_dir / "tokenizer.json").write_text(
        json.dumps({"kind": "local-dress-rehearsal", "vocab_size": 256}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "tokenizer-report.json").write_text(
        json.dumps(
            {
                "kind": "local-dress-rehearsal",
                "round_trips": [{"text": "abc", "round_trip": True}],
                "coverage": "synthetic ids only; full run trains byte-level BPE",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_data_report(config, output_dir, {"mode": "local-dress-rehearsal"})
    _write_environment(output_dir)
    _write_cost(output_dir, {"paid_compute": False, "estimated_cost_usd": 0.0})
    _write_pretrain_report(
        config,
        output_dir,
        {
            "first_result": result.to_dict(),
            "resume_result": resume_result.to_dict(),
            "data_offset_tokens": data_offset_tokens,
        },
        {"paid_compute": False, "estimated_cost_usd": 0.0},
        _local_git_commit(repo_root),
        _local_git_dirty(repo_root),
    )


def _local_git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _local_git_dirty(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else True
