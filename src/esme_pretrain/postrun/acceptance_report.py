from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esme_pretrain.postrun.eval_checkpoints import file_sha256, select_checkpoint
from esme_pretrain.pretrain_run import EXPECTED_ARTIFACTS


@dataclass(frozen=True)
class BaseAcceptanceReportConfig:
    run_dir: Path
    eval_path: Path
    output_path: Path


def build_base_acceptance_report(config: BaseAcceptanceReportConfig) -> dict[str, Any]:
    run_dir = config.run_dir
    if not run_dir.exists():
        raise ValueError(f"run directory does not exist: {run_dir}")
    missing = [name for name in EXPECTED_ARTIFACTS if not (run_dir / name).exists()]
    if missing:
        raise ValueError(f"run directory is missing required artifacts: {', '.join(missing)}")
    if not config.eval_path.exists():
        raise ValueError(f"eval JSON does not exist: {config.eval_path}")

    eval_payload = _read_json(config.eval_path, "eval JSON")
    checkpoints = list(eval_payload.get("checkpoints") or [])
    if not checkpoints:
        raise ValueError("eval JSON has no checkpoint results")
    selection = dict(eval_payload.get("selection") or select_checkpoint(checkpoints))

    inventory = _inventory(run_dir)
    run_summary = _read_json(run_dir / "run-summary.json", "run-summary.json")
    cost = _read_json(run_dir / "cost.json", "cost.json")
    tokenizer_report = _read_json(run_dir / "tokenizer-report.json", "tokenizer-report.json")
    final_metrics = _final_metrics(run_dir / "metrics.jsonl")
    throughput_summary = _throughput_summary(run_dir / "throughput.csv")
    final_step = _final_step(run_summary, final_metrics)
    final_tokens = _final_tokens(run_summary, final_metrics, final_step)
    samples_present = bool((run_dir / "samples.md").read_text(encoding="utf-8").strip())
    payload = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "eval": str(config.eval_path),
        "output": str(config.output_path),
        "artifacts": inventory,
        "run_status": _run_status(run_summary),
        "final_step": final_step,
        "final_tokens": final_tokens,
        "training": {
            "train_loss_first": _pick(run_summary, "train_loss_first"),
            "train_loss_last": _pick(run_summary, "train_loss_last"),
            "train_loss_min": _pick(run_summary, "train_loss_min"),
            "val_loss_first": _pick(run_summary, "val_loss_first"),
            "val_loss_last": _pick(run_summary, "val_loss_last"),
            "grad_norm_last": _pick(run_summary, "grad_norm_last"),
            "peak_memory_gb": _pick(run_summary, "peak_memory_gb"),
        },
        "cost": {
            "estimated_cost_usd": _pick(cost, "estimated_cost_usd", "cost_usd"),
            "paid_compute": _pick(cost, "paid_compute"),
            "wandb_run_id": _pick(run_summary, "wandb_run_id", "result.wandb_run_id"),
            "wandb_run_url": _pick(run_summary, "wandb_run_url", "result.wandb_run_url"),
        },
        "fixed_eval": {
            "best_checkpoint": selection["best_checkpoint"],
            "best_ce_loss": selection["best_ce_loss"],
            "best_bits_per_byte": selection.get("best_bits_per_byte"),
            "recommended_checkpoint": selection["recommended_checkpoint"],
            "margin_ce_loss": selection["margin_ce_loss"],
            "reason": selection["reason"],
            "checkpoints": checkpoints,
        },
        "samples_present": samples_present,
        "tokenizer_round_trip": _tokenizer_round_trip_evidence(tokenizer_report),
        "throughput": throughput_summary,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(_render_markdown(payload), encoding="utf-8")
    return payload


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _inventory(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return rows


def _final_metrics(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    payload = json.loads(lines[-1])
    return payload if isinstance(payload, dict) else {}


def _throughput_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"rows": 0}
    last = rows[-1]
    return {
        "rows": len(rows),
        "last_tokens_per_second": _float_or_none(last.get("tokens_per_second")),
        "last_step_time_ms": _float_or_none(last.get("step_time_ms")),
    }


def _run_status(run_summary: dict[str, Any]) -> str:
    explicit = _pick(run_summary, "status", "result.status")
    if explicit is not None:
        return str(explicit)
    if _pick(run_summary, "steps_completed") is not None:
        return "accepted_10b_base"
    return "unknown"


def _final_step(run_summary: dict[str, Any], final_metrics: dict[str, Any]) -> Any:
    return _pick(
        run_summary,
        "final_step",
        "steps_completed",
        "result.steps_completed",
        "result.steps_completed_total",
        default=final_metrics.get("step"),
    )


def _final_tokens(
    run_summary: dict[str, Any], final_metrics: dict[str, Any], final_step: Any
) -> Any:
    explicit = _pick(
        run_summary,
        "final_tokens",
        "tokens_seen",
        "total_tokens",
        "result.tokens_seen",
        "result.total_tokens",
    )
    if explicit is not None:
        return explicit
    tokens_per_step = final_metrics.get("tokens")
    if final_step is not None and tokens_per_step is not None:
        return int(final_step) * int(tokens_per_step)
    return tokens_per_step


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _pick(payload: dict[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        value: Any = payload
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return default


def _tokenizer_round_trip_evidence(report: dict[str, Any]) -> dict[str, Any]:
    round_trips = list(report.get("round_trips") or [])
    return {
        "count": len(round_trips),
        "all_passed": bool(round_trips) and all(bool(row.get("round_trip")) for row in round_trips),
        "examples": round_trips[:3],
        "coverage": report.get("coverage"),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 214M 10B Base Acceptance Report",
        "",
        f"- Run directory: `{payload['run_dir']}`",
        f"- Eval JSON: `{payload['eval']}`",
        f"- Run status: `{payload['run_status']}`",
        f"- Final step: `{payload['final_step']}`",
        f"- Final tokens: `{payload['final_tokens']}`",
        f"- Estimated cost USD: `{payload['cost']['estimated_cost_usd']}`",
        f"- W&B run: `{payload['cost']['wandb_run_url']}`",
        "",
        "## Training Summary",
        "",
        f"- Train loss: `{payload['training']['train_loss_first']}` -> "
        f"`{payload['training']['train_loss_last']}` "
        f"(min `{payload['training']['train_loss_min']}`)",
        f"- Validation loss: `{payload['training']['val_loss_first']}` -> "
        f"`{payload['training']['val_loss_last']}`",
        f"- Final grad norm: `{payload['training']['grad_norm_last']}`",
        f"- Peak memory GB: `{payload['training']['peak_memory_gb']}`",
        "- Last measured throughput: "
        f"`{payload['throughput'].get('last_tokens_per_second')}` tok/s",
        "",
        "## Fixed Validation Eval",
        "",
        f"- Eval tokens: `{payload['fixed_eval']['checkpoints'][0]['eval_tokens']}`",
        f"- Eval bytes: `{payload['fixed_eval']['checkpoints'][0]['eval_bytes']}`",
        f"- Eval batches: `{payload['fixed_eval']['checkpoints'][0]['eval_batches']}`",
        f"- Best checkpoint: `{payload['fixed_eval']['best_checkpoint']}`",
        f"- Best CE loss: `{payload['fixed_eval']['best_ce_loss']}`",
        f"- Best bits per byte: `{payload['fixed_eval']['best_bits_per_byte']}`",
        f"- Recommended checkpoint: `{payload['fixed_eval']['recommended_checkpoint']}`",
        f"- Final-vs-best margin: `{payload['fixed_eval']['margin_ce_loss']}`",
        f"- Decision: {payload['fixed_eval']['reason']}",
        "",
        "| checkpoint | step | CE loss | bpb | perplexity | eval tokens | eval bytes | sha256 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["fixed_eval"]["checkpoints"]:
        lines.append(
            f"| `{Path(row['path']).name}` | {row['checkpoint_step']} | "
            f"{row['ce_loss']:.6f} | {row['bits_per_byte']:.6f} | "
            f"{row['perplexity']:.3f} | {row['eval_tokens']} | "
            f"{row['eval_bytes']} | `{row['checkpoint_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Samples present: `{payload['samples_present']}`",
            "- Tokenizer round trips: "
            f"`{payload['tokenizer_round_trip']['count']}` "
            f"(all passed: `{payload['tokenizer_round_trip']['all_passed']}`)",
            f"- Cost estimate USD: `{payload['cost']['estimated_cost_usd']}`",
            f"- W&B run ID: `{payload['cost']['wandb_run_id']}`",
            f"- W&B run URL: `{payload['cost']['wandb_run_url']}`",
            "",
            "## Artifact Inventory",
            "",
            "| artifact | bytes | sha256 |",
            "| --- | ---: | --- |",
        ]
    )
    for row in payload["artifacts"]:
        lines.append(f"| `{row['name']}` | {row['bytes']} | `{row['sha256']}` |")
    lines.append("")
    return "\n".join(lines)
