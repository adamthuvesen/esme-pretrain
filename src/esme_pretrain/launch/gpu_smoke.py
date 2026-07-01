#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from esme_pretrain.launch.modal_image import build_pretrain_modal_image
from esme_pretrain.pretrain_run import PretrainLaunchConfig, load_pretrain_config

REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_GPU = os.environ.get("PRETRAIN_SMOKE_GPU", "H100!")
SMOKE_TIMEOUT_SECONDS = int(float(os.environ.get("PRETRAIN_SMOKE_TIMEOUT_SECONDS", "900")))
DEFAULT_LEDGER_PATH = Path("runs/pretrain-214m-b200/gpu-smoke-ledger.json")
DEFAULT_OUTPUT_DIR = Path("runs/pretrain-214m-b200")

try:
    import modal
except ImportError:  # pragma: no cover - Modal is supplied by the launch command.
    modal = None


if modal is not None:  # pragma: no cover - exercised on Modal, not in unit tests.
    image = build_pretrain_modal_image(
        repo_root_src=str(REPO_ROOT / "src"),
        modal_module=modal,
    )
    app = modal.App("esme-pretrain-214m-b200-gpu-smoke", image=image)

    @app.function(gpu=SMOKE_GPU, timeout=SMOKE_TIMEOUT_SECONDS)
    def run_smoke(config_payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        return _run_smoke_body(config_payload, params)

    @app.local_entrypoint()
    def main(
        config: str = "configs/pretrain_214m_b200.json",
        max_steps: int = 3,
        micro_batch_size: int = 0,
        grad_accum_steps: int = 0,
        spend_cap_usd: float = 10.0,
        ledger_path: str = str(DEFAULT_LEDGER_PATH),
        json: bool = True,
    ) -> None:
        """Run one short, spend-capped GPU smoke.

        The smoke is a few steps and its result feeds the spend ledger inline, so
        the entrypoint blocks on .remote() rather than spawning a detached run.
        """
        raise SystemExit(
            launch(
                [
                    "--config",
                    config,
                    "--max-steps",
                    str(max_steps),
                    "--micro-batch-size",
                    str(micro_batch_size),
                    "--grad-accum-steps",
                    str(grad_accum_steps),
                    "--spend-cap-usd",
                    str(spend_cap_usd),
                    "--ledger-path",
                    ledger_path,
                    "--json" if json else "--no-json",
                ]
            )
        )


def launch(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one approval-bounded 214M B200 pretrain GPU smoke with a spend ledger."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain_214m_b200.json"))
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--grad-accum-steps", type=int, default=0)
    parser.add_argument("--spend-cap-usd", type=float, default=10.0)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", dest="json_output", action="store_true", default=True)
    parser.add_argument("--no-json", dest="json_output", action="store_false")
    parser.add_argument(
        "--ledger-check-only",
        action="store_true",
        help="Validate and reserve nothing; useful for local tests.",
    )
    args = parser.parse_args(argv)

    try:
        config = load_pretrain_config(args.config)
        gpu_profile = _selected_smoke_profile(config, SMOKE_GPU)
        micro_batch = args.micro_batch_size or int(
            config.payload["optimizer"]["training"]["micro_batch_size"]
        )
        grad_accum = args.grad_accum_steps or int(
            config.payload["optimizer"]["training"]["grad_accum_steps"]
        )
        reserved_cost = _reserved_cost_usd(
            timeout_seconds=SMOKE_TIMEOUT_SECONDS,
            usd_per_hour=float(gpu_profile["usd_per_hour"]),
        )
        reservation = reserve_smoke_attempt(
            args.ledger_path,
            gpu=SMOKE_GPU,
            reserved_cost_usd=reserved_cost,
            spend_cap_usd=args.spend_cap_usd,
            params={
                "config": args.config.as_posix(),
                "max_steps": args.max_steps,
                "micro_batch_size": micro_batch,
                "grad_accum_steps": grad_accum,
                "timeout_seconds": SMOKE_TIMEOUT_SECONDS,
            },
        )
    except ValueError as error:
        print(f"pretrain gpu smoke refused: {error}", file=sys.stderr)
        return 2

    if args.ledger_check_only:
        print(_format_payload(reservation, args.json_output))
        return 0
    if modal is None:
        mark_smoke_attempt(
            args.ledger_path,
            reservation["attempt_id"],
            status="failed",
            actual_cost_usd=reserved_cost,
            result={"error": "modal is not installed"},
        )
        print("pretrain gpu smoke failed: modal is not installed", file=sys.stderr)
        return 2

    params = {
        "gpu": SMOKE_GPU,
        "max_steps": args.max_steps,
        "micro_batch_size": micro_batch,
        "grad_accum_steps": grad_accum,
        "commit": _local_git_commit(),
        "dirty": _local_git_dirty(),
    }
    wall_start = time.perf_counter()
    try:
        result = run_smoke.remote(config.payload, params)
    except Exception as error:  # noqa: BLE001 - preserve the failure in the ledger.
        actual_cost = min(
            reserved_cost,
            _reserved_cost_usd(
                timeout_seconds=time.perf_counter() - wall_start,
                usd_per_hour=float(gpu_profile["usd_per_hour"]),
            ),
        )
        mark_smoke_attempt(
            args.ledger_path,
            reservation["attempt_id"],
            status="failed",
            actual_cost_usd=actual_cost,
            result={"error": repr(error)},
        )
        raise

    result["client_wall_seconds"] = time.perf_counter() - wall_start
    result["reserved_cost_usd"] = reserved_cost
    result["ledger_attempt_id"] = reservation["attempt_id"]
    actual_cost = float(result["estimated_gpu_cost_usd"])
    mark_smoke_attempt(
        args.ledger_path,
        reservation["attempt_id"],
        status="complete",
        actual_cost_usd=actual_cost,
        result={
            "gpu": result["gpu"],
            "status": result["status"],
            "steady_tokens_per_second": result["result"]["steady_tokens_per_second"],
            "peak_memory_gb": result["result"]["peak_memory_gb"],
            "estimated_gpu_cost_usd": actual_cost,
        },
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_gpu = SMOKE_GPU.replace("!", "bang").lower()
    output_path = args.output_dir / f"gpu-smoke-{safe_gpu}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["local_output_path"] = str(output_path)
    print(_format_payload(result, args.json_output))
    return 0


def _run_smoke_body(config_payload: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    import torch

    from esme_pretrain.modeling.backbone import BackboneConfig
    from esme_pretrain.modeling.pretrain_checkpoint import load_pretrain_checkpoint
    from esme_pretrain.training.data_stream import Batch
    from esme_pretrain.training.metrics_logger import RunLogger, WandbSettings
    from esme_pretrain.training.pretrain import PretrainConfig, run_pretrain

    model_config = BackboneConfig.from_dict(config_payload["model"])
    optimizer = config_payload["optimizer"]
    training = optimizer["training"]
    gpu = params["gpu"]
    started = time.perf_counter()
    print(
        f"pretrain gpu smoke: start gpu={gpu} "
        f"micro_batch={params['micro_batch_size']} grad_accum={params['grad_accum_steps']}",
        flush=True,
    )

    torch.manual_seed(int(training["seed"]))
    device = torch.device("cuda")
    micro_batch = int(params["micro_batch_size"])
    grad_accum = int(params["grad_accum_steps"])
    window = torch.randint(
        0,
        model_config.vocab_size,
        (micro_batch, model_config.context_length + 1),
        device=device,
        dtype=torch.long,
    )
    fixed = Batch(input_ids=window[:, :-1], targets=window[:, 1:])

    class _RepeatLoader:
        def __init__(self) -> None:
            self.skip_tokens = 0

        def __iter__(self):
            while True:
                yield fixed

    output_dir = Path("/tmp/pretrain-214m-b200-gpu-smoke")
    train_config = PretrainConfig(
        model=model_config,
        max_steps=int(params["max_steps"]),
        micro_batch_size=micro_batch,
        grad_accum_steps=grad_accum,
        learning_rate=float(optimizer["learning_rate"]),
        min_lr_ratio=float(optimizer["min_lr_ratio"]),
        lr_schedule=str(optimizer["lr_schedule"]),
        decay_fraction=float(optimizer["decay_fraction"]),
        warmup_steps=min(1, int(params["max_steps"])),
        weight_decay=float(optimizer["weight_decay"]),
        grad_clip=float(optimizer["grad_clip"]),
        dtype=training["dtype"],
        device="cuda",
        use_compile=bool(training["compile"]),
        use_fused_optimizer=bool(training["fused_optimizer"]),
        seed=int(training["seed"]),
        log_interval=1,
        eval_interval=0,
        checkpoint_interval=0,
        sample_interval=0,
        output_dir=output_dir,
    )
    logger = RunLogger(output_dir, WandbSettings(enabled=False))
    print("pretrain gpu smoke: train begin", flush=True)
    result = run_pretrain(train_config, _RepeatLoader(), logger=logger)
    logger.finish()
    print(
        f"pretrain gpu smoke: train complete steady_tok_s={result.steady_tokens_per_second}",
        flush=True,
    )

    loss_trace = []
    for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if "loss" in record and "tokens_per_second" in record:
            loss_trace.append(float(record["loss"]))

    checkpoint = load_pretrain_checkpoint(Path(result.final_checkpoint), map_location="cpu")
    torch.cuda.empty_cache()
    resume_micro_batch = min(4, micro_batch)
    resume_window = torch.randint(
        0,
        model_config.vocab_size,
        (resume_micro_batch, model_config.context_length + 1),
        device=device,
        dtype=torch.long,
    )
    resume_fixed = Batch(input_ids=resume_window[:, :-1], targets=resume_window[:, 1:])

    class _ResumeLoader:
        def __init__(self) -> None:
            self.skip_tokens = 0

        def __iter__(self):
            while True:
                yield resume_fixed

    resume_config = replace(
        train_config,
        max_steps=result.start_step + result.steps_completed + 1,
        micro_batch_size=resume_micro_batch,
        resume_from=Path(result.final_checkpoint),
        use_compile=False,
        output_dir=Path("/tmp/pretrain-214m-b200-gpu-smoke-resume"),
    )
    resume_logger = RunLogger(resume_config.output_dir, WandbSettings(enabled=False))
    print("pretrain gpu smoke: resume begin", flush=True)
    resumed = run_pretrain(resume_config, _ResumeLoader(), logger=resume_logger)
    resume_logger.finish()
    print("pretrain gpu smoke: resume complete", flush=True)

    elapsed = time.perf_counter() - started
    profile = config_payload["runtime"]["gpu_profiles"][gpu]
    projected_cost = _project_cost(
        train_tokens=int(config_payload["budgets"]["train_token_budget"]),
        tokens_per_second=float(result.steady_tokens_per_second or 0.0),
        usd_per_hour=float(profile["usd_per_hour"]),
    )
    train_token_budget = int(config_payload["budgets"]["train_token_budget"])
    loss_finite = all(_is_finite(value) for value in loss_trace)
    loss_decreasing = len(loss_trace) >= 2 and loss_trace[-1] < loss_trace[0]
    resume_ok = (
        resumed.start_step == result.steps_completed
        and resumed.steps_completed == 1
        and _is_finite(resumed.train_loss_last)
    )
    return {
        "status": "smoke_complete",
        "gpu": gpu,
        "commit": params.get("commit"),
        "dirty": params.get("dirty"),
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": _nvidia_smi("memory.total"),
        },
        "config": {
            "run_id": config_payload["run_id"],
            "model": model_config.to_dict(),
            "max_steps": train_config.max_steps,
            "micro_batch_size": train_config.micro_batch_size,
            "grad_accum_steps": train_config.grad_accum_steps,
            "tokens_per_step": train_config.tokens_per_step,
            "compile": train_config.use_compile,
            "dtype": train_config.dtype,
            "fused_optimizer": train_config.use_fused_optimizer,
        },
        "result": result.to_dict(),
        "loss_trend": {
            "first": loss_trace[0] if loss_trace else None,
            "last": loss_trace[-1] if loss_trace else None,
            "is_finite": loss_finite,
            "is_decreasing": loss_decreasing,
            "trace": loss_trace,
        },
        "checkpoint_resume": {
            "ok": resume_ok,
            "checkpoint_step": checkpoint.step,
            "data_offset_tokens": checkpoint.data_offset_tokens,
            "resumed_from_step": resumed.start_step,
            "continued_steps": resumed.steps_completed,
            "resume_micro_batch": resume_micro_batch,
            "resumed_loss_last": resumed.train_loss_last,
        },
        "projection": {
            "train_token_budget": train_token_budget,
            "usd_per_hour": profile["usd_per_hour"],
            "estimated_usd_per_1b_tokens": round(
                projected_cost * 1_000_000_000 / train_token_budget, 2
            )
            if projected_cost is not None
            else None,
            "projected_full_cost_usd": round(projected_cost, 2)
            if projected_cost is not None
            else None,
        },
        "remote_wall_seconds": elapsed,
        "estimated_gpu_cost_usd": round(elapsed * float(profile["usd_per_hour"]) / 3600.0, 4),
    }


def _selected_smoke_profile(config: PretrainLaunchConfig, gpu: str) -> dict[str, Any]:
    profiles = config.payload["runtime"]["gpu_profiles"]
    if gpu not in profiles:
        raise ValueError(f"{gpu} is not listed in runtime.gpu_profiles")
    if gpu not in {"H100!", "H200", "B200"}:
        raise ValueError(f"{gpu} is not an approved 214M smoke GPU")
    return dict(profiles[gpu])


def _reserved_cost_usd(*, timeout_seconds: float, usd_per_hour: float) -> float:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if usd_per_hour <= 0:
        raise ValueError("usd_per_hour must be positive")
    return round(timeout_seconds * usd_per_hour / 3600.0, 4)


def reserve_smoke_attempt(
    ledger_path: Path,
    *,
    gpu: str,
    reserved_cost_usd: float,
    spend_cap_usd: float,
    params: dict[str, Any],
) -> dict[str, Any]:
    ledger = load_smoke_ledger(ledger_path)
    used = ledger_spend_used_usd(ledger)
    if used + reserved_cost_usd > spend_cap_usd:
        raise ValueError(
            "smoke spend cap would be exceeded: "
            f"used ${used:.4f} + reserved ${reserved_cost_usd:.4f} > cap ${spend_cap_usd:.2f}"
        )
    attempt_id = f"{int(time.time())}-{gpu.replace('!', 'bang').lower()}"
    attempt = {
        "attempt_id": attempt_id,
        "gpu": gpu,
        "status": "reserved",
        "reserved_cost_usd": reserved_cost_usd,
        "actual_cost_usd": None,
        "params": params,
        "created_at_unix": time.time(),
    }
    ledger["attempts"].append(attempt)
    ledger["spend_cap_usd"] = spend_cap_usd
    write_smoke_ledger(ledger_path, ledger)
    return {
        "status": "reserved",
        "attempt_id": attempt_id,
        "gpu": gpu,
        "reserved_cost_usd": reserved_cost_usd,
        "spend_used_before_usd": used,
        "spend_cap_usd": spend_cap_usd,
    }


def mark_smoke_attempt(
    ledger_path: Path,
    attempt_id: str,
    *,
    status: str,
    actual_cost_usd: float,
    result: dict[str, Any],
) -> None:
    ledger = load_smoke_ledger(ledger_path)
    for attempt in ledger["attempts"]:
        if attempt["attempt_id"] == attempt_id:
            attempt["status"] = status
            attempt["actual_cost_usd"] = round(actual_cost_usd, 4)
            attempt["result"] = result
            attempt["completed_at_unix"] = time.time()
            write_smoke_ledger(ledger_path, ledger)
            return
    raise ValueError(f"ledger attempt not found: {attempt_id}")


def load_smoke_ledger(ledger_path: Path) -> dict[str, Any]:
    if not ledger_path.exists():
        return {"spend_cap_usd": None, "attempts": []}
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("attempts"), list):
        raise ValueError("smoke ledger is malformed")
    return payload


def write_smoke_ledger(ledger_path: Path, ledger: dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")


def ledger_spend_used_usd(ledger: dict[str, Any]) -> float:
    total = 0.0
    for attempt in ledger.get("attempts", []):
        actual = attempt.get("actual_cost_usd")
        total += float(actual if actual is not None else attempt["reserved_cost_usd"])
    return round(total, 4)


def _project_cost(
    *, train_tokens: int, tokens_per_second: float, usd_per_hour: float
) -> float | None:
    if tokens_per_second <= 0:
        return None
    return train_tokens / tokens_per_second * usd_per_hour / 3600.0


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value and abs(value) != float("inf")


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _format_payload(payload: dict[str, Any], json_output: bool) -> str:
    if json_output:
        return json.dumps(payload, indent=2, sort_keys=True)
    return str(payload)


def _local_git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _local_git_dirty() -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else True


if __name__ == "__main__":
    raise SystemExit(launch())
