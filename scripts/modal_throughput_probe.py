#!/usr/bin/env python3
"""Throughput probe on a Modal GPU.

Runs MHA backbone configs with RoPE/RMSNorm/SwiGLU, FlashAttention via SDPA,
bf16, fused AdamW, and grad accumulation. It reports tokens/sec, MFU, and step
time without downloading data.

    uv run --with modal modal run scripts/modal_throughput_probe.py
    PROBE_GPU=H100 uv run --with modal modal run scripts/modal_throughput_probe.py --compile

The GPU is set by the PROBE_GPU env var (e.g. A100, A100-80GB, H100, L40S) so the
cost/speed of different hardware can be compared. --compile adds a torch.compile
150M pass. Each run writes runs/throughput-probe/probe-results-<gpu>.json.

This short probe blocks on .remote() and returns the result inline rather than
spawning a detached run.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from esme_pretrain.launch.modal_image import build_pretrain_modal_image

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_GPU = os.environ.get("PROBE_GPU", "A100")

try:
    import modal
except ImportError:  # pragma: no cover - Modal is supplied by the launch command.
    modal = None


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


if modal is not None:  # pragma: no cover - exercised on Modal, not in unit tests.
    image = build_pretrain_modal_image(
        repo_root_src=str(REPO_ROOT / "src"),
        modal_module=modal,
    )
    app = modal.App("esme-pretrain-throughput-probe", image=image)

    @app.function(gpu=PROBE_GPU, timeout=20 * 60)
    def run_probe(
        micro_batches: dict[str, int], measured_steps: int, compile_150m: bool
    ) -> dict[str, Any]:
        import torch

        from esme_pretrain.modeling.backbone import PROBE_CONFIGS
        from esme_pretrain.training.throughput import ProbeConfig, run_throughput_probe

        results: list[dict[str, Any]] = []
        for name in ("124M", "150M", "350M"):
            config = ProbeConfig(
                model=PROBE_CONFIGS[name],
                micro_batch_size=micro_batches[name],
                grad_accum_steps=2,
                warmup_steps=10,
                measured_steps=measured_steps,
                dtype="bfloat16",
                device="cuda",
            )
            results.append(run_throughput_probe(config).to_dict())

        if compile_150m:
            compiled = run_throughput_probe(
                ProbeConfig(
                    model=PROBE_CONFIGS["150M"],
                    micro_batch_size=micro_batches["150M"],
                    grad_accum_steps=2,
                    warmup_steps=10,
                    measured_steps=measured_steps,
                    dtype="bfloat16",
                    device="cuda",
                    use_compile=True,
                )
            ).to_dict()
            compiled["model_name"] = "150M-compiled"
            results.append(compiled)

        environment = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "nvidia_driver": _nvidia_smi("driver_version"),
            "gpu_memory_total": _nvidia_smi("memory.total"),
            "sm_clock_max_mhz": _nvidia_smi("clocks.max.sm"),
            "sm_clock_current_mhz": _nvidia_smi("clocks.current.sm"),
            "compute_capability": _nvidia_smi("compute_cap"),
        }
        return {"results": results, "environment": environment}

    @app.local_entrypoint()
    def main(compile: bool = False, measured_steps: int = 40) -> None:
        # A100-40GB-safe micro-batches at context 1024 (peak <= 23 GB, fits every
        # GPU >= 40 GB; H100-80GB / L40S-48GB have room to spare).
        micro_batches = {"124M": 24, "150M": 16, "350M": 8}
        payload = run_probe.remote(micro_batches, measured_steps, compile)

        output_dir = REPO_ROOT / "runs" / "throughput-probe"
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu_slug = PROBE_GPU.lower().replace("-", "").replace(" ", "")
        out_path = output_dir / f"probe-results-{gpu_slug}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(json.dumps(payload, indent=2))
        print(f"\ngpu: {PROBE_GPU}\nwrote {out_path}")
        for row in payload["results"]:
            mfu = row["mfu"]
            mfu_str = f"{mfu * 100:.1f}%" if mfu is not None else "n/a"
            print(
                f"{row['model_name']:>14}: "
                f"{row['tokens_per_second']:>12,.0f} tok/s  "
                f"step {row['step_time_ms']:>7.1f} ms  "
                f"MFU {mfu_str:>6}  "
                f"peak {row['peak_memory_gb']:.1f} GB"
            )
