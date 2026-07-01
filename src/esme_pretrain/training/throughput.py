"""Training-step throughput and MFU probe for dense MHA configs.

This measures the *compute ceiling* of one efficient training step: bf16 autocast,
FlashAttention via SDPA, a fused AdamW optimizer, and gradient accumulation. It
feeds the model synthetic on-device token batches so the number reflects the
forward/backward/optimizer cost without a data-loader in the way -- token *values*
do not change the FLOPs, so random ids give the same throughput as real text. The
data-loader's separate cost is analysed in docs/internal/scaleup-probe.md rather than
measured here.

Outputs tokens/sec, step time, and Model FLOPs Utilization (MFU) for
cost-per-billion-token projections.
"""

from __future__ import annotations

import platform
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone
from esme_pretrain.torch import torch
from esme_pretrain.training.device_profile import peak_tflops_for_device


@dataclass(frozen=True)
class ProbeConfig:
    model: BackboneConfig
    micro_batch_size: int
    grad_accum_steps: int = 2
    warmup_steps: int = 10
    measured_steps: int = 40
    context_length: int | None = None  # defaults to model.context_length
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    device: str = "cuda"
    seed: int = 0
    use_fused_optimizer: bool = True
    use_compile: bool = False
    device_peak_tflops: float | None = None

    @property
    def effective_context(self) -> int:
        return self.context_length or self.model.context_length

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps * self.effective_context


@dataclass
class ProbeResult:
    model_name: str
    parameter_total: int
    parameter_non_embedding: int
    context_length: int
    micro_batch_size: int
    grad_accum_steps: int
    measured_steps: int
    tokens_processed: int
    elapsed_seconds: float
    tokens_per_second: float
    step_time_ms: float
    flops_per_token: float
    achieved_tflops: float
    mfu: float | None
    device: str
    device_name: str
    device_peak_tflops: float | None
    dtype: str
    fused_optimizer: bool
    compiled: bool
    peak_memory_gb: float | None
    sm_clock_mhz: int | None
    torch_version: str
    cuda_version: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported probe dtype: {name}")
    return mapping[name]


def _peak_tflops(config: ProbeConfig, device_name: str) -> float | None:
    if config.device_peak_tflops is not None:
        return config.device_peak_tflops
    return peak_tflops_for_device(device_name)


def _sm_clock_mhz(device: torch.device) -> int | None:
    # Best-effort: torch.cuda.clock_rate needs nvidia-ml-py, which may be absent.
    # The Modal wrapper also records SM clocks via nvidia-smi, so failure here is fine.
    if device.type != "cuda":
        return None
    try:
        return int(torch.cuda.clock_rate(device.index or 0))  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError, ImportError):
        return None


def run_throughput_probe(config: ProbeConfig) -> ProbeResult:
    if config.micro_batch_size < 1:
        raise ValueError("micro_batch_size must be at least 1")
    if config.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    if config.measured_steps < 1:
        raise ValueError("measured_steps must be at least 1")

    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    is_cuda = device.type == "cuda"
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda device requested but torch.cuda.is_available() is False")

    if is_cuda:
        torch.set_float32_matmul_precision("high")

    dtype = _resolve_dtype(config.dtype)
    autocast_enabled = is_cuda and dtype in (torch.bfloat16, torch.float16)
    notes: list[str] = []

    model = DenseBackbone(config.model).to(device)
    model.train()

    compiled = False
    if config.use_compile:
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            compiled = True
        except Exception as error:  # noqa: BLE001 - compile is best-effort headroom
            notes.append(f"torch.compile unavailable, ran eager: {error}")

    fused = config.use_fused_optimizer and is_cuda
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=fused)

    context = config.effective_context
    vocab = config.model.vocab_size
    # One synthetic batch held on-device; reused so timing isolates compute.
    input_ids = torch.randint(
        0, vocab, (config.micro_batch_size, context), device=device, dtype=torch.long
    )
    targets = torch.randint(
        0, vocab, (config.micro_batch_size, context), device=device, dtype=torch.long
    )

    def one_optimizer_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.grad_accum_steps):
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(input_ids)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, vocab), targets.reshape(-1)
                    )
            else:
                logits = model(input_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, vocab), targets.reshape(-1)
                )
            (loss / config.grad_accum_steps).backward()
        optimizer.step()

    for _ in range(config.warmup_steps):
        one_optimizer_step()
    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    for _ in range(config.measured_steps):
        one_optimizer_step()
    if is_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens = config.measured_steps * config.tokens_per_optimizer_step
    tokens_per_second = tokens / elapsed
    step_time_ms = (elapsed / config.measured_steps) * 1000.0
    flops_per_token = config.model.flops_per_token(context)
    achieved_tflops = (tokens_per_second * flops_per_token) / 1e12

    device_name = torch.cuda.get_device_name(device) if is_cuda else platform.processor() or "cpu"
    peak_tflops = _peak_tflops(config, device_name) if is_cuda else None
    mfu = (achieved_tflops / peak_tflops) if peak_tflops else None
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if is_cuda else None
    params = config.model.parameter_count()

    return ProbeResult(
        model_name=config.model.name,
        parameter_total=params["total"],
        parameter_non_embedding=params["non_embedding"],
        context_length=context,
        micro_batch_size=config.micro_batch_size,
        grad_accum_steps=config.grad_accum_steps,
        measured_steps=config.measured_steps,
        tokens_processed=tokens,
        elapsed_seconds=elapsed,
        tokens_per_second=tokens_per_second,
        step_time_ms=step_time_ms,
        flops_per_token=flops_per_token,
        achieved_tflops=achieved_tflops,
        mfu=mfu,
        device=str(device),
        device_name=device_name,
        device_peak_tflops=peak_tflops,
        dtype=config.dtype,
        fused_optimizer=fused,
        compiled=compiled,
        peak_memory_gb=peak_memory_gb,
        sm_clock_mhz=_sm_clock_mhz(device),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda if is_cuda else None,
        notes=notes,
    )
