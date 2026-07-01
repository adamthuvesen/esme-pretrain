"""Save / load checkpoints for the trainable :class:`DenseBackbone`.

Production checkpoints use one compact, explicit schema. A checkpoint round-trips model weights,
optimizer state, step counter, backbone config, data-loader stream offset, and
RNG state, so a run can resume exactly where it stopped — including the corpus
position — or be reloaded for eval / export.

The data offset is the number of **tokens** the run had consumed from the stream when
the checkpoint was written. On resume the loop fast-forwards the (deterministic) token
stream by that many tokens, so a preempted run continues toward the corpus tail
instead of silently re-reading already-seen tokens from the head. Counting in tokens
(not batches) keeps resume correct even if the resumed run uses a different batch size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone
from esme_pretrain.torch import torch

# v2 adds data_offset_tokens + rng_state for stream-faithful resume. The repo is
# local-only with no persisted checkpoints to keep compatible, so this is a clean bump.
PRETRAIN_CHECKPOINT_FORMAT = 2


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the eager module behind a ``torch.compile`` wrapper (if any)."""
    return getattr(model, "_orig_mod", model)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot Python / NumPy / torch (+CUDA) RNG state for a faithful resume."""
    import random

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ModuleNotFoundError:
        # NumPy is optional in this env (runtime.py guards its absence); nothing to save.
        pass
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore RNG state captured by :func:`capture_rng_state` (no-op if absent)."""
    if not state:
        return
    import random

    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        # torch.get_rng_state returns a ByteTensor; set_rng_state needs it on CPU.
        torch.set_rng_state(_as_cpu_byte_tensor(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_cpu_byte_tensor(s) for s in state["torch_cuda"]])
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ModuleNotFoundError:
            pass


def _as_cpu_byte_tensor(value: Any) -> torch.Tensor:
    # weights_only torch.load rebuilds tensors fine; this guards any non-CPU/byte edge.
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(device="cpu", dtype=torch.uint8)


@dataclass(frozen=True)
class LoadedPretrainCheckpoint:
    model: DenseBackbone
    config: BackboneConfig
    optimizer_state: dict[str, Any]
    step: int
    metrics: dict[str, Any]
    # Tokens consumed from the stream by the time of this checkpoint; the resume offset
    # the loop fast-forwards to. 0 for checkpoints written before any step.
    data_offset_tokens: int = 0
    rng_state: dict[str, Any] = field(default_factory=dict)


def save_pretrain_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    config: BackboneConfig,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: dict[str, Any] | None = None,
    data_offset_tokens: int = 0,
    rng_state: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": PRETRAIN_CHECKPOINT_FORMAT,
        "config": config.to_dict(),
        "model_state": _unwrap(model).state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "metrics": dict(metrics or {}),
        "data_offset_tokens": int(data_offset_tokens),
        "rng_state": dict(rng_state or {}),
    }
    # Atomic write: a crash mid-save must not corrupt the previous good checkpoint.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_pretrain_checkpoint(
    path: Path, *, map_location: str | torch.device = "cpu"
) -> LoadedPretrainCheckpoint:
    if not Path(path).exists():
        raise ValueError(f"checkpoint does not exist: {path}")
    # weights_only=False: the payload carries RNG state (Python/NumPy tuples), which
    # are not plain tensors. This loads only our own checkpoints (local, trusted).
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != PRETRAIN_CHECKPOINT_FORMAT:
        raise ValueError("unsupported pretrain checkpoint format")
    config = BackboneConfig.from_dict(payload["config"])
    model = DenseBackbone(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return LoadedPretrainCheckpoint(
        model=model,
        config=config,
        optimizer_state=payload["optimizer_state"],
        step=int(payload["step"]),
        metrics=dict(payload["metrics"]),
        data_offset_tokens=int(payload.get("data_offset_tokens", 0)),
        rng_state=dict(payload.get("rng_state") or {}),
    )
