"""Shared transformer primitives for the DenseBackbone stack.

RMSNorm, rotary position embeddings (RoPE), and the SwiGLU MLP.
"""

from __future__ import annotations

from esme_pretrain.torch import torch

F = torch.nn.functional


class RMSNorm(torch.nn.Module):
    """Root-mean-square layer norm, computed in fp32.

    Used both for the residual-stream norms and, with ``dim = head_dim``, for the
    optional QK-norm on attention queries/keys.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        hidden_fp32 = hidden.float()
        normed = hidden_fp32 * torch.rsqrt(hidden_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(dtype) * self.weight


def build_rope_cache(
    context_length: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the cos/sin tables for rotary embeddings, shape ``[T, head_dim]``."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(context_length, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # [T, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # query/key: [B, H, T, head_dim]; cos/sin: [T, head_dim] -> broadcast.
    cos = cos.to(query.dtype)[None, None, :, :]
    sin = sin.to(query.dtype)[None, None, :, :]
    query_out = (query * cos) + (_rotate_half(query) * sin)
    key_out = (key * cos) + (_rotate_half(key) * sin)
    return query_out, key_out


class SwiGLU(torch.nn.Module):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))``, no biases."""

    def __init__(self, dim: int, feedforward_dim: int) -> None:
        super().__init__()
        self.w_gate = torch.nn.Linear(dim, feedforward_dim, bias=False)
        self.w_up = torch.nn.Linear(dim, feedforward_dim, bias=False)
        self.w_down = torch.nn.Linear(feedforward_dim, dim, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(hidden)) * self.w_up(hidden))
