"""Trainable dense decoder for the accepted 214M pretrain run.

Decoder-only GQA baseline with RoPE, RMSNorm, SwiGLU, QK-norm, and optional z-loss.
See ``docs/architecture.md`` for design rationale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from esme_pretrain.modeling.layers import RMSNorm, SwiGLU, apply_rope, build_rope_cache
from esme_pretrain.torch import torch

F = torch.nn.functional

# Config keys removed after the accepted 214M run, with the only value they ever
# held. Checkpoints saved before the cleanup still carry them; they load only if
# the feature was disabled, so the weights are unaffected.
REMOVED_DISABLED_CONFIG_KEYS: dict[str, Any] = {
    "logit_soft_cap": 0.0,
    "mtp_predict_tokens": 0,
}


def _drop_removed_disabled_keys(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    for key, disabled_value in REMOVED_DISABLED_CONFIG_KEYS.items():
        if key not in cleaned:
            continue
        if cleaned[key] != disabled_value:
            raise ValueError(
                f"legacy backbone config key {key!r} has non-disabled value "
                f"{cleaned[key]!r}; this checkpoint used a feature the current "
                "model code no longer implements"
            )
        del cleaned[key]
    return cleaned


@dataclass(frozen=True)
class BackboneConfig:
    """Config for the trainable dense backbone.

    ``feedforward_dim`` follows the standard SwiGLU convention (~8/3 * embedding_dim,
    rounded to a hardware-friendly multiple), so the three SwiGLU matrices match a
    classic 4*d feed-forward in FLOPs. The stability options (``qk_norm`` and
    ``z_loss_weight``) default to the accepted GQA baseline.
    """

    name: str
    vocab_size: int
    context_length: int
    embedding_dim: int
    layers: int
    heads: int
    feedforward_dim: int
    kv_heads: int | None = None
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    qk_norm: bool = True
    z_loss_weight: float = 1e-4
    attention_kind: str = "gqa"  # "mha" | "gqa"

    def __post_init__(self) -> None:
        if self.embedding_dim % self.heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")
        if self.effective_kv_heads < 1:
            raise ValueError("kv_heads must be at least 1")
        if self.heads % self.effective_kv_heads != 0:
            raise ValueError("heads must be divisible by kv_heads")
        if self.attention_kind == "mha" and self.effective_kv_heads != self.heads:
            raise ValueError("attention_kind='mha' requires kv_heads to equal heads")
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2")
        if self.z_loss_weight < 0:
            raise ValueError("z_loss_weight must be non-negative (0 disables)")
        if self.attention_kind not in ATTENTION_VARIANTS:
            raise ValueError(
                f"attention_kind must be one of {sorted(ATTENTION_VARIANTS)}, "
                f"got {self.attention_kind!r}"
            )

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.heads

    @property
    def effective_kv_heads(self) -> int:
        return self.heads if self.kv_heads is None else self.kv_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BackboneConfig:
        payload = _drop_removed_disabled_keys(payload)
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(payload) - fields
        if unknown:
            raise ValueError(f"unknown backbone config keys: {sorted(unknown)}")
        return cls(**payload)

    def parameter_count(self) -> dict[str, int]:
        """Exact parameter counts for the modules below (no biases anywhere)."""
        d = self.embedding_dim
        d_ff = self.feedforward_dim
        embedding = self.vocab_size * d
        kv_dim = self.effective_kv_heads * self.head_dim
        attention = 2 * d * d  # wq, wo
        attention += 2 * d * kv_dim  # wk, wv
        if self.qk_norm:
            attention += 2 * self.head_dim  # one RMSNorm weight vector for Q and for K
        mlp = 3 * d * d_ff  # gate, up, down (SwiGLU)
        block_norms = 2 * d  # two residual-stream RMSNorm weight vectors per block
        per_block = attention + mlp + block_norms
        non_embedding = self.layers * per_block + d  # + final RMSNorm
        # lm_head shares the embedding matrix when tied, so it adds no params.
        lm_head = 0 if self.tie_embeddings else embedding
        total = embedding + non_embedding + lm_head
        return {"total": total, "non_embedding": non_embedding, "embedding": embedding}

    def flops_per_token(self, context_length: int | None = None) -> float:
        """Forward+backward model FLOPs per token (the standard 6N + 12*L*d*ctx estimate)."""
        ctx = context_length if context_length is not None else self.context_length
        n = self.parameter_count()["total"]
        return 6.0 * n + 12.0 * self.layers * self.embedding_dim * ctx


class CausalSelfAttention(torch.nn.Module):
    """The attention interface every variant implements.

    An attention variant maps a pre-normed residual stream to an attention output
    of the same shape, given the precomputed RoPE tables:
    ``forward(hidden[B, T, D], cos[T, head_dim], sin[T, head_dim]) -> [B, T, D]``.
    Swapping MHA and GQA changes which subclass :func:`build_attention` returns;
    nothing else in the model moves. Subclasses take a :class:`BackboneConfig`.
    """

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MultiHeadAttention(CausalSelfAttention):
    """Multi-head self-attention: RoPE + optional QK-norm + SDPA-flash."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        if config.effective_kv_heads != config.heads:
            raise ValueError("MultiHeadAttention requires kv_heads to equal heads")
        self.heads = config.heads
        self.head_dim = config.head_dim
        d = config.embedding_dim
        self.wq = torch.nn.Linear(d, d, bias=False)
        self.wk = torch.nn.Linear(d, d, bias=False)
        self.wv = torch.nn.Linear(d, d, bias=False)
        self.wo = torch.nn.Linear(d, d, bias=False)
        # QK-norm normalizes each head's query/key vector (length head_dim) before
        # RoPE, which stabilizes attention logits at ~no cost.
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        query = self.wq(hidden).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        key = self.wk(hidden).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        value = self.wv(hidden).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query, key = apply_rope(query, key, cos, sin)
        # is_causal lets SDPA skip the explicit mask and pick a fused causal kernel.
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch, seq, self.heads * self.head_dim)
        return self.wo(attention)


class GroupedQueryAttention(CausalSelfAttention):
    """Grouped-query attention: full query heads with fewer shared KV heads.

    The implementation repeats K/V heads before SDPA rather than relying on a
    version-specific ``enable_gqa`` flag, so CPU tests and pinned Modal torch use
    the same code path.
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.kv_heads = config.effective_kv_heads
        self.head_dim = config.head_dim
        self.kv_repeat = self.heads // self.kv_heads
        d = config.embedding_dim
        kv_dim = self.kv_heads * self.head_dim
        self.wq = torch.nn.Linear(d, d, bias=False)
        self.wk = torch.nn.Linear(d, kv_dim, bias=False)
        self.wv = torch.nn.Linear(d, kv_dim, bias=False)
        self.wo = torch.nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden.shape
        query = self.wq(hidden).view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        key = self.wk(hidden).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.wv(hidden).view(batch, seq, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query, key = apply_rope(query, key, cos, sin)
        if self.kv_repeat != 1:
            key = key.repeat_interleave(self.kv_repeat, dim=1)
            value = value.repeat_interleave(self.kv_repeat, dim=1)
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        attention = attention.transpose(1, 2).reshape(batch, seq, self.heads * self.head_dim)
        return self.wo(attention)


ATTENTION_VARIANTS: dict[str, type[CausalSelfAttention]] = {
    "mha": MultiHeadAttention,
    "gqa": GroupedQueryAttention,
}


def build_attention(config: BackboneConfig) -> CausalSelfAttention:
    return ATTENTION_VARIANTS[config.attention_kind](config)


class DecoderBlock(torch.nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.embedding_dim, config.rms_norm_eps)
        self.attention = build_attention(config)
        self.feedforward_norm = RMSNorm(config.embedding_dim, config.rms_norm_eps)
        self.feedforward = SwiGLU(config.embedding_dim, config.feedforward_dim)

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), cos, sin)
        return hidden + self.feedforward(self.feedforward_norm(hidden))


class DenseBackbone(torch.nn.Module):
    """The trainable decoder-only transformer for the accepted 214M run."""

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = torch.nn.ModuleList(DecoderBlock(config) for _ in range(config.layers))
        self.final_norm = RMSNorm(config.embedding_dim, config.rms_norm_eps)
        self.lm_head = torch.nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        cos, sin = build_rope_cache(
            config.context_length, config.head_dim, config.rope_theta, torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: torch.nn.Module) -> None:
        # Standard transformer init: normal(0, 0.02) for linear weights and embeddings
        # (all linears are bias-free). RMSNorm weights keep their ones() init. The
        # residual-path output projections are rescaled afterward (see below).
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        # Residual-init trick: scale each residual-path output projection (attention
        # ``wo`` and MLP ``w_down``) by 1/sqrt(2 * n_layers). Both blocks write to the
        # residual stream once each, so 2*n_layers additive contributions accumulate;
        # this keeps the residual-stream variance ~constant at init instead of growing
        # with depth.
        scale = (2.0 * self.config.layers) ** -0.5
        with torch.no_grad():
            for block in self.blocks:
                block.attention.wo.weight.mul_(scale)
                block.feedforward.w_down.weight.mul_(scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        if input_ids.ndim != 2:
            raise ValueError("input ids must have shape [batch, sequence]")
        seq = input_ids.shape[1]
        if seq > self.config.context_length:
            raise ValueError(f"sequence length {seq} exceeds context {self.config.context_length}")
        cos = self.rope_cos[:seq]
        sin = self.rope_sin[:seq]
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, cos, sin)
        return self.lm_head(self.final_norm(hidden))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Greedy (or low-temperature) autoregressive sampling for sample logging.

        No KV cache — this is for short qualitative samples, not serving.
        """
        was_training = self.training
        self.eval()
        ctx = self.config.context_length
        generated = input_ids
        for _ in range(max_new_tokens):
            conditioned = generated[:, -ctx:]
            next_logits = self(conditioned)[:, -1, :]
            if temperature <= 0.0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    kth = torch.topk(next_logits, top_k, dim=-1).values[:, -1, None]
                    next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))
                probabilities = F.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_id), dim=1)
        if was_training:
            self.train()
        return generated


def language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    z_loss_weight: float = 0.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Next-token cross-entropy plus optional z-loss on raw logits."""
    raw_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index

    log_z = torch.logsumexp(raw_logits, dim=-1)  # [N], the CE denominator
    # clamp_min(0) keeps gather in range for ignored positions; they are masked out.
    target_logits = raw_logits.gather(-1, flat_targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    negative_log_likelihood = (log_z - target_logits)[valid]
    if negative_log_likelihood.numel() == 0:
        # No supervised positions (degenerate batch): a graph-connected zero, and z-loss
        # has nothing to penalize, so skip it (its mean over zero elements is NaN).
        components = {"ce_loss": 0.0}
        total = raw_logits.sum() * 0.0
        components["total_loss"] = float(total.detach())
        return total, components

    cross_entropy = negative_log_likelihood.mean()
    components = {"ce_loss": float(cross_entropy.detach())}
    total = cross_entropy
    if z_loss_weight > 0.0:
        z_loss = z_loss_weight * (log_z[valid] ** 2).mean()
        total = total + z_loss
        components["z_loss"] = float(z_loss.detach())
    components["total_loss"] = float(total.detach())
    return total, components


# vocab=32768 is the accepted byte-level BPE size for the 214M run.
# The probe configs test the deep/thin small-model recipe: keep head_dim=64
# and 3:1 GQA, then reinvest width into depth.
BACKBONE_VOCAB_SIZE = 32768

BACKBONE_CONFIGS: dict[str, BackboneConfig] = {
    "214M": BackboneConfig(
        name="214M",
        vocab_size=BACKBONE_VOCAB_SIZE,
        context_length=1024,
        embedding_dim=768,
        layers=30,
        heads=12,
        kv_heads=4,
        feedforward_dim=2048,
        attention_kind="gqa",
    ),
}


def baseline_config(name: str = "214M", **overrides: Any) -> BackboneConfig:
    """Fetch a named production preset, optionally overriding individual fields."""
    if name not in BACKBONE_CONFIGS:
        raise ValueError(f"unknown backbone preset {name!r}; known: {sorted(BACKBONE_CONFIGS)}")
    config = BACKBONE_CONFIGS[name]
    return replace(config, **overrides) if overrides else config


def _probe_config(**fields: Any) -> BackboneConfig:
    return BackboneConfig(
        attention_kind="mha",
        qk_norm=False,
        z_loss_weight=0.0,
        **fields,
    )


# Throughput-probe shapes (MHA, no QK-norm), separate from the accepted 214M preset.
PROBE_CONFIGS: dict[str, BackboneConfig] = {
    "124M": _probe_config(
        name="probe-124M",
        vocab_size=BACKBONE_VOCAB_SIZE,
        context_length=1024,
        embedding_dim=768,
        layers=12,
        heads=12,
        feedforward_dim=2048,
    ),
    "150M": _probe_config(
        name="probe-150M",
        vocab_size=BACKBONE_VOCAB_SIZE,
        context_length=1024,
        embedding_dim=896,
        layers=12,
        heads=14,
        feedforward_dim=2432,
    ),
    "350M": _probe_config(
        name="probe-350M",
        vocab_size=BACKBONE_VOCAB_SIZE,
        context_length=1024,
        embedding_dim=1024,
        layers=24,
        heads=16,
        feedforward_dim=2816,
    ),
}
