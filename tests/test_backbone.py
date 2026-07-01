from __future__ import annotations

import pytest

from esme_pretrain.modeling.backbone import (
    BACKBONE_CONFIGS,
    BackboneConfig,
    DenseBackbone,
    baseline_config,
    build_attention,
    language_model_loss,
    soft_cap_logits,
)
from esme_pretrain.torch import torch


def _tiny(**overrides) -> BackboneConfig:
    base = dict(
        name="tiny",
        vocab_size=128,
        context_length=32,
        embedding_dim=64,
        layers=2,
        heads=4,
        feedforward_dim=128,
    )
    base.update(overrides)
    return BackboneConfig(**base)


def test_150m_preset_matches_probe_param_target() -> None:
    config = baseline_config("150M")
    total = config.parameter_count()["total"]
    assert total == 136_693_824
    assert 120_000_000 < total < 155_000_000
    assert config.head_dim == 64
    assert config.embedding_dim == 576
    assert config.layers == 30
    assert config.heads == 9
    assert config.attention_kind == "gqa"
    assert config.kv_heads == 3


def test_214m_preset_matches_pretrain_param_target() -> None:
    config = baseline_config("214M")
    total = config.parameter_count()["total"]
    assert total == 213_960_192
    assert config.head_dim == 64
    assert config.embedding_dim == 768
    assert config.layers == 30
    assert config.heads == 12
    assert config.attention_kind == "gqa"
    assert config.kv_heads == 4
    assert config.feedforward_dim == 2048


def test_forward_shape() -> None:
    config = _tiny()
    model = DenseBackbone(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, config.vocab_size)


def test_sequence_longer_than_context_raises() -> None:
    model = DenseBackbone(_tiny(context_length=8))
    with pytest.raises(ValueError, match="exceeds context"):
        model(torch.randint(0, 128, (1, 9)))


def test_soft_cap_bounds_logits() -> None:
    # soft_cap_logits is the bounding primitive; forward() now returns RAW logits
    # (so the z-loss can regularize them), and the cap is applied in the loss/generate.
    capped = soft_cap_logits(torch.tensor([[100.0, -100.0, 0.0]]), 5.0)
    assert float(capped.abs().max()) <= 5.0
    # cap <= 0 disables capping (identity passthrough).
    raw = torch.tensor([[100.0, -100.0]])
    assert torch.equal(soft_cap_logits(raw, 0.0), raw)


def test_forward_returns_raw_uncapped_logits() -> None:
    # Regression guard for the z-loss fix: forward() must NOT soft-cap, so logsumexp
    # over its output is unbounded and the z-loss stays a real regularizer.
    config = _tiny(logit_soft_cap=5.0)
    model = DenseBackbone(config)
    # Push the LM head to large magnitudes so a cap, if present, would clamp to 5.
    with torch.no_grad():
        model.lm_head.weight.mul_(50.0)
    logits = model(torch.randint(0, config.vocab_size, (2, 16)))
    assert float(logits.detach().abs().max()) > 5.0


def test_qk_norm_adds_head_dim_params() -> None:
    with_norm = _tiny(qk_norm=True).parameter_count()["total"]
    without_norm = _tiny(qk_norm=False).parameter_count()["total"]
    # Two RMSNorm(head_dim) vectors per layer.
    config = _tiny()
    assert with_norm - without_norm == config.layers * 2 * config.head_dim


def test_param_count_formula_matches_built_module() -> None:
    config = _tiny()
    model = DenseBackbone(config)
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
    assert total == config.parameter_count()["total"]


def test_gqa_param_count_formula_matches_built_module() -> None:
    config = _tiny(attention_kind="gqa", kv_heads=2)
    model = DenseBackbone(config)
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel()
    assert total == config.parameter_count()["total"]


def test_gqa_forward_shape() -> None:
    config = _tiny(attention_kind="gqa", kv_heads=2)
    model = DenseBackbone(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, config.vocab_size)


def test_gqa_reduces_key_value_projection_params() -> None:
    mha = _tiny().parameter_count()["total"]
    gqa = _tiny(attention_kind="gqa", kv_heads=2).parameter_count()["total"]
    config = _tiny()
    expected_savings = (
        config.layers * 2 * config.embedding_dim * (config.heads - 2) * config.head_dim
    )
    assert mha - gqa == expected_savings


def test_tied_embeddings_share_storage() -> None:
    model = DenseBackbone(_tiny(tie_embeddings=True))
    assert model.lm_head.weight is model.token_embedding.weight


def test_cross_entropy_matches_torch_reference() -> None:
    # The fused (single-logsumexp) CE must equal F.cross_entropy, including ignored
    # positions, so the memory optimization is provably loss-preserving.
    torch.manual_seed(0)
    logits = torch.randn(4, 6, 50)
    targets = torch.randint(0, 50, (4, 6))
    targets[0, 0] = -100  # an ignored position
    ours, _ = language_model_loss(logits, targets, z_loss_weight=0.0)
    reference = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 50).float(), targets.reshape(-1), ignore_index=-100
    )
    assert torch.allclose(ours, reference, atol=1e-5)


def test_z_loss_increases_total_loss() -> None:
    config = _tiny()
    model = DenseBackbone(config)
    logits = model(torch.randint(0, config.vocab_size, (2, 16)))
    targets = torch.randint(0, config.vocab_size, (2, 16))
    plain, plain_parts = language_model_loss(logits, targets, z_loss_weight=0.0)
    with_z, z_parts = language_model_loss(logits, targets, z_loss_weight=1e-2)
    assert "z_loss" not in plain_parts
    assert z_parts["z_loss"] > 0.0
    assert z_parts["total_loss"] > plain_parts["total_loss"]
    assert plain_parts["ce_loss"] == pytest.approx(z_parts["ce_loss"], rel=1e-6)


def test_cross_entropy_uses_soft_capped_logits() -> None:
    # CE must match F.cross_entropy on the *capped* logits (the distribution the
    # model is trained on), proving the loss applies the cap to the CE path itself
    # now that forward() returns raw logits.
    torch.manual_seed(0)
    raw = torch.randn(4, 6, 50) * 8.0  # large enough that the cap actually bites
    targets = torch.randint(0, 50, (4, 6))
    cap = 5.0
    ours, _ = language_model_loss(raw, targets, z_loss_weight=0.0, logit_soft_cap=cap)
    capped = soft_cap_logits(raw.reshape(-1, 50).float(), cap)
    reference = torch.nn.functional.cross_entropy(capped, targets.reshape(-1))
    assert torch.allclose(ours, reference, atol=1e-5)


def test_z_loss_regularizes_raw_logits_not_capped() -> None:
    # The fix: z-loss must have a *meaningful* gradient to the RAW pre-cap logits.
    # Soft-capping makes logsumexp(capped) bounded (<= log(V) + cap), so a z-loss on
    # capped logits collapses; on raw logits it stays a real penalty. We assert the
    # raw-logit z-loss gradient is orders of magnitude larger than the capped one.
    torch.manual_seed(0)
    vocab = 256
    # Logits that genuinely exceed the cap (std 30 vs cap 5), so tanh saturates and
    # the capped-path z-loss gradient collapses — the regime the review measured at
    # scale (raw z-grad ~6e-2 vs capped ~9e-4).
    cap = 5.0
    z_weight = 1e-2
    base = torch.randn(2, 16, vocab) * 30.0
    targets = torch.randint(0, vocab, (2, 16))

    # Real path: z-loss computed on raw logits (what the fixed loss does).
    raw = base.clone().requires_grad_(True)
    _, _ = language_model_loss(raw, targets, z_loss_weight=z_weight, logit_soft_cap=cap)
    z_raw = z_weight * (torch.logsumexp(raw.reshape(-1, vocab).float(), dim=-1) ** 2).mean()
    (z_grad_raw,) = torch.autograd.grad(z_raw, raw)

    # Broken path: z-loss computed on the soft-capped logits (the bug we fixed).
    capped_in = base.clone().requires_grad_(True)
    capped = soft_cap_logits(capped_in.reshape(-1, vocab).float(), cap)
    z_capped = z_weight * (torch.logsumexp(capped, dim=-1) ** 2).mean()
    (z_grad_capped,) = torch.autograd.grad(z_capped, capped_in)

    raw_norm = float(z_grad_raw.norm())
    capped_norm = float(z_grad_capped.norm())
    assert raw_norm > 0.0
    # The capped-path gradient is throttled by the tanh saturation; raw is far larger.
    assert raw_norm > 50.0 * capped_norm


def test_generate_extends_sequence() -> None:
    model = DenseBackbone(_tiny())
    prompt = torch.randint(0, 128, (1, 4))
    out = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], prompt)


def test_mla_attention_placeholder_is_reserved() -> None:
    config = _tiny(attention_kind="mla")
    with pytest.raises(NotImplementedError, match="future ablation"):
        build_attention(config)


def test_mtp_head_placeholder_is_reserved() -> None:
    with pytest.raises(NotImplementedError, match="future ablation"):
        DenseBackbone(_tiny(mtp_predict_tokens=2))


def test_baseline_has_no_mtp_head() -> None:
    model = DenseBackbone(_tiny())
    assert model.mtp_head is None


def test_unknown_attention_kind_rejected() -> None:
    with pytest.raises(ValueError, match="attention_kind"):
        _tiny(attention_kind="bogus")


def test_mha_rejects_reduced_kv_heads() -> None:
    with pytest.raises(ValueError, match="mha"):
        _tiny(attention_kind="mha", kv_heads=2)


def test_gqa_requires_kv_heads_to_divide_query_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        _tiny(attention_kind="gqa", kv_heads=3)


def test_config_round_trip_and_rejects_unknown_keys() -> None:
    config = _tiny()
    restored = BackboneConfig.from_dict(config.to_dict())
    assert restored == config
    with pytest.raises(ValueError, match="unknown backbone config keys"):
        BackboneConfig.from_dict({**config.to_dict(), "bogus": 1})


def test_baseline_config_override() -> None:
    overridden = baseline_config("150M", context_length=512)
    assert overridden.context_length == 512
    assert overridden.embedding_dim == BACKBONE_CONFIGS["150M"].embedding_dim
