from __future__ import annotations

import pytest

from esme_pretrain.modeling.backbone import (
    BACKBONE_CONFIGS,
    BackboneConfig,
    DenseBackbone,
    baseline_config,
    language_model_loss,
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


def test_forward_returns_raw_logits() -> None:
    config = _tiny()
    model = DenseBackbone(config)
    with torch.no_grad():
        model.lm_head.weight.mul_(50.0)
    logits = model(torch.randint(0, config.vocab_size, (2, 16)))
    assert float(logits.detach().abs().max()) > 5.0


def test_attention_is_causal() -> None:
    # Changing a later token must not change logits at any earlier position. This is
    # the property that keeps the next-token loss from seeing its own answer; without
    # it, training and eval would silently leak future tokens.
    torch.manual_seed(0)
    model = DenseBackbone(_tiny())
    model.eval()
    ids = torch.randint(0, 128, (1, 12))
    altered = ids.clone()
    altered[0, -1] = (int(altered[0, -1]) + 1) % 128
    with torch.no_grad():
        base = model(ids)
        after = model(altered)
    assert torch.allclose(base[:, :-1], after[:, :-1], atol=1e-6)
    assert not torch.allclose(base[:, -1], after[:, -1], atol=1e-6)


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


def test_generate_extends_sequence() -> None:
    model = DenseBackbone(_tiny())
    prompt = torch.randint(0, 128, (1, 4))
    out = model.generate(prompt, max_new_tokens=6, temperature=0.0)
    assert out.shape == (1, 10)
    assert torch.equal(out[:, :4], prompt)


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
