from __future__ import annotations

from esme_pretrain.modeling.backbone import PROBE_CONFIGS, BackboneConfig, DenseBackbone
from esme_pretrain.torch import torch
from esme_pretrain.training.throughput import ProbeConfig, run_throughput_probe


def test_probe_config_param_counts_land_near_targets() -> None:
    totals = {name: cfg.parameter_count()["total"] for name, cfg in PROBE_CONFIGS.items()}
    assert 105_000_000 < totals["124M"] < 115_000_000
    assert 140_000_000 < totals["150M"] < 155_000_000
    assert 335_000_000 < totals["350M"] < 350_000_000


def test_forward_shape_and_causality_path() -> None:
    config = BackboneConfig(
        name="tiny",
        vocab_size=64,
        context_length=16,
        embedding_dim=32,
        layers=2,
        heads=4,
        feedforward_dim=64,
        attention_kind="mha",
        qk_norm=False,
        z_loss_weight=0.0,
    )
    model = DenseBackbone(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(input_ids)
    assert logits.shape == (2, 8, config.vocab_size)


def test_flops_per_token_is_dominated_by_6n() -> None:
    config = PROBE_CONFIGS["150M"]
    n = config.parameter_count()["total"]
    flops = config.flops_per_token(1024)
    attention_term = 12 * config.layers * config.embedding_dim * 1024
    assert flops == 6 * n + attention_term
    assert flops > 6 * n


def test_tiny_cpu_probe_runs_and_reports_tokens_per_second() -> None:
    config = ProbeConfig(
        model=BackboneConfig(
            name="tiny",
            vocab_size=128,
            context_length=32,
            embedding_dim=64,
            layers=2,
            heads=4,
            feedforward_dim=128,
            attention_kind="mha",
            qk_norm=False,
            z_loss_weight=0.0,
        ),
        micro_batch_size=2,
        grad_accum_steps=2,
        warmup_steps=1,
        measured_steps=2,
        dtype="float32",
        device="cpu",
        use_fused_optimizer=False,
    )
    result = run_throughput_probe(config)
    assert result.tokens_processed == 2 * (2 * 2 * 32)
    assert result.tokens_per_second > 0
    assert result.step_time_ms > 0
    assert result.mfu is None
    assert result.flops_per_token > 0
