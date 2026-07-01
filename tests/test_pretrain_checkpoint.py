from __future__ import annotations

from pathlib import Path

import pytest

from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone
from esme_pretrain.modeling.pretrain_checkpoint import (
    load_pretrain_checkpoint,
    save_pretrain_checkpoint,
)
from esme_pretrain.torch import torch


def _tiny() -> BackboneConfig:
    return BackboneConfig(
        name="tiny",
        vocab_size=128,
        context_length=32,
        embedding_dim=64,
        layers=2,
        heads=4,
        feedforward_dim=128,
    )


def test_checkpoint_round_trip_preserves_logits(tmp_path: Path) -> None:
    config = _tiny()
    model = DenseBackbone(config)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_pretrain_checkpoint(
        path, model=model, config=config, step=7, optimizer=optimizer, metrics={"loss": 1.23}
    )

    loaded = load_pretrain_checkpoint(path)
    assert loaded.step == 7
    assert loaded.config == config
    assert loaded.metrics["loss"] == pytest.approx(1.23)
    assert loaded.optimizer_state is not None

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        before = model(input_ids)
        after = loaded.model(input_ids)
    assert torch.allclose(before, after, atol=1e-6)


def test_optimizer_state_round_trips_for_resume(tmp_path: Path) -> None:
    config = _tiny()
    model = DenseBackbone(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Take a step so the optimizer has real state to reload.
    logits = model(torch.randint(0, config.vocab_size, (2, 16)))
    logits.sum().backward()
    optimizer.step()

    path = tmp_path / "checkpoint.pt"
    save_pretrain_checkpoint(path, model=model, config=config, step=1, optimizer=optimizer)
    loaded = load_pretrain_checkpoint(path)

    fresh = torch.optim.AdamW(DenseBackbone(config).parameters(), lr=1e-3)
    fresh.load_state_dict(loaded.optimizer_state)
    assert fresh.state_dict()["param_groups"][0]["lr"] == pytest.approx(1e-3)


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_pretrain_checkpoint(tmp_path / "nope.pt")


def test_unsupported_format_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 999}, path)
    with pytest.raises(ValueError, match="unsupported pretrain checkpoint format"):
        load_pretrain_checkpoint(path)


def test_checkpoint_load_rejects_extra_keys(tmp_path: Path) -> None:
    config = _tiny()
    path = tmp_path / "checkpoint.pt"
    save_pretrain_checkpoint(path, model=DenseBackbone(config), config=config, step=1)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["legacy_note"] = "do not accept unversioned schema drift"
    torch.save(payload, path)

    with pytest.raises(ValueError, match="unexpected keys.*legacy_note"):
        load_pretrain_checkpoint(path)
