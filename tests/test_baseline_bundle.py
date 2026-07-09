from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone
from esme_pretrain.postrun.bundle import load_bundle
from esme_pretrain.postrun.eval_checkpoints import file_sha256
from esme_pretrain.postrun.export_bundle import ExportConfig, export_checkpoint
from esme_pretrain.torch import torch
from esme_pretrain.training.checkpointing import save_pretrain_checkpoint


def _small_config() -> BackboneConfig:
    return BackboneConfig(
        name="small-test",
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        z_loss_weight=0.0,
    )


def _install_fake_tokenizers(monkeypatch, tokenizer_path: Path, *, vocab_size: int) -> None:
    class FakeTokenizer:
        @classmethod
        def from_file(cls, path: str):
            assert path == str(tokenizer_path)
            return cls()

        def get_vocab_size(self) -> int:
            return vocab_size

        def token_to_id(self, token: str) -> int | None:
            special_tokens = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
            return special_tokens.get(token)

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizer))


def _export_small_bundle(tmp_path: Path, monkeypatch) -> tuple[Path, DenseBackbone, BackboneConfig]:
    config = _small_config()
    checkpoint = tmp_path / "checkpoint.pt"
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"kind":"synthetic"}', encoding="utf-8")
    model = DenseBackbone(config)
    save_pretrain_checkpoint(checkpoint, model=model, config=config, step=9)
    _install_fake_tokenizers(monkeypatch, tokenizer, vocab_size=config.vocab_size)
    bundle_dir = tmp_path / "export"
    export_checkpoint(
        ExportConfig(
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            output_dir=bundle_dir,
            export_format="llm-infer",
        )
    )
    return bundle_dir, model, config


def test_load_bundle_round_trip_preserves_logits(tmp_path: Path, monkeypatch) -> None:
    bundle_dir, model, config = _export_small_bundle(tmp_path, monkeypatch)

    loaded = load_bundle(bundle_dir)

    input_ids = torch.arange(0, 16, dtype=torch.long).view(2, 8) % config.vocab_size
    with torch.no_grad():
        expected = model(input_ids)
        actual = loaded.model(input_ids)

    assert loaded.config == config
    assert loaded.checkpoint_step == 9
    assert loaded.tokenizer_path == bundle_dir / "tokenizer.json"
    assert loaded.weights_sha256 == file_sha256(bundle_dir / "weights.pt")
    assert loaded.manifest["model_config"] == config.to_dict()
    assert not loaded.model.training
    assert torch.allclose(expected, actual, atol=1e-6)


def test_load_bundle_rejects_missing_files(tmp_path: Path, monkeypatch) -> None:
    bundle_dir, _, _ = _export_small_bundle(tmp_path, monkeypatch)
    (bundle_dir / "tokenizer.json").unlink()

    with pytest.raises(ValueError, match="missing required files.*tokenizer.json"):
        load_bundle(bundle_dir)


def test_load_bundle_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bundle directory does not exist"):
        load_bundle(tmp_path / "nope")


def test_load_bundle_rejects_tampered_weights(tmp_path: Path, monkeypatch) -> None:
    bundle_dir, _, _ = _export_small_bundle(tmp_path, monkeypatch)
    with (bundle_dir / "weights.pt").open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch for weights.pt"):
        load_bundle(bundle_dir)


def test_load_bundle_rejects_config_drift(tmp_path: Path, monkeypatch) -> None:
    bundle_dir, _, config = _export_small_bundle(tmp_path, monkeypatch)
    drifted = config.to_dict()
    drifted["layers"] = config.layers + 1
    config_path = bundle_dir / "config.json"
    config_path.write_text(json.dumps(drifted, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["config"]["sha256"] = file_sha256(config_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest model_config does not match config.json"):
        load_bundle(bundle_dir)


def test_load_bundle_rejects_foreign_format(tmp_path: Path, monkeypatch) -> None:
    bundle_dir, _, _ = _export_small_bundle(tmp_path, monkeypatch)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"] = "other_format_v9"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle format is not llm_pretrain_dense_v1"):
        load_bundle(bundle_dir)
