"""Producer-side contract test for the llm_pretrain_dense_v1 export bundle.

These tests pin the public artifact contract that downstream loaders
(llm-infer, esme-posttrain) consume: the file set, manifest fields, config.json
keys, weights.pt payload fields, and state-dict key names. A failure here means
an export change would break a downstream loader. Do not loosen a pinned name;
bump the format version and follow docs/bundle-format.md instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone
from esme_pretrain.postrun.bundle import load_bundle
from esme_pretrain.postrun.eval_checkpoints import file_sha256
from esme_pretrain.postrun.export_bundle import (
    BUNDLE_SCHEMA_VERSION,
    CANONICAL_BUNDLE_FORMAT,
    ExportConfig,
    export_checkpoint,
)
from esme_pretrain.torch import torch
from esme_pretrain.training.checkpointing import save_pretrain_checkpoint

EXPECTED_BUNDLE_FILES = frozenset(
    {"manifest.json", "config.json", "tokenizer.json", "weights.pt", "README.md"}
)

EXPECTED_CONFIG_KEYS = frozenset(
    {
        "name",
        "vocab_size",
        "context_length",
        "embedding_dim",
        "layers",
        "heads",
        "feedforward_dim",
        "kv_heads",
        "rope_theta",
        "rms_norm_eps",
        "tie_embeddings",
        "qk_norm",
        "z_loss_weight",
        "attention_kind",
    }
)

# The exact state-dict key names for a 2-layer tied GQA model with QK-norm.
# llm-infer resolves these names in its loader alias tables; renaming any of
# them is a breaking format change.
EXPECTED_STATE_DICT_KEYS = frozenset(
    {
        "token_embedding.weight",
        "lm_head.weight",
        "final_norm.weight",
    }
    | {
        f"blocks.{layer}.{suffix}"
        for layer in (0, 1)
        for suffix in (
            "attention_norm.weight",
            "attention.wq.weight",
            "attention.wk.weight",
            "attention.wv.weight",
            "attention.wo.weight",
            "attention.q_norm.weight",
            "attention.k_norm.weight",
            "feedforward_norm.weight",
            "feedforward.w_gate.weight",
            "feedforward.w_up.weight",
            "feedforward.w_down.weight",
        )
    }
)


def _contract_config() -> BackboneConfig:
    return BackboneConfig(
        name="contract-fixture",
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        layers=2,
        heads=4,
        kv_heads=2,
        feedforward_dim=32,
        qk_norm=True,
        tie_embeddings=True,
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


@pytest.fixture()
def contract_bundle(tmp_path: Path, monkeypatch) -> tuple[Path, BackboneConfig]:
    torch.manual_seed(20260710)
    config = _contract_config()
    checkpoint = tmp_path / "checkpoint.pt"
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"kind":"synthetic-fixture"}', encoding="utf-8")
    model = DenseBackbone(config)
    save_pretrain_checkpoint(checkpoint, model=model, config=config, step=7)
    _install_fake_tokenizers(monkeypatch, tokenizer, vocab_size=config.vocab_size)
    bundle_dir = tmp_path / "bundle"
    export_checkpoint(
        ExportConfig(
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            output_dir=bundle_dir,
        )
    )
    return bundle_dir, config


def test_bundle_file_set_is_exact(contract_bundle) -> None:
    bundle_dir, _ = contract_bundle
    assert {path.name for path in bundle_dir.iterdir()} == set(EXPECTED_BUNDLE_FILES)


def test_manifest_contract(contract_bundle) -> None:
    bundle_dir, config = contract_bundle
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == BUNDLE_SCHEMA_VERSION == 1
    assert manifest["format"] == CANONICAL_BUNDLE_FORMAT == "llm_pretrain_dense_v1"
    assert manifest["tokenizer"] == {"path": "tokenizer.json", "format": "tokenizers-json"}
    assert manifest["checkpoint_step"] == 7
    assert manifest["model_config"] == config.to_dict()
    assert manifest["source_checkpoint_sha256"]

    for name in ("config", "tokenizer", "weights", "readme"):
        entry = manifest["files"][name]
        assert file_sha256(bundle_dir / entry["path"]) == entry["sha256"]


def test_config_json_key_contract(contract_bundle) -> None:
    bundle_dir, config = contract_bundle
    payload = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    assert set(payload) == set(EXPECTED_CONFIG_KEYS)
    assert payload == config.to_dict()


def test_weights_payload_contract(contract_bundle) -> None:
    bundle_dir, config = contract_bundle
    payload = torch.load(bundle_dir / "weights.pt", map_location="cpu", weights_only=False)

    assert payload["format_version"] == BUNDLE_SCHEMA_VERSION
    assert payload["format"] == CANONICAL_BUNDLE_FORMAT
    assert payload["metadata"]["key_format"] == CANONICAL_BUNDLE_FORMAT
    assert payload["model_config"] == config.to_dict()
    assert payload["checkpoint_step"] == 7
    assert payload["source_checkpoint_sha256"]
    assert set(payload["state_dict"]) == set(EXPECTED_STATE_DICT_KEYS)


def test_state_dict_tensors_are_floating_point(contract_bundle) -> None:
    bundle_dir, _ = contract_bundle
    payload = torch.load(bundle_dir / "weights.pt", map_location="cpu", weights_only=False)
    non_float = [
        name for name, tensor in payload["state_dict"].items() if not tensor.is_floating_point()
    ]
    assert non_float == []


def test_load_bundle_rejects_future_schema_version(contract_bundle) -> None:
    bundle_dir, _ = contract_bundle
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = BUNDLE_SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported bundle schema_version"):
        load_bundle(bundle_dir)


def test_load_bundle_rejects_missing_schema_version(contract_bundle) -> None:
    bundle_dir, _ = contract_bundle
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["schema_version"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported bundle schema_version"):
        load_bundle(bundle_dir)


def test_load_bundle_rejects_future_weights_format_version(contract_bundle) -> None:
    bundle_dir, _ = contract_bundle
    weights_path = bundle_dir / "weights.pt"
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    payload["format_version"] = BUNDLE_SCHEMA_VERSION + 1
    torch.save(payload, weights_path)
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["weights"]["sha256"] = file_sha256(weights_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported weights.pt format_version"):
        load_bundle(bundle_dir)


def test_load_bundle_rejects_tampered_tokenizer(contract_bundle) -> None:
    bundle_dir, _ = contract_bundle
    with (bundle_dir / "tokenizer.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="hash mismatch for tokenizer.json"):
        load_bundle(bundle_dir)
