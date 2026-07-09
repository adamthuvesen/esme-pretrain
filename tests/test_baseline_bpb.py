from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from esme_pretrain.baselines import models as baseline_models
from esme_pretrain.baselines.bpb import hash_texts, slice_bpb
from esme_pretrain.baselines.config import FinewebValidationSlice, HFDatasetSlice
from esme_pretrain.baselines.models import (
    EncodedText,
    load_slice_texts,
    partitioned_byte_counts,
)
from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone
from esme_pretrain.torch import torch


def _small_config() -> BackboneConfig:
    return BackboneConfig(
        name="small-test",
        vocab_size=32,
        context_length=4,
        embedding_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        z_loss_weight=0.0,
    )


class CharEvalModel:
    """Tiny char-level EvalModel over a real DenseBackbone."""

    def __init__(self, name: str = "char") -> None:
        self.name = name
        config = _small_config()
        self.context_length = config.context_length
        self.eos_id = 0
        self.provenance = {"kind": "fixture"}
        torch.manual_seed(7)
        self._module = DenseBackbone(config).eval()
        self._vocab_size = config.vocab_size

    def encode(self, text: str) -> EncodedText:
        ids = [(ord(char) % (self._vocab_size - 1)) + 1 for char in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return EncodedText(ids=ids, byte_counts=partitioned_byte_counts(text, offsets))

    def module(self) -> torch.nn.Module:
        return self._module


def test_partitioned_byte_counts_cover_multibyte_text() -> None:
    text = "aøb"
    offsets = [(0, 1), (1, 2), (2, 3)]

    counts = partitioned_byte_counts(text, offsets)

    assert counts == [1, 2, 1]
    assert sum(counts) == len(text.encode("utf-8"))


def test_partitioned_byte_counts_absorb_trimmed_whitespace() -> None:
    # GPT-2-style trim_offsets: the offsets skip the space, but the partition
    # assigns its byte to the following token.
    text = "hi there"
    offsets = [(0, 2), (3, 8)]

    counts = partitioned_byte_counts(text, offsets)

    assert counts == [3, 5]
    assert sum(counts) == len(text.encode("utf-8"))


def test_partitioned_byte_counts_handle_overlapping_offsets() -> None:
    # One codepoint split across two tokens must not double-count its bytes.
    text = "aøb"
    offsets = [(0, 1), (1, 2), (1, 2), (2, 3)]

    counts = partitioned_byte_counts(text, offsets)

    assert sum(counts) == len(text.encode("utf-8"))


def test_partitioned_byte_counts_reject_tokenless_text() -> None:
    with pytest.raises(ValueError, match="no tokens for non-empty text"):
        partitioned_byte_counts("abc", [])


def test_slice_bpb_matches_formula_and_hashes_text() -> None:
    model = CharEvalModel()
    texts = ["abcdefghij", "chars åäö and more text"]

    result = slice_bpb(model, texts, slice_name="fixture", batch_size=1, device="cpu")

    assert result.document_count == 2
    assert result.raw_bytes == sum(len(t.encode("utf-8")) for t in texts)
    assert result.text_sha256 == hash_texts(texts)
    assert result.eval_tokens == result.eval_batches * model.context_length
    assert result.bits_per_byte == pytest.approx(
        result.ce_loss * result.eval_tokens / math.log(2) / result.eval_bytes
    )
    assert result.perplexity == pytest.approx(math.exp(result.ce_loss))


def test_slice_bpb_is_deterministic() -> None:
    texts = ["abcdefghij", "second document"]

    first = slice_bpb(CharEvalModel(), texts, slice_name="fixture", batch_size=1, device="cpu")
    second = slice_bpb(CharEvalModel(), texts, slice_name="fixture", batch_size=1, device="cpu")

    assert first == second


def test_slice_bpb_text_hash_is_model_independent() -> None:
    texts = ["abcdefghij", "second document"]

    char_result = slice_bpb(CharEvalModel("a"), texts, slice_name="s", batch_size=1, device="cpu")

    assert char_result.text_sha256 == hash_texts(texts)


def test_slice_bpb_rejects_byte_undercount() -> None:
    class LossyModel(CharEvalModel):
        def encode(self, text: str) -> EncodedText:
            encoded = super().encode(text)
            byte_counts = list(encoded.byte_counts)
            byte_counts[0] = 0
            return EncodedText(ids=encoded.ids, byte_counts=byte_counts)

    with pytest.raises(ValueError, match="byte accounting mismatch"):
        slice_bpb(LossyModel(), ["abcdefghij"], slice_name="s", batch_size=1, device="cpu")


def test_slice_bpb_rejects_too_little_text() -> None:
    with pytest.raises(ValueError, match="did not produce any full eval batches"):
        slice_bpb(CharEvalModel(), ["ab"], slice_name="s", batch_size=4, device="cpu")


def test_load_slice_texts_fineweb_uses_validation_split(monkeypatch, tmp_path: Path) -> None:
    slice_cfg = FinewebValidationSlice(
        name="fineweb_edu_validation",
        pretrain_config=tmp_path / "pretrain.json",
        document_budget=2,
    )
    captured: dict[str, object] = {}

    def fake_load_pretrain_config(path: Path):
        captured["config_path"] = path
        return SimpleNamespace(payload={})

    def fake_document_text_stream(config, *, split: str):
        captured["split"] = split
        yield from ["doc one", "doc two", "doc three"]

    monkeypatch.setattr(baseline_models, "load_pretrain_config", fake_load_pretrain_config)
    monkeypatch.setattr(baseline_models, "document_text_stream", fake_document_text_stream)

    texts = load_slice_texts(slice_cfg)

    assert texts == ["doc one", "doc two"]
    assert captured["split"] == "validation"
    assert captured["config_path"] == slice_cfg.pretrain_config


def test_load_slice_texts_fails_on_short_stream(monkeypatch, tmp_path: Path) -> None:
    slice_cfg = FinewebValidationSlice(
        name="fineweb_edu_validation",
        pretrain_config=tmp_path / "pretrain.json",
        document_budget=5,
    )
    monkeypatch.setattr(
        baseline_models, "load_pretrain_config", lambda path: SimpleNamespace(payload={})
    )
    monkeypatch.setattr(
        baseline_models, "document_text_stream", lambda config, *, split: iter(["only one"])
    )

    with pytest.raises(ValueError, match="produced 1 documents; config requires 5"):
        load_slice_texts(slice_cfg)


def test_load_slice_texts_hf_dataset_pins_revision(monkeypatch) -> None:
    slice_cfg = HFDatasetSlice(
        name="pile_test",
        source="monology/pile-uncopyrighted",
        subset=None,
        split="test",
        revision="deadbeef",
        text_field="text",
        document_budget=2,
    )
    captured: dict[str, object] = {}

    def fake_load_dataset(source, *, name, split, revision, streaming):
        captured.update(
            source=source, name=name, split=split, revision=revision, streaming=streaming
        )
        return iter([{"text": "pile doc 1"}, {"text": "pile doc 2"}, {"text": "pile doc 3"}])

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    texts = load_slice_texts(slice_cfg)

    assert texts == ["pile doc 1", "pile doc 2"]
    assert captured == {
        "source": "monology/pile-uncopyrighted",
        "name": None,
        "split": "test",
        "revision": "deadbeef",
        "streaming": True,
    }


def test_load_slice_texts_hf_dataset_rejects_missing_field(monkeypatch) -> None:
    slice_cfg = HFDatasetSlice(
        name="pile_test",
        source="x",
        subset=None,
        split="test",
        revision="r",
        text_field="text",
        document_budget=1,
    )
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: iter([{"content": "no text field"}])),
    )

    with pytest.raises(ValueError, match="missing field 'text'"):
        load_slice_texts(slice_cfg)
