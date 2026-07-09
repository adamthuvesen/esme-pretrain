from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from esme_pretrain.baselines.config import GateConfig, load_baseline_eval_config
from esme_pretrain.baselines.harness import (
    evaluate_gate,
    run_downstream,
    score_loglikelihood,
    score_loglikelihood_rolling,
)
from esme_pretrain.baselines.models import EncodedText, partitioned_byte_counts
from esme_pretrain.baselines.run import run_baseline_eval, run_gate
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
    def __init__(self) -> None:
        self.name = "char"
        config = _small_config()
        self.context_length = config.context_length
        self.eos_id = 2
        self.provenance = {"kind": "fixture"}
        torch.manual_seed(11)
        self._module = DenseBackbone(config).eval()
        self._vocab_size = config.vocab_size

    def encode(self, text: str) -> EncodedText:
        ids = [(ord(char) % (self._vocab_size - 3)) + 3 for char in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        return EncodedText(ids=ids, byte_counts=partitioned_byte_counts(text, offsets))

    def module(self) -> torch.nn.Module:
        return self._module


def test_score_loglikelihood_matches_direct_forward() -> None:
    model = CharEvalModel()
    context, continuation = "ab", "cd"

    logprob, is_greedy = score_loglikelihood(model, context, continuation)

    full = model.encode(context).ids + model.encode(continuation).ids
    input_ids = torch.tensor([full[:-1]], dtype=torch.long)
    with torch.no_grad():
        log_probs = torch.log_softmax(model.module()(input_ids)[0].float(), dim=-1)
    expected = 0.0
    expected_greedy = True
    for position, token_id in zip(range(1, 3), full[2:], strict=True):
        expected += float(log_probs[position, token_id])
        expected_greedy = expected_greedy and int(log_probs[position].argmax()) == token_id

    assert logprob == pytest.approx(expected)
    assert is_greedy == expected_greedy


def test_score_loglikelihood_empty_context_uses_eos() -> None:
    model = CharEvalModel()

    logprob, _ = score_loglikelihood(model, "", "a")

    token_id = model.encode("a").ids[0]
    input_ids = torch.tensor([[model.eos_id]], dtype=torch.long)
    with torch.no_grad():
        log_probs = torch.log_softmax(model.module()(input_ids)[0].float(), dim=-1)

    assert logprob == pytest.approx(float(log_probs[0, token_id]))


def test_score_loglikelihood_rejects_oversized_continuation() -> None:
    model = CharEvalModel()

    with pytest.raises(ValueError, match="longer than the model context"):
        score_loglikelihood(model, "a", "abcde")


def test_score_loglikelihood_rolling_covers_all_tokens() -> None:
    model = CharEvalModel()
    text = "abcdefgh"

    total = score_loglikelihood_rolling(model, text)

    ids = model.encode(text).ids
    tokens = [model.eos_id, *ids]
    expected = 0.0
    window = model.context_length
    with torch.no_grad():
        for start in range(0, len(tokens) - 1, window):
            chunk = tokens[start : start + window + 1]
            input_ids = torch.tensor([chunk[:-1]], dtype=torch.long)
            log_probs = torch.log_softmax(model.module()(input_ids)[0].float(), dim=-1)
            for position, token_id in enumerate(chunk[1:]):
                expected += float(log_probs[position, token_id])

    assert total == pytest.approx(expected)


def _gate_config(published: dict[str, float], average: float) -> GateConfig:
    return GateConfig(
        model="cerebras",
        published=published,
        published_average=average,
        per_task_tolerance=0.01,
    )


def test_evaluate_gate_passes_within_tolerance() -> None:
    gate = _gate_config({"piqa": 0.6, "arc_easy": 0.4}, 0.5)
    measured = {"piqa": {"acc": 0.605}, "arc_easy": {"acc": 0.395}}

    result = evaluate_gate(measured, gate)

    assert result["passed"] is True
    assert result["per_task"]["piqa"]["delta"] == pytest.approx(0.005)
    assert result["average"]["measured"] == pytest.approx(0.5)


def test_evaluate_gate_fails_outside_tolerance() -> None:
    gate = _gate_config({"piqa": 0.6, "arc_easy": 0.4}, 0.5)
    measured = {"piqa": {"acc": 0.62}, "arc_easy": {"acc": 0.4}}

    result = evaluate_gate(measured, gate)

    assert result["passed"] is False
    assert result["per_task"]["piqa"]["within_tolerance"] is False


def test_evaluate_gate_requires_all_tasks() -> None:
    gate = _gate_config({"piqa": 0.6, "arc_easy": 0.4}, 0.5)

    with pytest.raises(ValueError, match="missing a measured result for task 'arc_easy'"):
        evaluate_gate({"piqa": {"acc": 0.6}}, gate)


def _install_fake_lm_eval(monkeypatch, *, accuracies: dict[str, float], version: str = "0.4.12"):
    calls: list[dict] = []

    class FakeLM:
        pass

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {
            "results": {
                task: {"acc,none": acc, "acc_stderr,none": 0.001}
                for task, acc in accuracies.items()
            }
        }

    fake = SimpleNamespace(
        __version__=version,
        api=SimpleNamespace(model=SimpleNamespace(LM=FakeLM)),
        simple_evaluate=fake_simple_evaluate,
    )
    monkeypatch.setitem(sys.modules, "lm_eval", fake)
    return calls


def _write_test_config(tmp_path: Path, *, bundle_dir: Path) -> Path:
    payload = {
        "schema_version": 1,
        "device": "cpu",
        "dtype": "float32",
        "max_context": 4,
        "bpb_batch_size": 1,
        "text_slices": {
            "fineweb_edu_validation": {
                "kind": "fineweb_edu_validation",
                "pretrain_config": "configs/pretrain_214m_b200.json",
                "document_budget": 2,
            }
        },
        "models": {
            "esme": {"kind": "bundle", "path": str(bundle_dir)},
            "cerebras": {"kind": "hf", "repo": "cerebras/Cerebras-GPT-256M", "revision": "abc"},
        },
        "downstream": {
            "harness": "lm-eval",
            "version": "0.4.12",
            "tasks": ["piqa", "arc_easy"],
            "num_fewshot": 0,
            "batch_size": 1,
        },
        "gate": {
            "model": "cerebras",
            "published": {"piqa": 0.6, "arc_easy": 0.4},
            "published_average": 0.5,
            "per_task_tolerance": 0.01,
        },
        "tolerances": {"bpb": 1e-6, "accuracy": 0.0},
    }
    path = tmp_path / "baseline_eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_downstream_rejects_version_mismatch(monkeypatch, tmp_path: Path) -> None:
    _install_fake_lm_eval(monkeypatch, accuracies={"piqa": 0.6, "arc_easy": 0.4}, version="0.4.99")
    config_path = _write_test_config(tmp_path, bundle_dir=tmp_path / "missing")
    config = load_baseline_eval_config(config_path)

    with pytest.raises(ValueError, match="does not match pinned"):
        run_downstream(config.models["cerebras"], config)


def test_run_gate_writes_passing_result(monkeypatch, tmp_path: Path) -> None:
    calls = _install_fake_lm_eval(monkeypatch, accuracies={"piqa": 0.6, "arc_easy": 0.4})
    config_path = _write_test_config(tmp_path, bundle_dir=tmp_path / "missing")
    config = load_baseline_eval_config(config_path)
    output = tmp_path / "gate.json"

    payload = run_gate(config, output_path=output)

    assert payload["passed"] is True
    assert payload["config_sha256"] == config.config_sha256
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True
    assert calls[0]["model"] == "hf"
    assert "revision=abc" in calls[0]["model_args"]
    assert calls[0]["tasks"] == ["piqa", "arc_easy"]


def test_run_baseline_eval_requires_gate_for_bundle(monkeypatch, tmp_path: Path) -> None:
    _install_fake_lm_eval(monkeypatch, accuracies={"piqa": 0.6, "arc_easy": 0.4})
    config_path = _write_test_config(tmp_path, bundle_dir=tmp_path / "missing")
    config = load_baseline_eval_config(config_path)

    with pytest.raises(ValueError, match="requires --gate"):
        run_baseline_eval(config, model_key="esme", output_path=tmp_path / "esme.json")


def test_run_baseline_eval_rejects_failed_gate(monkeypatch, tmp_path: Path) -> None:
    _install_fake_lm_eval(monkeypatch, accuracies={"piqa": 0.6, "arc_easy": 0.4})
    config_path = _write_test_config(tmp_path, bundle_dir=tmp_path / "missing")
    config = load_baseline_eval_config(config_path)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(
        json.dumps({"passed": False, "config_sha256": config.config_sha256}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="gate did not pass"):
        run_baseline_eval(
            config, model_key="esme", output_path=tmp_path / "esme.json", gate_path=gate_path
        )


def test_run_baseline_eval_rejects_stale_gate_config(monkeypatch, tmp_path: Path) -> None:
    _install_fake_lm_eval(monkeypatch, accuracies={"piqa": 0.6, "arc_easy": 0.4})
    config_path = _write_test_config(tmp_path, bundle_dir=tmp_path / "missing")
    config = load_baseline_eval_config(config_path)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps({"passed": True, "config_sha256": "0" * 64}), encoding="utf-8")

    with pytest.raises(ValueError, match="different baseline eval config"):
        run_baseline_eval(
            config, model_key="esme", output_path=tmp_path / "esme.json", gate_path=gate_path
        )


def test_run_baseline_eval_scores_bundle_end_to_end(monkeypatch, tmp_path: Path) -> None:
    from esme_pretrain.baselines import run as baseline_run
    from esme_pretrain.postrun.export_bundle import ExportConfig, export_checkpoint
    from esme_pretrain.training.checkpointing import save_pretrain_checkpoint

    config_model = _small_config()
    checkpoint = tmp_path / "checkpoint.pt"
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text('{"kind":"synthetic"}', encoding="utf-8")
    save_pretrain_checkpoint(
        checkpoint, model=DenseBackbone(config_model), config=config_model, step=3
    )

    class FakeEncoding:
        def __init__(self, text: str) -> None:
            self.ids = [(ord(char) % 28) + 4 for char in text]
            self.offsets = [(i, i + 1) for i in range(len(text))]

    class FakeTokenizer:
        @classmethod
        def from_file(cls, path: str):
            return cls()

        def get_vocab_size(self) -> int:
            return config_model.vocab_size

        def token_to_id(self, token: str) -> int | None:
            return {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}.get(token)

        def encode(self, text: str) -> FakeEncoding:
            return FakeEncoding(text)

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizer))
    bundle_dir = tmp_path / "export"
    export_checkpoint(
        ExportConfig(
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer_path,
            output_dir=bundle_dir,
            export_format="llm-infer",
        )
    )

    _install_fake_lm_eval(monkeypatch, accuracies={"piqa": 0.6, "arc_easy": 0.4})
    monkeypatch.setattr(
        baseline_run, "load_slice_texts", lambda slice_cfg: ["abcdefgh", "ijklmnop"]
    )
    config_path = _write_test_config(tmp_path, bundle_dir=bundle_dir)
    config = load_baseline_eval_config(config_path)
    gate_path = tmp_path / "gate.json"
    run_gate(config, output_path=gate_path)

    payload = run_baseline_eval(
        config, model_key="esme", output_path=tmp_path / "esme.json", gate_path=gate_path
    )

    assert payload["model"]["kind"] == "bundle"
    assert payload["model"]["weights_sha256"]
    assert payload["gate"] == {"required": True, "gate_path": str(gate_path), "passed": True}
    bpb_entry = payload["bpb"]["fineweb_edu_validation"]
    assert bpb_entry["document_count"] == 2
    assert bpb_entry["bits_per_byte"] > 0
    assert bpb_entry["raw_bytes"] == 16
    assert payload["downstream"]["average"] == pytest.approx(0.5)
    assert payload["context_length"] == 4
    assert json.loads((tmp_path / "esme.json").read_text(encoding="utf-8")) == payload


def test_run_baseline_eval_rejects_unknown_model(monkeypatch, tmp_path: Path) -> None:
    config_path = _write_test_config(tmp_path, bundle_dir=tmp_path / "missing")
    config = load_baseline_eval_config(config_path)

    with pytest.raises(ValueError, match="unknown model 'gpt5'"):
        run_baseline_eval(config, model_key="gpt5", output_path=tmp_path / "x.json")
