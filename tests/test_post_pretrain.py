from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone
from esme_pretrain.postrun import eval_checkpoints
from esme_pretrain.postrun.acceptance_report import (
    BaseAcceptanceReportConfig,
    build_base_acceptance_report,
)
from esme_pretrain.postrun.eval_checkpoints import (
    FixedEvalBatch,
    evaluate_checkpoint,
    hash_eval_batches,
    select_checkpoint,
    token_byte_counts,
)
from esme_pretrain.postrun.export_bundle import (
    CANONICAL_BUNDLE_FORMAT,
    ExportConfig,
    export_checkpoint,
)
from esme_pretrain.torch import torch
from esme_pretrain.training.checkpointing import save_pretrain_checkpoint


def _tiny_config() -> BackboneConfig:
    return BackboneConfig(
        name="tiny",
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        z_loss_weight=0.0,
    )


def _fixed_batches(config: BackboneConfig) -> list[FixedEvalBatch]:
    window = torch.arange(0, 36, dtype=torch.long).view(4, 9) % config.vocab_size
    return [
        FixedEvalBatch(
            input_ids=window[:, :-1].clone(),
            targets=window[:, 1:].clone(),
            target_byte_counts=torch.ones((4, 8), dtype=torch.long),
        )
    ]


def _save_checkpoint(path: Path, config: BackboneConfig, *, step: int, offset: float = 0.0) -> None:
    model = DenseBackbone(config)
    if offset:
        with torch.no_grad():
            model.token_embedding.weight.add_(offset)
    save_pretrain_checkpoint(path, model=model, config=config, step=step)


def test_two_tiny_checkpoints_evaluate_on_identical_token_batches(tmp_path: Path) -> None:
    config = _tiny_config()
    batches = _fixed_batches(config)
    first = tmp_path / "checkpoint-step1.pt"
    second = tmp_path / "checkpoint-step2.pt"
    _save_checkpoint(first, config, step=1)
    _save_checkpoint(second, config, step=2, offset=0.01)

    before_hash = hash_eval_batches(batches)
    first_result = evaluate_checkpoint(first, batches, device="cpu", expected_config=config)
    second_result = evaluate_checkpoint(second, batches, device="cpu", expected_config=config)
    after_hash = hash_eval_batches(batches)

    assert before_hash == after_hash
    assert first_result.eval_tokens == second_result.eval_tokens == 32
    assert first_result.eval_bytes == second_result.eval_bytes == 32
    assert first_result.eval_batches == second_result.eval_batches == 1
    assert first_result.checkpoint_step == 1
    assert second_result.checkpoint_step == 2
    assert first_result.bits_per_byte == pytest.approx(
        first_result.ce_loss * first_result.eval_tokens / math.log(2) / first_result.eval_bytes
    )


def test_token_byte_counts_use_utf8_lengths() -> None:
    class Encoding:
        offsets = [(0, 1), (1, 2), (2, 3)]

    assert token_byte_counts("aøb", Encoding(), token_count=3) == [1, 2, 1]


def test_tokenized_validation_stream_uses_shared_corpus_stream(monkeypatch) -> None:
    config = object()
    captured: dict[str, object] = {}

    def fake_document_text_stream(stream_config, *, split: str):
        captured["config"] = stream_config
        captured["split"] = split
        yield "abc"

    class FakeTokenizer:
        def encode(self, text: str):
            assert text == "abc"
            return type("Encoding", (), {"ids": [7, 8], "offsets": [(0, 1), (1, 3)]})()

    monkeypatch.setattr(eval_checkpoints, "document_text_stream", fake_document_text_stream)

    tokens = list(eval_checkpoints.tokenized_validation_stream(config, FakeTokenizer(), eos_id=0))

    assert tokens == [(7, 1), (8, 2), (0, 0)]
    assert captured == {"config": config, "split": "validation"}


def test_select_checkpoint_recommends_final_when_within_margin() -> None:
    selection = select_checkpoint(
        [
            {
                "path": "runs/checkpoint-step21500.pt",
                "checkpoint_step": 21500,
                "ce_loss": 3.0,
            },
            {"path": "runs/checkpoint.pt", "checkpoint_step": 22000, "ce_loss": 3.019},
        ]
    )

    assert selection["recommended_checkpoint"] == "runs/checkpoint.pt"
    assert selection["within_final_margin"] is True
    assert selection["margin_ce_loss"] == pytest.approx(0.019)


def test_select_checkpoint_recommends_best_when_final_misses_margin() -> None:
    selection = select_checkpoint(
        [
            {
                "path": "runs/checkpoint-step21500.pt",
                "checkpoint_step": 21500,
                "ce_loss": 3.0,
            },
            {"path": "runs/checkpoint.pt", "checkpoint_step": 22000, "ce_loss": 3.021},
        ]
    )

    assert selection["recommended_checkpoint"] == "runs/checkpoint-step21500.pt"
    assert selection["within_final_margin"] is False


def test_acceptance_report_fails_on_missing_required_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(json.dumps({"checkpoints": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required artifacts"):
        build_base_acceptance_report(
            BaseAcceptanceReportConfig(
                run_dir=run_dir,
                eval_path=eval_path,
                output_path=tmp_path / "report.md",
            )
        )


def test_acceptance_report_succeeds_with_fixture_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_required_run_artifacts(run_dir)
    eval_path = tmp_path / "base-eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "path": str(run_dir / "checkpoint.pt"),
                        "checkpoint_step": 3,
                        "checkpoint_sha256": "abc",
                        "ce_loss": 2.5,
                        "perplexity": 12.18,
                        "eval_tokens": 32,
                        "eval_bytes": 17,
                        "bits_per_byte": 6.79,
                        "eval_batches": 1,
                        "runtime_seconds": 0.1,
                    }
                ],
                "selection": {
                    "recommended_checkpoint": str(run_dir / "checkpoint.pt"),
                    "recommended_step": 3,
                    "best_checkpoint": str(run_dir / "checkpoint.pt"),
                    "best_ce_loss": 2.5,
                    "best_bits_per_byte": 6.79,
                    "final_checkpoint": str(run_dir / "checkpoint.pt"),
                    "final_ce_loss": 2.5,
                    "final_bits_per_byte": 6.79,
                    "margin_ce_loss": 0.0,
                    "within_final_margin": True,
                    "reason": "final checkpoint is within 0.02 CE",
                },
            }
        ),
        encoding="utf-8",
    )

    payload = build_base_acceptance_report(
        BaseAcceptanceReportConfig(
            run_dir=run_dir,
            eval_path=eval_path,
            output_path=run_dir / "base-acceptance-report.md",
        )
    )

    assert payload["samples_present"] is True
    assert payload["fixed_eval"]["best_bits_per_byte"] == pytest.approx(6.79)
    assert payload["tokenizer_round_trip"]["all_passed"] is True
    assert (
        (run_dir / "base-acceptance-report.md")
        .read_text(encoding="utf-8")
        .startswith("# 214M 10B Base Acceptance Report")
    )


def test_export_round_trip_preserves_logits(tmp_path: Path, monkeypatch) -> None:
    config = _tiny_config()
    checkpoint = tmp_path / "checkpoint.pt"
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"kind":"synthetic"}', encoding="utf-8")
    model = DenseBackbone(config)
    save_pretrain_checkpoint(checkpoint, model=model, config=config, step=9)
    _install_fake_tokenizers(monkeypatch, tokenizer, vocab_size=config.vocab_size)

    input_ids = torch.arange(0, 16, dtype=torch.long).view(2, 8) % config.vocab_size
    with torch.no_grad():
        expected = model(input_ids)

    output_dir = tmp_path / "export"
    manifest = export_checkpoint(
        ExportConfig(
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            output_dir=output_dir,
            export_format="llm-infer",
        )
    )
    weights = torch.load(output_dir / "weights.pt", map_location="cpu", weights_only=False)
    exported_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    reloaded_config = BackboneConfig.from_dict(exported_config)
    reloaded = DenseBackbone(reloaded_config)
    reloaded.load_state_dict(weights["state_dict"])
    reloaded.eval()

    with torch.no_grad():
        actual = reloaded(input_ids)

    assert manifest["format"] == CANONICAL_BUNDLE_FORMAT
    assert manifest["target"] == "llm-infer"
    assert manifest["weights_format"] == CANONICAL_BUNDLE_FORMAT
    assert manifest["tokenizer"] == {"path": "tokenizer.json", "format": "tokenizers-json"}
    assert manifest["model_config"] == config.to_dict()
    assert exported_config == manifest["model_config"]
    assert weights["model_config"] == exported_config
    assert "embedding_dim" in exported_config
    assert "feedforward_dim" in exported_config
    assert "layers" in exported_config
    assert "heads" in exported_config
    assert "tie_embeddings" in exported_config
    assert "hidden_size" not in exported_config
    assert manifest["llm_infer_config"]["hidden_size"] == config.embedding_dim
    assert manifest["llm_infer_config"]["intermediate_size"] == config.feedforward_dim
    assert manifest["llm_infer_config"] == weights["llm_infer_config"]
    assert weights["format"] == CANONICAL_BUNDLE_FORMAT
    assert weights["key_format"] == CANONICAL_BUNDLE_FORMAT
    assert weights["metadata"]["key_format"] == CANONICAL_BUNDLE_FORMAT
    assert weights["metadata"]["target"] == "llm-infer"
    assert weights["state_dict_key"] == "dense_backbone"
    assert "native esme-pretrain DenseBackbone config" in readme
    assert manifest["files"]["weights"]["sha256"]
    assert torch.allclose(expected, actual, atol=1e-6)


def test_export_rejects_tokenizer_vocab_drift(tmp_path: Path, monkeypatch) -> None:
    config = _tiny_config()
    checkpoint = tmp_path / "checkpoint.pt"
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text('{"kind":"synthetic"}', encoding="utf-8")
    save_pretrain_checkpoint(checkpoint, model=DenseBackbone(config), config=config, step=9)
    _install_fake_tokenizers(monkeypatch, tokenizer, vocab_size=config.vocab_size + 1)

    with pytest.raises(ValueError, match="tokenizer vocab size does not match"):
        export_checkpoint(
            ExportConfig(
                checkpoint_path=checkpoint,
                tokenizer_path=tokenizer,
                output_dir=tmp_path / "export",
                export_format="llm-infer",
            )
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


def _write_required_run_artifacts(run_dir: Path) -> None:
    (run_dir / "config.json").write_text(
        json.dumps({"run_id": "fixture", "model": {"name": "214M"}}),
        encoding="utf-8",
    )
    (run_dir / "tokenizer.json").write_text('{"version":"1.0","model":{}}', encoding="utf-8")
    (run_dir / "data-report.json").write_text(
        json.dumps({"budgeted_tokens": 96, "truncated_tokens": 96}),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text('{"step": 3, "tokens": 96}\n', encoding="utf-8")
    (run_dir / "throughput.csv").write_text(
        "step,tokens,tokens_per_second,mfu,step_time_ms\n3,96,123.4,0.01,56.7\n",
        encoding="utf-8",
    )
    (run_dir / "checkpoint.pt").write_bytes(b"fixture-checkpoint")
    (run_dir / "samples.md").write_text("## step 3\n\nprompt: 'abc'\n", encoding="utf-8")
    (run_dir / "environment.txt").write_text("python=3.11\n", encoding="utf-8")
    (run_dir / "scaleup-pretrain-report.md").write_text("# fixture\n", encoding="utf-8")
    (run_dir / "tokenizer-report.json").write_text(
        json.dumps(
            {
                "round_trips": [{"text": "abc", "round_trip": True}],
                "coverage": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "cost.json").write_text(
        json.dumps({"paid_compute": False, "estimated_cost_usd": 0.0}),
        encoding="utf-8",
    )
    (run_dir / "run-summary.json").write_text(
        json.dumps({"status": "pretrain_complete", "final_step": 3, "final_tokens": 96}),
        encoding="utf-8",
    )
