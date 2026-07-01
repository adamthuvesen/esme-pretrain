from pathlib import Path

import pytest

from esme_pretrain.tokenization.lab import TokenizerLabConfig, parse_vocab_sizes, run_tokenizer_lab


def test_parse_vocab_sizes_rejects_empty_parts() -> None:
    with pytest.raises(ValueError, match="comma-separated"):
        parse_vocab_sizes("48,")


def test_tokenizer_lab_reports_compression_metrics_and_loss_shape(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("tokenizer lab text repeats tokenizer lab text\n" * 16, encoding="utf-8")

    result = run_tokenizer_lab(
        TokenizerLabConfig(
            input_path=corpus,
            context_length=8,
            token_budget=128,
            vocab_sizes=(32,),
            steps=2,
        )
    )

    assert len(result.comparisons) == 2
    baseline, learned = result.comparisons
    assert baseline.tokenizer == "char"
    assert baseline.compression_ratio == 1
    assert learned.tokenizer == "pair_merge"
    assert learned.vocab_size == 32
    assert learned.tokens < baseline.tokens
    assert learned.chars_per_token > baseline.chars_per_token
    assert learned.compression_ratio > baseline.compression_ratio
    assert learned.unknown_tokens == 0
    assert "unseen characters raise ValueError" in learned.coverage
    assert learned.train_rows > 0
    assert learned.validation_rows > 0
    assert learned.train_loss_initial > 0
    assert learned.train_loss_final > 0
    assert learned.validation_loss_initial > 0
    assert learned.validation_loss_final > 0
    assert learned.round_trip_examples[0]["input"] == learned.round_trip_examples[0]["decoded"]
