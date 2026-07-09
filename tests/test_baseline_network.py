"""Opt-in networked checks: real tokenizers and the real Cerebras gate.

Run with `uv run --extra baselines pytest -m network`. These tests download
baseline models and task data; they are excluded from the default suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.network

CEREBRAS = ("cerebras/Cerebras-GPT-256M", "abcf5974334502fa0d98811ea6d0cc81d4af9ecc")
PYTHIA = ("EleutherAI/pythia-160m", "50f5173d932e8e61f858120bcb800b97af589f46")

SAMPLE_TEXTS = [
    "Hello world, this is a plain ASCII document.\n",
    "Multibyte content: åäö øñ 中文 — with punctuation…\n",
    "  leading spaces, trailing spaces  ",
    "tabs\tand\nnewlines\r\nmixed",
]


@pytest.mark.parametrize("repo_revision", [CEREBRAS, PYTHIA], ids=["cerebras", "pythia"])
def test_real_tokenizer_byte_partition_is_exact(repo_revision: tuple[str, str]) -> None:
    transformers = pytest.importorskip("transformers")
    from esme_pretrain.baselines.models import partitioned_byte_counts

    repo, revision = repo_revision
    tokenizer = transformers.AutoTokenizer.from_pretrained(repo, revision=revision)
    backend = tokenizer.backend_tokenizer

    for text in SAMPLE_TEXTS:
        encoding = backend.encode(text)
        counts = partitioned_byte_counts(text, encoding.offsets)
        assert sum(counts) == len(text.encode("utf-8")), text


def test_cerebras_gate_reproduces_published_numbers(tmp_path: Path) -> None:
    """The pre-registered acceptance gate: measured 0-shot within ±0.01 of published."""
    pytest.importorskip("lm_eval")
    from esme_pretrain.baselines.config import load_baseline_eval_config
    from esme_pretrain.baselines.run import run_gate

    config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline_eval.json"
    config = load_baseline_eval_config(config_path)

    payload = run_gate(config, output_path=tmp_path / "gate.json")

    failed = [
        f"{task}: measured {entry['measured']:.4f} vs published {entry['published']:.4f}"
        for task, entry in payload["gate"]["per_task"].items()
        if not entry["within_tolerance"]
    ]
    assert payload["passed"], f"gate failed: {failed}"
