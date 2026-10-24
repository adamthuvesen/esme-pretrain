from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from esme_pretrain.data.token_shards import manifest_path_for, write_token_shards
from esme_pretrain.modeling.backbone import BackboneConfig
from esme_pretrain.torch import torch
from esme_pretrain.training.eval_batch import hash_eval_batches
from esme_pretrain.training.metrics import RunLogger, WandbSettings
from esme_pretrain.training.pretrain import PretrainConfig, run_pretrain
from esme_pretrain.training.shard_loader import ShardBatchLoader, fixed_validation_batches

EOS_ID = 0
VOCAB_SIZE = 128


class WhitespaceIdTokenizer:
    def encode_batch(self, texts: list[str]) -> list[SimpleNamespace]:
        return [SimpleNamespace(ids=[int(word) for word in text.split()]) for text in texts]


def _documents(count: int, *, seed: int = 0) -> list[str]:
    generator = torch.Generator().manual_seed(seed)
    documents = []
    for _ in range(count):
        length = int(torch.randint(1, 12, (1,), generator=generator))
        ids = torch.randint(1, VOCAB_SIZE, (length,), generator=generator).tolist()
        documents.append(" ".join(str(token_id) for token_id in ids))
    return documents


def _flat_stream(documents: list[str]) -> list[int]:
    flat: list[int] = []
    for text in documents:
        flat.extend(int(word) for word in text.split())
        flat.append(EOS_ID)
    return flat


def _write(output_dir: Path, documents: list[str], **overrides) -> dict:
    kwargs = dict(
        eos_id=EOS_ID,
        vocab_size=VOCAB_SIZE,
        tokenizer_sha256="0" * 64,
        split="train",
        output_dir=output_dir,
        shard_tokens=53,
        provenance={"source": "fixture"},
    )
    kwargs.update(overrides)
    return write_token_shards(documents, WhitespaceIdTokenizer(), **kwargs)


def _loader(output_dir: Path, **overrides) -> ShardBatchLoader:
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        tokenizer_sha256="0" * 64,
        batch_size=2,
        context_length=4,
        device="cpu",
    )
    kwargs.update(overrides)
    return ShardBatchLoader(manifest_path_for(output_dir, "train"), **kwargs)


def _rows(loader: ShardBatchLoader) -> list[list[int]]:
    rows = []
    for batch in loader:
        full = torch.cat([batch.input_ids, batch.targets[:, -1:]], dim=1)
        assert torch.equal(batch.input_ids[:, 1:], batch.targets[:, :-1])
        rows.extend(full.tolist())
    return rows


def test_round_trip_reproduces_unshuffled_windows_per_shard(tmp_path: Path) -> None:
    documents = _documents(60)
    flat = _flat_stream(documents)
    manifest = _write(tmp_path, documents, shard_tokens=53)
    assert manifest["total_tokens"] == len(flat)
    assert manifest["documents"] == len(documents)

    window = 5
    expected = []
    for shard_start in range(0, len(flat), 53):
        shard = flat[shard_start : shard_start + 53]
        expected.extend(
            shard[offset : offset + window] for offset in range(0, len(shard) - window + 1, window)
        )
    loader = _loader(tmp_path, batch_size=1, verify_hashes=True)
    assert loader.window_count == len(expected)
    assert loader.dropped_tail_tokens == len(flat) - len(expected) * window
    assert _rows(loader) == expected
    assert loader.tokens_yielded == len(expected) * window


def test_shuffled_order_is_deterministic_and_covers_every_window_once(tmp_path: Path) -> None:
    _write(tmp_path, _documents(60))
    unshuffled = _rows(_loader(tmp_path, batch_size=1))
    first = _rows(_loader(tmp_path, batch_size=1, shuffle_seed=7))
    again = _rows(_loader(tmp_path, batch_size=1, shuffle_seed=7))
    other = _rows(_loader(tmp_path, batch_size=1, shuffle_seed=8))

    assert first == again
    assert first != unshuffled
    assert first != other
    assert sorted(first) == sorted(unshuffled)


def test_skip_tokens_resumes_the_same_order_without_repeats_or_gaps(tmp_path: Path) -> None:
    _write(tmp_path, _documents(60))
    full = _rows(_loader(tmp_path, batch_size=1, shuffle_seed=3))
    skipped_windows = 6

    resumed_loader = _loader(tmp_path, batch_size=3, shuffle_seed=3)
    resumed_loader.skip_tokens = skipped_windows * resumed_loader.window
    resumed = _rows(resumed_loader)

    remaining = len(full) - skipped_windows
    assert resumed == full[skipped_windows : skipped_windows + remaining - remaining % 3]


def test_skip_tokens_must_be_a_whole_number_of_windows(tmp_path: Path) -> None:
    _write(tmp_path, _documents(60))
    loader = _loader(tmp_path)
    with pytest.raises(ValueError, match="whole number of 5-token windows"):
        loader.skip_tokens = 7


def test_truncated_shard_file_is_rejected(tmp_path: Path) -> None:
    manifest = _write(tmp_path, _documents(60))
    shard_path = tmp_path / manifest["shards"][0]["path"]
    shard_path.write_bytes(shard_path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="has 104 bytes; manifest records 53 tokens"):
        _loader(tmp_path)


def test_out_of_range_token_id_is_rejected_at_write(tmp_path: Path) -> None:
    documents = ["5 6 7", f"8 {VOCAB_SIZE} 9"]
    with pytest.raises(ValueError, match=f"token id {VOCAB_SIZE} at stream position 5"):
        _write(tmp_path, documents)
    assert not manifest_path_for(tmp_path, "train").exists()


def test_max_tokens_truncates_at_exactly_that_many_tokens(tmp_path: Path) -> None:
    documents = ["1 2 3", "4 5 6", "7 8 9"]
    manifest = _write(tmp_path, documents, max_tokens=6)

    assert manifest["total_tokens"] == 6
    assert manifest["documents"] == 2
    shard = tmp_path / manifest["shards"][0]["path"]
    assert torch.frombuffer(bytearray(shard.read_bytes()), dtype=torch.int16).tolist() == [
        1,
        2,
        3,
        EOS_ID,
        4,
        5,
    ]
    on_disk = json.loads(manifest_path_for(tmp_path, "train").read_text())
    assert on_disk["max_tokens"] == 6


class RecordingShardLoader(ShardBatchLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seen_input_ids: list[list[int]] = []

    def __iter__(self):
        for batch in super().__iter__():
            self.seen_input_ids.extend(batch.input_ids.tolist())
            yield batch


def test_run_pretrain_resume_continues_the_shard_window_order(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shards"
    _write(shard_dir, _documents(400, seed=1), shard_tokens=500)
    model = BackboneConfig(
        name="small-test",
        vocab_size=VOCAB_SIZE,
        context_length=16,
        embedding_dim=64,
        layers=2,
        heads=4,
        feedforward_dim=128,
    )

    def config(output_dir: Path, **overrides) -> PretrainConfig:
        return PretrainConfig(
            model=model,
            max_steps=6,
            micro_batch_size=4,
            grad_accum_steps=2,
            learning_rate=1e-3,
            warmup_steps=2,
            device="cpu",
            use_compile=False,
            use_fused_optimizer=False,
            log_interval=1,
            checkpoint_interval=3,
            output_dir=output_dir,
            **overrides,
        )

    def loader() -> RecordingShardLoader:
        return RecordingShardLoader(
            manifest_path_for(shard_dir, "train"),
            vocab_size=VOCAB_SIZE,
            tokenizer_sha256="0" * 64,
            batch_size=4,
            context_length=16,
            device="cpu",
            shuffle_seed=11,
        )

    run_dir = tmp_path / "run"
    first_loader = loader()
    assert first_loader.window_count >= 6 * 8
    logger = RunLogger(run_dir, WandbSettings(enabled=False))
    first = run_pretrain(config(run_dir), first_loader, logger=logger)
    logger.finish()
    assert first.steps_completed == 6

    resumed_loader = loader()
    logger = RunLogger(run_dir, WandbSettings(enabled=False))
    resumed = run_pretrain(
        config(run_dir, resume_from=run_dir / "checkpoint-step3.pt"), resumed_loader, logger=logger
    )
    logger.finish()

    assert resumed.start_step == 3
    assert resumed.steps_completed == 3
    assert resumed_loader.skip_tokens == 3 * 8 * 17
    windows_per_step = 8
    first_tail = first_loader.seen_input_ids[3 * windows_per_step : 6 * windows_per_step]
    assert resumed_loader.seen_input_ids[: 3 * windows_per_step] == first_tail


def test_loader_refuses_shards_from_another_vocabulary_or_tokenizer(tmp_path: Path) -> None:
    _write(tmp_path, _documents(40, seed=2))

    with pytest.raises(ValueError, match="the model uses"):
        _loader(tmp_path, vocab_size=VOCAB_SIZE + 1)
    with pytest.raises(ValueError, match="were written with tokenizer"):
        _loader(tmp_path, tokenizer_sha256="1" * 64)


def test_fixed_validation_batches_match_sequential_packing_and_check_the_hash(
    tmp_path: Path,
) -> None:
    _write(tmp_path, _documents(200, seed=3), shard_tokens=10_000)
    reference = _rows(_loader(tmp_path, batch_size=2))[:6]
    kwargs = dict(
        vocab_size=VOCAB_SIZE,
        tokenizer_sha256="0" * 64,
        context_length=4,
        batch_size=2,
        eval_token_budget=24,
    )

    batches = fixed_validation_batches(
        manifest_path_for(tmp_path, "train"), expected_token_batch_sha256=None, **kwargs
    )
    windows = [
        row
        for batch in batches
        for row in torch.cat([batch.input_ids, batch.targets[:, -1:]], dim=1).tolist()
    ]
    assert windows == reference
    digest = hash_eval_batches(batches)
    again = fixed_validation_batches(
        manifest_path_for(tmp_path, "train"), expected_token_batch_sha256=digest, **kwargs
    )
    assert len(again) == 3
    with pytest.raises(ValueError, match="validation batches hash to"):
        fixed_validation_batches(
            manifest_path_for(tmp_path, "train"), expected_token_batch_sha256="0" * 64, **kwargs
        )
