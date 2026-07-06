from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import esme_pretrain.training.pretrain as pretrain_module
from esme_pretrain.modeling.backbone import BackboneConfig
from esme_pretrain.torch import torch
from esme_pretrain.training.checkpointing import load_pretrain_checkpoint
from esme_pretrain.training.data_stream import Batch, StreamingBatchLoader, synthetic_token_stream
from esme_pretrain.training.metrics import RunLogger, WandbSettings
from esme_pretrain.training.pretrain import PretrainConfig, run_pretrain
from esme_pretrain.training.runtime import cosine_lr, wsd_lr


def _model() -> BackboneConfig:
    return BackboneConfig(
        name="small-test",
        vocab_size=128,
        context_length=16,
        embedding_dim=64,
        layers=2,
        heads=4,
        feedforward_dim=128,
    )


def _loader(seed: int) -> StreamingBatchLoader:
    return StreamingBatchLoader(
        synthetic_token_stream(128, seed=seed),
        batch_size=4,
        context_length=16,
        device="cpu",
    )


def _config(output_dir: Path, **overrides) -> PretrainConfig:
    base = dict(
        model=_model(),
        max_steps=8,
        micro_batch_size=4,
        grad_accum_steps=2,
        learning_rate=1e-3,
        warmup_steps=2,
        device="cpu",
        use_compile=False,
        use_fused_optimizer=False,
        log_interval=2,
        output_dir=output_dir,
    )
    base.update(overrides)
    return PretrainConfig(**base)


def test_cosine_lr_warmup_then_decay() -> None:
    kwargs = dict(warmup_steps=3, max_steps=10, max_lr=1.0, min_lr=0.1)
    assert cosine_lr(0, **kwargs) < cosine_lr(1, **kwargs)  # warmup ramps up
    assert cosine_lr(2, **kwargs) == 1.0  # peak at end of warmup
    assert cosine_lr(2, **kwargs) > cosine_lr(6, **kwargs) > cosine_lr(9, **kwargs)  # decays
    assert math.isclose(cosine_lr(10, **kwargs), 0.1)  # floors at min_lr


def test_wsd_lr_warmup_stable_then_decays() -> None:
    kwargs = dict(warmup_steps=3, max_steps=10, max_lr=1.0, min_lr=0.1, decay_fraction=0.2)
    # decay_fraction 0.2 of 10 steps -> decay over the last 2 steps (decay_start=8).
    assert wsd_lr(0, **kwargs) < wsd_lr(1, **kwargs)  # warmup ramps up
    assert wsd_lr(2, **kwargs) == 1.0  # peak reached at end of warmup
    assert wsd_lr(3, **kwargs) == 1.0  # stable phase holds peak LR
    assert wsd_lr(7, **kwargs) == 1.0  # still stable just before decay starts
    assert wsd_lr(8, **kwargs) == 1.0  # decay begins (progress 0 -> still peak)
    assert wsd_lr(8, **kwargs) > wsd_lr(9, **kwargs)  # decays through the tail
    assert math.isclose(wsd_lr(10, **kwargs), 0.1)  # floors at min_lr at the end


def test_wsd_lr_with_zero_decay_holds_peak_until_the_end() -> None:
    kwargs = dict(warmup_steps=2, max_steps=10, max_lr=1.0, min_lr=0.1, decay_fraction=0.0)
    # decay_fraction 0.0 -> round(0)=0 -> max(1, 0)=1 step of decay at the very end.
    assert wsd_lr(5, **kwargs) == 1.0  # stable all the way through
    assert wsd_lr(9, **kwargs) == 1.0  # last in-range step still at peak
    assert math.isclose(wsd_lr(10, **kwargs), 0.1)  # floors at min_lr


def test_loop_runs_and_writes_durable_metrics(tmp_path: Path) -> None:
    config = _config(
        tmp_path, eval_interval=4, eval_batches=2, sample_interval=4, sample_new_tokens=3
    )
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    result = run_pretrain(config, _loader(1), eval_loader=_loader(2), logger=logger)
    logger.finish()

    assert result.steps_completed == 8
    assert math.isfinite(result.train_loss_first)
    assert math.isfinite(result.train_loss_last)
    assert result.val_loss_last is not None
    assert result.steady_tokens_per_second is not None and result.steady_tokens_per_second > 0
    assert result.wandb_status == "disabled"

    # Durable local record exists regardless of W&B.
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "throughput.csv").read_text().startswith("step,tokens,tokens_per_second")
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert lines
    record = json.loads(lines[0])
    assert "loss" in record and "lr" in record and "tokens_per_second" in record

    reloaded = load_pretrain_checkpoint(tmp_path / "checkpoint.pt")
    assert reloaded.step == 8


def test_loop_stops_cleanly_when_token_stream_exhausts(tmp_path: Path) -> None:
    # One optimizer step consumes 4 * 2 * (16 + 1) = 136 stream tokens. The finite
    # stream has exactly one step worth of tokens, so step 1 must stop cleanly
    # instead of throwing StopIteration after a partial second accumulation.
    config = _config(tmp_path, max_steps=5, grad_accum_steps=2)
    finite_loader = StreamingBatchLoader(
        synthetic_token_stream(128, seed=1, total_tokens=config.stream_tokens_per_step),
        batch_size=4,
        context_length=16,
        device="cpu",
    )
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    result = run_pretrain(config, finite_loader, logger=logger)
    logger.finish()

    assert result.steps_completed == 1
    assert result.start_step == 0
    assert any("data exhausted at step 1" in note for note in result.notes)
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "run-summary.json").exists()
    assert load_pretrain_checkpoint(tmp_path / "checkpoint.pt").step == 1


def test_loop_accepts_plain_batch_iterables(tmp_path: Path) -> None:
    config = _config(tmp_path, max_steps=1, grad_accum_steps=1, warmup_steps=0)
    batch = Batch(
        input_ids=torch.randint(0, 128, (4, 16), dtype=torch.long),
        targets=torch.randint(0, 128, (4, 16), dtype=torch.long),
    )
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    result = run_pretrain(config, [batch], logger=logger)
    logger.finish()

    assert result.steps_completed == 1
    assert load_pretrain_checkpoint(tmp_path / "checkpoint.pt").step == 1


def test_resume_continues_from_saved_step(tmp_path: Path) -> None:
    first = _config(tmp_path, max_steps=4)
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    run_pretrain(first, _loader(1), logger=logger)
    logger.finish()

    resumed = _config(tmp_path, max_steps=7, resume_from=tmp_path / "checkpoint.pt")
    logger2 = RunLogger(tmp_path, WandbSettings(enabled=False))
    result = run_pretrain(resumed, _loader(1), logger=logger2)
    logger2.finish()

    assert result.start_step == 4
    assert result.steps_completed == 3
    assert load_pretrain_checkpoint(tmp_path / "checkpoint.pt").step == 7


def test_resume_rejects_shape_compatible_config_drift(tmp_path: Path) -> None:
    first = _config(tmp_path, max_steps=1, warmup_steps=0)
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    run_pretrain(first, _loader(1), logger=logger)
    logger.finish()

    drifted = _config(
        tmp_path,
        max_steps=2,
        resume_from=tmp_path / "checkpoint.pt",
        model=replace(_model(), z_loss_weight=0.0),
    )
    with pytest.raises(ValueError, match="resume checkpoint config does not match"):
        run_pretrain(drifted, _loader(1))


def test_non_finite_loss_aborts_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_loss(logits, _targets, **_kwargs):
        loss = logits.sum() * float("nan")
        return loss, {"ce_loss": float("nan"), "total_loss": float("nan")}

    monkeypatch.setattr(pretrain_module, "language_model_loss", fake_loss)
    config = _config(tmp_path, max_steps=1, grad_accum_steps=1, warmup_steps=0)
    batch = Batch(
        input_ids=torch.randint(0, 128, (4, 16), dtype=torch.long),
        targets=torch.randint(0, 128, (4, 16), dtype=torch.long),
    )
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    try:
        with pytest.raises(RuntimeError, match="non-finite train loss"):
            run_pretrain(config, [batch], logger=logger)
    finally:
        logger.finish()
    assert not (tmp_path / "checkpoint.pt").exists()


def test_periodic_checkpoint_cadence_uses_completed_steps(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        max_steps=4,
        checkpoint_interval=2,
        log_interval=1,
        eval_interval=0,
        sample_interval=0,
    )
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    run_pretrain(config, _loader(1), logger=logger)
    logger.finish()

    assert load_pretrain_checkpoint(tmp_path / "checkpoint-step2.pt").step == 2
    assert load_pretrain_checkpoint(tmp_path / "checkpoint-step4.pt").step == 4
    assert not (tmp_path / "checkpoint-step3.pt").exists()


def test_checkpoint_load_rejects_missing_model_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format_version": 2,
            "config": _model().to_dict(),
            "optimizer_state": None,
            "step": 1,
            "metrics": {},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="missing keys.*model_state"):
        load_pretrain_checkpoint(checkpoint)


class _RecordingLoader:
    """Wraps a StreamingBatchLoader and records the exact batches it yields.

    ``skip_tokens`` is forwarded so run_pretrain's resume fast-forward works; the
    recorded id sequences let a test assert a resumed run continues the stream
    instead of re-reading the corpus head.
    """

    def __init__(self, seed: int, *, batch_size: int = 4, context_length: int = 16) -> None:
        self._seed = seed
        self._batch_size = batch_size
        self._context_length = context_length
        self.skip_tokens = 0
        self.records: list[list[int]] = []

    def __iter__(self):
        inner = StreamingBatchLoader(
            synthetic_token_stream(128, seed=self._seed),
            batch_size=self._batch_size,
            context_length=self._context_length,
            device="cpu",
            skip_tokens=self.skip_tokens,
        )
        for batch in inner:
            self.records.append(batch.input_ids.flatten().tolist())
            yield batch


def test_resume_continues_token_stream_not_from_head(tmp_path: Path) -> None:
    # grad_accum=2, micro_batch 4, context 16 -> window 17 -> 4*2*17 = 136 stream
    # tokens/step. 2 steps consume 4 batches = 272 tokens. After resume the loop must
    # skip those 272 tokens and continue, never re-reading batch 0..3.
    first = _config(tmp_path, max_steps=2, grad_accum_steps=2)
    first_loader = _RecordingLoader(seed=7)
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    run_pretrain(first, first_loader, logger=logger)
    logger.finish()

    # The checkpoint must persist the consumed-token offset (2 steps * 136 tokens).
    saved = load_pretrain_checkpoint(tmp_path / "checkpoint.pt")
    assert saved.data_offset_tokens == 272
    assert saved.rng_state  # RNG snapshot present for a faithful resume

    resumed = _config(
        tmp_path, max_steps=4, grad_accum_steps=2, resume_from=tmp_path / "checkpoint.pt"
    )
    resume_loader = _RecordingLoader(seed=7)
    logger2 = RunLogger(tmp_path, WandbSettings(enabled=False))
    result = run_pretrain(resumed, resume_loader, logger=logger2)
    logger2.finish()

    assert result.start_step == 2
    # The loader was fast-forwarded past the 272 consumed tokens.
    assert resume_loader.skip_tokens == 272

    # Independent reference: the first 8 batches of a from-scratch stream (same seed).
    reference = StreamingBatchLoader(
        synthetic_token_stream(128, seed=7), batch_size=4, context_length=16, device="cpu"
    )
    ref_seq: list[list[int]] = []
    rit = iter(reference)
    for _ in range(8):
        ref_seq.append(next(rit).input_ids.flatten().tolist())
    rit.close()

    # The first run consumed reference batches 0..3.
    assert first_loader.records[:4] == ref_seq[:4]
    # The resumed run continued at reference batch 4 — it did NOT restart from 0..3.
    assert resume_loader.records[0] == ref_seq[4]
    assert all(record not in ref_seq[:4] for record in resume_loader.records)


def test_resume_checkpoint_offset_keeps_prior_tokens_when_batch_shape_changes(
    tmp_path: Path,
) -> None:
    first = _config(tmp_path, max_steps=2, micro_batch_size=4, grad_accum_steps=2)
    first_loader = _RecordingLoader(seed=11, batch_size=4)
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    run_pretrain(first, first_loader, logger=logger)
    logger.finish()

    saved = load_pretrain_checkpoint(tmp_path / "checkpoint.pt")
    assert saved.data_offset_tokens == 272

    resumed = _config(
        tmp_path,
        max_steps=4,
        micro_batch_size=2,
        grad_accum_steps=1,
        resume_from=tmp_path / "checkpoint.pt",
    )
    resume_loader = _RecordingLoader(seed=11, batch_size=2)
    logger2 = RunLogger(tmp_path, WandbSettings(enabled=False))
    run_pretrain(resumed, resume_loader, logger=logger2)
    logger2.finish()

    reloaded = load_pretrain_checkpoint(tmp_path / "checkpoint.pt")
    assert resume_loader.skip_tokens == 272
    assert reloaded.data_offset_tokens == 272 + 2 * resumed.stream_tokens_per_step


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        PretrainConfig(model=_model(), max_steps=4, micro_batch_size=1, warmup_steps=10)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        PretrainConfig(model=_model(), max_steps=4, micro_batch_size=1, min_lr_ratio=2.0)
    with pytest.raises(ValueError, match="grad_clip"):
        PretrainConfig(model=_model(), max_steps=4, micro_batch_size=1, grad_clip=0.0)


def test_float16_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path, dtype="float16", device="cpu")
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    with pytest.raises(ValueError, match="float16 is unsupported"):
        run_pretrain(config, _loader(1), logger=logger)
    logger.finish()


def test_wandb_disabled_falls_back_to_local_only(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path, WandbSettings(enabled=False))
    assert logger.wandb_status == "disabled"
    assert logger.wandb_run is None
    logger.log(0, {"loss": 1.0, "tokens_per_second": 100.0, "mfu": None, "step_time_ms": 1.0})
    logger.finish()
    assert (tmp_path / "metrics.jsonl").exists()
    assert len((tmp_path / "throughput.csv").read_text().strip().splitlines()) == 2  # header + row


def test_wandb_resume_settings_are_forwarded_and_persisted(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id="x99drn15", url="https://wandb.test/runs/x99drn15")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    logger = RunLogger(
        tmp_path,
        WandbSettings(
            project="esme-pretrain",
            run_name="pretrain_214m_b200",
            run_id="x99drn15",
            resume="allow",
            mode="disabled",
        ),
    )
    logger.finish()

    assert captured["id"] == "x99drn15"
    assert captured["resume"] == "allow"
    assert logger.wandb_run_id == "x99drn15"
    assert (tmp_path / "wandb-run-id.txt").read_text(encoding="utf-8").strip() == "x99drn15"
