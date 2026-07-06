# esme-pretrain

`esme-pretrain` trains `Esme-214M-Base`, a 214M-parameter dense decoder-only
language model, from scratch on FineWeb-Edu `sample-10BT`.

It covers the full base-model path: data preparation, tokenizer training, model
code, training checks, checkpoint evaluation, reporting, and export to
`llm-infer`.

The accepted base checkpoint comes from one run, `pretrain_214m_b200`, recorded in
[`docs/status.md`](docs/status.md) and
[`docs/run-cards/pretrain-214m-b200.md`](docs/run-cards/pretrain-214m-b200.md).
Those docs, the locked config, fixed checkpoint eval, bits-per-byte reporting,
acceptance report, export bundle, and telemetry plots are the evidence trail for
the base checkpoint.

At 214M parameters, Esme keeps the full LLM lifecycle cheap enough to run end to
end and small enough to debug. It still goes through real training, evaluation,
export, post-training, and inference.

For the model and training design, read
[`docs/architecture.md`](docs/architecture.md). Then run the local checks in
[Quickstart](#quickstart).

## Current State

`Esme-214M-Base` is the current base checkpoint: 213,960,192 parameters trained
on a nominal `10B`-token budget: `10,229,514,240` tokens over `26,015`
optimizer steps.

The base model is ready for downstream work. The next step is post-training in
`esme-posttrain`, not another pretraining launch.

## Training Telemetry

Telemetry from the `pretrain_214m_b200` run, plotted from the run's
`metrics.jsonl` and `throughput.csv`:

![Train and validation loss vs training tokens](assets/fig-pretrain-loss-vs-tokens.svg)

![Throughput and MFU stability over the run](assets/fig-pretrain-throughput-mfu.svg)

## What Is Here

- Data tools for local text files and FineWeb-Edu streaming splits.
- A byte-level BPE tokenizer contract with digit splitting.
- Production model code in
  [`DenseBackbone`](src/esme_pretrain/modeling/backbone.py) and
  `BackboneConfig`.
- Training code with checkpoint/resume checks, fixed validation batches, local
  metrics, and optional W&B logging.
- Training entrypoints for the current 214M B200 shape, launched with an
  explicit `--approved` flag.
- Post-training evaluation, bits-per-byte reporting, and `llm-infer` export.

Active code lives in [`src/esme_pretrain/`](src/esme_pretrain/).

## Quickstart

Install the dev environment:

```bash
uv sync --extra dev
```

Check the repo:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run esme-pretrain status --json
uv run esme-pretrain doctor
```

`doctor` expects no `origin` remote or one containing `adamthuvesen/esme-pretrain`
by default. For a fork or mirror, pass the expected owner/repo substring:

```bash
uv run esme-pretrain doctor --expected-origin <owner/repo>
```

Check the pretraining launch config without touching data or GPUs:

```bash
uv run esme-pretrain pretrain-214m-b200 --config configs/pretrain_214m_b200.json --dry-run --json
```

That command checks the pinned config, dataset revision, split rule, GPU profile,
token budget, artifact manifest, and Modal command. It does not download data or
start training.

## Common Commands

```bash
# Prepare a local text dataset into packed tokens
uv run esme-pretrain prepare-data \
  --input <text-file> --output-dir data/processed/<name> \
  --context-length 1024 --token-budget <tokens> --json

# Evaluate a checkpoint on the fixed validation batches
uv run esme-pretrain eval-checkpoints \
  --config configs/pretrain_214m_b200.json --tokenizer <run-dir>/tokenizer.json \
  --checkpoint <run-dir>/checkpoint.pt --eval-token-budget 10000000 \
  --output <run-dir>/base-eval.json --json

# Turn an eval into an acceptance report
uv run esme-pretrain base-acceptance-report \
  --run-dir <run-dir> --eval <run-dir>/base-eval.json \
  --output <run-dir>/base-acceptance-report.md --json

# Export the selected checkpoint for llm-infer
uv run esme-pretrain export \
  --checkpoint <selected-checkpoint.pt> --tokenizer <run-dir>/tokenizer.json \
  --format llm-infer --output exports/pretrain-214m-b200 --json
```

## Full Pretraining

A full run streams FineWeb-Edu and trains on rented GPUs via Modal. The
entrypoint only launches with an explicit `--approved` flag. Use a detached Modal
launch so a local disconnect does not stop training.

B200 was picked because measurements on H100, H200, and B200 showed it had the
lowest cost per token for this run.

## Documentation

- Current state: [`docs/status.md`](docs/status.md)
- Model and training design: [`docs/architecture.md`](docs/architecture.md)

## Related Repositories

These repositories are separate codebases connected by model artifacts and
measurement questions:

- [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain): trains
  `Esme-214M-Base` from scratch.
- [`esme-posttrain`](https://github.com/adamthuvesen/esme-posttrain): adapts
  the base checkpoint with SFT, DPO, and verifier-backed RLVR.
- [`llm-infer`](https://github.com/adamthuvesen/llm-infer): loads, serves, and
  benchmarks exported Esme checkpoints.
- [`llm-rlvr`](https://github.com/adamthuvesen/llm-rlvr): provides a reusable
  RLVR harness with text-to-SQL as the reference task.
- [`grpo-decomp`](https://github.com/adamthuvesen/grpo-decomp): measures where
  GRPO gains come from, separating reliability from new capability.

## References

- Lozhkov et al., [_FineWeb-Edu: the Finest Collection of Educational Content_](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), 2024.
- Qwen Team, [_Qwen3 Technical Report_](https://arxiv.org/abs/2505.09388), 2025.
- Liu et al., [_MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases_](https://arxiv.org/abs/2402.14905), 2024.
- Allal et al., [_SmolLM2: When Smol Goes Big_](https://arxiv.org/abs/2502.02737), 2025.
