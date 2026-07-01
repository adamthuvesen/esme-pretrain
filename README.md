# esme-pretrain

`Esme-214M-Base` is a language model trained from scratch. `esme-pretrain` owns
its base-model pretraining path: data preparation, tokenizer training, model
code, training launch checks, checkpoint evaluation, acceptance reporting, and
export to `llm-infer`.

## Current State

`Esme-214M-Base` is the accepted base checkpoint.

The accepted run is a 213,960,192-parameter dense decoder-only transformer
trained from scratch on FineWeb-Edu `sample-10BT`. The public training label is
`10B` tokens; the exact configured budget is `10,229,514,240` tokens over
`26,015` optimizer steps.

The base-model handoff is complete. The next model work is posttraining in
`esme-posttrain`, not another pretraining launch.

## What Is Here

- Data tools for local text files and FineWeb-Edu streaming splits.
- A byte-level BPE tokenizer contract with digit splitting.
- Production model code in
  [`DenseBackbone`](src/esme_pretrain/modeling/backbone.py) and
  `BackboneConfig`.
- Training code with checkpoint/resume checks, fixed validation batches, local
  metrics, and optional W&B logging.
- Approval-gated launch code for the accepted `214M B200` run shape.
- Post-run evaluation, acceptance reporting, bits-per-byte reporting, and
  `llm-infer` export.

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

Validate the accepted pretraining launch surface without spend:

```bash
uv run esme-pretrain pretrain-214m-b200 --config configs/pretrain_214m_b200.json --dry-run --json
```

That command checks the pinned config, dataset revision, split rule, GPU profile,
token budget, artifact manifest, and Modal command. It must not download
FineWeb-Edu data or start training.

## Common Commands

Prepare a local text dataset:

```bash
uv run esme-pretrain prepare-data --input <text-file> --output-dir data/processed/<name> --context-length 1024 --token-budget <tokens> --json
```

Evaluate a checkpoint:

```bash
uv run esme-pretrain eval-checkpoints --config configs/pretrain_214m_b200.json --tokenizer <run-dir>/tokenizer.json --checkpoint <run-dir>/checkpoint.pt --eval-token-budget 10000000 --output <run-dir>/base-eval.json --json
```

Generate an acceptance report:

```bash
uv run esme-pretrain base-acceptance-report --run-dir <run-dir> --eval <run-dir>/base-eval.json --output <run-dir>/base-acceptance-report.md --json
```

Export for `llm-infer`:

```bash
uv run esme-pretrain export --checkpoint <selected-checkpoint.pt> --tokenizer <run-dir>/tokenizer.json --format llm-infer --output exports/pretrain-214m-b200 --json
```

## Full Pretraining

Full pretraining is approval-gated. Do not launch FineWeb-Edu, ClimbMix, Modal,
GPU, W&B write, or paid API work without explicit chat approval.

A full launch needs the exact command, hardware target, cost cap, approval
record, and `--approved` flag. Long paid runs should use detached Modal launch
so a local laptop disconnect does not stop training.

## Files That Stay Out Of Git

Keep runtime and secret material ignored:

- `runs/`
- `checkpoints/`
- `exports/`
- `wandb/`
- `.env*`
- raw or processed training data

Config changes that alter the dataset source, token budget, model shape,
tokenizer contract, GPU choice, or cost cap are new run decisions.

## Documentation

- Current state: [`docs/status.md`](docs/status.md)
- Model and training design: [`docs/architecture.md`](docs/architecture.md)

Detailed approval and measurement records live under `docs/` for operators who
need to reproduce the accepted run.

## Related Repos

- `esme-posttrain`: SFT and simple-task RL for the accepted base checkpoint.
- `llm-rlvr`: SFT and execution-verified RLVR experiments.
- `grpo-decomp`: measurement work for RLVR gains.
- `llm-infer`: loading, serving, and benchmarking exported checkpoints.

## References

- Lozhkov et al., [_FineWeb-Edu: the Finest Collection of Educational Content_](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), 2024.
- Qwen Team, [_Qwen3 Technical Report_](https://arxiv.org/abs/2505.09388), 2025.
- Liu et al., [_MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases_](https://arxiv.org/abs/2402.14905), 2024.
- Allal et al., [_SmolLM2: When Smol Goes Big_](https://arxiv.org/abs/2502.02737), 2025.
