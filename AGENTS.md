# esme-pretrain - Agent Instructions

`esme-pretrain` is public at `github.com/adamthuvesen/esme-pretrain`. It owns
base-model pretraining from raw text through checkpoint eval and export.

## Read First

| Work type | Read |
| --- | --- |
| Current state, accepted run, next step | `docs/status.md` |
| Model, data, training, eval, export design | `docs/architecture.md` |
| Export bundle contract and version policy | `docs/bundle-format.md` |
| 214M B200 config, budget, artifacts, abort rules | `docs/run-cards/pretrain-214m-b200.md` |
| GPU selection and Modal safety | `docs/internal/pretrain-214m-b200-hardware-measurements.md` (local, untracked) |
| CLI surface and quick commands | `README.md` |

If docs and code disagree, fix the stale doc in the same change.

## Workflow

- Work from the repo root unless Adam explicitly asks for an isolated worktree.
- Do not push or open a PR without explicit approval.
- Commit accepted local work directly to `main`; keep commits small and conventional.
- Before committing, run the checks that match the change and leave `git status --short` clean except ignored runtime files.
- Use `uv run ...` for project commands.

## Spend And Data

- No FineWeb-Edu, ClimbMix, Modal, GPU, W&B write, or paid API work without an
  explicit run card and chat approval.
- Any new corpus or data recipe needs a streaming data audit (per-stage
  before/after counts, contamination check) and a pre-registered ablation plan
  before its run card.
- Full pretrain launch requires the exact command, hardware, cost cap, and `--approved` flag.
- Use detached Modal launch for long paid runs; local laptop disconnects must not kill training.
- Keep run outputs, checkpoints, exports, W&B state, `.env*`, and raw/processed data out of git.

## Code Rules

- Use clear domain names: corpus, tokenizer, packed tokens, checkpoint, run card, eval, export.
- Make data loss, skipped records, malformed artifacts, resume drift, and config mismatches fail loudly.
- Keep post-run eval deterministic: same validation token batches for every checkpoint.
- Keep `llm-infer` export correctness-first: manifest, config, tokenizer, weights, hashes, and provenance must match the source checkpoint.
- Avoid trivializing small models or fixtures. Use small, rehearsal, fixture, or synthetic fixture as appropriate.
- Production model code is `modeling/backbone.py` (`DenseBackbone` + `BackboneConfig`).

## Gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run esme-pretrain status --json
uv run esme-pretrain pretrain-214m-b200 --config configs/pretrain_214m_b200.json --dry-run --json
```
