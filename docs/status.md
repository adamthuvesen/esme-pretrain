# Status

`Esme-214M-Base` is the current base checkpoint from the `214M B200` 10B-token
pretraining run.

It is a 213,960,192-parameter dense decoder-only transformer trained from
scratch on FineWeb-Edu `sample-10BT`. The public label is `10B` tokens; the
exact configured budget is `10,229,514,240` tokens over `26,015` optimizer
steps.

## Current Run Shape

- Config: `configs/pretrain_214m_b200.json`
- Dataset: `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`, revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- Split: deterministic source-document hash split, validation remainder `0`
  modulo `100`.
- Tokenizer: digit-split byte-level BPE, vocab `32768`.
- Model: 30 layers, `d_model=768`, 12 query heads, 4 KV heads, `d_ff=2048`,
  context `1024`.
- Export target: `llm-infer` bundle format `llm_pretrain_dense_v1`.

## Handoff

The base checkpoint has evaluation, bits-per-byte reporting, a summary report,
and an exported `llm-infer` bundle. The next model work is posttraining in
`esme-posttrain`, not another pretraining launch.

## Baseline Comparison

The baseline harness (2026-07-10 run, `configs/baseline_eval.json`) scored the
exported base bundle against Cerebras-GPT-256M and Pythia-160M. The gate
reproduced Cerebras's published 0-shot table within ±0.002 before any Esme
score was recorded. Esme won 6 of 7 downstream tasks (average 0.408 vs 0.347
and 0.365) and the FineWeb-Edu bits-per-byte slice; the Pile slice went to the
Pile-trained baselines. Results table: README "How Good Is It?". Raw JSONs
live under gitignored `out/`.

## Safety

- Full pretraining remains approval-gated in code and docs.
- The dry-run command must never download data or start Modal.
- Runtime artifacts belong under ignored paths such as `runs/`, `checkpoints/`,
  `exports/`, and `wandb/`.
- Secrets belong in ignored `.env*` files and must not be committed.

## Local Gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run esme-pretrain status --json
uv run esme-pretrain pretrain-214m-b200 --config configs/pretrain_214m_b200.json --dry-run --json
```
