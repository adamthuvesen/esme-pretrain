# B200 Throughput Benchmark Run Card

Status: **complete** (2026-10-24). Result: the production training step runs
`1.57x` faster on one B200. See [Result](#result).

## Question

The base model run trained at `215,714` tokens/sec on one B200, about 15 percent
MFU against the B200's dense BF16 peak. A GPU smoke of the same step shape on a
fixed on-device batch measured `290,838` tokens/sec, 26 percent faster. This benchmark measures where the gap comes from and how much of it
the throughput changes close:

- Pre-tokenized uint16 shards with a shuffled, seekable loader, instead of
  on-the-fly tokenization of the Hub stream.
- One host read per optimizer step instead of about three per micro-batch.
- SDPA's native GQA (`enable_gqa`) instead of materialized `repeat_interleave`.
- Forward and loss compiled as one graph, so the loss's fp32 upcast fuses into
  its reductions instead of copying the logits to fp32.

## Design

One B200 container runs every variant in sequence, so all variants share the
GPU, driver, and image. Each variant is a subprocess with its own source tree:

- `before`: `git archive 5cd1851 src`, the code before the throughput work.
- `after`: the committed working tree at launch. The launcher refuses a dirty tree.

Each variant runs the production step shape from
`configs/pretrain_214m_b200.json`: micro batch `24`, grad accum `16`,
context `1024`, bf16, `torch.compile`, fused AdamW. That is `393,216` tokens per
optimizer step. It runs `10` warmup steps and `40` measured steps. The result is
the median, p10, and p90 tokens/sec over the measured steps, plus peak memory and
the loss trace.

| Variant | Tree | Data path | Measures |
| --- | --- | --- | --- |
| `before-fixed` | before | fixed on-device batch | Old compute ceiling; reproduces the smoke |
| `after-fixed` | after | fixed on-device batch | Loop, GQA, and loss changes in isolation |
| `before-stream` | before | Hub stream, tokenized on the fly | The base model run's path; reproduces the gap |
| `after-stream` | after | Hub stream, tokenized on the fly | Whether the loop changes alone fix the gap |
| `after-shards` | after | pre-tokenized shards, shuffled | The full new path |
| `before-fixed-repeat` | before | fixed on-device batch | Drift control across the session |

Both trees ran under the harness locally in `--rehearsal` mode (tiny model, CPU)
and reached the same final loss (`9.8754`), and the shard variant ran on the
real shards. The rehearsal checks the harness, not GPU throughput.

## Pre-Registered Predictions

Written before any measurement. The result report scores each one.

1. `before-fixed` reproduces the smoke's `290,838` tokens/sec within 5 percent.
2. `before-stream` reproduces the base model run's `215,714` tokens/sec within
   5 percent.
3. `after-fixed` beats `before-fixed` by at least 3 percent. Its peak memory is
   at least 2 GB lower, because the fp32 copy of the `24 x 1024 x 32768` logits
   (3.2 GB) is no longer materialized.
4. `after-shards` is within 3 percent of `after-fixed`: the data path is no
   longer the bottleneck.
5. `after-stream` is within 5 percent of `before-stream`: the gap is the
   tokenizing data path, not the training loop.

Control: `before-fixed-repeat` must be within 2 percent of `before-fixed`.
If it is not, the session was noise-limited, and differences smaller than the
drift are reported as unresolved.

## Data

- Shards: `40,000,000` train-split tokens (`36,767` documents) from
  `sample/10BT/000_00000.parquet` at revision
  `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, written locally by
  `esme-pretrain pretokenize`. Shard sha256 `5c8690d4dfad...`, tokenizer sha256
  `1cba351ea632...` (the base model run's `tokenizer.json`). The run needs
  `19,200` windows, and the shards hold `39,024`.
- Stream variants read the Hub stream unauthenticated, as the base model run did.
  Local streaming stalled twice on server-closed connections, so a stream
  variant can time out. It is then reported as `timeout`, not retried.

## Commands

```bash
# 1. Create the benchmark volume (Modal write, no GPU).
uv run modal volume create esme-pretrain-bench

# 2. Upload the shards and the tokenizer (Modal write, about 80 MB, no GPU).
uv run modal volume put esme-pretrain-bench data/processed/shards-bench-40m /shards
uv run modal volume put esme-pretrain-bench runs/pretrain-214m-b200/dl/tokenizer.json /tokenizer.json
uv run modal volume ls esme-pretrain-bench /shards

# 3. No-spend check: config, commits, and the worst-case budget.
uv run python scripts/modal_throughput_bench.py --spend-cap-usd 10 --check-only

# 4. The benchmark (rents one B200).
uv run python scripts/modal_throughput_bench.py --spend-cap-usd 10 --approved
```

Step 2's `ls` must show `/shards/train-manifest.json` and
`/shards/train-00000.bin`. The benchmark reads those paths.

## Budget

- Expected: about `0.5` B200-hours (six variants at about 4 minutes each,
  including compile, plus container start).
- Hard cap: `1.25` B200-hours, the Modal function timeout. Each variant has an
  11-minute subprocess timeout, so all six fit inside the function timeout even
  if every variant stalls. The launcher refuses to start when the timeout's
  worst case exceeds `--spend-cap-usd`.
- The app runs detached, so a laptop disconnect does not stop it, and nothing
  already measured is lost.
- No W&B, no retries, no other GPUs.

## Abort Rules

- Any command differs from this card, or the tree at launch is dirty.
- Step 3 does not print `"status": "check_ok"`.
- A variant fails or times out: the run continues with the rest and reports it.
  It is not re-run without a new approval.
- A loss is non-finite in any variant: the loop raises, and that variant is
  reported as `failed`.

## Artifacts

- Per variant, as it finishes: `esme-pretrain-bench:/results/<run-id>/<variant>.json`,
  committed to the volume. The launcher prints the run id first. If the client
  disconnects, fetch them with
  `uv run modal volume get esme-pretrain-bench /results/<run-id> runs/throughput-bench-b200/`.
- Local result: `runs/throughput-bench-b200/throughput-bench-<commit>.json`
  (gitignored).
- Committed evidence: the result JSON at
  [`docs/results/throughput-bench-b200.json`](../results/throughput-bench-b200.json),
  and the [Result](#result) section below, which scores each prediction.
- MFU convention: achieved TFLOPs over the B200's `2,250` dense BF16 TFLOPs,
  with FLOPs per token from `BackboneConfig.flops_per_token`
  (see `docs/architecture.md`).

## Result

Run id `0e0b699-20261024T154847`, about `0.17` B200-hours. Every variant
imported its assigned source tree. Raw rows:
[`docs/results/throughput-bench-b200.json`](../results/throughput-bench-b200.json).

| Variant | Median tokens/sec | p10-p90 | MFU | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| `before-fixed` | 344,906 | 344,198-345,528 | 24.0% | 40.69 GB |
| `after-fixed` | 421,183 | 420,117-422,108 | 29.3% | 27.29 GB |
| `before-stream` | 266,478 | 245,356-276,657 | 18.6% | 40.69 GB |
| `after-stream` | 322,962 | 315,601-328,523 | 22.5% | 27.29 GB |
| `after-shards` | 418,466 | 415,599-422,784 | 29.1% | 27.29 GB |
| `before-fixed-repeat` | 343,457 | 342,227-345,377 | 23.9% | 40.69 GB |

The drift control, `before-fixed-repeat`, is `0.4` percent below
`before-fixed`, inside the 2 percent bound. The before tree's peak memory,
`40.69` GB, equals the base model run's recorded peak, an independent check that
`before` is the production code.

**Headline.** The production path went from `266,478` tokens/sec (stream,
before) to `418,466` (shards, after): `1.57x`, MFU `18.6` -> `29.1` percent,
peak memory `40.69` -> `27.29` GB. The two changes compose:

- Loop, GQA, and compiled loss, on the streaming path: `266,478` -> `322,962`,
  `1.21x` (`1.22x` on the fixed batch).
- Data path, on the after code, stream -> shards: `322,962` -> `418,466`, `1.30x`.

**Predictions.**

1. Not confirmed. `before-fixed` is `344,906`, `18.6` percent above the June
   smoke's `290,838`. The smoke ran older code and an older image, and this run
   cannot separate the two.
2. Not confirmed. `before-stream` is `266,478`, `23.5` percent above the base
   model run's `215,714`. That run also logged to W&B and read the Hub for
   26,015 steps, and streaming throughput varies by about 12 percent between
   sessions, so stream numbers compare only within one session.
3. Confirmed. `after-fixed` beats `before-fixed` by `22.1` percent, and peak
   memory drops by `13.4` GB, more than the `3.2` GB fp32 logits copy alone.
   The three loop changes ran as one bundle, so their separate shares are not
   measured.
4. Confirmed. `after-shards` is `0.6` percent below `after-fixed`. Shards leave
   the data path within noise of the compute ceiling.
5. Not confirmed. `after-stream` beats `before-stream` by `21.2` percent, so the
   loop changes also speed up the streaming path. Streaming stays at about `77`
   percent of the fixed-batch rate under both trees. The data path is a
   bottleneck, but not the only one, and this benchmark does not explain the
   interaction.

**Method note.** Modal mounts the benchmark's own package at `/root`, which
Python puts ahead of `PYTHONPATH`, so a variant can silently import the wrong
source tree. The runner starts each variant with `python -P` and refuses to
measure unless `esme_pretrain` was imported from that variant's tree.
