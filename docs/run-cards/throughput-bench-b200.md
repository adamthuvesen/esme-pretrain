# B200 Throughput Benchmark Run Card

Status: **draft, not approved.** This card approves nothing until Adam approves
the exact commands below in chat.

## Question

The accepted 214M run trained at `215,714` tokens/sec on one B200. A GPU smoke of
the same step shape on a fixed on-device batch measured `290,838` tokens/sec.
That is a 26 percent gap, which is about 15 percent MFU against the B200's dense
BF16 peak. This benchmark measures where the gap comes from and how much of it
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
| `before-stream` | before | Hub stream, tokenized on the fly | The accepted run's path; reproduces the gap |
| `after-stream` | after | Hub stream, tokenized on the fly | Whether the loop changes alone fix the gap |
| `after-shards` | after | pre-tokenized shards, shuffled | The full new path |
| `before-fixed-repeat` | before | fixed on-device batch | Drift control across the session |

Both trees ran under the harness locally in `--rehearsal` mode (tiny model, CPU)
and reached the same final loss (`9.8754`), and the shard variant ran on the
real shards. The rehearsal checks the harness, not GPU throughput.

## Pre-Registered Predictions

Written before any measurement. The result report scores each one.

1. `before-fixed` reproduces the smoke's `290,838` tokens/sec within 5 percent.
2. `before-stream` reproduces the accepted run's `215,714` tokens/sec within
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
  `1cba351ea632...` (the accepted run's `tokenizer.json`). The run needs
  `19,200` windows, and the shards hold `39,024`.
- Stream variants read the Hub stream unauthenticated, as the accepted run did.
  Local streaming stalled twice on server-closed connections, so a stream
  variant can time out. It is then reported as `timeout`, not retried.

## Commands

Each command below is a separate approval item.

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
- Committed evidence at acceptance: the result JSON copied to
  `docs/results/throughput-bench-b200.json`, plus a results section in this
  card that scores each prediction. This follows the retrospective's rule that
  acceptance commits its own numbers.
- MFU convention: achieved TFLOPs over the B200's `2,250` dense BF16 TFLOPs,
  with FLOPs per token from `BackboneConfig.flops_per_token`
  (see `docs/architecture.md`).
