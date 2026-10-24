# 214M B200 Pretrain Run Card

`214M B200` is the current dense 10B pretrain run shape. Its exported model
artifact is `Esme-214M-Base`, the base model used by posttraining and serving.

## Run

- **Run:** `pretrain_214m_b200`
- **Produces:** `Esme-214M-Base`
- **Config:** `configs/pretrain_214m_b200.json`
- **Dataset:** FineWeb-Edu `sample-10BT`, source `HuggingFaceFW/fineweb-edu`, revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- **Split:** deterministic document hash, 1 percent validation bucket, seed `0`.
- **Train budget:** `10,229,514,240` tokens (`26015` steps; rounded public label: 10B).
- **Validation budget:** `50,000,000` tokens.
- **Tokenizer training budget:** `50,000,000` tokens.
- **Hard read budget:** `10,329,514,240` tokens.

This file and the checked-in config do not approve a new FineWeb-Edu download,
Modal job, GPU, W&B run, or paid compute. The 10B base training run is complete.
Future reruns still require explicit approval.

## Model

- **Shape:** `30L x 768`, `12Q/4KV`, `d_ff=2048`, context `1024`.
- **Expected params:** `213,960,192`.
- **Tokenizer:** digit-split byte-level BPE, vocab `32768`, special tokens
  `<pad>`, `<bos>`, `<eos>`, `<unk>`.
- **Architecture:** RoPE, RMSNorm, SwiGLU, pre-norm, no biases, GQA, QK-norm,
  tied embeddings, z-loss `0.0001`, logit soft-cap disabled.
- **Optimizer/runtime recipe:** WSD schedule, bf16, `torch.compile`, SDPA,
  fused AdamW.

## Hardware

Selected dry-run profile: `B200`.

A measurement pass ran the same training shape (micro batch `24`, grad accum
`16`) on H100, H200, and B200. The measurements are recorded in
`configs/pretrain_214m_b200.json`:

| GPU | Tokens/sec | Cost per token, relative to B200 |
| --- | ---: | ---: |
| H100 | 165,070 | 1.11 |
| H200 | 191,391 | 1.11 |
| B200 | 290,838 | 1.00 |

At their hourly prices, H200 had to beat H100 by about 15 percent in throughput
to cost less per token, and B200 by about 58 percent. B200 cleared that bar and
has the lowest cost per token.

## Launch Contract

Dry-run:

```bash
uv run esme-pretrain pretrain-214m-b200 --config configs/pretrain_214m_b200.json --dry-run --json
```

Future full runs require explicit chat approval of the exact command and a
runtime spend stop. The current base reached `26015` steps / `10.23B` target
tokens before the finite `sample-10BT` stream exhausted. Its artifacts live on
the Modal volume `llm-pretrain-214m-b200`.

```bash
PRETRAIN_GPU='B200' PRETRAIN_TIMEOUT_HOURS=24 uv run --with modal==1.5.0 modal run --detach scripts/modal_pretrain.py --config configs/pretrain_214m_b200.json --approved --json
```

The script refuses the full Modal job unless `--approved` is present. The
`modal run --detach` form keeps the ephemeral app alive if the local laptop
disconnects. `run_pretrain_launch.spawn(...)` is called by the launcher, and the
function-call ID is returned by default. With `--no-spawn`, `.get()` is awaited
only for a short inline smoke.

## Required Artifacts

The run directory `runs/pretrain-214m-b200/pretrain_214m_b200/` should contain:

- `config.json`
- `tokenizer.json`
- `tokenizer-report.json`
- `data-report.json`
- `metrics.jsonl`
- `throughput.csv`
- `checkpoint.pt`
- `environment.txt`
- `cost.json`
- `run-summary.json`
- `scaleup-pretrain-report.md`

## Accepted Result

The run completed all `26,015` steps and stopped when the finite `sample-10BT`
stream was exhausted. Accepted numbers, quoted here so they are citable from a
fresh clone (run outputs are gitignored):

- Train loss `10.5339` -> `2.7360` (min `2.5944`). Training-time validation
  loss `10.4642` -> `2.7426`. Final grad norm `0.1215`. This run's in-loop eval
  drew new validation batches at each eval, so each point is a real held-out
  loss with sampling noise in the trend. The post-run fixed-validation eval
  below is the reference number.
- Steady throughput `215,714` tokens/sec (peak `239,908`) on one B200; peak
  GPU memory `40.69` GB. The full series are plotted in the committed
  telemetry figures under `assets/`.
- Post-run fixed-validation eval over `10,002,432` tokens and `44,478,593`
  bytes (`token_batch_sha256` `1e1bb2aa...`): CE loss `2.7756`, bits per byte
  `0.9005`, perplexity `16.05`. The independent baseline harness measured
  `0.9012` bits per byte on its own FineWeb-Edu slice, an agreement of about
  `0.0007` across two code paths.

Source artifacts: `run-summary.json`
(sha256 `7ebf1b5aca34a89702348a7bd54637253d859692720aa869c37cd4fbcbc3571e`)
and the eval of record, `base-eval-bpb-fix.json`
(sha256 `cb6af9fa55fedfe4a6e81a93fd976c4ba4621ed20201926f67a6fb2889a40fdd`),
both in the run directory above. The eval JSON records its device (`mps`) and
the fixed-batch hash.

Decoded qualitative samples can be produced from the accepted checkpoint
without changing the accepted artifact set:

```bash
uv run esme-pretrain sample \
  --checkpoint runs/pretrain-214m-b200/pretrain_214m_b200/checkpoint.pt \
  --tokenizer runs/pretrain-214m-b200/pretrain_214m_b200/tokenizer.json \
  --prompt "Once upon a time" --max-new-tokens 64 \
  --output runs/pretrain-214m-b200/pretrain_214m_b200/samples.md --json
```

## Abort Rules

- A new full pretrain command and runtime spend stop have not been explicitly approved.
- The dry-run payload is not `ready_for_pretrain_launch`.
- Runtime spend reaches `$100` or projected spend would exceed the `$100` absolute cap.
- FineWeb-Edu source, revision, license, or deterministic split differs from this config.
- Tokenizer training cannot write tokenizer evidence and pass round-trip checks.
- Loss becomes NaN/inf, fails to trend down in the early sanity window, or validation loss cannot be computed.
- Throughput is low enough that the run will miss the approved `$100` cap.
- Checkpoint save/resume fails or resume would re-read the corpus head.
- Required artifacts cannot be written to the Modal Volume and local output mirror.
- Any implementation change would alter dataset, current 10B token budget, model config, tokenizer choice, selected GPU profile, or cost cap.
