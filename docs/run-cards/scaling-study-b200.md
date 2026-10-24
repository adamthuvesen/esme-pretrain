# Scaling Study Run Card (Phase 2)

Status: **design draft, not approved.** This card approves no spend. It fixes the
question, the design, and the predictions before any study code or data exists.
The launch commands are added after the infrastructure below is built and
rehearsed, and the card then goes back for approval.

## Question

`Esme-214M-Base` is one run. Its design makes two claims it never tests:

1. Its loss follows a predictable scaling law in parameters and tokens under
   this exact recipe (data, tokenizer, architecture family, optimizer, WSD).
2. Deep-and-thin beats wide-and-shallow at this scale, the MobileLLM thesis
   cited in `docs/architecture.md`.

The study tests both on small models and then uses the existing 214M run as a
held-out point. A law fitted only on small models predicts the 214M run's
fixed-validation loss. The study pre-registers the fit method, so the known
answer (`2.7756` nats) cannot tune it.

## Design

### Depth family (fit points)

One width (`d_model=768`, 12 query heads, 4 KV heads, `d_ff=2048`), so the 214M
model (`30` layers) is the family's natural endpoint. Only depth varies.

| Arm | Layers | Non-embedding params | Total params |
| --- | ---: | ---: | ---: |
| L6 | 6 | 37.8M | 62.9M |
| L12 | 12 | 75.5M | 100.7M |
| L18 | 18 | 113.3M | 138.4M |
| held out: 214M | 30 | 188.7M | 214.0M |

Each size runs one constant-LR main run to `40 x N` tokens (N = non-embedding
params), with its own 20 percent cooldown at the end. Cooldown branches fork
from the main run's stable phase and decay over the last 20 percent of `5x`,
`10x`, and `20x N`. That gives 4 loss points per size and 12 fit points in all,
for about one full run's compute per size. Hägele et al. (2024, arXiv
2405.18392) report that constant LR plus cooldown scales like cosine and allows
"fewer but reusable training runs". The study checks that claim on this recipe
itself (prediction 1) and does not rely on it.

### Depth vs width (one size)

At about 75M non-embedding params and `20 x N` tokens:

| Arm | Shape | Non-embedding | Total |
| --- | --- | ---: | ---: |
| deep-thin | `d576 x 21L` (9 q / 3 kv heads) | 74.3M | 93.2M |
| middle | `d768 x 12L` (reuses the family's L12 `20x` branch) | 75.5M | 100.7M |
| wide-shallow | `d1152 x 5L` (18 q / 6 kv heads) | 70.8M | 108.5M |
| deep-thin, seed 1 | repeat of deep-thin | 74.3M | 93.2M |

The seed repeat measures run-to-run noise, so arm differences smaller than it
are reported as unresolved.

### Fixed across every arm

- Data: FineWeb-Edu `sample-10BT` at revision `87f09149...`, the same
  deterministic split, pre-tokenized into uint16 shards, one seeded window
  shuffle. Every run reads a prefix of the same permutation, so smaller budgets
  see a subset of larger ones.
- Optimizer and batch: the 214M recipe. AdamW, LR `6e-4`, `min_lr_ratio`
  `0.1`, weight decay `0.1`, betas `0.9/0.95`, grad clip `1.0`, `393,216`
  tokens per step, bf16, compile.
- Warmup: `200` steps for every arm (the 214M run used `450`). With 450, the
  smallest branch (L6 at `5x N`, 480 steps) would cool down inside its warmup.
- Loss metric: CE on the first 10M validation tokens, read from validation
  shards and hash-checked against the accepted eval's `token_batch_sha256`
  (`1e1bb2aa...`). So every point, including the 214M one, uses the token
  windows behind the accepted `2.7756`.

### LR check (gate before the study)

Same LR across sizes is a known limitation without muP. Before the study, L6
trains for `0.5B` tokens at LR `3e-4`, `6e-4`, and `1.2e-3`. If `6e-4` is more
than `0.02` nats worse than the best, the study stops and the LR choice goes
back to Adam.

## Fit Protocol (pre-registered)

- Form: `L(N, D) = E + A / N^alpha + B / D^beta` (Hoffmann et al. 2022,
  approach 3), with N = non-embedding params and D = training tokens.
- Fit: Huber loss (`delta = 1e-3`) on `log L`, L-BFGS from a grid of
  initializations, best fit kept. Besiroglu et al. (2024) showed that this
  approach is sensitive to optimizer details and reported uncertainty, so the
  fit script and its settings are committed before the study runs.
- Uncertainty: bootstrap over the 12 fit points (1,000 resamples) gives a
  90 percent interval for the 214M prediction.
- The fit uses no 214M data. The 214M actual value is read only after the
  prediction is written to `docs/results/`.

## Pre-Registered Predictions

1. **Instrument.** A from-scratch WSD run of L12 at `10x N` ends within `0.01`
   nats of the cooldown branch forked from the L12 main run at the same token
   count. If not, branched points are not trusted as full runs, and the report
   says so.
2. **Held-out prediction.** The fit predicts the 214M fixed-validation CE
   within `0.03` nats of `2.7756`, and the actual value falls inside the
   90 percent bootstrap interval.
3. **Depth vs width.** Deep-thin beats wide-shallow by at least `0.02` nats, by
   more than the seed-repeat gap. The middle arm falls between them.

Known differences between the 214M run and the study: no data shuffle, the old
training loop (same math, see the throughput run card), warmup `450`, and a
token budget of about `54 x N`. That extrapolates beyond the fit grid in both N
(1.7x the largest fit size) and D. If prediction 2 fails, the report separates
these explanations as far as the data allow. It does not tune the fit.

## Compute

FLOPs from `BackboneConfig.flops_per_token` times training tokens. B200-hours
assume `430` achieved TFLOP/s for these smaller models. The 214M run reached
about `650` on the current code. The LR check measures the real rate first.

| Part | FLOPs | B200-hours |
| --- | ---: | ---: |
| GPU smoke on the Python 3.13 image | small | ~0.1 |
| LR check (3 x L6 x 0.5B tokens) | 6.5e17 | 0.4 |
| Depth family: 3 main runs + 9 cooldown branches | 8.5e18 | 5.6 |
| Instrument check (L12 scratch at `10x`) | 5.4e17 | 0.4 |
| Depth vs width: 3 new runs (middle is reused) | 3.1e18 | 2.0 |
| Final evals (16 checkpoints x 10M tokens) | small | ~0.2 |
| **Total** | **~1.3e19** | **~8.7** |

That is about the compute of the original 214M run (`1.3e19`). Arms run in
parallel on up to 4 B200s under the shared account quota. The wall time is
about 3.5 hours, set by the L18 main run.

## Engineering Before Approval (no spend)

1. Study config: every arm's shape, token budget, branch points, seed, and LR.
2. Cooldown branches. A run starts from a checkpoint's weights, optimizer
   state, and data offset under a new schedule. It refuses unless the fork step
   is in the stable phase of both schedules at the same LR, so a branch can
   never hide schedule drift. The main run saves checkpoints at the exact
   branch steps.
3. Shard checks. `ShardBatchLoader` users verify the manifest's vocab and
   tokenizer hash against the model and run.
4. Validation from shards. The first 10M validation tokens come from shards,
   and the loader refuses unless they hash to `1e1bb2aa...`.
5. Shard staging. Each container copies the shard set from the volume to local
   disk before training, because the shuffled reads are random.
6. Study runner on Modal. Arms launch detached, each arm's results go to the
   volume as they finish, at most 4 GPUs at a time, with a spend cap.
7. Fit script. Its test must recover known parameters from synthetic data.
8. Data. About 5B train tokens and 10M validation tokens pre-tokenized locally
   (CPU, free) and uploaded to a study volume.
9. A CPU rehearsal of one full arm (main run, branch, eval) on a tiny model.

## Artifacts

- `docs/results/scaling-study/`: per-arm JSON (losses, throughput, provenance),
  the fit, the bootstrap interval, and the written prediction before the 214M
  comparison.
- A results section in this card that scores each prediction.
- Plots: loss vs tokens per size, the fitted surface with the 214M point, and
  depth vs width.
