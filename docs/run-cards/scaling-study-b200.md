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
- Fit: Huber loss (`delta = 1e-3`) on `log L`, minimized with L-BFGS (float64)
  in `src/esme_pretrain/study/scaling_law.py`. The solver works on centered
  `log N` and `log D` and divides the loss by `delta^2`. Both change the
  conditioning, not the model or the optimum. Without them, L-BFGS stopped
  early in a flat valley on synthetic data.
- Initialization: the Hoffmann grid (4,500 starts). All starts run 200 L-BFGS
  iterations as one batched problem, then the 4 best run alone to convergence,
  and the lowest objective wins. The batched screen is a heuristic. It found the
  same optimum as screens keeping 8 and 32 candidates on 6 synthetic datasets.
  Besiroglu et al. (2024) showed this approach is sensitive to optimizer
  details, which is why these settings are fixed here before the study runs.
- Uncertainty: bootstrap over the 12 fit points (1,000 resamples, seed 0) gives
  a 90 percent interval for the 214M prediction. Each resample refits from the
  4 polished full-data solutions, not the full grid. A resample whose optimum
  lies in a basin the full data never reached could be missed.
- Check: on noiseless synthetic data from known parameters on this grid, the
  fit recovers each parameter to 1e-8 and the 214M prediction to 5e-12 nats
  (`tests/test_scaling_law.py`). With noise `sigma = 0.003`, the 90 percent
  interval covered the true value in 9 of 10 noise seeds.
- Command: `esme-pretrain scaling-fit --points <points.json>
  --predict-non-embedding-params 188794368 --predict-tokens 10229514240
  --output <fit.json>`.
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

## Infrastructure (built, no spend)

- `configs/scaling_study.json`: every arm's shape, token budget, branch points,
  seed, and LR. `study/plan.py` expands it into runs and refuses any branch
  that forks outside the main run's peak-LR phase.
- `PretrainConfig.branch_from`: a branch starts from a checkpoint's weights,
  optimizer state, and data offset. It may change only `max_steps`. A CPU test
  shows a branch equals a from-scratch run of its schedule bit for bit.
- `ShardBatchLoader` requires the model's vocab and the tokenizer's sha256.
  `fixed_validation_batches` refuses validation tokens that do not hash to
  `1e1bb2aa...`.
- `launch/scaling_study.py`: stages run separately (`smoke`, `lr-gate`,
  `study`). Every arm is spawned at launch into a detached app, each in its own
  single-use B200 container. Modal runs at most 4 arms at once
  (`max_containers`), so a client disconnect neither stops nor drops an arm.
  The held-out 214M eval is its own `heldout` stage. It refuses to run without
  the written fit output (`--prediction`) and records that file's sha256 in its
  row, so the actual value cannot be read before the prediction exists. It
  also refuses a fit that predicts any point other than the 214M one, a fit
  whose points are not exactly the 12 runs of the `heldout.fit_arms` (L6, L12,
  L18), a fit whose settings differ from the protocol above, and a checkpoint
  whose model is not the base config's or whose step is not the full
  10.23B-token run. Before any GPU starts, it checks that the checkpoint is on
  `llm-pretrain-214m-b200`, the volume that holds the accepted run. The base
  config's `runtime.modal_volume` (`esme-pretrain-214m-b200`) names a volume
  that does not exist. The `study` stage refuses
  any LR that `docs/results/scaling-study/lr-gate.json` puts more than `0.02`
  nats behind the best gate row. Each container stops itself once
  its budget runs out: 2x its FLOPs estimate plus 20 minutes, at most 5 hours.
  Containers copy the shards to local disk, and rows go to the volume as runs
  finish. `--check-only` prices a stage.
- Scoring: every run and the held-out 214M checkpoint are scored by
  `validation_ce` in true fp32. Training enables TF32 process-wide, so the
  scorer sets `float32_matmul_precision("highest")` for the eval and restores
  it afterwards. The accepted `2.7756` eval also ran in fp32.
- `esme-pretrain scaling-fit --results-dir <dir> --arm L6 --arm L12 --arm L18`
  reads the study's rows directly, one point per run, labeled `arm/run`. Each
  row names its arm's planned runs (`arm_runs`), and the fit refuses an arm
  that is missing one.
- `tests/test_scaling_study.py` trains a tiny arm (main and branch) end to end
  on CPU, scores it, and writes its rows.

## Commands

```bash
# Data (done 2026-10-24; CPU and upload only). 5B train and 20M validation
# tokens from sample-10BT files 000-006 at revision 87f09149.
uv run --extra baselines esme-pretrain pretokenize --split validation --max-tokens 20000000 \
  --config configs/pretrain_214m_b200.json --tokenizer runs/pretrain-214m-b200/dl/tokenizer.json \
  --output-dir data/processed/study-shards --corpus-parquet <file> ...
uv run --extra baselines esme-pretrain pretokenize --split train --max-tokens 5000000000 ...
uv run modal volume create esme-pretrain-study
uv run modal volume put esme-pretrain-study data/processed/study-shards /shards
uv run modal volume put esme-pretrain-study runs/pretrain-214m-b200/dl/tokenizer.json /tokenizer.json

# Stages, each priced first with --check-only.
uv run python scripts/modal_scaling_study.py --stage smoke --spend-cap-usd 20 --approved
uv run python scripts/modal_scaling_study.py --stage lr-gate --spend-cap-usd 20 --approved
uv run python scripts/modal_scaling_study.py --stage study --spend-cap-usd <cap> --approved

# Fit and held-out prediction, after the study rows are fetched from the volume.
uv run modal volume get esme-pretrain-study /results/<study-run-id> runs/scaling-study/
uv run esme-pretrain scaling-fit --results-dir runs/scaling-study/<study-run-id> \
  --arm L6 --arm L12 --arm L18 --predict-non-embedding-params 188794368 \
  --predict-tokens 10229514240 --output docs/results/scaling-study/fit.json
git add docs/results/scaling-study/fit.json && git commit -m "..."   # prediction first

# Only then: score the 214M checkpoint on the same validation batches.
uv run python scripts/modal_scaling_study.py --stage heldout \
  --prediction docs/results/scaling-study/fit.json --spend-cap-usd 20 --approved
```

## Artifacts

- `docs/results/scaling-study/`: per-arm JSON (losses, throughput, provenance),
  the fit, the bootstrap interval, and the written prediction before the 214M
  comparison.
- A results section in this card that scores each prediction.
- Plots: loss vs tokens per size, the fitted surface with the 214M point, and
  depth vs width.

## Gate Results (2026-10-24)

**Smoke** (run `smoke-51870ae-20261024T171214`, about 10 minutes of one B200, on
the Python 3.13 image). The main run checkpointed at its fork. The branch resumed
from step 16 at stream offset `6,297,600` tokens (16 x 393,216 x 1025/1024).
Both runs were scored on validation batches hashing to `1e1bb2aa...`. L6 trained
at `1.61M` tokens/sec, 31 percent MFU, about 700 TFLOP/s. That beats the
430 assumed above, so every study estimate here is conservative. Rows:
[`docs/results/scaling-study/smoke.json`](../results/scaling-study/smoke.json).

**LR gate: failed** (run `lr-gate-51870ae-20261024T172303`, three B200s for
about 6 minutes each). Rows:
[`docs/results/scaling-study/lr-gate.json`](../results/scaling-study/lr-gate.json).

| L6, 0.5B tokens | Validation CE |
| --- | ---: |
| LR `3e-4` | 3.7176 |
| LR `6e-4` (the 214M recipe) | 3.5476 |
| LR `1.2e-3` | 3.4542 |

`6e-4` is `0.093` nats worse than the best LR, past the `0.02` stop rule. The
best LR is also the edge of the grid, so the optimum may lie higher. Per this
card, the study stops, and the LR choice goes back to Adam. These three runs
were scored with TF32 matmuls, before the fp32 scoring fix. All three share that
path, so the comparison between them holds.


## Amendment 1: Per-Shape LR Sweep (2026-10-24, before any sweep run)

The LR gate failed, so the LR choice went back to Adam. He chose to tune the LR
per shape rather than keep one LR for all sizes. One LR across sizes puts each
size at a different distance from its own optimum. That bends the fitted
exponents, which is the bias the gate exposed.

**Sweep** (`lr-gate` stage, 0.5B tokens per run, seed 0, all scored in fp32):

| Shape | Non-embedding params | LRs |
| --- | ---: | --- |
| `d768-L6` | 37.8M | `2.4e-3` (adds to the gate's `3e-4`, `6e-4`, `1.2e-3`) |
| `d768-L12` | 75.5M | `6e-4`, `1.2e-3`, `2.4e-3` |
| `d768-L18` | 113.3M | `6e-4`, `1.2e-3`, `2.4e-3` |
| `d768-L30` (the 214M shape) | 188.8M | `6e-4`, `1.2e-3`, `2.4e-3` |

**Selection rule** (fixed before the sweep runs):

1. Each shape uses the LR with the lowest validation CE at 0.5B tokens.
2. If that LR is the highest one swept for the shape, add one run at `2x` that
   LR and pick again. Do not extend a second time; report a best LR that is
   still at the edge.
3. Study arms take their shape's selected LR. Shapes outside the sweep (the
   depth-vs-width arms, 70.8M and 74.3M) take the LR of the swept shape
   nearest in non-embedding params, which is L12.
4. The `study` stage enforces this: it refuses any run whose LR is more than
   `0.02` nats behind the best gate row of its shape (or of the nearest swept
   shape). It reads `lr-gate.json` and `lr-sweep.json`.
5. The L30 runs do not enter the fit. They measure how far `6e-4`, the held-out
   214M run's LR, is from the best LR at that shape. The report states that gap
   next to prediction 2 either way.

The L6 rows from the gate were scored with TF32 matmuls; the new L6 run is
scored in fp32. If L6's best two LRs are within `0.005` nats, the report flags
that comparison as inside the scoring difference.

**Compute:** 10 runs. The estimate at the 430 TFLOP/s planning assumption is
3.3 B200-hours; at measured throughput (L6 about 700, the 214M run about 520
TFLOP/s) it is about 2 to 2.5 B200-hours plus container start-up. The worst
case, if every container runs to its timeout, is about 10 B200-hours.

### Sweep Results (run `lr-gate-5d97495-20261024T193042`)

Ten B200 runs, about 2.6 B200-hours in all. Rows:
[`docs/results/scaling-study/lr-sweep.json`](../results/scaling-study/lr-sweep.json).
Validation CE at 0.5B tokens:

| Shape | `6e-4` | `1.2e-3` | `2.4e-3` | `6e-4` minus best |
| --- | ---: | ---: | ---: | ---: |
| L6 (37.8M) | 3.5476* | 3.4542* | **3.4171** | 0.131 |
| L12 (75.5M) | 3.4425 | 3.3674 | **3.3382** | 0.104 |
| L18 (113.3M) | 3.3888 | 3.3130 | **3.2983** | 0.090 |
| L30 (188.8M) | 3.3389 | 3.2698 | **3.2519** | 0.087 |

\* From the LR gate, scored with TF32. L6's best two differ by `0.037` nats,
well above the `0.005` flag, so the scoring difference does not decide L6.

Every shape is best at the highest LR swept, so rule 2 adds one run per shape
at `4.8e-3`.

## Amendment 2: Rules Fixed Before the `4.8e-3` Runs (2026-10-24)

1. **Ties.** A `4.8e-3` run that beats `2.4e-3` by less than `0.005` nats is a
   tie, and ties go to the lower LR. A `4.8e-3` run that stops on a non-finite
   loss or gradient (the trainer raises on both) loses to `2.4e-3`.
2. **Divergence in a study arm.** A study run diverges if it stops on a
   non-finite loss, or if its 50-step mean train loss rises more than `0.5`
   nats above its running minimum after warmup. These two thresholds have no
   external source; they are fixed here, before any study data. A diverged arm
   reruns once at its shape's next lower swept LR, and the report names it.
   That rerun is the one exception to the `0.02`-nat guard in rule 4 of
   Amendment 1. The launcher does not yet encode it; it is added only if a
   divergence happens, in the same change that records it.
3. **Horizon bias.** The sweep picks each LR at 0.5B tokens: 13x N for L6, 6.6x
   for L12, 4.4x for L18, and 2.6x for L30. The study's main runs go to 40x N,
   which stretches the horizon 3x for L6 and 9x for L18. The best LR falls as
   the horizon grows, so a 0.5B pick is likely more too-high for larger shapes.
   That would bias the fitted N exponent low. The report gives each arm's
   residuals at its `5x` branch (close to the sweep horizon) and at `40x`. A
   residual trend that tracks shape is this bias.
4. **Reading the L30 gap.** The `0.087`-nat cost of `6e-4` at L30 is measured at
   0.5B tokens. A too-low LR costs less as training gets longer, so this is an
   upper bound on the 214M run's handicap at 10.23B tokens. The report never
   subtracts it from prediction 2's residual. If the 214M actual is worse than
   the prediction by less than this gap, the report reads the miss as
   consistent with the LR handicap, not as caused by it.
5. **Noise floor.** All 12 fit points use seed 0. Until the deep-thin repeat
   runs, the only noise estimate is the `0.005`-nat scoring flag.
