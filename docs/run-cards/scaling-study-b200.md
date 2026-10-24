# Scaling Study Run Card

Status: learning-rate selection **done**. The study, the fit, and the 214M
prediction are **not yet run**.

## Question

`Esme-214M-Base` is one run. Its design rests on two claims it never tests:

1. Its loss follows a predictable scaling law in parameters and tokens under
   this recipe (data, tokenizer, architecture family, optimizer, WSD).
2. Deep-and-thin beats wide-and-shallow at this scale, the MobileLLM thesis
   cited in [`docs/architecture.md`](../architecture.md).

The study tests both on small models and uses the existing 214M run as a
held-out point. A law fitted only on small models predicts the 214M run's
fixed-validation loss. The fit method is fixed in advance, so the known answer
(`2.7756` nats) cannot tune it.

## Design

### Depth family (fit points)

One width (`d_model=768`, 12 query heads, 4 KV heads, `d_ff=2048`), so the 214M
model (`30` layers) is the family's natural endpoint. Only depth varies.

| Arm | Layers | Non-embedding params | Total params | LR |
| --- | ---: | ---: | ---: | --- |
| L6 | 6 | 37.8M | 62.9M | `4.8e-3` |
| L12 | 12 | 75.5M | 100.7M | `2.4e-3` |
| L18 | 18 | 113.3M | 138.4M | `2.4e-3` |
| held out: 214M | 30 | 188.8M | 214.0M | `6e-4` (as trained) |

Each size runs one main run to `40 x N` tokens (N = non-embedding params), with
a 20 percent cooldown at the end. Cooldown branches fork from the main run's
constant-LR phase and decay over the last 20 percent of `5x`, `10x`, and `20x N`.
That gives 4 loss points per size and 12 fit points in all, for about one full
run's compute per size. Hägele et al. (2024, arXiv 2405.18392) report that
constant LR plus cooldown scales like cosine and allows "fewer but reusable
training runs". Prediction 1 checks that claim on this recipe.

### Depth vs width (one size)

At about 75M non-embedding params and `20 x N` tokens, all at L12's LR:

| Arm | Shape | Non-embedding | Total |
| --- | --- | ---: | ---: |
| deep-thin | `d576 x 21L` (9 q / 3 kv heads) | 74.3M | 93.2M |
| middle | `d768 x 12L` (reuses the family's L12 `20x` branch) | 75.5M | 100.7M |
| wide-shallow | `d1152 x 5L` (18 q / 6 kv heads) | 70.8M | 108.5M |
| deep-thin, seed 1 | repeat of deep-thin | 74.3M | 93.2M |

The seed repeat measures run-to-run noise. Arm differences smaller than it are
reported as unresolved.

### Fixed across every arm

- Data: FineWeb-Edu `sample-10BT` at revision `87f09149...`, the same
  deterministic split, pre-tokenized into uint16 shards, one seeded window
  shuffle. Every run reads a prefix of the same permutation, so smaller budgets
  see a subset of larger ones.
- Optimizer and batch: the 214M recipe except the LR. AdamW, `min_lr_ratio`
  `0.1`, weight decay `0.1`, betas `0.9/0.95`, grad clip `1.0`, `393,216`
  tokens per step, bf16, compile. Each shape uses the LR from
  [Learning Rate Selection](#learning-rate-selection).
- Warmup: `200` steps for every arm (the 214M run used `450`). With 450, the
  smallest branch (L6 at `5x N`, 480 steps) would cool down inside its warmup.
- Loss metric: CE in fp32 on the first 10M validation tokens, read from
  validation shards and hash-checked against the 214M eval's
  `token_batch_sha256` (`1e1bb2aa...`). Every point, including the 214M one,
  uses the token windows behind `2.7756`.

## Learning Rate Selection

One LR for every size puts each size at a different distance from its own best
LR, which bends the fitted exponents. So each shape gets its own LR, picked by
a sweep at 0.5B tokens per run, seed 0. The grid is `6e-4`, `1.2e-3`, and
`2.4e-3` for each swept shape, plus `3e-4` for L6 from the original gate.

### Rules

1. Each shape uses the LR with the lowest validation CE.
2. If that LR is the highest one swept, add one run at `2x` and pick again. Do
   not extend a second time. Report a best LR that is still at the edge.
3. A `2x` run that beats the previous best by less than `0.005` nats is a tie,
   and ties go to the lower LR. A run that stops on a non-finite loss or
   gradient loses.
4. Shapes outside the sweep (the depth-vs-width arms) take the LR of the swept
   shape nearest in non-embedding params, which is L12.
5. The 214M shape (`d768-L30`) is swept but does not enter the fit. It measures
   how far `6e-4`, the 214M run's LR, is from the best LR at that shape.
6. The `study` stage refuses any run whose LR is more than `0.02` nats behind
   the best sweep row of its shape (or of the nearest swept shape).

### Result

Validation CE at 0.5B tokens. Rows:
[`lr-gate.json`](../results/scaling-study/lr-gate.json),
[`lr-sweep.json`](../results/scaling-study/lr-sweep.json),
[`lr-sweep-extension.json`](../results/scaling-study/lr-sweep-extension.json).

| Shape | `3e-4` | `6e-4` | `1.2e-3` | `2.4e-3` | `4.8e-3` | Selected |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| L6 (37.8M) | 3.7176* | 3.5476* | 3.4542* | 3.4171 | **3.4089** | `4.8e-3`, at the grid edge |
| L12 (75.5M) | | 3.4425 | 3.3674 | 3.3382 | 3.3365 | `2.4e-3` (tie, rule 3) |
| L18 (113.3M) | | 3.3888 | 3.3130 | **3.2983** | 3.3412 | `2.4e-3` |
| L30 (188.8M) | | 3.3389 | 3.2698 | **3.2519** | 3.2690 | `2.4e-3` |

\* Scored with TF32 matmuls. Every other cell is fp32. L6's top two LRs are
both fp32, and the TF32 cells trail them by `0.037` nats or more, so the scoring
path does not decide L6's pick.

At this fixed width, the best LR is lower for deeper models: at least `4.8e-3`
at L6, `2.4e-3` to `4.8e-3` at L12, and `2.4e-3` at L18 and L30, on a 2x grid
with one seed. At the 214M shape, `6e-4` is `0.087` nats behind the best LR.
Short runs favor high LRs, so that gap likely overestimates what `6e-4` cost the
full 10.23B-token run.

The selection cost about 3.9 B200-hours over 17 runs.

## Fit Protocol

- Form: `L(N, D) = E + A / N^alpha + B / D^beta` (Hoffmann et al. 2022,
  approach 3), with N = non-embedding params and D = training tokens.
- Fit: Huber loss (`delta = 1e-3`) on `log L`, minimized with L-BFGS (float64)
  in [`study/scaling_law.py`](../../src/esme_pretrain/study/scaling_law.py). The
  solver works on centered `log N` and `log D` and divides the loss by
  `delta^2`. Both change the conditioning, not the model or the optimum.
  Without them, L-BFGS stops early in a flat valley on synthetic data.
- Initialization: the Hoffmann grid (4,500 starts). All starts run 200 L-BFGS
  iterations as one batched problem, then the 4 best run alone to convergence,
  and the lowest objective wins. The batched screen is a heuristic. It found the
  same optimum as screens keeping 8 and 32 candidates on 6 synthetic datasets.
  Besiroglu et al. (2024) showed this approach is sensitive to optimizer
  details, which is why these settings are fixed before the study runs.
- Uncertainty: bootstrap over the 12 fit points (1,000 resamples, seed 0) gives
  a 90 percent interval for the 214M prediction. Each resample refits from the
  4 polished full-data solutions, not the full grid. A resample whose optimum
  lies in a basin the full data never reached could be missed.
- Check: on noiseless synthetic data from known parameters on this grid, the
  fit recovers each parameter to 1e-8 and the 214M prediction to 5e-12 nats
  ([`tests/test_scaling_law.py`](../../tests/test_scaling_law.py)). With noise
  `sigma = 0.003`, the 90 percent interval covered the true value in 9 of 10
  noise seeds.
- The fit uses no 214M data. The 214M value is read only after the prediction
  is committed to `docs/results/`.

## Predictions

1. **Instrument.** A from-scratch WSD run of L12 at `10x N` ends within `0.01`
   nats of the cooldown branch forked from the L12 main run at the same token
   count. If not, branched points are not trusted as full runs, and the report
   says so.
2. **Held-out prediction.** The fit predicts the 214M fixed-validation CE
   within `0.03` nats of `2.7756`, and the actual value falls inside the
   90 percent bootstrap interval.
3. **Depth vs width.** Deep-thin beats wide-shallow by at least `0.02` nats, by
   more than the seed-repeat gap. The middle arm falls between them.

### How the results are read

- **Known gaps between the 214M run and the study.** The 214M run used LR
  `6e-4`, no data shuffle, warmup `450`, and about `54 x N` tokens. The
  prediction extrapolates beyond the fit grid in both N (1.7x the largest fit
  size) and D. If prediction 2 fails, the report separates these explanations
  as far as the data allow. It does not tune the fit.
- **The 214M LR gap.** The `0.087`-nat cost of `6e-4` at 0.5B tokens likely
  overestimates the cost at 10.23B tokens, and it is never subtracted from
  prediction 2's residual. If the 214M actual
  is worse than predicted by less than this gap, the report reads the miss as
  consistent with the LR gap, not caused by it.
- **Horizon bias.** LRs were picked at 0.5B tokens: 13x N for L6, 6.6x for
  L12, 4.4x for L18. Main runs go to 40x N, which stretches the horizon 3x for
  L6 and 9x for L18. The best LR falls as training gets longer, so a 0.5B pick
  is likely more too-high for larger shapes, which would bias the fitted N
  exponent low. The report gives each arm's residuals at its `5x` branch (near
  the sweep horizon) and at `40x`. A residual trend that tracks shape is this
  bias.
- **Divergence.** A study run diverges if it stops on a non-finite loss, or if
  its 50-step mean train loss rises more than `0.5` nats above its running
  minimum after warmup. These thresholds have no external source and are fixed
  before any study data. A diverged arm reruns once at its shape's next lower
  swept LR, the one exception to rule 6, and the report names it.
- **Noise floor.** All 12 fit points use seed 0. Until the deep-thin repeat
  runs, the only noise estimate is the `0.005`-nat tie threshold.

## Compute

FLOPs from `BackboneConfig.flops_per_token` times training tokens. B200-hours
at 700 achieved TFLOP/s, the rate L6 measured in the sweep:

| Part | FLOPs | B200-hours |
| --- | ---: | ---: |
| Depth family: 3 main runs + 9 cooldown branches | 8.6e18 | 3.4 |
| Instrument check (L12 scratch at `10x`) | 5.4e17 | 0.2 |
| Depth vs width: 3 new runs (middle is reused) | 3.1e18 | 1.3 |
| **Total** | **1.23e19** | **~4.9** |

On the same formula, the 214M run itself used `1.60e19`. Arms run in
parallel on up to 4 B200s, and the L18 main run sets the wall time at about
2.5 hours. Each container has a timeout of twice its estimate plus 20 minutes.
If every container ran to its timeout, the total would be about 12 B200-hours.

## How the Protocol Is Enforced

- [`configs/scaling_study.json`](../../configs/scaling_study.json) holds every
  arm's shape, token budget, branch points, seed, and LR.
  [`study/plan.py`](../../src/esme_pretrain/study/plan.py) expands it into runs
  and refuses a branch that forks outside the main run's constant-LR phase.
- A cooldown branch (`PretrainConfig.branch_from`) may change only `max_steps`.
  A CPU test shows a branch equals a from-scratch run of its schedule bit for
  bit.
- The shard loader refuses shards whose vocab or tokenizer sha256 differ from
  the run's, and scoring refuses validation tokens that do not hash to
  `1e1bb2aa...`.
- [`launch/scaling_study.py`](../../src/esme_pretrain/launch/scaling_study.py)
  runs stages separately: `smoke`, `lr-gate`, `study`, and `heldout`.
  - Every arm is spawned into a detached app, each in its own single-use B200
    container, at most 4 at once. A client disconnect neither stops nor drops an
    arm. Rows are written to the volume as runs finish.
  - The `study` stage enforces rule 6 of the LR selection.
  - The `heldout` stage scores the 214M checkpoint only when the prediction
    file is committed in `HEAD`, predicts the 214M point, was fit on exactly the
    12 depth-family runs, and used the settings above. It also checks that the
    checkpoint has the base model's config and finished all 26,015 steps.
  - `--check-only` prices a stage. A real launch needs `--approved` and a spend
    cap that covers the worst case.
- `esme-pretrain scaling-fit` reads the study's rows directly, one point per
  run, and refuses an arm that is missing a planned run.

## Commands

```bash
# Data: 5B train and 20M validation tokens from sample-10BT files 000-006 at
# revision 87f09149, then upload to the study volume.
uv run --extra baselines esme-pretrain pretokenize --split validation --max-tokens 20000000 \
  --config configs/pretrain_214m_b200.json --tokenizer <tokenizer.json> \
  --output-dir data/processed/study-shards --corpus-parquet <file> ...
uv run --extra baselines esme-pretrain pretokenize --split train --max-tokens 5000000000 ...
uv run modal volume create esme-pretrain-study
uv run modal volume put esme-pretrain-study data/processed/study-shards /shards
uv run modal volume put esme-pretrain-study <tokenizer.json> /tokenizer.json

# Each stage: price it first, then launch.
uv run python scripts/modal_scaling_study.py --stage study --spend-cap-usd <cap> --check-only
uv run python scripts/modal_scaling_study.py --stage study --spend-cap-usd <cap> --approved

# Fit, then commit the prediction.
uv run modal volume get esme-pretrain-study /results/<study-run-id> runs/scaling-study/
uv run esme-pretrain scaling-fit --results-dir runs/scaling-study/<study-run-id> \
  --arm L6 --arm L12 --arm L18 --predict-non-embedding-params 188794368 \
  --predict-tokens 10229514240 --output docs/results/scaling-study/fit.json
git add docs/results/scaling-study/fit.json && git commit -m "..."

# Only then: score the 214M checkpoint on the same validation batches.
uv run python scripts/modal_scaling_study.py --stage heldout \
  --prediction docs/results/scaling-study/fit.json --spend-cap-usd <cap> --approved
```

## Results

- **Smoke** (run `smoke-51870ae-20261024T171214`): a main run checkpointed at
  its fork, and its branch resumed from step 16 at the matching data offset.
  Both were scored on validation batches hashing to `1e1bb2aa...`. L6 trained
  at `1.61M` tokens/sec, 31 percent MFU. Rows:
  [`smoke.json`](../results/scaling-study/smoke.json).
- **Learning-rate selection:** see [Result](#result).
- **Study, fit, and 214M prediction:** not yet run.

## Protocol Record

Every rule above was written before the data it governs. In order, all on
2026-10-24:

1. The card fixed the design, fit protocol, and predictions, with one LR
   (`6e-4`) for every size and an L6 LR gate: if `6e-4` trailed the best of
   `3e-4`, `6e-4`, and `1.2e-3` by more than `0.02` nats, the study would stop.
2. The gate stopped the study: `6e-4` trailed `1.2e-3` by `0.093` nats, at the
   grid edge. The author chose a per-shape sweep. Rules 1, 2, 4, 5, and 6 of the LR
   selection were added before any sweep run.
3. Every shape was best at `2.4e-3`, the grid edge. Rule 3, the divergence
   rule, the horizon-bias diagnostic, and the reading of the 214M LR gap were
   added before the `4.8e-3` runs.
4. The selected LRs were set on the study arms.
