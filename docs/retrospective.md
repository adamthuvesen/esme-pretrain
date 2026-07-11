# Esme, End to End

A 214M-parameter language model was trained from scratch, posttrained through
SFT, DPO, and verifier-rewarded RL, measured with a placebo control to locate
where the RL gains actually came from, and served at 95.6x a naive baseline.
The work spans five public repos connected by one versioned artifact
contract. This retrospective walks the chain stage by stage: what was
decided, what it took in tokens and compute, what the evidence shows, and
what would change in a second iteration.

The rule this document follows is the same one the repos follow: no claim
without an artifact. Every number below links to a committed doc, config, or
machine-written result record. Where evidence was lost or never recorded, the
gap is stated instead of papered over. Cost appears as decisions (which GPU,
what caps) and never as amounts; tokens and compute are the axis that
transfers.

The chain:

```
FineWeb-Edu sample-10BT
     |  data + tokenizer
Esme-214M-Base ........ baseline scorecard (vs Cerebras, Pythia)
     |  SFT, then DPO
Esme-214M-Chat ........ served by llm-infer (95.6x, gated)
     |  GRPO on Countdown-Lite
Esme-214M-RL .......... placebo-tested gains (grpo-decomp)
```

## The chain in tokens and compute

| Stage | Params | Tokens | Compute |
| --- | --- | --- | --- |
| Pretrain (Esme-214M-Base) | 213,960,192 | 10,229,514,240 train | ~1.3e19 FLOPs (6ND, computed) |
| SFT, multi-turn | same backbone | 38,205,562 supervised / 54,630,414 trained | small next to pretrain |
| DPO (Esme-214M-Chat) | same backbone | 122,880 pair updates; token count derived, see below | small |
| RLVR GRPO (Esme-214M-RL) | same backbone | 5,300,000 rollout-token budget per run | small |
| Six-seed placebo study | same backbone | 5.3M x 12 runs (6 real reward + 6 placebo) | small |

Reference points from the baseline scorecard: Cerebras-GPT-256M trained on
5.1B tokens, Pythia-160M on roughly 300B (about 22x Esme's compute).

Two things stand out in this table. First, the whole posttraining chain
(SFT, DPO, and RL together) rides on well under 1 percent of the pretrain
budget. Second, DPO is the one stage with no recorded token count. The run
processed 122,880 preference pairs (960 optimizer steps at an effective batch
of 128, two epochs over 61,440 pairs), each pair two sequences capped at
1,024 tokens, so the ceiling is about 252M token positions and the true count
is well below it. That number is derived from the
[committed config](https://github.com/adamthuvesen/esme-posttrain/blob/main/configs/esme-214m-chat-dpo.json)
because nothing recorded it at run time. Section 11 returns to this.

## 1. Data and tokenizer

The corpus is FineWeb-Edu `sample-10BT` at a pinned revision, streamed with a
quality filter, and split once: a deterministic hash of the source document id
sends 1 percent of documents to validation. Training, tokenizer training, and
post-run eval all share that split, so no stage can leak validation text into
another. The [config](../configs/pretrain_214m_b200.json) and
[run card](run-cards/pretrain-214m-b200.md) pin all of it.

The tokenizer is a digit-split byte-level BPE with a 32,768 vocab, trained on
a 50M-token prefix of the training split, with round-trip and coverage checks
required before use. Digit splitting makes numbers tokenize as single-digit
sequences, which matters for a model this small doing arithmetic later in the
chain. The choice was measured: a probe model was trained for each candidate,
and the losses were compared in a tokenizer lab that has since been retired. The
method survived; the
per-candidate loss table did not. No artifact records which alternatives were
scored, and that loss is the first entry in a pattern this document keeps
returning to.

## 2. Pretrain

`Esme-214M-Base` is a 30-layer, `d_model=768` dense decoder with grouped-query
attention (12 query heads, 4 KV heads), QK-norm, SwiGLU, tied embeddings, and
a z-loss, trained for one epoch over the corpus: 10,229,514,240 tokens in
26,015 optimizer steps. The deep-and-thin geometry follows what Qwen3,
MobileLLM, and SmolLM2 report for small models.
[`docs/architecture.md`](architecture.md) records the shape and, just as
useful, the rejected alternatives with reasons: MLA, MoE, multi-token
prediction, sliding-window attention, and logit soft-capping all have a
written "no" with a rationale.

The hardware decision was measured. The same training shape ran on H100, H200,
and B200; the [run card](run-cards/pretrain-214m-b200.md) states the decision
rule up front (B200 had to beat H100 throughput by about 58 percent to win on
cost per token) and B200 cleared it. A spend cap and abort rules were written
before launch and held. The launch itself requires an explicit `--approved`
flag and runs detached on Modal, so a laptop disconnect cannot kill training.

The run completed every step and stopped when the finite stream ran out
("data exhausted at step 26015"). Train loss went from 10.53 to 2.74,
validation tracked it, and throughput held near 215,714 tokens/sec on one
B200. The accepted numbers, including the post-run fixed-validation eval (CE
2.7756, 0.9001 bits per byte over 10M validation tokens), are quoted with
artifact hashes in the run card's
[Accepted Result](run-cards/pretrain-214m-b200.md) section, and the loss and
throughput series are plotted in the committed
[telemetry figures](../README.md#training-telemetry). One honesty note on
those figures: the MFU curve is computed from tokens/sec and FLOPs-per-token
at plot time. No raw MFU measurement exists.

Not everything survived. The acceptance eval and report were produced locally
after the run and later lost; they were regenerated in July 2026 from the
checkpoint, deterministically, and the regenerated bits per byte agreed with
an independent measurement from the baseline harness to about 0.001. The raw
GPU-smoke records behind the B200 decision are gone for good (recreating them
means renting three GPUs again), though the measured numbers live on in the
committed config. Losing instrument readings while keeping the conclusions is
survivable exactly once; section 11 makes it a rule instead of a habit.

## 3. Is the base model any good?

Every number so far compares Esme to itself. The baseline scorecard is the
external reference: `Esme-214M-Base` against
[Cerebras-GPT-256M](https://huggingface.co/cerebras/Cerebras-GPT-256M)
(a similar training budget) and
[Pythia-160M](https://huggingface.co/EleutherAI/pythia-160m) (about 22x the
compute), on identical inputs, deterministic fp32 forward passes, everything
pinned by revision or bundle hash.

The harness had to earn trust before producing a single Esme number. The
comparison code refuses to score Esme downstream until it reproduces
Cerebras's published 0-shot table; it matched every task within 0.002 against
a stated tolerance of 0.01. That ordering is enforced in the
[committed harness](../src/esme_pretrain/baselines/run.py), and the working
notes fixed the expectations before any result existed (beat Cerebras, lose
to Pythia). Those notes live with the run rather than in git, which is its
own small lesson about where to keep pre-registrations.

The expectation about Pythia was wrong. Esme won 6 of 7 downstream tasks
against both baselines, averaging 0.408 against Cerebras's 0.347 and Pythia's
0.365. The full table is in the
[README](../README.md#how-good-is-it), backed by the pinned
[config](../configs/baseline_eval.json). The shape of the one loss says as
much as the wins: LAMBADA tests fiction, FineWeb-Edu barely contains fiction,
and on bits per byte each model wins its home text (Esme 0.901 on FineWeb-Edu
validation, Pythia 0.902 on Pile test, with Esme at 1.283 there). Esme is an
educational-domain base model, and the scorecard says so plainly rather than
averaging the domain gap away.

Of everything in this project, this stage is the one to replicate untouched:
fix the expectations, validate the instrument against known answers, then
measure.

## 4. The artifact contract

Five repos exchange models as `llm_pretrain_dense_v1` bundles: five files,
sha256 for each, and one version declared in three places that move together
(the manifest's `schema_version`, the weights file's `format_version`, and
the `_v1` suffix in the name). [`docs/bundle-format.md`](bundle-format.md)
is the contract; a
[producer-side test](../tests/test_bundle_contract.py) pins the exact file
set, manifest fields, config keys, and state-dict key names, and both
consumers ([llm-infer](https://github.com/adamthuvesen/llm-infer),
[esme-posttrain](https://github.com/adamthuvesen/esme-posttrain)) reject a
declared version they do not support. When the contract landed, a fresh
export loaded in llm-infer with a maximum logit difference of 3e-08, and a
bundle edited to declare v2 was refused by both loaders.

The wrinkle worth telling: the contract was retrofitted after the full chain
had shipped, and the first local rebuild of the base bundle came out with an
empty provenance block because the run's metadata files were not sitting next
to the checkpoint at export time. Both were cheap to fix and would have been
cheaper on day one. Version the artifact format before the second repo
consumes it, and treat provenance as part of the export, since it silently
reflects whatever directory the exporter happens to see.

## 5. SFT and DPO

SFT turned the base model into a multi-turn chat model on an 85/15 mix of
smol-smoltalk and tulu-3-personas, both revision-pinned, with the
function-calling and hardest-reasoning subsets dropped as beyond 214M
capacity. Selection is mechanical: the checkpoint with the best matched
held-out response loss ships, and a small `no_robots` eval acts as an
out-of-distribution tripwire that can crash training but never picks the
winner. The accepted run selected step 6300 at a matched response loss of
1.35637 over 38.2M supervised tokens, quoted with artifact hashes in the
[run card](https://github.com/adamthuvesen/esme-posttrain/blob/main/run_cards/esme-214m-sft-multiturn.md).

DPO then polished the chat model into `Esme-214M-Chat` with one offline pass
over ultrafeedback_binarized: beta 0.5, two epochs, effective batch 128. The
accepted checkpoint (step 600) reached 0.674 preference accuracy over 959
held-out pairs, with the chosen-sequence likelihood staying above the
rejected one throughout, which was the collapse signal being watched. Numbers
and hashes are in that
[run card](https://github.com/adamthuvesen/esme-posttrain/blob/main/run_cards/esme-214m-chat-dpo.md)
too. The acceptance criteria at this scale are qualitative (repetition,
length, likelihood, and coherence proxies), on the written argument that
benchmark suites meant for much larger models say little at 214M.

The honest gap in this stage was chat_eval, and the first version of this
document misstated it. The eval had in fact run, on Modal on 2026-06-28,
against the accepted SFT and DPO checkpoints; what never happened was
retrieval. Its results sat on the Modal volume, unretrieved and uncommitted,
so "it was never run" (this document's original claim) was really "it
ran, and nobody went back for the numbers": the exact off-machine-artifact
failure section 9 warns about, live in this document's own claims. The
results were downloaded and pinned on 2026-07-10 in the
[run card's Chat Eval Result section](https://github.com/adamthuvesen/esme-posttrain/blob/main/run_cards/esme-214m-chat-dpo.md#chat-eval-result-sft-vs-dpo):
DPO reduces 3-gram repetition under both decoders with no degenerate length
shift, and no LLM judge is configured, so the record carries generation
metrics, not judge scores. The gap is closed; chat quality no longer rests
on preference accuracy and read transcripts alone.

## 6. RLVR: reinforcement learning against a verifier

The RL stage trained `Esme-214M-Chat` on Countdown-Lite, a constrained
arithmetic task where a deterministic Python verifier scores each completion
(using exact `Fraction` arithmetic, never `eval`) on a ladder from invalid
through well-formed to exactly solved. The objective is GRPO in its plain
form: group-normalized REINFORCE with a baseline plus a KL term. The
[run card](https://github.com/adamthuvesen/esme-posttrain/blob/main/run_cards/esme-214m-rl.md)
works through why there is no PPO-style clipping term: with one gradient step
per rollout batch, the importance ratio is identically 1, so a clipping term
would be dead code.

Under a 5.3M rollout-token budget per run, the accepted checkpoint moved
pass@1 from 3.33 to 16.67 percent, and the valid-expression rate from 5.83 to
99.38 percent. The committed docs are as specific about the limits as the
gains: all five newly solved tasks sit in the easy band, medium and hard
stayed at zero, and token entropy fell from 2.02 to 0.35. On fresh held-out
tasks the format transfers almost perfectly (99 percent valid) while exact
arithmetic at three numbers stays at zero even with quadruple the token
budget, per the committed
[transfer](https://github.com/adamthuvesen/esme-posttrain/blob/main/docs/rlvr-countdown-heldout-transfer.md)
and
[token-budget](https://github.com/adamthuvesen/esme-posttrain/blob/main/docs/rlvr-countdown-3number-tokenbudget.md)
studies. The first full run also died to a SIGTERM during its after-eval and
had to be redone; the excluded run is recorded in the committed
[study report](https://github.com/adamthuvesen/esme-posttrain/blob/main/studies/rlvr-placebo.report.md)
rather than quietly forgotten, and resume-from-artifacts is still unbuilt.

A sibling repo, [llm-rlvr](https://github.com/adamthuvesen/llm-rlvr), runs
the same discipline at 3B scale on text-to-SQL, where GRPO over an SFT
cold-start lifts Spider execution accuracy from 0.678 to 0.779 (3-seed mean,
committed result records). It is a separate substrate and its results stay in
its own README; the relevant part here is that the evaluation rules
(per-seed CIs, never pooling rollouts across seeds) are the same ones the
Esme chain uses.

## 7. Where the RL gains actually come from

RL curves go up. The question [grpo-decomp](https://github.com/adamthuvesen/grpo-decomp)
asks is what the increase is made of, and its instrument is a placebo: a
third training arm identical to the real one except the reward is drawn from
a seeded random number generator, blind to the completion. Any gain the
placebo shows is training-process artifact. Only the correct-versus-placebo
comparison is confirmatory; everything else is descriptive and labeled that
way.

Three studies, all committed as machine-written result records:

On GSM8K with Qwen2.5-Math-1.5B over six seeds, correct reward beats placebo
by 3.9 points of pass@1 (CI 2.3 to 5.6). But pass@8 coverage moves 0.7 points
with a CI spanning zero, and the per-problem decomposition attributes 0.0
percent of the gain to problems the base model could not already solve at
pass@8. GRPO made the model reliable at what it could already do. The
[findings doc](https://github.com/adamthuvesen/grpo-decomp/blob/main/results/FINDINGS.md)
also shows why six seeds: at three seeds the CI still straddled zero, and the
flashy seed-0 numbers regressed by nearly half.

On Countdown with a general Qwen2.5-1.5B, the same recipe finds 46.5 points
over placebo and 10.9 percent genuinely new capability. When the task has
headroom, the instrument detects it, which is what makes the GSM8K zero
believable.

On Esme itself, six real-reward seeds against six placebo seeds separate
valid-expression rate 85.4 versus 0.8 percent. At 214M, RL's dominant effect
is format acquisition, and the placebo proves the training process alone
produces almost none of it. The
[committed summary](https://github.com/adamthuvesen/grpo-decomp/blob/main/results/esme-countdown/sampled_multiseed_summary.json)
matches the esme-posttrain study report seed for seed, and the held-out task
fixture is pinned by a test on both sides of the repo pair.

The standing caveat is scope: the GSM8K result is within one model family.
The cross-family arm is named future work in every result file, and until it
runs, the claim stays fenced.

## 8. Serving

[llm-infer](https://github.com/adamthuvesen/llm-infer) serves Esme bundles
through a paged-KV engine with continuous batching, prefix caching,
speculative decode, and CUDA-graph decode. Its core rule is that throughput
only counts after the generated tokens match a full-recompute fp32 reference;
one benchmark cell was in fact forfeited over a 0.108-logit divergence,
slightly outside the documented tie tolerance.

The headline: 3,776.8 tokens/sec at 256 concurrent chat requests against
39.5 tokens/sec for naive sequential HF generation in the same container on
the same A100, which is 95.6x. (Full precision gives 95.71x; the headline
rounds both inputs, and this document quotes the repo's committed number.)
The one committed
[result record](https://github.com/adamthuvesen/llm-infer/blob/main/assets/esme-batch-curve.json)
backs the whole curve, and [docs/benchmark.md](https://github.com/adamthuvesen/llm-infer/blob/main/docs/benchmark.md)
documents the method.

Two decisions in this repo generalize. First, torch.compile was measured
against hand-captured CUDA graphs for decode and lost badly (batch-8 decode
118.7 versus 425.0 tokens/sec, with a 9x startup penalty), so the manual
graphs shipped; the alternative got a measurement, a doc, and a rejection,
same as the pretrain architecture choices. Second, the repo states its own
ceiling: vLLM on the same host is 3.2x faster at batch 1 and 6.6 to 13.7x at
high batch. The claim is the distance closed above a naive floor under a
correctness gate, and printing the ceiling keeps the 95.6x honest.

## 9. What would change next time

Three changes, each traceable to a specific failure above.

**Commit derived evidence at acceptance time.** grpo-decomp commits every
machine-written result record and was fully citable from a fresh clone the
whole way through. Nearly every other stage kept its accepted numbers only in
gitignored run directories, and this retrospective forced a round of
after-the-fact commits (the SFT, DPO, and pretrain accepted-result sections
all landed in July 2026, long after the runs). The fix is one convention: the
acceptance step writes its numbers into a committed record as part of
accepting.

**Treat off-machine artifacts as unaccepted.** The pretrain acceptance eval
existed only on a laptop and was lost; the GPU-smoke records existed only on
a laptop and are gone for good; the first bundle rebuild lost its provenance
because the metadata sat on a Modal volume instead of next to the checkpoint.
A run is done when its artifacts are downloaded and committed or hashed, and
that belongs in the run card's definition of done.

**Record every axis you will later compare on.** DPO has no token count and
had to be bounded from its config. No posttrain stage recorded compute. MFU
is reconstructed at plot time. Each is one line of code at run time and a
derivation with a footnote afterward.

If a fourth fits: run the measurements you build, and go back for what they
write. chat_eval turned out to have run after all: its results sat
unretrieved on a Modal volume while this document called it never-run (the
correction is in section 5), and they are now pinned in the posttrain run
card. The tokenizer lab's comparison table was never saved and stays lost.
Both started as process without numbers; only one could be recovered.

## 10. How this document was written

A claim-to-artifact inventory came first: roughly 70 rows across the five
repos, each claim mapped to the committed doc or result record that backs it,
with 16 gaps logged. This document was then written from the inventory, and
the gaps appear above at the point where they weaken a claim. Four
cross-repo agreement checks came out clean, including the six-seed placebo
numbers matching exactly between esme-posttrain and grpo-decomp and the
regenerated pretrain eval agreeing with the baseline harness to about 0.001.

The repos, in chain order:
[esme-pretrain](https://github.com/adamthuvesen/esme-pretrain),
[esme-posttrain](https://github.com/adamthuvesen/esme-posttrain),
[llm-infer](https://github.com/adamthuvesen/llm-infer),
[llm-rlvr](https://github.com/adamthuvesen/llm-rlvr),
[grpo-decomp](https://github.com/adamthuvesen/grpo-decomp).
