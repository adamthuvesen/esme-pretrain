# Scale-Up Probe Notes

These notes record the throughput and sizing probes behind the current 214M B200
pretrain configuration. The current run card is
[`docs/run-cards/pretrain-214m-b200.md`](../run-cards/pretrain-214m-b200.md).

## Current Use

- The current training shape is `configs/pretrain_214m_b200.json`.
- The active model implementation is `src/esme_pretrain/modeling/backbone.py`.
- Probe configs in `PROBE_CONFIGS` remain useful for quick throughput comparisons.
- Full training decisions still require an approved run card and explicit spend cap.

## Probe Takeaways

- A deep/thin dense model with GQA, QK norm, tied embeddings, and byte-level BPE
  was the right risk-adjusted choice for the 214M run.
- The measured B200 profile gave the best cost-per-token among the tested options.
- FlashAttention-compatible dimensions and fused optimizer paths mattered more
  than small architectural novelty for this scale.
- The current run uses a 32,768-token vocabulary, context length 1024, and the
  30-layer / 768-width / 12-query-head / 4-KV-head shape recorded in the run card.

## Reproducibility

Use the current dry run to validate the launch surface without downloading data or
starting paid work:

```bash
uv run esme-pretrain pretrain-214m-b200 --config configs/pretrain_214m_b200.json --dry-run --json
```

Runtime artifacts, checkpoints, and raw datasets stay outside git.
