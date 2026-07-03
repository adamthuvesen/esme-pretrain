# 214M B200 Hardware Measurements

These internal notes preserve the hardware measurements behind the `B200`
selection for `Esme-214M-Base`. The public run shape is
[`docs/run-cards/pretrain-214m-b200.md`](../run-cards/pretrain-214m-b200.md).

## Selected Profile

`B200` is the selected profile. H200 cleared the throughput threshold over
H100! but only barely improved cost per token. B200 cleared the threshold and
had the lowest measured cost per token.

| GPU | Measurement record | Micro batch | Grad accum | Steady tokens/sec | Peak memory | $/1B tokens | 10.23B projected cost | Loss | Checkpoint/resume |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| H100! | `runs/pretrain-214m-b200/gpu-smoke-h100bang.json` | 24 | 16 | 165,070 | 40.69 GB | 6.65 | 68.00 | finite, 10.5629 -> 10.3786 | ok |
| H200 | `runs/pretrain-214m-b200/gpu-smoke-h200.json` | 24 | 16 | 191,391 | 40.69 GB | 6.60 | 67.55 | finite, 10.5629 -> 10.3784 | ok |
| B200 | `runs/pretrain-214m-b200/gpu-smoke-b200.json` | 24 | 16 | 290,838 | 40.69 GB | 5.97 | 61.06 | finite, 10.5629 -> 10.3776 | ok |

Actual measured setup spend was `$0.8583`.

## Long-Job Safety

- Full launch command uses `modal run --detach`.
- `scripts/modal_pretrain.py` uses `run_pretrain_launch.spawn(...).get()` for
  the remote training call instead of `.remote(...)`.
- Configured timeout is `24h`, at Modal's per-function maximum.
- Selected B200 projection is `9.77h`, under the timeout.
- Checkpoint/resume remains on the Modal Volume. The full run writes periodic
  checkpoints and `_resume_checkpoint()` resumes from the latest checkpoint.
- Retries remain intentionally disabled in config (`allow_retries=false`)
  because the selected measured B200 run fits inside one 24h function. If a
  future selected profile exceeds 24h, the full launch should stay blocked
  until chunked retry/resume orchestration is added.

## Evidence Paths

- Config: `configs/pretrain_214m_b200.json`
- Run card: `docs/run-cards/pretrain-214m-b200.md`
- Dry-run evidence path: `runs/pretrain-214m-b200/dry-run-214m.json`
- H100! record: `runs/pretrain-214m-b200/gpu-smoke-h100bang.json`
- H200 record: `runs/pretrain-214m-b200/gpu-smoke-h200.json`
- B200 record: `runs/pretrain-214m-b200/gpu-smoke-b200.json`
- Local dress rehearsal evidence path:
  `runs/pretrain-214m-b200/local-dress-rehearsal/`
- Local tokenizer evidence path:
  `runs/pretrain-214m-b200/local-tokenizer-smoke/`

The dry-run must remain `will_download_data=false` and
`will_start_modal_job=false`. The full command must remain approval-gated with
`--approved`.
