# Status

`Esme-214M-Base` is trained, evaluated, and exported. It is a 213,960,192-parameter
dense decoder trained on `10,229,514,240` FineWeb-Edu tokens over `26,015` steps
([run card](run-cards/pretrain-214m-b200.md)). Posttraining and serving work
from its export bundle in the [related repositories](../README.md#related-repositories).

## Experiments

| Experiment | State | Result |
| --- | --- | --- |
| [Base model pretraining](run-cards/pretrain-214m-b200.md) | Done | CE `2.7756`, `0.9005` bits per byte on the fixed validation set. Wins 6 of 7 0-shot tasks against Cerebras-GPT-256M and Pythia-160M. |
| [Throughput benchmark](run-cards/throughput-bench-b200.md) | Done | `1.57x` tokens/sec on one B200, MFU 18.6 to 29.1 percent. |
| [Scaling study](run-cards/scaling-study-b200.md): LR selection | Done | Deeper models want a lower LR: `4.8e-3` at 38M, `2.4e-3` at 113M and 189M. |
| [Scaling study](run-cards/scaling-study-b200.md): fit and 214M prediction | Next | Not run. |

## Next

Run the scaling study's `study` stage, fit the law, commit the 214M prediction,
and then score the base checkpoint with the `heldout` stage.

## Rules

- Paid compute (Modal, GPUs, W&B writes) needs a run card and explicit approval.
  Launchers refuse to start without `--approved` and a spend cap that covers
  their worst case.
- Dry runs never download data or start Modal.
- Run outputs, checkpoints, exports, W&B state, and `.env*` files stay out of git.

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run esme-pretrain status --json
uv run esme-pretrain pretrain-214m-b200 --config configs/pretrain_214m_b200.json --dry-run --json
```
