Critical
- None

High
- None

Medium
- None

Low
- **Checkpoint schema regression coverage was dropped**: `tests/test_pretrain_checkpoint.py:27`. The remaining checkpoint tests cover writer-to-loader round trips from the current `save_pretrain_checkpoint` path, but the loader still treats format `2` as the current on-disk schema (`src/esme_pretrain/training/checkpointing.py:26`). The branch removed the hand-built format-2 payload regression case, so future writer/loader schema drift could pass tests while breaking ignored accepted-run checkpoints used by eval, resume, or export. This is a coverage risk, not a current runtime failure.

Areas Reviewed & Found Clean
- Branch scope: reviewed `main...HEAD` across 25 changed paths; changes are mostly terminology cleanup and accepted-214M surface pruning.
- Correctness: no current behavior regressions found in model config validation, CLI status/doctor/data surfaces, tokenizer errors, Modal local rehearsal, checkpoint load/save logic, or status payload shape.
- DS/ML: 214M backbone shape remains locked at 213,960,192 parameters with GQA/QK-norm/z-loss; pretrain dry run still reports no launch blockers and preserves the $100 cap.
- Security/spend safety: no new secret handling, subprocess interpolation, remote writes, Modal launch bypass, or paid-work bypass found.
- Performance: no hot-path model/training algorithm changes found; throughput probe presets remain under `PROBE_CONFIGS`.
- Documentation drift: renamed `tiny`/`pilot`/`conventional`/`stage0` surfaces were swept; no stale changed-surface references found in the reviewed docs and code.
- Verification run: `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest`, `uv run esme-pretrain status --json`, and `uv run esme-pretrain pretrain-214m-b200 --config configs/pretrain_214m_b200.json --dry-run --json` all passed.

Summary
| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 1 |

Resolution
- Added `test_current_checkpoint_schema_loads_from_disk_payload` to cover loading a current-format checkpoint payload written directly to disk, without depending on the checkpoint writer.
