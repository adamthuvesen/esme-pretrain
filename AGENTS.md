# esme-pretrain - Agent Instructions

`esme-pretrain` is public at `github.com/adamthuvesen/esme-pretrain`. It owns
base-model pretraining from raw text through checkpoint eval and export.

## Read First

| Work type | Read |
| --- | --- |
| Current state, accepted run, next step | `docs/status.md` |
| Model, data, training, eval, export design | `docs/architecture.md` |
| Export bundle contract and version policy | `docs/bundle-format.md` |
| 214M B200 config, budget, artifacts, abort rules | `docs/run-cards/pretrain-214m-b200.md` |
| Scaling study: design, predictions, fit method, commands | `docs/run-cards/scaling-study-b200.md` |
| GPU selection and Modal safety | `docs/run-cards/pretrain-214m-b200.md` (Hardware, Modal Safety) |
| CLI surface and quick commands | `README.md` |

If docs and code disagree, fix the stale doc in the same change.

## Workflow

- Use `uv run ...` for project commands. Run `make check` before you commit.
- Paid work (Modal, GPU, FineWeb-Edu or ClimbMix streaming, W&B writes, paid APIs) needs
  an explicit run card and chat approval of the exact command, hardware, and cost cap
  before you pass `--approved`. The code enforces the flag and the caps, not the approval.

## Code Rules

- Use clear domain names: corpus, tokenizer, packed tokens, checkpoint, run card, eval, export.
- Make data loss, skipped records, malformed artifacts, resume drift, and config mismatches fail loudly.
- Put run card constraints in the typed config (`pretrain_config.py`, `data/corpus_config.py`),
  not in launcher code or prose.
- Keep post-run eval deterministic: same validation token batches for every checkpoint.
- Keep `llm-infer` export correctness-first: manifest, config, tokenizer, weights, hashes, and provenance must match the source checkpoint.
- Avoid trivializing small models or fixtures. Use small, rehearsal, fixture, or synthetic fixture as appropriate.
- Production model code is `modeling/backbone.py` (`DenseBackbone` + `BackboneConfig`).

## Gates

```bash
make check    # Ruff, strict basedpyright, import-linter, fast offline CPU tests; first CI job
make fix      # ruff format and safe lint fixes
make slow     # slow offline tests (statistical fits); second CI job
uv run esme-pretrain pretrain-214m-b200 --config configs/pretrain_214m_b200.json --dry-run --json
```

Prove changes with `make verify` (`verify-esme-pretrain` skill at `.agents/skills/verify-esme-pretrain/`).

## Enforced Rules

`tests/test_repo_rules.py` fails when this section is missing, a row cites no `path::name` check or make target, or a cited file does not define the cited name.

| Rule | Enforced by | Added |
| --- | --- | --- |
| Every corpus recipe has a data audit run card before any run reads it | `src/esme_pretrain/data/corpus_config.py`, `tests/test_pretrain_214m_b200.py::test_pretrain_config_rejects_unpinned_dataset_revision`, `tests/test_repo_rules.py::test_every_audited_corpus_recipe_cites_an_existing_run_card` | 2026-11-06 |
| A pretrain run card stays within the $100 absolute spend cap, with no retries | `src/esme_pretrain/pretrain_config.py`, `tests/test_pretrain_214m_b200.py::test_run_card_schema_enforces_the_spend_cap` | 2026-11-06 |
| Paid launchers refuse without the approval flag and a cap that covers the worst case | `src/esme_pretrain/launch/budget.py`, `tests/test_launch_approval.py::test_every_paid_launch_asks_for_approval_before_modal`, `tests/test_scaling_study.py::test_study_stage_refuses_a_spend_cap_below_its_worst_case` | 2026-10-24 |
| Study owns its protocol without CLI, launch, or post-run imports | `make imports`, `pyproject.toml::Study owns its protocol without CLI, launch, or post-run dependencies` | 2026-11-04 |
| Launch never imports CLI policy | `make imports`, `pyproject.toml::Launch must not depend on CLI policy` | 2026-11-04 |
| Keep modeling independent, and keep data and training below operation policy | `make imports`, `pyproject.toml::Modeling and torch are independent of data and operations`, `pyproject.toml::Artifact and data domains do not depend on operations`, `pyproject.toml::Training does not depend on operation policy` | 2026-11-04 |
| Type-check every production module, script, and test in strict mode | `make types`, `pyproject.toml::[tool.basedpyright]` | 2026-11-04 |
| Fit complete planned arms with matching run and evaluation identities | `src/esme_pretrain/study/results.py`, `tests/test_study_results.py::test_partial_rows_cannot_declare_themselves_complete`, `tests/test_study_results.py::test_fit_refuses_incomparable_evaluation_identity` | 2026-11-04 |
| Apply the registered LR gate only to complete, finite, comparable evidence | `src/esme_pretrain/study/lr_gate.py`, `tests/test_study_evidence.py::test_lr_gate_rejects_invalid_evidence` | 2026-11-04 |
| Use validated study rows for heldout predictions | `src/esme_pretrain/study/prediction.py`, `tests/test_study_evidence.py::test_heldout_requires_matching_evidence` | 2026-11-04 |
| Keep run outputs, checkpoints, data, W&B state, and `.env*` out of git | `.gitignore`, `tests/test_repo_rules.py::test_no_run_outputs_or_data_are_tracked` | 2026-10-24 |
| The verify drives must observe FAIL against a wrong expectation | `.agents/skills/verify-esme-pretrain/verify::control`, `make verify` | 2026-11-06 |
| Process calls, network clients, Modal, and W&B stay in their named owner files | `make lint`, `pyproject.toml::[tool.ruff.lint.flake8-tidy-imports.banned-api]`, `pyproject.toml::[tool.ruff.lint.per-file-ignores]`, `tests/test_repo_rules.py::test_ban_owners_use_only_the_bans_they_own` | 2026-11-07 |
| No `# pyright:` file comment weakens strict mode, no `noqa` silences an import ban or is bare, and every inline ignore gives a reason | `tests/test_strict_mode.py::test_no_file_downgrades_type_checking_mode`, `pyproject.toml::[tool.ruff.lint]` | 2026-11-07 |
