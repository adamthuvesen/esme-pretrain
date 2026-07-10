# Export Bundle Format: `llm_pretrain_dense_v1`

This is the artifact contract between `esme-pretrain` (producer) and the
downstream loaders in `llm-infer` and `esme-posttrain` (consumers). A bundle is
one directory written by `esme-pretrain export`. The producer-side contract
test is [`tests/test_bundle_contract.py`](../tests/test_bundle_contract.py);
`llm-infer` covers the consumer side with its own loader and parity tests.

## Version

One version number covers the whole contract, written in three places that
move together:

- `manifest.json` `schema_version: 1`
- `weights.pt` `format_version: 1`
- the `_v1` suffix in the format name `llm_pretrain_dense_v1`

The constant is `BUNDLE_SCHEMA_VERSION` in
[`postrun/export_bundle.py`](../src/esme_pretrain/postrun/export_bundle.py).

## Files

| File | Contents |
| --- | --- |
| `manifest.json` | Bundle metadata, file hashes, provenance |
| `config.json` | Native `BackboneConfig` fields as JSON |
| `tokenizer.json` | `tokenizers` JSON file, copied byte-for-byte from the run |
| `weights.pt` | torch-saved payload with the state dict and metadata |
| `README.md` | Human-readable summary generated from the manifest |

No other files. The tokenizer must have vocab size matching `config.json` and
the special tokens `<pad>`, `<bos>`, `<eos>`, `<unk>`; export refuses to write
a bundle otherwise.

## manifest.json

Required fields:

- `schema_version`: `1`
- `format`: `"llm_pretrain_dense_v1"`
- `tokenizer`: `{"path": "tokenizer.json", "format": "tokenizers-json"}`
- `checkpoint_step`, `source_checkpoint`, `source_checkpoint_sha256`
- `model_config`: byte-identical to the contents of `config.json`
- `files`: `config`, `tokenizer`, `weights`, and `readme` entries, each with
  `path` and `sha256`

The manifest also carries `llm_infer_config` (a consumer-friendly view of the
architecture fields) and `run_metadata` (run summary, cost, and launch status
JSONs found next to the source checkpoint).

## config.json

Exactly the `BackboneConfig` fields: `name`, `vocab_size`, `context_length`,
`embedding_dim`, `layers`, `heads`, `feedforward_dim`, `kv_heads`,
`rope_theta`, `rms_norm_eps`, `tie_embeddings`, `qk_norm`, `z_loss_weight`,
`attention_kind`.

## weights.pt

A torch-saved dict with `format_version: 1`, `format`, `metadata.key_format`
(both the format name), `state_dict`, `model_config` (matching `config.json`),
`checkpoint_step`, and the source checkpoint path and sha256.

State-dict key names are part of the contract; `llm-infer` resolves them by
name:

- `token_embedding.weight`, `final_norm.weight`, `lm_head.weight` (present
  even when embeddings are tied)
- per layer `i`: `blocks.{i}.attention_norm.weight`,
  `blocks.{i}.attention.wq/wk/wv/wo.weight`,
  `blocks.{i}.attention.q_norm/k_norm.weight` (QK-norm models),
  `blocks.{i}.feedforward_norm.weight`,
  `blocks.{i}.feedforward.w_gate/w_up/w_down.weight`

## Compatibility Policy

- Adding a new optional field to `manifest.json` or the `weights.pt` payload
  is allowed without a version bump. Consumers must ignore fields they do not
  know.
- Any change that would break a v1 consumer is a new format: renamed or
  removed fields, changed state-dict key names, a changed file set, or changed
  semantics of an existing field. Bump all three version markers in one
  change (`llm_pretrain_dense_v2`, `schema_version: 2`, `format_version: 2`)
  and never write a changed layout under the v1 name.
- Consumers reject versions they do not support. `load_bundle` in this repo
  requires `schema_version` and `format_version` to equal `1` exactly.
  `llm-infer` rejects a bundle that declares a version it does not support and
  accepts bundles that predate the version fields.
- Every load verifies the manifest sha256 of `config.json`, `tokenizer.json`,
  and `weights.pt`; an altered file fails loudly.
