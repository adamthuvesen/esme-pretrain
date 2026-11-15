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

## Downstream Extension Fields

`esme-pretrain` writes none of these fields. `esme-posttrain` adds them when it
exports a post-trained bundle under the same `llm_pretrain_dense_v1` name, as
the compatibility policy below allows. `llm-infer` reads them. This section
records both sides as the code stands. The code is the authority; fix this
section when it changes.

Producer: `esme-posttrain` `export/bundle_writer.py::write_dense_bundle`. Its
callers supply the stage fields: `export/dense_bundle.py::_stage_metadata` (the
DPO chat bundle) and `rl/bundle.py::write_rlvr_bundle` (the RLVR bundle).
`export/chat_template.py` owns the chat template object.

Consumer: `llm-infer` `llm_infer/model/runtime.py::_load_bundle_runtime` and the
`_dense_*` helpers it calls.

| Field | Written by esme-posttrain | Read by llm-infer |
| --- | --- | --- |
| `model.id`, `model.name`, `model.stage` | Every bundle. `stage` is `"dpo"` or `"rlvr"`. esme-pretrain writes `model: {format, family}` instead. | `_dense_model_id`: the served model id is `model.id`, else `model.name`, else top-level `model_id`, else the bundle directory name. `stage` is not read. |
| `chat_template` | DPO chat bundle only. The RLVR bundle omits it. | `_dense_chat_template`: top-level `chat_template`, else `tokenizer.chat_template`. It must be an object. |
| `eos_token_ids` | Every bundle: the `<eos>` id. | `_dense_eos_ids`, see below. |
| `decoding.eos_token_ids` | Every bundle: the same list as `eos_token_ids`. | `_dense_eos_ids`, see below. |
| `decoding.default_add_special_tokens` | Every bundle: `false`. | Not read. |
| `provenance` | Every bundle: training source, checkpoint hashes, run ids. Its keys differ by stage. | Not read. The runtime keeps the whole manifest as metadata. |

### Chat bundles

A bundle is a chat bundle when it has a `chat_template`. llm-infer requires:

- `chat_template.roles`: an object that maps each role in a served conversation
  to a pattern with one `{content}` placeholder. llm-infer renders each message
  with its role pattern and joins the results with no separator.
- `chat_template.generation_prompt`: a non-empty string, appended when a
  request asks for a generation prompt.
- EOS ids in `eos_token_ids` or `decoding.eos_token_ids`. Load fails when a chat
  bundle has neither.

Optional: `model`, `provenance`, and the `add_special_tokens` keys. llm-infer
does not read the other `chat_template` keys that esme-posttrain writes (`id`,
`format`, `example`).

### Chat template versions

The current template is `esme_newline_v2`. Its role patterns are
`system\n{content}\n`, `user\n{content}\n`, and `assistant\n{content}<eos>`, and
its generation prompt is `assistant\n`. Training ends each assistant turn with
the `<eos>` token id. The v2 assistant pattern ends with the literal text
`<eos>`, and llm-infer tokenizes the rendered text with
`add_special_tokens: false`. Served history matches training only when the
tokenizer encodes the text `<eos>` as the single `<eos>` id. That holds when
`tokenizer.json` registers `<eos>` as a special token in `added_tokens`. The
esme-pretrain tokenizer trainer registers `<pad>`, `<bos>`, `<eos>`, and
`<unk>` that way (`launch/modal_tokenizer.py`). esme-posttrain export refuses a
tokenizer that fails this check.

`esme_newline_v1` bundles still load. Their assistant pattern is
`assistant\n{content}\n`, so served history has a newline where training has
`<eos>`. `esme-posttrain upgrade-chat-template --bundle-dir <dir>` rewrites a
v1 bundle's `chat_template` to v2. It changes only `manifest.json`, which no
`files.*.sha256` entry covers, so the bundle still verifies. `--dry-run`
prints the diff and writes nothing.

### EOS ids

esme-posttrain writes the EOS list twice, at the top level and under
`decoding`, with equal values. llm-infer reads both. When both are present,
they must be equal lists of integers, or load fails. A non-empty list is the
stop set. With no non-empty list, a chat bundle fails at load, and any other
bundle uses the tokenizer's `<eos>` id. This is the case for an esme-pretrain
base bundle. A bundle with no EOS list and no `<eos>` token fails at load.

### add_special_tokens

llm-infer reads `add_special_tokens` from three places and uses the first one
that is set: `tokenizer.add_special_tokens`, then top-level
`add_special_tokens`, then `chat_template.add_special_tokens`. The value must
be a boolean. With none set, it uses `true`. esme-posttrain sets
`chat_template.add_special_tokens: false` in the chat bundle. It also writes
`decoding.default_add_special_tokens: false`, which llm-infer does not read. An
RLVR bundle has no `chat_template`, so llm-infer encodes its prompts with
`add_special_tokens: true`.

## config.json

Exactly the `BackboneConfig` fields: `name`, `vocab_size`, `context_length`,
`embedding_dim`, `layers`, `heads`, `feedforward_dim`, `kv_heads`,
`rope_theta`, `rms_norm_eps`, `tie_embeddings`, `qk_norm`, `z_loss_weight`,
`attention_kind`.

In v1, `attention_kind` is preserved as the serialized `"gqa"` architecture
tag. Only grouped-query attention is implemented by the current producer.

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
