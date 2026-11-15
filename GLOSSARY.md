# Esme Bundle Contract

Terms for the export bundle that `esme-pretrain` writes and `esme-posttrain` and `llm-infer` read. This repo owns the contract terms. The other two repos point here. Full contract: `docs/bundle-format.md`.

## Contract terms

**Bundle**:
One directory from `esme-pretrain export`: `manifest.json`, `config.json`, `tokenizer.json`, `weights.pt`, `README.md`. Posttrain also writes bundles under the same format name.
_Avoid_: checkpoint (a training file, not a bundle), export

**Bundle format**:
The name `llm_pretrain_dense_v1`. One version covers `manifest.json` `schema_version`, `weights.pt` `format_version`, and the `_v1` suffix. They change together (`src/esme_pretrain/postrun/export_bundle.py::BUNDLE_SCHEMA_VERSION`).

**Manifest**:
`manifest.json` in a bundle: format, file sha256 hashes, provenance. Not the token shard manifest.

**Base bundle / chat bundle**:
A chat bundle has a `chat_template`. A base bundle, such as the esme-pretrain export, has none.

**Chat template**:
The `chat_template` object in a chat bundle manifest. It holds `roles` patterns and a `generation_prompt`. Posttrain writes it. Training and serving must render the same text.

**Chat template version**:
`esme_newline_v1` ends an assistant turn with a newline. `esme_newline_v2` (current) ends it with the text `<eos>`. v1 bundles still load. Upgrade with `esme-posttrain upgrade-chat-template`.

**EOS (`<eos>`)**:
The end-of-document and end-of-assistant-turn token. The text `<eos>` must encode to the one `<eos>` id.
_Avoid_: end token, stop token

**EOS ids**:
The stop set that serving uses: `eos_token_ids` and `decoding.eos_token_ids`. Both are written, and they must be equal. A base bundle has none and falls back to the tokenizer `<eos>`.

**Special tokens**:
The four locked tokens `<pad>`, `<bos>`, `<eos>`, `<unk>`. Export refuses a tokenizer that lacks any of them. Posttrain also requires `<eos>` to be special (`require_eos_text_is_eos_id` in the posttrain chat template export).

**add_special_tokens**:
The tokenizer encode flag. Chat bundles set it to `false`. With no value set, serving uses `true`.

**Extension fields**:
Optional manifest fields. Posttrain adds them. Pretrain writes none. `llm-infer` reads `chat_template`, `eos_token_ids`, `decoding.eos_token_ids`, and `model.id`/`model.name`. It does not read `model.stage` or `decoding.default_add_special_tokens`. Adding a field needs no version bump (`docs/bundle-format.md`).

## Pretraining terms

**Token shard manifest**:
The index of a pre-tokenized shard set (`src/esme_pretrain/data/token_shards.py::TokenShardManifest`). Different file from the bundle manifest.

**Packing**:
Documents joined with `<eos>` and cut into fixed windows of `context_length + 1`. There is no document attention mask.

**Run card**:
The typed spec for one paid run: config, budget, abort rules. The code reads it (`src/esme_pretrain/pretrain_config.py`). No run starts without one.

**Accepted run**:
The one pretrain profile that policy allows: `src/esme_pretrain/pretrain_run.py::AcceptedRun`. Its result is `Esme-214M-Base`, 26,015 steps (`docs/status.md`).

## Name drift

- "manifest" names two files: bundle `manifest.json` and the token shard manifest.
- `schema_version` and `format_version` are one version in two files.
- The `<eos>` text in the v2 template and the `<eos>` token id are two forms of one thing.
