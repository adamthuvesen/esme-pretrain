# A chat template version never changes in place

Status: accepted

In 2026-11, post-training and serving rendered the assistant turn differently. Training ended it with the `<eos>` token id. Serving history ended it with a newline (`esme_newline_v1`). The model saw text in serving that it never saw in training.

The fix is `esme_newline_v2`. Its assistant pattern ends with the text `<eos>`, and serving tokenizes with `add_special_tokens: false`. This matches training only if the tokenizer encodes the text `<eos>` as the single `<eos>` id. So `tokenizer.json` must register `<eos>` in `added_tokens` as special. `esme-posttrain` refuses to export a tokenizer that fails this check (the `require_eos_text_is_eos_id` check).

A template change is a new template id, never an edit of an existing one. Old bundles keep loading. `esme-posttrain upgrade-chat-template --bundle-dir <dir>` rewrites a v1 `chat_template` to v2. It touches only `manifest.json`, which no `files.*.sha256` covers, so the bundle still verifies.

## Consequences

The template has one code owner in the posttrain repo. Its contract text lives in `docs/bundle-format.md`.
