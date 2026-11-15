# One bundle version in three places; a breaking change is a new format

Status: accepted

Two other repos read the bundle: `esme-posttrain` and `llm-infer`. A silent layout change breaks them without a clear error. So one version covers the whole contract. It appears as `schema_version` in `manifest.json`, `format_version` in `weights.pt`, and the `_v1` suffix in `llm_pretrain_dense_v1`. The constant is `src/esme_pretrain/postrun/export_bundle.py::BUNDLE_SCHEMA_VERSION`.

A new optional field needs no bump. Consumers ignore unknown fields. Renamed or removed fields, changed state-dict key names, a changed file set, or changed field meaning make a new format. Bump all three markers in one change. Never write a changed layout under the v1 name.

## Consequences

`src/esme_pretrain/postrun/bundle.py` rejects any version other than the one it supports. Posttrain adds fields under the v1 name, and `docs/bundle-format.md` lists them. The producer-side test is `tests/test_bundle_contract.py`.
