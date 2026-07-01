from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esme_pretrain.modeling.pretrain_checkpoint import load_pretrain_checkpoint
from esme_pretrain.postrun.eval_checkpoints import file_sha256
from esme_pretrain.torch import torch

EXPORT_FORMAT_VERSION = 1
CANONICAL_BUNDLE_FORMAT = "llm_pretrain_dense_v1"
LLM_INFER_TARGET = "llm-infer"
EXPORT_SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_path: Path
    tokenizer_path: Path
    output_dir: Path
    export_format: str = LLM_INFER_TARGET


def export_checkpoint(config: ExportConfig) -> dict[str, Any]:
    if config.export_format != LLM_INFER_TARGET:
        raise ValueError(f"--format must be {LLM_INFER_TARGET}")
    if not config.checkpoint_path.exists():
        raise ValueError(f"checkpoint does not exist: {config.checkpoint_path}")
    if not config.tokenizer_path.exists():
        raise ValueError(f"tokenizer does not exist: {config.tokenizer_path}")

    loaded = load_pretrain_checkpoint(config.checkpoint_path)
    _validate_export_tokenizer(config.tokenizer_path, loaded.config.vocab_size)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    config_payload = loaded.config.to_dict()
    llm_infer_config = _llm_infer_model_config(config_payload)
    weights_payload = {
        "format_version": EXPORT_FORMAT_VERSION,
        "format": CANONICAL_BUNDLE_FORMAT,
        "key_format": CANONICAL_BUNDLE_FORMAT,
        "metadata": {
            "key_format": CANONICAL_BUNDLE_FORMAT,
            "target": config.export_format,
            "state_dict_key": "dense_backbone",
        },
        "state_dict_key": "dense_backbone",
        "state_dict": loaded.model.state_dict(),
        "model_config": config_payload,
        "llm_infer_config": llm_infer_config,
        "checkpoint_step": loaded.step,
        "source_checkpoint": str(config.checkpoint_path),
        "source_checkpoint_sha256": file_sha256(config.checkpoint_path),
    }
    weights_path = config.output_dir / "weights.pt"
    torch.save(weights_payload, weights_path)

    config_path = config.output_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True), encoding="utf-8")
    tokenizer_out = config.output_dir / "tokenizer.json"
    shutil.copyfile(config.tokenizer_path, tokenizer_out)

    inferred = _sibling_run_metadata(config.checkpoint_path)
    manifest = {
        "schema_version": 1,
        "format": CANONICAL_BUNDLE_FORMAT,
        "target": config.export_format,
        "weights_format": CANONICAL_BUNDLE_FORMAT,
        "model_family": "DenseBackbone",
        "model": {"format": CANONICAL_BUNDLE_FORMAT, "family": "DenseBackbone"},
        "tokenizer": {"path": "tokenizer.json", "format": "tokenizers-json"},
        "checkpoint_step": loaded.step,
        "source_checkpoint": str(config.checkpoint_path),
        "source_checkpoint_sha256": weights_payload["source_checkpoint_sha256"],
        "files": {
            "config": {"path": "config.json", "sha256": file_sha256(config_path)},
            "tokenizer": {"path": "tokenizer.json", "sha256": file_sha256(tokenizer_out)},
            "weights": {"path": "weights.pt", "sha256": file_sha256(weights_path)},
            "readme": {"path": "README.md"},
        },
        "model_config": config_payload,
        "llm_infer_config": llm_infer_config,
        "run_metadata": inferred,
    }
    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    readme_path = config.output_dir / "README.md"
    readme_path.write_text(_readme(manifest), encoding="utf-8")
    manifest["files"]["readme"]["sha256"] = file_sha256(readme_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _llm_infer_model_config(config: dict[str, Any]) -> dict[str, Any]:
    kv_heads = config["heads"] if config["kv_heads"] is None else config["kv_heads"]
    return {
        "format": CANONICAL_BUNDLE_FORMAT,
        "name": config["name"],
        "vocab_size": config["vocab_size"],
        "context_length": config["context_length"],
        "hidden_size": config["embedding_dim"],
        "intermediate_size": config["feedforward_dim"],
        "num_hidden_layers": config["layers"],
        "num_attention_heads": config["heads"],
        "num_key_value_heads": kv_heads,
        "rms_norm_eps": config["rms_norm_eps"],
        "rope_theta": config["rope_theta"],
        "tie_word_embeddings": config["tie_embeddings"],
        "attention_kind": config["attention_kind"],
        "qk_norm": config["qk_norm"],
        "z_loss_weight": config["z_loss_weight"],
    }


def _validate_export_tokenizer(tokenizer_path: Path, vocab_size: int) -> None:
    try:
        from tokenizers import Tokenizer
    except ModuleNotFoundError as error:
        raise ValueError(
            "tokenizers is required to validate tokenizer.json before export"
        ) from error

    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as error:  # noqa: BLE001 - tokenizers raises several parse errors.
        raise ValueError(f"tokenizer is not a valid tokenizers JSON: {tokenizer_path}") from error
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(
            "tokenizer vocab size does not match checkpoint config: "
            f"{tokenizer.get_vocab_size()} != {vocab_size}"
        )
    missing = [token for token in EXPORT_SPECIAL_TOKENS if tokenizer.token_to_id(token) is None]
    if missing:
        raise ValueError(f"tokenizer is missing required special tokens: {missing}")


def _sibling_run_metadata(checkpoint_path: Path) -> dict[str, Any]:
    run_dir = checkpoint_path.parent
    metadata: dict[str, Any] = {}
    skipped: list[str] = []
    for name in ("run-summary.json", "cost.json", "launch-status.json"):
        path = run_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            skipped.append(f"{name}: {error}")
            continue
        if isinstance(payload, dict):
            metadata[name] = payload
        else:
            skipped.append(f"{name}: expected object, got {type(payload).__name__}")
    if skipped:
        metadata["_skipped_run_metadata"] = skipped
    return metadata


def _readme(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# llm-infer Export Bundle",
            "",
            "This bundle is the correctness-first DenseBackbone export from esme-pretrain.",
            "",
            f"- Bundle format: `{manifest['format']}`",
            f"- Target adapter: `{manifest['target']}`",
            f"- Weights format: `{manifest['weights_format']}`",
            f"- Checkpoint step: `{manifest['checkpoint_step']}`",
            f"- Source checkpoint SHA256: `{manifest['source_checkpoint_sha256']}`",
            "",
            "Files:",
            "",
            "- `manifest.json`: bundle metadata and hashes",
            "- `config.json`: native esme-pretrain DenseBackbone config",
            "- `tokenizer.json`: tokenizer artifact copied byte-for-byte",
            "- `weights.pt`: torch-saved DenseBackbone state dict and metadata",
            "",
        ]
    )
