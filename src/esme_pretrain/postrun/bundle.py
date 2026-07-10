from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esme_pretrain.modeling.backbone import BackboneConfig, DenseBackbone
from esme_pretrain.postrun.eval_checkpoints import file_sha256
from esme_pretrain.postrun.export_bundle import BUNDLE_SCHEMA_VERSION, CANONICAL_BUNDLE_FORMAT
from esme_pretrain.torch import torch

REQUIRED_BUNDLE_FILES = ("weights.pt", "config.json", "tokenizer.json", "manifest.json")
HASHED_BUNDLE_FILES = ("config", "tokenizer", "weights")


@dataclass(frozen=True)
class LoadedBundle:
    bundle_dir: Path
    model: DenseBackbone
    config: BackboneConfig
    tokenizer_path: Path
    manifest: dict[str, Any]
    checkpoint_step: int
    weights_sha256: str


def load_bundle(bundle_dir: Path, *, device: str = "cpu") -> LoadedBundle:
    """Load an exported llm-infer bundle back into a DenseBackbone.

    Every integrity failure raises ValueError: missing files, manifest hash
    mismatches, unknown bundle format, unsupported schema version, and config
    drift between config.json, manifest.json, and weights.pt.
    """
    if not bundle_dir.is_dir():
        raise ValueError(f"bundle directory does not exist: {bundle_dir}")
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle_dir / name).exists()]
    if missing:
        raise ValueError(f"bundle is missing required files: {missing} in {bundle_dir}")

    manifest = _read_json(bundle_dir / "manifest.json")
    if manifest.get("format") != CANONICAL_BUNDLE_FORMAT:
        raise ValueError(
            f"bundle format is not {CANONICAL_BUNDLE_FORMAT}: {manifest.get('format')!r}"
        )
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema_version: {manifest.get('schema_version')!r} "
            f"(this loader supports {BUNDLE_SCHEMA_VERSION})"
        )
    file_hashes = _verify_manifest_hashes(bundle_dir, manifest)

    config_payload = _read_json(bundle_dir / "config.json")
    if manifest.get("model_config") != config_payload:
        raise ValueError("manifest model_config does not match config.json")
    config = BackboneConfig.from_dict(config_payload)

    weights = torch.load(bundle_dir / "weights.pt", map_location="cpu", weights_only=False)
    if not isinstance(weights, dict) or "state_dict" not in weights:
        raise ValueError(f"weights.pt does not contain a state_dict: {bundle_dir}")
    if weights.get("format_version") != BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported weights.pt format_version: {weights.get('format_version')!r} "
            f"(this loader supports {BUNDLE_SCHEMA_VERSION})"
        )
    if weights.get("model_config") != config_payload:
        raise ValueError("weights.pt model_config does not match config.json")

    model = DenseBackbone(config)
    model.load_state_dict(weights["state_dict"])
    model = model.to(torch.device(device)).float()
    model.eval()
    return LoadedBundle(
        bundle_dir=bundle_dir,
        model=model,
        config=config,
        tokenizer_path=bundle_dir / "tokenizer.json",
        manifest=manifest,
        checkpoint_step=int(weights.get("checkpoint_step", -1)),
        weights_sha256=file_hashes["weights"],
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"bundle file is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"bundle file must contain a JSON object: {path}")
    return payload


def _verify_manifest_hashes(bundle_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"manifest is missing the files section: {bundle_dir}")
    verified: dict[str, str] = {}
    for name in HASHED_BUNDLE_FILES:
        entry = files.get(name)
        if not isinstance(entry, dict) or "path" not in entry or "sha256" not in entry:
            raise ValueError(f"manifest files.{name} must record path and sha256")
        actual = file_sha256(bundle_dir / str(entry["path"]))
        if actual != entry["sha256"]:
            raise ValueError(
                f"bundle file hash mismatch for {entry['path']}: "
                f"manifest {entry['sha256']} != actual {actual}"
            )
        verified[name] = actual
    return verified
