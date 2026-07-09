"""Orchestration for baseline gate and per-model baseline eval runs."""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from esme_pretrain import __version__
from esme_pretrain.baselines.bpb import slice_bpb
from esme_pretrain.baselines.config import (
    BaselineEvalConfig,
    BundleModel,
    FinewebValidationSlice,
    HFModel,
)
from esme_pretrain.baselines.harness import evaluate_gate, run_downstream
from esme_pretrain.baselines.models import EsmeBundleModel, build_eval_model, load_slice_texts
from esme_pretrain.torch import torch

RESULT_SCHEMA_VERSION = 1
GATE_SCHEMA_VERSION = 1


def run_gate(config: BaselineEvalConfig, *, output_path: Path) -> dict[str, Any]:
    """Score the gate model downstream and compare to the published table."""
    start = time.perf_counter()
    spec = config.models[config.gate.model]
    if not isinstance(spec, HFModel):
        raise ValueError("gate.model must be an hf model with published numbers")
    downstream = run_downstream(spec, config)
    gate = evaluate_gate(downstream["tasks"], config.gate)
    payload = {
        "schema_version": GATE_SCHEMA_VERSION,
        "config": str(config.config_path),
        "config_sha256": config.config_sha256,
        "device": config.device,
        "dtype": config.dtype,
        "downstream": downstream,
        "gate": gate,
        "passed": gate["passed"],
        "provenance": _tool_versions(),
        "runtime_seconds": round(time.perf_counter() - start, 6),
    }
    _write_json(output_path, payload)
    return payload


def run_baseline_eval(
    config: BaselineEvalConfig,
    *,
    model_key: str,
    output_path: Path,
    gate_path: Path | None = None,
) -> dict[str, Any]:
    """Score one configured model on every text slice and the downstream tasks."""
    start = time.perf_counter()
    if model_key not in config.models:
        known = ", ".join(sorted(config.models))
        raise ValueError(f"unknown model {model_key!r}; configured models: {known}")
    spec = config.models[model_key]

    gate_reference: dict[str, Any] | None = None
    if isinstance(spec, BundleModel):
        if gate_path is None:
            raise ValueError(
                "evaluating a bundle model requires --gate pointing at a passing "
                "baseline-gate result; run baseline-gate first"
            )
        gate_reference = _load_passing_gate(gate_path, config)

    model = build_eval_model(spec, max_context=config.max_context, device=config.device)

    bpb: dict[str, Any] = {}
    for slice_name in sorted(config.text_slices):
        slice_cfg = config.text_slices[slice_name]
        texts = load_slice_texts(slice_cfg)
        result = slice_bpb(
            model,
            texts,
            slice_name=slice_name,
            batch_size=config.bpb_batch_size,
            device=config.device,
        )
        entry = asdict(result)
        entry["source"] = _slice_source(slice_cfg)
        bpb[slice_name] = entry

    eval_model = model if isinstance(model, EsmeBundleModel) else None
    downstream = run_downstream(spec, config, eval_model=eval_model)

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "config": str(config.config_path),
        "config_sha256": config.config_sha256,
        "model": {"name": model_key, **model.provenance},
        "device": config.device,
        "dtype": config.dtype,
        "max_context": config.max_context,
        "context_length": model.context_length,
        "bpb_batch_size": config.bpb_batch_size,
        "bpb": bpb,
        "downstream": downstream,
        "gate": {
            "required": isinstance(spec, BundleModel),
            "gate_path": None if gate_path is None else str(gate_path),
            "passed": None if gate_reference is None else gate_reference["passed"],
        },
        "tolerances": config.tolerances,
        "provenance": _tool_versions(),
        "runtime_seconds": round(time.perf_counter() - start, 6),
    }
    _write_json(output_path, payload)
    return payload


def _load_passing_gate(gate_path: Path, config: BaselineEvalConfig) -> dict[str, Any]:
    if not gate_path.exists():
        raise ValueError(f"gate result does not exist: {gate_path}")
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"gate result is not valid JSON: {gate_path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"gate result must be a JSON object: {gate_path}")
    if payload.get("config_sha256") != config.config_sha256:
        raise ValueError(
            "gate result was produced with a different baseline eval config; "
            "re-run baseline-gate against the current config"
        )
    if payload.get("passed") is not True:
        raise ValueError(
            "gate did not pass: the harness has not reproduced the published "
            "numbers, so no Esme result can be produced"
        )
    return payload


def _slice_source(slice_cfg: FinewebValidationSlice | Any) -> dict[str, Any]:
    if isinstance(slice_cfg, FinewebValidationSlice):
        return {
            "kind": "fineweb_edu_validation",
            "pretrain_config": str(slice_cfg.pretrain_config),
            "document_budget": slice_cfg.document_budget,
        }
    return {
        "kind": "hf_dataset",
        "source": slice_cfg.source,
        "subset": slice_cfg.subset,
        "split": slice_cfg.split,
        "revision": slice_cfg.revision,
        "text_field": slice_cfg.text_field,
        "document_budget": slice_cfg.document_budget,
    }


def _tool_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "esme_pretrain": __version__,
        "torch": getattr(torch, "__version__", None),
    }
    for module_name in ("transformers", "lm_eval", "datasets", "tokenizers"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            versions[module_name] = None
            continue
        versions[module_name] = getattr(module, "__version__", None)
    return versions


def _write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
