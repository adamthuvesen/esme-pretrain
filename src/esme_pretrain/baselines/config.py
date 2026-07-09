from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from esme_pretrain.launch.validators import expect_object, require_keys

FINEWEB_SLICE_KIND = "fineweb_edu_validation"
HF_DATASET_SLICE_KIND = "hf_dataset"
BUNDLE_MODEL_KIND = "bundle"
HF_MODEL_KIND = "hf"
SUPPORTED_DTYPE = "float32"
SUPPORTED_HARNESS = "lm-eval"


@dataclass(frozen=True)
class FinewebValidationSlice:
    name: str
    pretrain_config: Path
    document_budget: int


@dataclass(frozen=True)
class HFDatasetSlice:
    name: str
    source: str
    subset: str | None
    split: str
    revision: str
    text_field: str
    document_budget: int


@dataclass(frozen=True)
class BundleModel:
    name: str
    path: Path


@dataclass(frozen=True)
class HFModel:
    name: str
    repo: str
    revision: str


@dataclass(frozen=True)
class DownstreamConfig:
    harness: str
    version: str
    tasks: tuple[str, ...]
    num_fewshot: int
    batch_size: int


@dataclass(frozen=True)
class GateConfig:
    model: str
    published: dict[str, float]
    published_average: float
    per_task_tolerance: float


@dataclass(frozen=True)
class BaselineEvalConfig:
    config_path: Path
    config_sha256: str
    device: str
    dtype: str
    max_context: int
    bpb_batch_size: int
    text_slices: dict[str, FinewebValidationSlice | HFDatasetSlice]
    models: dict[str, BundleModel | HFModel]
    downstream: DownstreamConfig
    gate: GateConfig
    tolerances: dict[str, float]


def load_baseline_eval_config(path: Path) -> BaselineEvalConfig:
    if not path.exists():
        raise ValueError(f"baseline eval config does not exist: {path}")
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"baseline eval config is not valid JSON: {path}") from error
    payload = expect_object(payload, "baseline eval config")
    require_keys(
        payload,
        {
            "schema_version",
            "device",
            "dtype",
            "max_context",
            "bpb_batch_size",
            "text_slices",
            "models",
            "downstream",
            "gate",
            "tolerances",
        },
        "baseline eval config",
    )
    if payload["schema_version"] != 1:
        raise ValueError(f"unsupported schema_version: {payload['schema_version']!r}")
    if payload["dtype"] != SUPPORTED_DTYPE:
        raise ValueError(f"dtype must be {SUPPORTED_DTYPE}: {payload['dtype']!r}")
    device = _require_str(payload["device"], "device")
    max_context = _require_positive_int(payload["max_context"], "max_context", minimum=2)
    bpb_batch_size = _require_positive_int(payload["bpb_batch_size"], "bpb_batch_size", minimum=1)

    text_slices = _validate_text_slices(expect_object(payload["text_slices"], "text_slices"))
    models = _validate_models(expect_object(payload["models"], "models"))
    downstream = _validate_downstream(expect_object(payload["downstream"], "downstream"))
    gate = _validate_gate(expect_object(payload["gate"], "gate"), models, downstream)
    tolerances = _validate_tolerances(expect_object(payload["tolerances"], "tolerances"))

    return BaselineEvalConfig(
        config_path=path,
        config_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        device=device,
        dtype=SUPPORTED_DTYPE,
        max_context=max_context,
        bpb_batch_size=bpb_batch_size,
        text_slices=text_slices,
        models=models,
        downstream=downstream,
        gate=gate,
        tolerances=tolerances,
    )


def _validate_text_slices(
    payload: dict[str, Any],
) -> dict[str, FinewebValidationSlice | HFDatasetSlice]:
    if not payload:
        raise ValueError("text_slices must define at least one slice")
    slices: dict[str, FinewebValidationSlice | HFDatasetSlice] = {}
    for name, value in payload.items():
        label = f"text_slices.{name}"
        block = expect_object(value, label)
        kind = block.get("kind")
        if kind == FINEWEB_SLICE_KIND:
            require_keys(block, {"kind", "pretrain_config", "document_budget"}, label)
            slices[name] = FinewebValidationSlice(
                name=name,
                pretrain_config=Path(
                    _require_str(block["pretrain_config"], f"{label}.pretrain_config")
                ),
                document_budget=_require_positive_int(
                    block["document_budget"], f"{label}.document_budget", minimum=1
                ),
            )
        elif kind == HF_DATASET_SLICE_KIND:
            require_keys(
                block,
                {"kind", "source", "subset", "split", "revision", "text_field", "document_budget"},
                label,
            )
            subset = block["subset"]
            if subset is not None:
                subset = _require_str(subset, f"{label}.subset")
            slices[name] = HFDatasetSlice(
                name=name,
                source=_require_str(block["source"], f"{label}.source"),
                subset=subset,
                split=_require_str(block["split"], f"{label}.split"),
                revision=_require_str(block["revision"], f"{label}.revision"),
                text_field=_require_str(block["text_field"], f"{label}.text_field"),
                document_budget=_require_positive_int(
                    block["document_budget"], f"{label}.document_budget", minimum=1
                ),
            )
        else:
            raise ValueError(f"{label} has unknown kind: {kind!r}")
    return slices


def _validate_models(payload: dict[str, Any]) -> dict[str, BundleModel | HFModel]:
    if not payload:
        raise ValueError("models must define at least one model")
    models: dict[str, BundleModel | HFModel] = {}
    for name, value in payload.items():
        label = f"models.{name}"
        block = expect_object(value, label)
        kind = block.get("kind")
        if kind == BUNDLE_MODEL_KIND:
            require_keys(block, {"kind", "path"}, label)
            path = Path(_require_str(block["path"], f"{label}.path"))
            models[name] = BundleModel(name=name, path=path)
        elif kind == HF_MODEL_KIND:
            require_keys(block, {"kind", "repo", "revision"}, label)
            models[name] = HFModel(
                name=name,
                repo=_require_str(block["repo"], f"{label}.repo"),
                revision=_require_str(block["revision"], f"{label}.revision"),
            )
        else:
            raise ValueError(f"{label} has unknown kind: {kind!r}")
    return models


def _validate_downstream(payload: dict[str, Any]) -> DownstreamConfig:
    require_keys(
        payload, {"harness", "version", "tasks", "num_fewshot", "batch_size"}, "downstream"
    )
    if payload["harness"] != SUPPORTED_HARNESS:
        raise ValueError(f"downstream.harness must be {SUPPORTED_HARNESS}: {payload['harness']!r}")
    tasks = payload["tasks"]
    if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) for t in tasks):
        raise ValueError("downstream.tasks must be a non-empty list of task names")
    if len(set(tasks)) != len(tasks):
        raise ValueError("downstream.tasks must not contain duplicates")
    num_fewshot = payload["num_fewshot"]
    if not isinstance(num_fewshot, int) or isinstance(num_fewshot, bool) or num_fewshot < 0:
        raise ValueError("downstream.num_fewshot must be a non-negative integer")
    return DownstreamConfig(
        harness=SUPPORTED_HARNESS,
        version=_require_str(payload["version"], "downstream.version"),
        tasks=tuple(tasks),
        num_fewshot=num_fewshot,
        batch_size=_require_positive_int(payload["batch_size"], "downstream.batch_size", minimum=1),
    )


def _validate_gate(
    payload: dict[str, Any],
    models: dict[str, BundleModel | HFModel],
    downstream: DownstreamConfig,
) -> GateConfig:
    require_keys(payload, {"model", "published", "published_average", "per_task_tolerance"}, "gate")
    model = _require_str(payload["model"], "gate.model")
    if model not in models:
        raise ValueError(f"gate.model is not a configured model: {model!r}")
    published = expect_object(payload["published"], "gate.published")
    if set(published) != set(downstream.tasks):
        raise ValueError("gate.published keys must exactly match downstream.tasks")
    published_scores = {
        task: _require_unit_float(score, f"gate.published.{task}")
        for task, score in published.items()
    }
    tolerance = payload["per_task_tolerance"]
    if not isinstance(tolerance, int | float) or isinstance(tolerance, bool) or tolerance <= 0:
        raise ValueError("gate.per_task_tolerance must be a positive number")
    return GateConfig(
        model=model,
        published=published_scores,
        published_average=_require_unit_float(
            payload["published_average"], "gate.published_average"
        ),
        per_task_tolerance=float(tolerance),
    )


def _validate_tolerances(payload: dict[str, Any]) -> dict[str, float]:
    require_keys(payload, {"bpb", "accuracy"}, "tolerances")
    tolerances: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
            raise ValueError(f"tolerances.{key} must be a non-negative number")
        tolerances[key] = float(value)
    return tolerances


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_positive_int(value: Any, label: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _require_unit_float(value: Any, label: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be a number between 0 and 1")
    return float(value)
