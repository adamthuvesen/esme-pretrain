"""Model and text-slice adapters for the baseline comparison harness."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from esme_pretrain.baselines.config import (
    BundleModel,
    FinewebValidationSlice,
    HFDatasetSlice,
    HFModel,
)
from esme_pretrain.data.corpus_stream import document_text_stream
from esme_pretrain.postrun.bundle import load_bundle
from esme_pretrain.postrun.eval_checkpoints import load_tokenizer
from esme_pretrain.pretrain_run import load_pretrain_config
from esme_pretrain.torch import torch


@dataclass(frozen=True)
class EncodedText:
    ids: list[int]
    byte_counts: list[int]


class EvalModel(Protocol):
    name: str
    context_length: int
    eos_id: int | None
    provenance: dict[str, Any]

    def encode(self, text: str) -> EncodedText: ...

    def module(self) -> torch.nn.Module: ...


def partitioned_byte_counts(text: str, offsets: Sequence[tuple[int, int]]) -> list[int]:
    """Per-token UTF-8 byte counts that partition the full text exactly.

    Token offsets from byte-level BPE tokenizers can trim whitespace or overlap,
    so raw offset spans under- or over-count bytes. Partitioning by token start
    boundaries assigns every byte to exactly one token: the counts always sum to
    ``len(text.encode("utf-8"))``.
    """
    if not offsets:
        if text:
            raise ValueError("tokenizer produced no tokens for non-empty text")
        return []
    starts = [int(start) for start, _end in offsets]
    boundaries = [0]
    for start in starts[1:]:
        boundaries.append(max(start, boundaries[-1]))
    boundaries.append(len(text))
    counts = []
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        counts.append(len(text[begin:end].encode("utf-8")))
    return counts


def _encoded_text(text: str, ids: Sequence[int], offsets: Sequence[tuple[int, int]]) -> EncodedText:
    if len(ids) != len(offsets):
        raise ValueError("tokenizer encoding ids and offsets must have the same length")
    return EncodedText(
        ids=[int(token_id) for token_id in ids],
        byte_counts=partitioned_byte_counts(text, offsets),
    )


class EsmeBundleModel:
    """Exported Esme bundle behind the shared eval interface."""

    def __init__(self, spec: BundleModel, *, max_context: int, device: str) -> None:
        bundle = load_bundle(spec.path, device=device)
        tokenizer = load_tokenizer(bundle.tokenizer_path)
        self.name = spec.name
        self.context_length = min(max_context, bundle.config.context_length)
        self.eos_id = tokenizer.token_to_id("<eos>")
        if self.eos_id is None:
            raise ValueError(f"bundle tokenizer is missing <eos>: {bundle.tokenizer_path}")
        self.provenance = {
            "kind": "bundle",
            "bundle_dir": str(spec.path),
            "weights_sha256": bundle.weights_sha256,
            "checkpoint_step": bundle.checkpoint_step,
        }
        self._tokenizer = tokenizer
        self._module = bundle.model

    def encode(self, text: str) -> EncodedText:
        encoding = self._tokenizer.encode(text)
        offsets = getattr(encoding, "offsets", None)
        if offsets is None:
            raise ValueError("bundle tokenizer encoding must expose offsets for bits-per-byte")
        return _encoded_text(text, encoding.ids, offsets)

    def module(self) -> torch.nn.Module:
        return self._module


class _LogitsOnly(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.inner(input_ids).logits


class HFCausalModel:
    """HuggingFace causal LM baseline behind the shared eval interface."""

    def __init__(self, spec: HFModel, *, max_context: int, device: str) -> None:
        transformers = _import_baseline_dependency("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                f"baseline tokenizer must be a fast tokenizer with offsets: {spec.repo}"
            )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            spec.repo, revision=spec.revision, torch_dtype=torch.float32
        )
        model = model.to(torch.device(device)).eval()
        native_context = getattr(model.config, "max_position_embeddings", None)
        if not isinstance(native_context, int) or native_context < 2:
            raise ValueError(f"baseline model does not declare a context length: {spec.repo}")
        self.name = spec.name
        self.context_length = min(max_context, native_context)
        self.eos_id = tokenizer.eos_token_id
        self.provenance = {"kind": "hf", "repo": spec.repo, "revision": spec.revision}
        self._backend = tokenizer.backend_tokenizer
        self._module = _LogitsOnly(model)

    def encode(self, text: str) -> EncodedText:
        encoding = self._backend.encode(text)
        return _encoded_text(text, encoding.ids, encoding.offsets)

    def module(self) -> torch.nn.Module:
        return self._module


def build_eval_model(
    spec: BundleModel | HFModel, *, max_context: int, device: str
) -> EsmeBundleModel | HFCausalModel:
    if isinstance(spec, BundleModel):
        return EsmeBundleModel(spec, max_context=max_context, device=device)
    return HFCausalModel(spec, max_context=max_context, device=device)


def load_slice_texts(slice_cfg: FinewebValidationSlice | HFDatasetSlice) -> list[str]:
    """Materialize the fixed document list for one text slice, in stream order."""
    if isinstance(slice_cfg, FinewebValidationSlice):
        launch_config = load_pretrain_config(slice_cfg.pretrain_config)
        stream = document_text_stream(launch_config, split="validation")
    else:
        stream = _hf_dataset_text_stream(slice_cfg)
    texts: list[str] = []
    for text in stream:
        if not isinstance(text, str) or not text:
            raise ValueError(f"text slice {slice_cfg.name} yielded an empty or non-string document")
        texts.append(text)
        if len(texts) >= slice_cfg.document_budget:
            break
    if len(texts) < slice_cfg.document_budget:
        raise ValueError(
            f"text slice {slice_cfg.name} produced {len(texts)} documents; "
            f"config requires {slice_cfg.document_budget}"
        )
    return texts


def _hf_dataset_text_stream(slice_cfg: HFDatasetSlice):
    datasets = _import_baseline_dependency("datasets")
    stream = datasets.load_dataset(
        slice_cfg.source,
        name=slice_cfg.subset,
        split=slice_cfg.split,
        revision=slice_cfg.revision,
        streaming=True,
    )
    for row in stream:
        if slice_cfg.text_field not in row:
            raise ValueError(
                f"text slice {slice_cfg.name} row is missing field {slice_cfg.text_field!r}"
            )
        yield row[slice_cfg.text_field]


def _import_baseline_dependency(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        raise ValueError(
            f"{module_name} is required for baseline eval; install the baselines extra: "
            "uv sync --extra baselines"
        ) from error
