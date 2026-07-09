"""lm-eval downstream scoring and the published-numbers gate."""

from __future__ import annotations

import importlib
from typing import Any

from esme_pretrain.baselines.config import (
    BaselineEvalConfig,
    BundleModel,
    GateConfig,
    HFModel,
)
from esme_pretrain.baselines.models import EsmeBundleModel, EvalModel
from esme_pretrain.torch import torch

ACC_KEYS = ("acc,none", "acc")
ACC_STDERR_KEYS = ("acc_stderr,none", "acc_stderr")


@torch.no_grad()
def score_loglikelihood(model: EvalModel, context: str, continuation: str) -> tuple[float, bool]:
    """Teacher-forced log-probability of ``continuation`` given ``context``.

    Returns ``(logprob, is_greedy)`` matching the lm-eval loglikelihood contract.
    """
    continuation_ids = model.encode(continuation).ids
    if not continuation_ids:
        raise ValueError("loglikelihood continuation tokenized to zero tokens")
    if len(continuation_ids) > model.context_length:
        raise ValueError(
            f"continuation is longer than the model context ({model.context_length} tokens)"
        )
    context_ids = model.encode(context).ids if context else []
    if not context_ids:
        if model.eos_id is None:
            raise ValueError("empty context requires an eos token id")
        context_ids = [int(model.eos_id)]

    full = (context_ids + continuation_ids)[-(model.context_length + 1) :]
    input_ids = torch.tensor([full[:-1]], dtype=torch.long)
    logits = model.module()(input_ids)
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)

    span = len(continuation_ids)
    target_positions = range(len(full) - 1 - span, len(full) - 1)
    total = 0.0
    is_greedy = True
    for position, token_id in zip(target_positions, continuation_ids, strict=True):
        total += float(log_probs[position, token_id])
        if int(log_probs[position].argmax()) != token_id:
            is_greedy = False
    return total, is_greedy


@torch.no_grad()
def score_loglikelihood_rolling(model: EvalModel, text: str) -> float:
    """Log-probability of the full text in disjoint windows with eos priming."""
    if model.eos_id is None:
        raise ValueError("rolling loglikelihood requires an eos token id")
    ids = model.encode(text).ids
    if not ids:
        raise ValueError("rolling loglikelihood text tokenized to zero tokens")
    tokens = [int(model.eos_id), *ids]
    total = 0.0
    window = model.context_length
    for start in range(0, len(tokens) - 1, window):
        chunk = tokens[start : start + window + 1]
        if len(chunk) < 2:
            break
        input_ids = torch.tensor([chunk[:-1]], dtype=torch.long)
        log_probs = torch.log_softmax(model.module()(input_ids)[0].float(), dim=-1)
        for position, token_id in enumerate(chunk[1:]):
            total += float(log_probs[position, token_id])
    return total


def build_esme_lm(model: EsmeBundleModel) -> Any:
    """Wrap the bundle model in an lm-eval LM subclass (lazy lm_eval import)."""
    lm_eval = _import_lm_eval()

    class EsmeLM(lm_eval.api.model.LM):
        def loglikelihood(self, requests) -> list[tuple[float, bool]]:
            return [score_loglikelihood(model, *request.args) for request in requests]

        def loglikelihood_rolling(self, requests) -> list[float]:
            return [score_loglikelihood_rolling(model, request.args[0]) for request in requests]

        def generate_until(self, requests):
            raise NotImplementedError("baseline eval tasks are loglikelihood-only")

    return EsmeLM()


def run_downstream(
    spec: BundleModel | HFModel,
    config: BaselineEvalConfig,
    *,
    eval_model: EvalModel | None = None,
) -> dict[str, Any]:
    """Run the configured lm-eval tasks for one model; returns per-task accuracies."""
    lm_eval = _import_lm_eval()
    installed = getattr(lm_eval, "__version__", None)
    if installed != config.downstream.version:
        raise ValueError(
            f"installed lm-eval {installed!r} does not match pinned "
            f"downstream.version {config.downstream.version!r}"
        )
    common: dict[str, Any] = {
        "tasks": list(config.downstream.tasks),
        "num_fewshot": config.downstream.num_fewshot,
        "batch_size": config.downstream.batch_size,
    }
    if isinstance(spec, HFModel):
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={spec.repo},revision={spec.revision},dtype=float32",
            device=config.device,
            **common,
        )
    else:
        if not isinstance(eval_model, EsmeBundleModel):
            raise ValueError("bundle models need a loaded EsmeBundleModel for downstream eval")
        results = lm_eval.simple_evaluate(model=build_esme_lm(eval_model), **common)

    tasks: dict[str, Any] = {}
    for task in config.downstream.tasks:
        tasks[task] = _extract_task_result(results, task)
    average = sum(entry["acc"] for entry in tasks.values()) / len(tasks)
    return {
        "harness": config.downstream.harness,
        "harness_version": installed,
        "num_fewshot": config.downstream.num_fewshot,
        "tasks": tasks,
        "average": average,
    }


def evaluate_gate(measured_tasks: dict[str, Any], gate: GateConfig) -> dict[str, Any]:
    """Compare measured accuracies to the published table within tolerance."""
    per_task: dict[str, Any] = {}
    all_within = True
    for task, published in sorted(gate.published.items()):
        if task not in measured_tasks:
            raise ValueError(f"gate is missing a measured result for task {task!r}")
        measured = float(measured_tasks[task]["acc"])
        delta = measured - published
        within = abs(delta) <= gate.per_task_tolerance
        all_within = all_within and within
        per_task[task] = {
            "measured": measured,
            "published": published,
            "delta": delta,
            "within_tolerance": within,
        }
    measured_average = sum(entry["measured"] for entry in per_task.values()) / len(per_task)
    average_delta = measured_average - gate.published_average
    average_within = abs(average_delta) <= gate.per_task_tolerance
    return {
        "model": gate.model,
        "per_task_tolerance": gate.per_task_tolerance,
        "per_task": per_task,
        "average": {
            "measured": measured_average,
            "published": gate.published_average,
            "delta": average_delta,
            "within_tolerance": average_within,
        },
        "passed": all_within and average_within,
    }


def _extract_task_result(results: dict[str, Any], task: str) -> dict[str, float]:
    task_results = results.get("results", {}).get(task)
    if not isinstance(task_results, dict):
        raise ValueError(f"lm-eval results are missing task {task!r}")
    acc = _first_metric(task_results, ACC_KEYS)
    if acc is None:
        raise ValueError(f"lm-eval results for {task!r} have no accuracy metric")
    entry: dict[str, float] = {"acc": acc}
    stderr = _first_metric(task_results, ACC_STDERR_KEYS)
    if stderr is not None:
        entry["stderr"] = stderr
    return entry


def _first_metric(task_results: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = task_results.get(key)
        if isinstance(value, int | float):
            return float(value)
    return None


def _import_lm_eval() -> Any:
    try:
        return importlib.import_module("lm_eval")
    except ModuleNotFoundError as error:
        raise ValueError(
            "lm_eval is required for downstream baseline eval; install the baselines "
            "extra: uv sync --extra baselines"
        ) from error
