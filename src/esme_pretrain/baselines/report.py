"""Markdown comparison report built only from per-model baseline result JSONs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = 1


def build_comparison(result_paths: list[Path], output_path: Path) -> dict[str, Any]:
    """Render the cross-model comparison; fails loudly on non-comparable inputs."""
    if len(result_paths) < 2:
        raise ValueError("baseline comparison needs at least two result files")
    results = [_read_result(path) for path in result_paths]

    names = [result["model"]["name"] for result in results]
    if len(set(names)) != len(names):
        raise ValueError(f"result files contain duplicate model names: {names}")

    config_hashes = {result["config_sha256"] for result in results}
    if len(config_hashes) != 1:
        raise ValueError(
            "result files were produced with different baseline eval configs; "
            "re-run every model against the same config"
        )

    slice_names = _shared_keys(results, "bpb", "text slices")
    for slice_name in slice_names:
        text_hashes = {result["bpb"][slice_name]["text_sha256"] for result in results}
        if len(text_hashes) != 1:
            raise ValueError(
                f"slice {slice_name!r} was not scored on identical text across models; "
                "text_sha256 differs, so bits-per-byte is not comparable"
            )

    task_names = _shared_keys(
        [result["downstream"] for result in results], "tasks", "downstream tasks"
    )

    lines = _render_markdown(results, sorted(slice_names), list(task_names))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "output": str(output_path),
        "models": names,
        "slices": sorted(slice_names),
        "tasks": list(task_names),
    }


def _read_result(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"result file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"result file is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"result file must contain a JSON object: {path}")
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unsupported result schema_version in {path}")
    for key in ("model", "config_sha256", "bpb", "downstream"):
        if key not in payload:
            raise ValueError(f"result file is missing {key!r}: {path}")
    return payload


def _shared_keys(payloads: list[dict[str, Any]], field: str, label: str) -> list[str]:
    first = payloads[0].get(field)
    if not isinstance(first, dict) or not first:
        raise ValueError(f"result files have no {label}")
    expected = set(first)
    for payload in payloads[1:]:
        actual = set(payload.get(field, {}))
        if actual != expected:
            raise ValueError(
                f"result files do not cover the same {label}: "
                f"{sorted(expected)} vs {sorted(actual)}"
            )
    return list(first)


def _render_markdown(
    results: list[dict[str, Any]], slice_names: list[str], task_names: list[str]
) -> list[str]:
    lines = [
        "# Baseline Comparison",
        "",
        "Generated from per-model baseline-eval result JSONs only.",
        "",
        "## Models",
        "",
        "| model | kind | provenance | context | device | dtype |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        model = result["model"]
        provenance = (
            f"{model.get('repo')}@{str(model.get('revision'))[:12]}"
            if model.get("kind") == "hf"
            else f"weights {str(model.get('weights_sha256'))[:12]}"
        )
        lines.append(
            f"| {model['name']} | {model.get('kind')} | {provenance} "
            f"| {result.get('context_length')} | {result.get('device')} | {result.get('dtype')} |"
        )

    for slice_name in slice_names:
        reference = results[0]["bpb"][slice_name]
        lines.extend(
            [
                "",
                f"## Bits Per Byte: {slice_name}",
                "",
                f"- documents: {reference['document_count']}",
                f"- raw bytes: {reference['raw_bytes']}",
                f"- text sha256: `{reference['text_sha256']}`",
                "",
                "| model | bits/byte | ce loss | perplexity | eval tokens | eval bytes |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        ranked = sorted(results, key=lambda r: r["bpb"][slice_name]["bits_per_byte"])
        for result in ranked:
            entry = result["bpb"][slice_name]
            lines.append(
                f"| {result['model']['name']} | {entry['bits_per_byte']:.4f} "
                f"| {entry['ce_loss']:.4f} | {entry['perplexity']:.2f} "
                f"| {entry['eval_tokens']} | {entry['eval_bytes']} |"
            )

    header = " | ".join(task_names)
    divider = " | ".join("---" for _ in task_names)
    lines.extend(
        [
            "",
            "## Downstream 0-Shot Accuracy",
            "",
            f"| model | {header} | average |",
            f"| --- | {divider} | --- |",
        ]
    )
    ranked = sorted(results, key=lambda r: r["downstream"]["average"], reverse=True)
    for result in ranked:
        tasks = result["downstream"]["tasks"]
        cells = " | ".join(f"{tasks[task]['acc']:.3f}" for task in task_names)
        lines.append(
            f"| {result['model']['name']} | {cells} | {result['downstream']['average']:.3f} |"
        )

    lines.extend(["", "## Gate Status", ""])
    for result in results:
        gate = result.get("gate", {})
        lines.append(
            f"- {result['model']['name']}: required={gate.get('required')} "
            f"passed={gate.get('passed')}"
        )
    return lines
