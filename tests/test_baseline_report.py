from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from esme_pretrain.baselines.report import build_comparison
from esme_pretrain.cli import main


def _result_payload(name: str, *, kind: str = "hf", bpb: float = 1.2, acc: float = 0.5) -> dict:
    model: dict[str, Any] = {"name": name, "kind": kind}
    if kind == "hf":
        model.update(repo=f"org/{name}", revision="a" * 40)
    else:
        model.update(bundle_dir=f"exports/{name}", weights_sha256="b" * 64, checkpoint_step=5)
    return {
        "schema_version": 1,
        "config": "configs/baseline_eval.json",
        "config_sha256": "c" * 64,
        "model": model,
        "device": "cpu",
        "dtype": "float32",
        "max_context": 1024,
        "context_length": 1024,
        "bpb_batch_size": 4,
        "bpb": {
            "fineweb_edu_validation": {
                "slice_name": "fineweb_edu_validation",
                "document_count": 500,
                "text_sha256": "d" * 64,
                "raw_bytes": 100000,
                "token_batch_sha256": "e" * 64,
                "eval_tokens": 20000,
                "eval_bytes": 90000,
                "eval_batches": 5,
                "ce_loss": 2.5,
                "perplexity": 12.18,
                "bits_per_byte": bpb,
                "source": {"kind": "fineweb_edu_validation"},
            }
        },
        "downstream": {
            "harness": "lm-eval",
            "harness_version": "0.4.12",
            "num_fewshot": 0,
            "tasks": {"piqa": {"acc": acc, "stderr": 0.01}, "arc_easy": {"acc": acc}},
            "average": acc,
        },
        "gate": {"required": kind == "bundle", "gate_path": None, "passed": None},
        "tolerances": {"bpb": 1e-6, "accuracy": 0.0},
        "provenance": {"esme_pretrain": "0.0.0"},
        "runtime_seconds": 1.0,
    }


def _write_results(tmp_path: Path, payloads: list[dict]) -> list[Path]:
    paths = []
    for payload in payloads:
        path = tmp_path / f"{payload['model']['name']}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    return paths


def test_build_comparison_renders_ranked_tables(tmp_path: Path) -> None:
    paths = _write_results(
        tmp_path,
        [
            _result_payload("esme", kind="bundle", bpb=1.1, acc=0.42),
            _result_payload("cerebras", bpb=1.3, acc=0.35),
        ],
    )
    output = tmp_path / "comparison.md"

    payload = build_comparison(paths, output)

    text = output.read_text(encoding="utf-8")
    assert payload["models"] == ["esme", "cerebras"]
    assert text.startswith("# Baseline Comparison")
    assert "## Bits Per Byte: fineweb_edu_validation" in text
    assert "## Downstream 0-Shot Accuracy" in text
    bpb_section = text.split("## Bits Per Byte")[1]
    assert bpb_section.index("| esme |") < bpb_section.index("| cerebras |")
    assert "weights bbbbbbbbbbbb" in text
    assert "org/cerebras@aaaaaaaaaaaa" in text


def test_build_comparison_rejects_text_hash_mismatch(tmp_path: Path) -> None:
    first = _result_payload("esme", kind="bundle")
    second = _result_payload("cerebras")
    second["bpb"]["fineweb_edu_validation"]["text_sha256"] = "f" * 64
    paths = _write_results(tmp_path, [first, second])

    with pytest.raises(ValueError, match="not scored on identical text"):
        build_comparison(paths, tmp_path / "comparison.md")


def test_build_comparison_rejects_config_mismatch(tmp_path: Path) -> None:
    first = _result_payload("esme", kind="bundle")
    second = _result_payload("cerebras")
    second["config_sha256"] = "9" * 64
    paths = _write_results(tmp_path, [first, second])

    with pytest.raises(ValueError, match="different baseline eval configs"):
        build_comparison(paths, tmp_path / "comparison.md")


def test_build_comparison_rejects_missing_task(tmp_path: Path) -> None:
    first = _result_payload("esme", kind="bundle")
    second = _result_payload("cerebras")
    del second["downstream"]["tasks"]["arc_easy"]
    paths = _write_results(tmp_path, [first, second])

    with pytest.raises(ValueError, match="do not cover the same downstream tasks"):
        build_comparison(paths, tmp_path / "comparison.md")


def test_build_comparison_rejects_missing_slice(tmp_path: Path) -> None:
    first = _result_payload("esme", kind="bundle")
    second = _result_payload("cerebras")
    second["bpb"]["pile_test"] = second["bpb"]["fineweb_edu_validation"]
    paths = _write_results(tmp_path, [first, second])

    with pytest.raises(ValueError, match="do not cover the same text slices"):
        build_comparison(paths, tmp_path / "comparison.md")


def test_build_comparison_needs_two_results(tmp_path: Path) -> None:
    paths = _write_results(tmp_path, [_result_payload("esme", kind="bundle")])

    with pytest.raises(ValueError, match="at least two result files"):
        build_comparison(paths, tmp_path / "comparison.md")


def test_baseline_compare_cli_json(tmp_path: Path, capsys) -> None:
    paths = _write_results(
        tmp_path,
        [_result_payload("esme", kind="bundle"), _result_payload("cerebras")],
    )
    output = tmp_path / "comparison.md"

    exit_code = main(
        [
            "baseline-compare",
            "--result",
            str(paths[0]),
            "--result",
            str(paths[1]),
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"] == ["esme", "cerebras"]
    assert output.exists()


def test_baseline_compare_cli_error_exit_code(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "baseline-compare",
            "--result",
            str(tmp_path / "missing.json"),
            "--result",
            str(tmp_path / "missing2.json"),
            "--output",
            str(tmp_path / "out.md"),
        ]
    )

    assert exit_code == 2
    assert "baseline-compare failed" in capsys.readouterr().err


def test_baseline_eval_cli_error_exit_code(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "baseline-eval",
            "--config",
            str(tmp_path / "missing.json"),
            "--model",
            "esme",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )

    assert exit_code == 2
    assert "baseline-eval failed" in capsys.readouterr().err


def test_baseline_gate_cli_error_exit_code(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "baseline-gate",
            "--config",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "gate.json"),
        ]
    )

    assert exit_code == 2
    assert "baseline-gate failed" in capsys.readouterr().err
