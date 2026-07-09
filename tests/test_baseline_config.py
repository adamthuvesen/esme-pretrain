from __future__ import annotations

import json
from pathlib import Path

import pytest

from esme_pretrain.baselines.config import (
    BundleModel,
    FinewebValidationSlice,
    HFDatasetSlice,
    HFModel,
    load_baseline_eval_config,
)

REPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "baseline_eval.json"


def _valid_payload() -> dict:
    return json.loads(REPO_CONFIG.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "baseline_eval.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_repo_config_loads() -> None:
    config = load_baseline_eval_config(REPO_CONFIG)

    assert config.device == "cpu"
    assert config.dtype == "float32"
    assert config.max_context == 1024
    assert set(config.text_slices) == {"fineweb_edu_validation", "pile_test"}
    assert isinstance(config.text_slices["fineweb_edu_validation"], FinewebValidationSlice)
    assert isinstance(config.text_slices["pile_test"], HFDatasetSlice)
    assert isinstance(config.models["esme"], BundleModel)
    assert isinstance(config.models["cerebras"], HFModel)
    assert isinstance(config.models["pythia"], HFModel)
    assert config.gate.model == "cerebras"
    assert set(config.gate.published) == set(config.downstream.tasks)
    assert len(config.downstream.tasks) == 7
    assert len(config.config_sha256) == 64


def test_missing_top_level_key_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["gate"]

    with pytest.raises(ValueError, match="missing required keys: gate"):
        load_baseline_eval_config(_write(tmp_path, payload))


def test_extra_top_level_key_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["surprise"] = True

    with pytest.raises(ValueError, match="unsupported keys: surprise"):
        load_baseline_eval_config(_write(tmp_path, payload))


def test_unknown_slice_kind_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["text_slices"]["pile_test"]["kind"] = "local_files"

    with pytest.raises(ValueError, match="unknown kind: 'local_files'"):
        load_baseline_eval_config(_write(tmp_path, payload))


def test_pile_slice_is_optional(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["text_slices"]["pile_test"]

    config = load_baseline_eval_config(_write(tmp_path, payload))

    assert set(config.text_slices) == {"fineweb_edu_validation"}


def test_empty_text_slices_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["text_slices"] = {}

    with pytest.raises(ValueError, match="at least one slice"):
        load_baseline_eval_config(_write(tmp_path, payload))


def test_gate_model_must_exist(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["gate"]["model"] = "gpt5"

    with pytest.raises(ValueError, match="gate.model is not a configured model"):
        load_baseline_eval_config(_write(tmp_path, payload))


def test_gate_published_must_match_tasks(tmp_path: Path) -> None:
    payload = _valid_payload()
    del payload["gate"]["published"]["piqa"]

    with pytest.raises(ValueError, match="published keys must exactly match"):
        load_baseline_eval_config(_write(tmp_path, payload))


def test_non_float32_dtype_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["dtype"] = "bfloat16"

    with pytest.raises(ValueError, match="dtype must be float32"):
        load_baseline_eval_config(_write(tmp_path, payload))


def test_slice_extra_key_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["text_slices"]["pile_test"]["cache_dir"] = "/tmp"

    with pytest.raises(ValueError, match="text_slices.pile_test has unsupported keys"):
        load_baseline_eval_config(_write(tmp_path, payload))


def test_duplicate_tasks_rejected(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["downstream"]["tasks"].append("piqa")

    with pytest.raises(ValueError, match="must not contain duplicates"):
        load_baseline_eval_config(_write(tmp_path, payload))
