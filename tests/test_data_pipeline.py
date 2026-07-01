import json

import pytest

from esme_pretrain.data.pipeline import (
    DataPipelineConfig,
    build_data_report,
    ingest_local_text,
    prepare_data,
)
from esme_pretrain.torch import torch


def test_missing_input_fails_loudly(tmp_path) -> None:
    with pytest.raises(ValueError, match="input path does not exist"):
        build_data_report(
            DataPipelineConfig(
                input_path=tmp_path / "missing.txt",
                context_length=8,
                token_budget=64,
            )
        )


def test_empty_input_fails_loudly(tmp_path) -> None:
    corpus = tmp_path / "empty.txt"
    corpus.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="input corpus is empty"):
        build_data_report(
            DataPipelineConfig(
                input_path=corpus,
                context_length=8,
                token_budget=64,
            )
        )


def test_directory_ingestion_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside the requested root", encoding="utf-8")
    (root / "escape.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes input root"):
        ingest_local_text(root)


def test_budget_enforcement_reports_deterministic_truncation(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 8, encoding="utf-8")

    report = build_data_report(
        DataPipelineConfig(
            input_path=corpus,
            context_length=8,
            token_budget=33,
        )
    )

    assert report.total_tokens == 80
    assert report.budgeted_tokens == 33
    assert report.truncated_tokens == 47
    assert report.packable_rows == 4
    assert report.splits.train_rows == 3
    assert report.splits.validation_rows == 1


def test_input_too_small_to_pack_fails_loudly(tmp_path) -> None:
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("abc", encoding="utf-8")

    with pytest.raises(ValueError, match="need more than 8 to pack"):
        build_data_report(
            DataPipelineConfig(
                input_path=corpus,
                context_length=8,
                token_budget=16,
            )
        )


def test_prepare_data_is_deterministic_for_same_seed(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("deterministic split\n" * 20, encoding="utf-8")
    config = DataPipelineConfig(
        input_path=corpus,
        context_length=8,
        token_budget=128,
        shard_rows=3,
        seed=23,
    )

    first = prepare_data(config, tmp_path / "first")
    second = prepare_data(config, tmp_path / "second")

    assert first.report.to_dict() == second.report.to_dict()
    first_train = torch.load(first.output_dir / first.train_shards[0].path, weights_only=True)
    second_train = torch.load(second.output_dir / second.train_shards[0].path, weights_only=True)
    assert torch.equal(first_train["inputs"], second_train["inputs"])
    assert torch.equal(first_train["targets"], second_train["targets"])


def test_prepare_data_writes_shards_manifest_and_report(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghijklmnopqrstuvwxyz" * 4, encoding="utf-8")

    result = prepare_data(
        DataPipelineConfig(
            input_path=corpus,
            context_length=8,
            token_budget=80,
            shard_rows=2,
        ),
        tmp_path / "prepared",
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    train_shards = manifest["splits"]["train"]["shards"]
    validation_shards = manifest["splits"]["validation"]["shards"]

    assert manifest["format"] == "packed-token-shards-v1"
    assert manifest["config"]["split_seed"] == 17
    assert report["token_budget"] == 80
    assert manifest["splits"]["train"]["rows"] == result.report.splits.train_rows
    assert manifest["splits"]["validation"]["rows"] == result.report.splits.validation_rows
    assert len(train_shards) == 4
    assert len(validation_shards) == 1
    assert (result.output_dir / train_shards[0]["path"]).exists()
    assert (result.output_dir / validation_shards[0]["path"]).exists()


def test_prepare_data_rejects_non_empty_output_dir(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 20, encoding="utf-8")
    output_dir = tmp_path / "prepared"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="output directory must be empty"):
        prepare_data(
            DataPipelineConfig(
                input_path=corpus,
                context_length=8,
                token_budget=64,
            ),
            output_dir,
        )


def test_prepare_data_rejects_output_path_that_is_a_file(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 20, encoding="utf-8")
    output_path = tmp_path / "prepared-file"
    output_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ValueError, match="output path is not a directory"):
        prepare_data(
            DataPipelineConfig(
                input_path=corpus,
                context_length=8,
                token_budget=64,
            ),
            output_path,
        )
