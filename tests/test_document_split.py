from types import SimpleNamespace

from esme_pretrain.data import corpus_stream
from esme_pretrain.data.document_split import (
    document_id_from_row,
    is_validation_document,
    passes_int_score_filter,
    select_documents,
)


def test_validation_split_is_deterministic() -> None:
    assert is_validation_document(
        "doc-a",
        seed=0,
        validation_modulo=100,
        validation_remainder=0,
    ) == is_validation_document(
        "doc-a",
        seed=0,
        validation_modulo=100,
        validation_remainder=0,
    )


def test_validation_split_changes_with_document_id() -> None:
    rows = [
        is_validation_document(f"id-{index}", seed=0, validation_modulo=100, validation_remainder=0)
        for index in range(500)
    ]
    assert any(rows)
    assert not all(rows)


def test_validation_split_uses_seed() -> None:
    doc_id = "stable-document-id"
    outcomes = {
        is_validation_document(doc_id, seed=seed, validation_modulo=100, validation_remainder=0)
        for seed in range(100)
    }
    assert len(outcomes) > 1


def test_int_score_filter_requires_minimum() -> None:
    assert passes_int_score_filter({"int_score": 3}, min_int_score=3)
    assert not passes_int_score_filter({"int_score": 2}, min_int_score=3)
    assert not passes_int_score_filter({}, min_int_score=3)


def test_document_id_prefers_id_field() -> None:
    row = {"id": "doc-1", "text": "fallback text"}
    assert document_id_from_row(row, id_field="id", text_field="text") == "doc-1"
    fallback = document_id_from_row({"text": "only text"}, id_field="id", text_field="text")
    assert fallback == "only text"


def test_select_documents_filters_and_partitions_the_split() -> None:
    dataset_cfg = {"filters": {"min_int_score": 3}, "id_field": "id", "text_field": "text"}
    split_cfg = {"seed": 0, "validation_modulo": 2, "validation_remainder": 0}
    rows = [{"id": f"doc-{i}", "text": f"text-{i}", "int_score": 3} for i in range(40)]
    rows.append({"id": "low", "text": "low quality", "int_score": 2})  # dropped by score filter
    rows.append({"id": "empty", "text": "", "int_score": 5})  # dropped: no text

    train = list(
        select_documents(rows, dataset_cfg=dataset_cfg, split_cfg=split_cfg, split="train")
    )
    validation = list(
        select_documents(rows, dataset_cfg=dataset_cfg, split_cfg=split_cfg, split="validation")
    )

    passing = {f"text-{i}" for i in range(40)}
    # The score-filtered and empty-text rows never reach either split.
    assert "low quality" not in passing and "low quality" not in train + validation
    assert all(text for text in train + validation)
    # Train and validation partition the score-passing documents exactly.
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == passing
    assert train and validation  # deterministic seed keeps both sides non-empty
    # The validation side matches the deterministic predicate document-for-document.
    expected_validation = {
        f"text-{i}"
        for i in range(40)
        if is_validation_document(f"doc-{i}", seed=0, validation_modulo=2, validation_remainder=0)
    }
    assert set(validation) == expected_validation
    # Any other split value yields both sides together.
    everything = list(
        select_documents(rows, dataset_cfg=dataset_cfg, split_cfg=split_cfg, split="all")
    )
    assert set(everything) == passing


def test_document_text_stream_loads_configured_dataset_and_reuses_split(monkeypatch) -> None:
    rows = [{"id": f"doc-{index}", "text": f"text-{index}", "int_score": 3} for index in range(20)]
    rows.append({"id": "low", "text": "low quality", "int_score": 2})
    captured: dict[str, object] = {}

    def load_dataset(source: str, **kwargs):
        captured.update({"source": source, **kwargs})
        return rows

    def fake_import(name: str):
        if name == "datasets":
            return SimpleNamespace(load_dataset=load_dataset)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(corpus_stream.importlib, "import_module", fake_import)
    config = SimpleNamespace(
        payload={
            "dataset": {
                "source": "HuggingFaceFW/fineweb-edu",
                "subset": "sample-10BT",
                "split": "train",
                "streaming": True,
                "revision": "revision",
                "filters": {"min_int_score": 3},
                "id_field": "id",
                "text_field": "text",
            },
            "split": {"seed": 0, "validation_modulo": 2, "validation_remainder": 0},
        }
    )

    train = list(corpus_stream.document_text_stream(config, split="train"))
    validation = list(corpus_stream.document_text_stream(config, split="validation"))

    assert captured == {
        "source": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "streaming": True,
        "revision": "revision",
    }
    assert "low quality" not in train + validation
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == {f"text-{index}" for index in range(20)}
