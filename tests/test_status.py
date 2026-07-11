from esme_pretrain.status import PIPELINE_STAGES, current_status


def test_pipeline_preserves_status_spine() -> None:
    assert [stage.name for stage in PIPELINE_STAGES] == [
        "raw text",
        "data report",
        "tokenizer",
        "packed tokens",
        "transformer",
        "training loop",
        "eval",
        "checkpoint export",
    ]
    assert PIPELINE_STAGES[-1].milestone == "llm-infer bundle export"


def test_status_is_honest_about_current_state() -> None:
    status = current_status()
    assert status.state == "214M B200 pretrain accepted"
    assert "completed the accepted 10B FineWeb-Edu B200 pretrain" in status.summary
    assert "esme-posttrain" in status.next_milestone
    assert PIPELINE_STAGES[6].status == "accepted"
    assert PIPELINE_STAGES[7].status == "accepted"
    assert status.run_card_path == "docs/run-cards/pretrain-214m-b200.md"
    assert "run card" in status.spend_policy
