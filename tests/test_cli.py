import json
from pathlib import Path

import pytest

from esme_pretrain.cli import main, run_doctor
from esme_pretrain.pretrain_run import load_pretrain_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_214M_B200 = Path("configs/pretrain_214m_b200.json")


def test_status_json_reports_state_and_pipeline(capsys) -> None:
    exit_code = main(["status", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state"] == "214M B200 pretrain accepted"
    assert payload["pipeline"][0]["name"] == "raw text"
    assert payload["pipeline"][-1]["name"] == "checkpoint export"
    assert payload["run_card_path"] == "docs/run-cards/pretrain-214m-b200.md"


def test_no_command_defaults_to_status(capsys) -> None:
    exit_code = main([])

    assert exit_code == 0
    assert "state: 214M B200 pretrain accepted" in capsys.readouterr().out


def test_doctor_passes_for_repo_scaffold() -> None:
    ok, checks = run_doctor(REPO_ROOT)

    assert ok
    assert {check.name for check in checks} >= {
        "python",
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "git remote",
    }


def test_pilot_fixture_is_committed_text() -> None:
    corpus = REPO_ROOT / "tests" / "fixtures" / "pilot_corpus.txt"
    text = corpus.read_text(encoding="utf-8")

    assert text.count("\n") >= 2
    assert "real data" in text


def test_data_report_cli_emits_json(tmp_path, capsys) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 12, encoding="utf-8")

    exit_code = main(
        [
            "data-report",
            "--input",
            str(corpus),
            "--context-length",
            "8",
            "--token-budget",
            "64",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["budgeted_tokens"] == 64
    assert payload["truncated_tokens"] == 56
    assert payload["splits"]["train_rows"] > payload["splits"]["validation_rows"]


def test_prepare_data_cli_emits_json_and_writes_manifest(tmp_path, capsys) -> None:
    corpus = tmp_path / "corpus.txt"
    output_dir = tmp_path / "prepared"
    corpus.write_text("abcdefghij" * 12, encoding="utf-8")

    exit_code = main(
        [
            "prepare-data",
            "--input",
            str(corpus),
            "--output-dir",
            str(output_dir),
            "--context-length",
            "8",
            "--token-budget",
            "64",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_path"].endswith("manifest.json")
    assert payload["split_sizes"]["train_rows"] > 0
    assert (output_dir / "manifest.json").exists()


def test_data_report_cli_returns_error_for_missing_input(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "data-report",
            "--input",
            str(tmp_path / "missing.txt"),
            "--context-length",
            "8",
            "--token-budget",
            "64",
            "--json",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "input path does not exist" in captured.err


def test_tokenizer_lab_cli_emits_json_with_comparisons(tmp_path, capsys) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("the tiny tokenizer lab repeats tiny text\n" * 12, encoding="utf-8")

    exit_code = main(
        [
            "tokenizer-lab",
            "--input",
            str(corpus),
            "--context-length",
            "8",
            "--token-budget",
            "96",
            "--vocab-sizes",
            "24",
            "--steps",
            "1",
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "comparisons" in payload
    assert [row["tokenizer"] for row in payload["comparisons"]] == ["char", "pair_merge"]
    learned = payload["comparisons"][1]
    assert learned["vocab_size"] == 24
    assert learned["tokens"] < payload["comparisons"][0]["tokens"]
    assert learned["compression_ratio"] > 1
    assert learned["train_loss_final"] > 0
    assert learned["validation_loss_final"] > 0
    assert (
        learned["round_trip_examples"][0]["input"] == learned["round_trip_examples"][0]["decoded"]
    )


def test_tokenizer_lab_cli_rejects_malformed_vocab_sizes(tmp_path, capsys) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabcabcabcabcabcabcabc", encoding="utf-8")

    exit_code = main(
        [
            "tokenizer-lab",
            "--input",
            str(corpus),
            "--context-length",
            "4",
            "--token-budget",
            "20",
            "--vocab-sizes",
            "8,nope",
            "--steps",
            "1",
            "--json",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "vocab sizes must be a comma-separated list of integers" in captured.err


def test_pretrain_214m_b200_cli_refuses_without_dry_run(capsys) -> None:
    exit_code = main(["pretrain-214m-b200", "--json"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--dry-run is required" in captured.err


def test_pretrain_214m_b200_cli_dry_run_is_guarded(capsys) -> None:
    config = load_pretrain_config(CONFIG_214M_B200)
    exit_code = main(["pretrain-214m-b200", "--dry-run", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ready_for_pretrain_launch"
    assert payload["run_id"] == "pretrain_214m_b200"
    assert payload["model"]["name"] == "214M"
    assert payload["runtime"]["selected_gpu"] == "B200"
    assert payload["runtime"]["estimated_cost_usd"] == pytest.approx(
        round(config.estimated_cost_usd, 2)
    )
    profile = config.selected_gpu_profile
    expected_hours = round(
        config.train_token_budget / profile["projected_tokens_per_second"] / 3600,
        2,
    )
    assert payload["runtime"]["selected_gpu_profile"]["expected_duration_hours"] == pytest.approx(
        expected_hours
    )
    assert payload["runtime"]["max_cost_usd"] == 100
    assert payload["will_download_data"] is False
    assert payload["will_start_modal_job"] is False
    assert "PRETRAIN_GPU='B200'" in payload["launch_command"]
    assert "modal run --detach" in payload["launch_command"]
    assert "--approved" in payload["launch_command"]


def test_export_cli_returns_error_for_missing_checkpoint(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "export",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
            "--format",
            "llm-infer",
            "--output",
            str(tmp_path / "out"),
            "--json",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "export failed" in captured.err
    assert "does not exist" in captured.err


def test_eval_checkpoints_cli_returns_error_for_missing_config(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "eval-checkpoints",
            "--config",
            str(tmp_path / "missing.json"),
            "--tokenizer",
            str(tmp_path / "tokenizer.json"),
            "--checkpoint",
            str(tmp_path / "checkpoint.pt"),
            "--eval-token-budget",
            "64",
            "--output",
            str(tmp_path / "eval.json"),
            "--json",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "eval-checkpoints failed" in captured.err


def test_base_acceptance_report_cli_returns_error_for_missing_run_dir(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "base-acceptance-report",
            "--run-dir",
            str(tmp_path / "missing-run"),
            "--eval",
            str(tmp_path / "eval.json"),
            "--output",
            str(tmp_path / "report.md"),
            "--json",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "base-acceptance-report failed" in captured.err
    assert "run directory does not exist" in captured.err
