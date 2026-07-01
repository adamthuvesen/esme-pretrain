from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from esme_pretrain.launch import modal_tokenizer
from esme_pretrain.launch.pretrain import (
    EXPECTED_ARTIFACTS,
    LAUNCH_APPROVAL_FLAG,
    build_pretrain_dry_run,
    load_pretrain_config,
)

CONFIG_214M_B200 = Path("configs/pretrain_214m_b200.json")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _modal_pretrain_module():
    spec = importlib.util.spec_from_file_location(
        "modal_pretrain_under_test",
        REPO_ROOT / "scripts" / "modal_pretrain.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load modal_pretrain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gpu_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "pretrain_gpu_smoke_under_test",
        REPO_ROOT / "scripts" / "pretrain_gpu_smoke.py",
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load pretrain_gpu_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pretrain_214m_b200_config_validates_conventional_keystone_contract() -> None:
    config = load_pretrain_config(CONFIG_214M_B200)

    assert config.payload["run_id"] == "pretrain_214m_b200"
    assert config.payload["run_card"] == "docs/run-cards/pretrain-214m-b200.md"
    assert config.payload["model"]["name"] == "214M"
    assert config.payload["model"]["layers"] == 30
    assert config.payload["model"]["embedding_dim"] == 768
    assert config.payload["model"]["heads"] == 12
    assert config.payload["model"]["kv_heads"] == 4
    assert config.payload["model"]["feedforward_dim"] == 2048
    assert config.payload["budgets"]["train_token_budget"] == 10_229_514_240
    assert config.payload["optimizer"]["training"]["micro_batch_size"] == 24
    assert config.payload["optimizer"]["training"]["grad_accum_steps"] == 16
    assert config.selected_gpu == "B200"
    assert set(config.payload["runtime"]["gpu_profiles"]) == {"H100!", "H200", "B200"}
    assert config.tokens_per_step == 24 * 16 * 1024
    assert config.train_steps == 26015
    assert config.estimated_cost_usd == pytest.approx(
        config.payload["budgets"]["train_token_budget"]
        / config.selected_gpu_profile["projected_tokens_per_second"]
        * config.selected_gpu_profile["usd_per_hour"]
        / 3600,
        rel=1e-6,
    )
    assert tuple(config.payload["artifacts"]["required_files"]) == EXPECTED_ARTIFACTS


def test_pretrain_214m_b200_dry_run_includes_gpu_projection_and_approval_gate() -> None:
    config = load_pretrain_config(CONFIG_214M_B200)
    payload = build_pretrain_dry_run(config)

    assert payload["status"] == "ready_for_pretrain_launch"
    assert payload["launch_blockers"] == []
    assert payload["requires_approval"] is True
    assert payload["approval_flag"] == LAUNCH_APPROVAL_FLAG
    assert payload["will_download_data"] is False
    assert payload["will_start_modal_job"] is False
    assert payload["parameter_count"]["total"] == 213_960_192
    assert payload["model"]["layers"] == 30
    assert payload["model"]["embedding_dim"] == 768
    assert payload["model"]["heads"] == 12
    assert payload["model"]["kv_heads"] == 4
    assert payload["model"]["feedforward_dim"] == 2048
    assert payload["budgets"]["train_token_budget"] == 10_229_514_240
    assert payload["runtime"]["selected_gpu"] == "B200"
    profile = config.selected_gpu_profile
    assert payload["runtime"]["selected_gpu_profile"]["modal_gpu"] == "B200"
    assert payload["runtime"]["selected_gpu_profile"]["expected_duration_hours"] == pytest.approx(
        round(config.train_token_budget / profile["projected_tokens_per_second"] / 3600, 2)
    )
    assert payload["runtime"]["estimated_cost_usd"] == pytest.approx(
        round(config.estimated_cost_usd, 2)
    )
    assert payload["runtime"]["estimated_usd_per_1b_tokens"] == pytest.approx(
        round(profile["usd_per_hour"] / profile["projected_tokens_per_second"] * 1e9 / 3600, 2)
    )
    assert "PRETRAIN_GPU='B200'" in payload["launch_command"]
    assert "modal run --detach" in payload["launch_command"]
    assert "--approved" in payload["launch_command"]
    assert "--approved-by-" + "adam" not in payload["launch_command"]


def test_pretrain_214m_b200_full_run_cap_is_100_usd() -> None:
    config = load_pretrain_config(CONFIG_214M_B200)
    payload = build_pretrain_dry_run(config)

    assert payload["runtime"]["max_cost_usd"] == 100
    assert payload["runtime"]["absolute_cost_cap_usd"] == 100
    assert payload["runtime"]["runtime_spend_stop_usd"] == 100
    assert payload["runtime"]["estimated_cost_usd"] < 100
    assert payload["runtime"]["timeout_hours"] <= 24
    assert payload["runtime"]["selected_gpu_profile"]["expected_duration_hours"] <= 24


def test_pretrain_gpu_smoke_ledger_refuses_cap_overrun(tmp_path: Path) -> None:
    module = _gpu_smoke_module()
    ledger_path = tmp_path / "ledger.json"

    first = module.reserve_smoke_attempt(
        ledger_path,
        gpu="H100!",
        reserved_cost_usd=6.0,
        spend_cap_usd=10.0,
        params={"max_steps": 3},
    )
    assert first["status"] == "reserved"

    try:
        module.reserve_smoke_attempt(
            ledger_path,
            gpu="H200",
            reserved_cost_usd=5.0,
            spend_cap_usd=10.0,
            params={"max_steps": 3},
        )
    except ValueError as error:
        assert "smoke spend cap would be exceeded" in str(error)
    else:
        raise AssertionError("ledger allowed an over-cap reservation")


def test_pretrain_gpu_smoke_ledger_releases_unused_reserve(tmp_path: Path) -> None:
    module = _gpu_smoke_module()
    ledger_path = tmp_path / "ledger.json"

    reserved = module.reserve_smoke_attempt(
        ledger_path,
        gpu="H100!",
        reserved_cost_usd=6.0,
        spend_cap_usd=10.0,
        params={"max_steps": 3},
    )
    module.mark_smoke_attempt(
        ledger_path,
        reserved["attempt_id"],
        status="complete",
        actual_cost_usd=1.25,
        result={"status": "smoke_complete"},
    )
    second = module.reserve_smoke_attempt(
        ledger_path,
        gpu="H200",
        reserved_cost_usd=5.0,
        spend_cap_usd=10.0,
        params={"max_steps": 3},
    )

    assert second["status"] == "reserved"
    assert second["spend_used_before_usd"] == 1.25


def test_pretrain_config_rejects_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_214M_B200.read_text(encoding="utf-8"))
    payload["runtime"]["max_cost_usd"] = 40
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        load_pretrain_config(config_path)
    except ValueError as error:
        assert "runtime.max_cost_usd" in str(error)
    else:
        raise AssertionError("config drift was accepted")


def _fake_tokenizers_namespace(captured: dict[str, object]):
    """Minimal `tokenizers` stand-in with digit-aware pre-tokenization.

    ``Digits``/``ByteLevel``/``Sequence`` implement enough of ``pre_tokenize_str``
    to prove the digit-split branch in ``_train_tokenizer`` actually splits across
    digit boundaries, without pulling in the Modal-image ``tokenizers`` dependency.
    """
    import re

    class FakeByteLevel:
        def __init__(self, add_prefix_space: bool = False) -> None:
            self.add_prefix_space = add_prefix_space

        @staticmethod
        def alphabet() -> list[str]:
            return []

        def pre_tokenize_str(self, text: str) -> list[tuple[str, tuple[int, int]]]:
            return [(text, (0, len(text)))]

    class FakeDigits:
        def __init__(self, individual_digits: bool = False) -> None:
            self.individual_digits = individual_digits

        def pre_tokenize_str(self, text: str) -> list[tuple[str, tuple[int, int]]]:
            pieces: list[tuple[str, tuple[int, int]]] = []
            cursor = 0
            for chunk in re.findall(r"\d|\D+", text):
                pieces.append((chunk, (cursor, cursor + len(chunk))))
                cursor += len(chunk)
            return pieces

    class FakeSequence:
        def __init__(self, members: list[object]) -> None:
            self.members = members

        def pre_tokenize_str(self, text: str) -> list[tuple[str, tuple[int, int]]]:
            pieces = [(text, (0, len(text)))]
            for member in self.members:
                expanded: list[tuple[str, tuple[int, int]]] = []
                for piece_text, _ in pieces:
                    expanded.extend(member.pre_tokenize_str(piece_text))
                pieces = expanded
            return pieces

    class FakeBPE:
        def __init__(self, unk_token: str | None = None) -> None:
            self.unk_token = unk_token

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeByteDecoder:
        def decode(self, tokens: list[str]) -> str:
            return "".join(tokens)

    class FakeTokenizer:
        def __init__(self, model: object) -> None:
            self.model = model
            self.pre_tokenizer = None
            self.decoder = None
            self._texts: list[str] = []

        def train_from_iterator(self, texts, trainer) -> None:
            captured["pre_tokenizer"] = self.pre_tokenizer

        def save(self, path: str) -> None:
            Path(path).write_text("{}", encoding="utf-8")

        def encode(self, text: str):
            pieces = self.pre_tokenizer.pre_tokenize_str(text)
            index = len(self._texts)
            self._texts.append(text)
            # Encode the text index in the first id so decode is unambiguous; the
            # remaining ids carry the piece count.
            ids = [index] + list(range(len(pieces)))
            return SimpleNamespace(ids=ids, _pieces=pieces)

        def decode(self, ids: list[int]) -> str:
            return self._texts[ids[0]] if ids else ""

        def get_vocab_size(self) -> int:
            return 32768

    pre_tokenizers = SimpleNamespace(
        ByteLevel=FakeByteLevel, Digits=FakeDigits, Sequence=FakeSequence
    )
    return {
        "tokenizers": SimpleNamespace(Tokenizer=FakeTokenizer),
        "tokenizers.models": SimpleNamespace(BPE=FakeBPE),
        "tokenizers.trainers": SimpleNamespace(BpeTrainer=FakeTrainer),
        "tokenizers.pre_tokenizers": pre_tokenizers,
        "tokenizers.decoders": SimpleNamespace(ByteLevel=FakeByteDecoder),
    }


def _train_with_fake_tokenizers(config, output_dir: Path, captured: dict[str, object]):
    fakes = _fake_tokenizers_namespace(captured)

    def fake_import(name: str):
        try:
            return fakes[name]
        except KeyError as error:
            raise AssertionError(f"unexpected import: {name}") from error

    import unittest.mock as mock

    with mock.patch.object(modal_tokenizer.importlib, "import_module", fake_import):
        return modal_tokenizer._train_tokenizer(
            config, output_dir, iter(["1234 abc"]), require_target_vocab=False
        )


def test_split_digits_pretokenizer_splits_multi_digit_numbers(tmp_path: Path) -> None:
    config = load_pretrain_config(CONFIG_214M_B200)
    assert config.payload["tokenizer"]["split_digits"] is True

    captured: dict[str, object] = {}
    _train_with_fake_tokenizers(config, tmp_path, captured)

    pre_tokenizer = captured["pre_tokenizer"]
    pieces = [text for text, _ in pre_tokenizer.pre_tokenize_str("1234")]
    assert pieces == ["1", "2", "3", "4"]  # no merges across digit boundaries


def test_without_split_digits_multi_digit_numbers_are_not_split(tmp_path: Path) -> None:
    config = load_pretrain_config(CONFIG_214M_B200)
    config.payload["tokenizer"]["split_digits"] = False

    captured: dict[str, object] = {}
    _train_with_fake_tokenizers(config, tmp_path, captured)

    pre_tokenizer = captured["pre_tokenizer"]
    pieces = [text for text, _ in pre_tokenizer.pre_tokenize_str("1234")]
    assert pieces == ["1234"]  # plain byte-level keeps the digit run intact


def test_invalid_trained_tokenizer_is_not_persisted(tmp_path: Path, monkeypatch) -> None:
    config = load_pretrain_config(CONFIG_214M_B200)
    config.payload["tokenizer"]["vocab_size"] = 32769
    fakes = _fake_tokenizers_namespace({})

    def fake_import(name: str):
        try:
            return fakes[name]
        except KeyError as error:
            raise AssertionError(f"unexpected import: {name}") from error

    monkeypatch.setattr(modal_tokenizer.importlib, "import_module", fake_import)

    try:
        modal_tokenizer._train_tokenizer(config, tmp_path, iter(["1234 abc"]))
    except RuntimeError as error:
        assert "tokenizer did not reach the configured vocab size" in str(error)
    else:
        raise AssertionError("invalid tokenizer was accepted")

    assert not (tmp_path / "tokenizer.json").exists()
    assert not (tmp_path / "tokenizer-report.json").exists()


def test_existing_tokenizer_is_loaded_instead_of_retrained(tmp_path: Path, monkeypatch) -> None:
    config = load_pretrain_config(CONFIG_214M_B200)
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer-report.json").write_text(
        json.dumps({"round_trips": [{"round_trip": True}]}), encoding="utf-8"
    )

    class FakeTokenizer:
        loaded_path: str | None = None

        @classmethod
        def from_file(cls, path: str):
            cls.loaded_path = path
            return cls()

        def get_vocab_size(self) -> int:
            return config.payload["tokenizer"]["vocab_size"]

    def fake_import(name: str):
        if name == "tokenizers":
            return SimpleNamespace(Tokenizer=FakeTokenizer)
        raise AssertionError(f"unexpected tokenizer training import: {name}")

    monkeypatch.setattr(modal_tokenizer.importlib, "import_module", fake_import)
    tokenizer, report = modal_tokenizer._load_or_train_tokenizer(config, tmp_path)

    assert isinstance(tokenizer, FakeTokenizer)
    assert FakeTokenizer.loaded_path == str(tmp_path / "tokenizer.json")
    assert report["source"] == "loaded_existing_tokenizer"
    persisted = json.loads((tmp_path / "tokenizer-report.json").read_text(encoding="utf-8"))
    assert persisted["source"] == "loaded_existing_tokenizer"


def test_bounded_tokenizer_texts_use_shared_corpus_stream(monkeypatch) -> None:
    config = SimpleNamespace(payload={"budgets": {"tokenizer_training_token_budget": 2}})
    captured: dict[str, object] = {}

    def fake_document_text_stream(stream_config, *, split: str):
        captured["config"] = stream_config
        captured["split"] = split
        yield "abcd"
        yield "efghi"
        yield "unused"

    monkeypatch.setattr(modal_tokenizer, "document_text_stream", fake_document_text_stream)

    assert list(modal_tokenizer._bounded_texts_for_tokenizer(config)) == ["abcd"]
    assert captured == {"config": config, "split": "train"}


def test_wandb_resume_run_id_prefers_persisted_id(tmp_path: Path) -> None:
    module = _modal_pretrain_module()
    (tmp_path / "wandb-run-id.txt").write_text("x99drn15\n", encoding="utf-8")

    assert module._wandb_resume_run_id(tmp_path) == "x99drn15"


def test_wandb_resume_run_id_falls_back_to_existing_wandb_run_dir(tmp_path: Path) -> None:
    module = _modal_pretrain_module()
    run_dir = tmp_path / "wandb" / "run-20260625_220548-x99drn15"
    run_dir.mkdir(parents=True)

    assert module._wandb_resume_run_id(tmp_path) == "x99drn15"


def test_modal_pretrain_refuses_without_approval(capsys) -> None:
    module = _modal_pretrain_module()
    exit_code = module.launch(["--config", str(CONFIG_214M_B200), "--json"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert LAUNCH_APPROVAL_FLAG in captured.err


def test_pretrain_config_rejects_split_digits_false(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_214M_B200.read_text(encoding="utf-8"))
    payload["tokenizer"]["split_digits"] = False
    config_path = tmp_path / "bad-tokenizer.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="split_digits"):
        load_pretrain_config(config_path)


def test_pretrain_config_rejects_unpinned_dataset_revision(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_214M_B200.read_text(encoding="utf-8"))
    payload["dataset"]["revision"] = "deadbeef"
    config_path = tmp_path / "bad-dataset.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="revision"):
        load_pretrain_config(config_path)


def test_pretrain_config_rejects_model_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_214M_B200.read_text(encoding="utf-8"))
    payload["model"]["layers"] = 29
    config_path = tmp_path / "bad-model.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline_config"):
        load_pretrain_config(config_path)
