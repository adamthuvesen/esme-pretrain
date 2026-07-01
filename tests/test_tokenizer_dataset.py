import pytest

from esme_pretrain.data.dataset import build_pilot_datasets, pack_token_ids
from esme_pretrain.tokenization.tokenizer import CharTokenizer, PairMergeTokenizer


def test_char_tokenizer_round_trips_and_rejects_unknown_text() -> None:
    tokenizer = CharTokenizer.from_text("pilot\n")

    encoded = tokenizer.encode("pilot")

    assert tokenizer.decode(encoded) == "pilot"
    with pytest.raises(ValueError, match="outside the pilot vocabulary"):
        tokenizer.encode("pilot!")


def test_pair_merge_tokenizer_round_trips_and_rejects_unknown_text() -> None:
    tokenizer = PairMergeTokenizer.from_text("banana bandana\n" * 3, target_vocab_size=12)

    encoded = tokenizer.encode("banana bandana\n")

    assert len(encoded) < len("banana bandana\n")
    assert tokenizer.decode(encoded) == "banana bandana\n"
    with pytest.raises(ValueError, match="outside the tokenizer vocabulary"):
        tokenizer.encode("banana!")


def test_pair_merge_tokenizer_vocab_is_deterministic() -> None:
    text = "the theater there then\n" * 4

    first = PairMergeTokenizer.from_text(text, target_vocab_size=14)
    second = PairMergeTokenizer.from_text(text, target_vocab_size=14)

    assert first.id_to_token == second.id_to_token
    assert first.merges == second.merges


def test_pack_token_ids_builds_next_token_windows() -> None:
    packed = pack_token_ids([0, 1, 2, 3, 4], context_length=2)

    assert packed.inputs.tolist() == [[0, 1], [2, 3]]
    assert packed.targets.tolist() == [[1, 2], [3, 4]]


def test_build_pilot_datasets_splits_train_and_validation() -> None:
    text = "abcd" * 40
    tokenizer = CharTokenizer.from_text(text)

    train, validation = build_pilot_datasets(text, tokenizer, context_length=8)

    assert train.rows > validation.rows
    assert train.inputs.shape[1] == 8
    assert validation.targets.shape[1] == 8
