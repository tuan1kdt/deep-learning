from vivlm.config import SPECIAL_TOKENS
from vivlm.tokenizer_train import fertility, train_bpe
from vivlm.tests.conftest import SAMPLE_VI


def test_special_token_ids(tiny_tokenizer):
    for i, t in enumerate(SPECIAL_TOKENS):
        assert tiny_tokenizer.token_to_id(t) == i


def test_roundtrip_vietnamese(tiny_tokenizer):
    s = "Xin chào, đây là tiếng Việt có dấu: ắẳẵế ơư đ!"
    ids = tiny_tokenizer.encode(s).ids
    assert tiny_tokenizer.decode(ids) == s


def test_special_tokens_atomic(tiny_tokenizer):
    ids = tiny_tokenizer.encode("<|user|> hi <|assistant|>").ids
    assert ids.count(1) == 1 and ids.count(2) == 1   # mỗi special = đúng 1 id


def test_fertility_positive(tiny_tokenizer):
    f = fertility(tiny_tokenizer, SAMPLE_VI)
    assert 1.0 <= f < 10.0
