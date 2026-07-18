from vivlm.config import (GPTConfig, PretrainConfig, SFTConfig, SCSTConfig,
                          SPECIAL_TOKENS, pick_device)


def test_special_tokens_order():
    assert SPECIAL_TOKENS == ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|image|>"]


def test_gpt_config_defaults():
    c = GPTConfig()
    assert (c.vocab_size, c.n_layer, c.n_head, c.d_model, c.context) == (20480, 12, 12, 768, 1024)
    assert c.d_model % c.n_head == 0


def test_pretrain_batch_math():
    c = PretrainConfig()
    # batch_tokens phải chia hết cho micro_batch * context (grad accum nguyên)
    assert c.batch_tokens % (c.micro_batch * c.gpt.context) == 0


def test_pick_device():
    assert pick_device() in ("cuda", "mps", "cpu")
