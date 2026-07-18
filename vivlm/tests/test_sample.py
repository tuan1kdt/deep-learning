import torch

from vivlm.config import GPTConfig, PretrainConfig
from vivlm.models.gpt import GPT
from vivlm.pretrain import save_ckpt
from vivlm.sample import load_model, generate_text


def _make_ckpt(tmp_path):
    cfg = PretrainConfig(gpt=GPTConfig(vocab_size=300, n_layer=1, n_head=2,
                                       d_model=16, context=32, mlp_hidden=32))
    m = GPT(cfg.gpt)
    opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    p = tmp_path / "ck.pt"
    save_ckpt(str(p), m, opt, 1, cfg, torch.Generator())
    return str(p)


def test_load_model_roundtrip(tmp_path):
    m = load_model(_make_ckpt(tmp_path), "cpu")
    assert m.cfg.vocab_size == 300 and not m.training


def test_generate_text_returns_string(tmp_path, tiny_tokenizer):
    m = load_model(_make_ckpt(tmp_path), "cpu")
    out = generate_text(m, tiny_tokenizer, "xin chào", max_new=8,
                        temperature=0.0)
    assert isinstance(out, str) and len(out) > 0
