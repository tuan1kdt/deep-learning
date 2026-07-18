import torch
import pytest
from vivlm.config import GPTConfig
from vivlm.models.gpt import GPT, RMSNorm, precompute_rope, apply_rope

TINY = GPTConfig(vocab_size=64, n_layer=2, n_head=4, d_model=32,
                 context=16, mlp_hidden=64)


def test_rmsnorm_unit_scale():
    x = torch.randn(2, 5, 8)
    y = RMSNorm(8)(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rope_preserves_norm():
    cos, sin = precompute_rope(8, 16)
    x = torch.randn(2, 4, 16, 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
    # vị trí 0 không xoay (góc 0)
    assert torch.allclose(x[:, :, 0], y[:, :, 0], atol=1e-6)


def test_forward_shapes_and_loss():
    m = GPT(TINY)
    idx = torch.randint(0, 64, (2, 10))
    tgt = torch.randint(0, 64, (2, 10))
    logits, loss = m(idx, tgt)
    assert logits.shape == (2, 10, 64)
    assert loss.isfinite()
    logits_only, loss_none = m(idx)
    assert logits_only.shape == (2, 1, 64) and loss_none is None


def test_causality():
    m = GPT(TINY).eval()
    idx = torch.randint(0, 64, (1, 10))
    idx2 = idx.clone()
    idx2[0, -1] = (idx2[0, -1] + 1) % 64          # đổi token CUỐI
    with torch.no_grad():
        a, _ = m(idx, idx)
        b, _ = m(idx2, idx2)
    # logits các vị trí TRƯỚC không đổi
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)


def test_loss_mask_ignore_index():
    m = GPT(TINY)
    idx = torch.randint(0, 64, (1, 8))
    tgt = torch.randint(0, 64, (1, 8))
    tgt[0, :4] = -100
    _, loss = m(idx, tgt)
    # loss thủ công chỉ trên 4 vị trí cuối
    logits, _ = m(idx)  # (1,1,64) — không dùng được; tính lại full
    full_logits, _ = m(idx, torch.zeros_like(idx))
    manual = torch.nn.functional.cross_entropy(
        full_logits[0, 4:], tgt[0, 4:])
    assert torch.allclose(loss, manual, atol=1e-5)


def test_weight_tying_and_param_count():
    m = GPT(TINY)
    assert m.lm_head.weight is m.tok_emb.weight
    full = GPT(GPTConfig())
    n = full.num_params()
    assert abs(n - 100_682_496) < 10_000, n     # ~100.7M


def test_generate_greedy_deterministic():
    m = GPT(TINY).eval()
    idx = torch.randint(0, 64, (1, 4))
    a = m.generate(idx.clone(), 5, temperature=0.0)
    b = m.generate(idx.clone(), 5, temperature=0.0)
    assert a.shape == (1, 9) and torch.equal(a, b)
