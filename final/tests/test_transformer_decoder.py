import torch

from final.models.transformer_decoder import TransformerDecoder

B, R, D, V, T = 2, 49, 16, 30, 5


def _decoder():
    return TransformerDecoder(vocab_size=V, d_model=D, num_layers=2,
                              num_heads=2, ffn_dim=32, dropout=0.0, max_len=10)


def test_shapes():
    dec = _decoder()
    logits, attn = dec(torch.randn(B, R, D), torch.randint(0, V, (B, T)))
    assert logits.shape == (B, T, V)
    assert attn.shape == (B, T, R)
    assert torch.allclose(attn.sum(-1), torch.ones(B, T), atol=1e-5)


def test_causal_tuong_lai_khong_anh_huong_qua_khu():
    """Đổi token cuối không được làm thay đổi logits các vị trí trước —
    nếu fail nghĩa là causal mask sai chiều (bug kinh điển)."""
    dec = _decoder().eval()
    feats = torch.randn(B, R, D)
    cap = torch.randint(0, V, (B, T))
    with torch.no_grad():
        logits1, _ = dec(feats, cap)
        cap2 = cap.clone()
        cap2[:, -1] = (cap2[:, -1] + 1) % V
        logits2, _ = dec(feats, cap2)
    assert torch.allclose(logits1[:, :-1], logits2[:, :-1], atol=1e-5)


def test_tie_weight():
    dec = _decoder()
    assert dec.fc.weight is dec.embedding.weight


def test_embedding_init_nho_khi_tie_weight():
    """Regression: init N(0,1) mặc định + final LayerNorm làm logits có
    std ~ sqrt(d_model) → CE ban đầu ~267 — phải init std=0.02 kiểu GPT."""
    dec = _decoder()
    assert dec.embedding.weight.std().item() < 0.1  # ~1.0 nếu init mặc định
    assert dec.embedding.weight[0].abs().sum().item() == 0.0  # hàng pad = 0 tại init
    assert dec.pos_embedding.weight.std().item() < 0.1


def test_qua_max_len_bi_chan():
    import pytest
    dec = _decoder()   # max_len=10
    with pytest.raises(AssertionError):
        dec(torch.randn(B, R, D), torch.randint(0, V, (B, 11)))
