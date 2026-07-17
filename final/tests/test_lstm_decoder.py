import torch

from final.models.image_proj import ImageProjection
from final.models.lstm_decoder import BahdanauAttention, LSTMDecoder

B, R, D, V, T = 2, 49, 16, 30, 5


def test_image_projection_shape():
    proj = ImageProjection(feat_dim=32, d_model=D, dropout=0.0)
    out = proj(torch.randn(B, R, 32))
    assert out.shape == (B, R, D)


def test_bahdanau_weights_tong_bang_1():
    attn = BahdanauAttention(d_model=D, attn_dim=8)
    ctx, w = attn(torch.randn(B, D), torch.randn(B, R, D))
    assert ctx.shape == (B, D) and w.shape == (B, R)
    assert torch.allclose(w.sum(-1), torch.ones(B), atol=1e-5)
    assert (w >= 0).all()


def test_lstm_decoder_shapes():
    dec = LSTMDecoder(vocab_size=V, d_model=D, attn_dim=8, dropout=0.0,
                      use_attention=True)
    logits, attn = dec(torch.randn(B, R, D), torch.randint(0, V, (B, T)))
    assert logits.shape == (B, T, V)
    assert attn.shape == (B, T, R)
    assert torch.allclose(attn.sum(-1), torch.ones(B, T), atol=1e-5)


def test_khong_attention_thi_trong_so_deu():
    dec = LSTMDecoder(vocab_size=V, d_model=D, attn_dim=8, dropout=0.0,
                      use_attention=False)
    _, attn = dec(torch.randn(B, R, D), torch.randint(0, V, (B, T)))
    assert torch.allclose(attn, torch.full((B, T, R), 1.0 / R), atol=1e-6)


def test_tie_weight_embedding_va_output():
    dec = LSTMDecoder(vocab_size=V, d_model=D, attn_dim=8, dropout=0.0,
                      use_attention=True)
    assert dec.fc.weight is dec.embedding.weight
