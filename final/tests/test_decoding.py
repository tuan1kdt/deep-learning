import torch
from torch import nn

from final.config import Config
from final.data.vocab import BOS_ID, EOS_ID
from final.models.caption_model import build_model
from final.models.decoding import beam_search, greedy_decode

V = 12


class StubModel(nn.Module):
    """Model giả ép sinh đúng chuỗi TARGET rồi EOS — kiểm tra cơ chế decode
    tách biệt khỏi chất lượng model."""

    TARGET = [5, 7, 9]

    def forward(self, feats, cap_in):
        B, T = cap_in.shape
        logits = torch.full((B, T, V), -10.0)
        for t in range(T):
            want = self.TARGET[t] if t < len(self.TARGET) else EOS_ID
            logits[:, t, want] = 10.0
        return logits, torch.full((B, T, 49), 1.0 / 49)


def test_greedy_sinh_dung_chuoi_va_dung_o_eos():
    seqs, attn = greedy_decode(StubModel(), torch.randn(2, 49, 8), max_len=10)
    assert seqs == [[5, 7, 9], [5, 7, 9]]              # không BOS/EOS trong output
    assert attn.shape[0] == 2 and attn.shape[2] == 49


def test_greedy_khong_vuot_max_len():
    class NeverEOS(StubModel):
        TARGET = list(range(4, 11)) * 5                 # không bao giờ ra EOS

    seqs, _ = greedy_decode(NeverEOS(), torch.randn(1, 49, 8), max_len=6)
    assert len(seqs[0]) == 5                            # max_len trừ BOS


def test_beam_1_bang_greedy_tren_model_that():
    torch.manual_seed(0)
    cfg = Config(decoder="lstm", d_model=16, attn_dim=8, dropout=0.0,
                 feat_dim=8, max_words=6)
    model = build_model(cfg, vocab_size=V).eval()
    feats = torch.randn(1, 49, 8)
    greedy_ids, _ = greedy_decode(model, feats, cfg.max_len)
    beam_ids = beam_search(model, feats, beam_size=1, max_len=cfg.max_len)
    assert beam_ids == greedy_ids[0]


def test_decode_khoi_phuc_trang_thai_train():
    cfg = Config(decoder="lstm", d_model=16, attn_dim=8, dropout=0.0,
                 feat_dim=8, max_words=6)
    model = build_model(cfg, vocab_size=V)
    feats = torch.randn(1, 49, 8)

    model.train()
    greedy_decode(model, feats, cfg.max_len)
    assert model.training is True                       # train được khôi phục

    beam_search(model, feats, beam_size=2, max_len=cfg.max_len)
    assert model.training is True

    model.eval()
    greedy_decode(model, feats, cfg.max_len)
    assert model.training is False                      # eval giữ nguyên


def test_build_model_transformer_va_dem_tham_so():
    cfg = Config(decoder="transformer", d_model=16, num_layers=1, num_heads=2,
                 ffn_dim=32, dropout=0.0, feat_dim=8, max_words=6)
    model = build_model(cfg, vocab_size=V)
    logits, attn = model(torch.randn(2, 49, 8), torch.full((2, 3), BOS_ID))
    assert logits.shape == (2, 3, V) and attn.shape == (2, 3, 49)
    total, trainable = model.count_parameters()
    assert total == trainable and total > 0
