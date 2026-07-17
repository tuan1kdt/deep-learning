import numpy as np
import torch

from final.scst import cider_rewards, sample_decode


def test_cider_rewards_thu_tu_dung():
    refs = [["a dog runs on grass"], ["a girl in red dress"]]
    r = cider_rewards(["a dog runs on grass", "blue elephant flies"], refs)
    assert r.shape == (2,)
    assert r[0] > r[1]                    # câu trùng ref phải điểm cao hơn


def test_sample_decode_logprob_am_va_dung_shape():
    from final.config import Config
    from final.models.caption_model import build_model

    torch.manual_seed(0)
    cfg = Config(decoder="lstm", d_model=16, attn_dim=8, dropout=0.0,
                 feat_dim=8, max_words=6)
    model = build_model(cfg, vocab_size=12)
    seqs, logps = sample_decode(model, torch.randn(3, 49, 8), cfg.max_len)
    assert len(seqs) == 3
    assert logps.shape == (3,)
    assert (logps <= 0).all()             # log-prob luôn ≤ 0
    assert logps.requires_grad            # cần gradient cho REINFORCE
