import torch

from final.data.dataset import CaptionDataset
from final.data.vocab import BOS_ID, EOS_ID, PAD_ID, build_vocab


def _make_ds(max_words=5):
    feats = torch.randn(2, 49, 8).to(torch.float16)   # feat_dim thu nhỏ cho test
    caps = [["a dog runs", "a dog sleeps"], ["a cat runs", "a cat sleeps"]]
    vocab = build_vocab([c for g in caps for c in g], min_freq=1)
    return CaptionDataset(feats, caps, vocab, max_words=max_words), vocab


def test_moi_cap_anh_caption_la_mot_mau():
    ds, _ = _make_ds()
    assert len(ds) == 4                                # 2 ảnh × 2 caption


def test_shape_va_dtype():
    ds, _ = _make_ds(max_words=5)
    item = ds[0]
    assert item["feat"].dtype == torch.float32        # cast từ fp16 khi lấy ra
    assert item["feat"].shape == (49, 8)
    assert item["cap_in"].shape == (6,)               # max_words + 1
    assert item["cap_out"].shape == (6,)


def test_cap_in_cap_out_lech_nhau_dung_mot_buoc():
    ds, vocab = _make_ds(max_words=5)
    item = ds[0]                                       # "a dog runs" của ảnh 0
    ids = vocab.encode("a dog runs", max_words=5)
    assert item["cap_in"].tolist() == [BOS_ID] + ids + [PAD_ID] * 2
    assert item["cap_out"].tolist() == ids + [EOS_ID] + [PAD_ID] * 2


def test_anh_dung_voi_caption():
    ds, _ = _make_ds()
    # item 2 và 3 phải là feature của ảnh 1
    assert torch.equal(ds[2]["feat"], ds[3]["feat"])
    assert not torch.equal(ds[0]["feat"], ds[2]["feat"])
