"""RawCaptionDataset là logic thuần — test bằng ảnh PIL giả, không cần network."""
import torch
from PIL import Image

from final.data.vocab import BOS_ID, EOS_ID, PAD_ID, Vocab
from final.finetune import RawCaptionDataset


def _vocab():
    return Vocab({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                  "a": 4, "dog": 5, "runs": 6})


def test_raw_dataset_shape_va_ma_hoa_caption():
    rows = [{"image": Image.new("RGB", (300, 280))},
            {"image": Image.new("RGB", (260, 320))}]
    caps = [["a dog runs", "a dog"], ["runs", "a"]]
    ds = RawCaptionDataset(rows, caps, _vocab(), max_words=5, train=False)
    assert len(ds) == 4  # 2 ảnh × 2 caption
    s = ds[0]
    assert s["image"].shape == (3, 224, 224)
    assert s["cap_in"].tolist() == [BOS_ID, 4, 5, 6, PAD_ID, PAD_ID]
    assert s["cap_out"].tolist() == [4, 5, 6, EOS_ID, PAD_ID, PAD_ID]


def test_raw_dataset_train_augment_ngau_nhien_eval_thi_khong():
    rows = [{"image": Image.effect_noise((320, 320), 64).convert("RGB")}]
    caps = [["a dog"]]
    torch.manual_seed(0)
    train_ds = RawCaptionDataset(rows, caps, _vocab(), 5, train=True)
    eval_ds = RawCaptionDataset(rows, caps, _vocab(), 5, train=False)
    assert not torch.equal(train_ds[0]["image"], train_ds[0]["image"])
    assert torch.equal(eval_ds[0]["image"], eval_ds[0]["image"])
