"""Dataset cho captioning: mỗi mẫu = (feature ảnh precomputed, 1 caption).

6000 ảnh × 5 caption = 30k mẫu train. Feature lưu fp16 cho nhẹ đĩa/RAM nhưng
trả ra fp32 — model train ổn định ở fp32, cast một lần tại đây thay vì rải
rác trong model.
"""
import json

import torch
from torch.utils.data import Dataset

from final.config import Config
from final.data.vocab import BOS_ID, EOS_ID, PAD_ID, Vocab


class CaptionDataset(Dataset):
    def __init__(self, features: torch.Tensor, captions: list[list[str]],
                 vocab: Vocab, max_words: int):
        assert features.shape[0] == len(captions)
        self.features = features
        self.vocab = vocab
        self.max_words = max_words
        # Trải phẳng (ảnh, caption) một lần lúc init — encode sẵn để __getitem__
        # không tokenize lại mỗi epoch (30k mẫu × 25 epoch).
        self.samples: list[tuple[int, list[int]]] = [
            (img_idx, vocab.encode(cap, max_words))
            for img_idx, group in enumerate(captions)
            for cap in group
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        img_idx, ids = self.samples[i]
        pad = [PAD_ID] * (self.max_words - len(ids))
        return {
            "feat": self.features[img_idx].to(torch.float32),
            "cap_in": torch.tensor([BOS_ID] + ids + pad, dtype=torch.long),
            "cap_out": torch.tensor(ids + [EOS_ID] + pad, dtype=torch.long),
        }


def load_caption_dataset(cfg: Config, split: str, vocab: Vocab) -> CaptionDataset:
    feats = torch.load(cfg.features_path(split))["features"]
    caps = json.loads(cfg.captions_path(split).read_text())
    return CaptionDataset(feats, caps, vocab, cfg.max_words)


def load_eval_data(cfg: Config, split: str):
    """Cho evaluate/visualize: feature fp32 (N,49,2048) + 5 refs thô mỗi ảnh."""
    feats = torch.load(cfg.features_path(split))["features"].to(torch.float32)
    refs = json.loads(cfg.captions_path(split).read_text())
    return feats, refs
