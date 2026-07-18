"""Dataset SFT: ảnh + chuỗi <|user|> prompt <|assistant|> response <|endoftext|>.

Quy ước labels: dịch sẵn 1 vị trí (labels[t] = token cần dự đoán tại t),
-100 trên prompt/pad — model CHỈ học phần trả lời (bản chất của SFT).
"""
import json

import torch
from PIL import Image
from torch.utils.data import Dataset


def preprocess_image(pil_img, size=224):
    """Resize + normalize về [-1,1] (mean=std=0.5, khớp SigLIP)."""
    img = pil_img.convert("RGB").resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.view(size, size, 3).permute(2, 0, 1).float() / 255.0
    return (t - 0.5) / 0.5


def encode_chat(tok, prompt, response, max_len=256):
    eos = tok.token_to_id("<|endoftext|>")
    prompt_ids = tok.encode(f"<|user|> {prompt} <|assistant|>").ids
    resp_ids = tok.encode(f" {response}").ids + [eos]
    seq = (prompt_ids + resp_ids)[: max_len + 1]      # +1 vì shift mất 1
    input_ids = seq[:-1]
    labels = seq[1:]
    n_mask = min(len(prompt_ids) - 1, len(labels))    # labels[t]=seq[t+1]:
    labels = [-100] * n_mask + labels[n_mask:]        # mask đến hết <|assistant|>
    return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, img_root, max_text_len=256,
                 img_size=224):
        self.recs = [json.loads(l) for l in
                     open(jsonl_path, encoding="utf-8") if l.strip()]
        self.tok = tokenizer
        self.img_root = img_root
        self.max_text_len = max_text_len
        self.img_size = img_size

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        px = preprocess_image(Image.open(f"{self.img_root}/{r['image']}"),
                              self.img_size)
        ids, labels = encode_chat(self.tok, r["prompt"], r["response"],
                                  self.max_text_len)
        return {"pixel_values": px,
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def collate(batch, pad_id):
    T = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "input_ids": ids, "labels": labels}
