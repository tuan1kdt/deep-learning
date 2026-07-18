import json

import torch
from PIL import Image

from vivlm.data.sft_dataset import (SFTDataset, collate, encode_chat,
                                    preprocess_image)
from vivlm.data.prepare_sft import CAPTION_PROMPTS


def test_caption_prompts():
    assert len(CAPTION_PROMPTS) == 5
    assert all(isinstance(p, str) and p for p in CAPTION_PROMPTS)


def test_encode_chat_masks_prompt(tiny_tokenizer):
    ids, labels = encode_chat(tiny_tokenizer, "Ảnh có gì?", "ba chiếc thuyền",
                              max_len=64)
    assert len(ids) == len(labels)
    asst_id = tiny_tokenizer.token_to_id("<|assistant|>")
    pos = ids.index(asst_id)
    assert all(l == -100 for l in labels[:pos])       # prompt mask hết
    assert labels[pos] != -100                         # từ sau <|assistant|> học
    assert labels[-1] == tiny_tokenizer.token_to_id("<|endoftext|>")
    # labels dịch 1: labels[t] == ids[t+1] tại vị trí không mask
    assert labels[pos] == ids[pos + 1]


def test_encode_chat_truncates(tiny_tokenizer):
    ids, labels = encode_chat(tiny_tokenizer, "a " * 200, "b " * 200, max_len=32)
    assert len(ids) <= 32


def test_preprocess_image_range():
    img = Image.new("RGB", (64, 100), (255, 0, 0))
    t = preprocess_image(img, 224)
    assert t.shape == (3, 224, 224)
    assert t.max() <= 1.0 and t.min() >= -1.0


def test_dataset_and_collate(tmp_path, tiny_tokenizer):
    (tmp_path / "images").mkdir()
    recs = []
    for i in range(3):
        name = f"images/x_{i}.jpg"
        Image.new("RGB", (50, 50), (i * 40, 0, 0)).save(tmp_path / name)
        recs.append({"image": name, "prompt": "Ảnh có gì?",
                     "response": f"cái thứ {i}", "source": "viic"})
    jl = tmp_path / "train.jsonl"
    jl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs),
                  encoding="utf-8")
    ds = SFTDataset(str(jl), tiny_tokenizer, str(tmp_path), max_text_len=64)
    assert len(ds) == 3
    batch = collate([ds[0], ds[1], ds[2]], pad_id=0)
    B, T = batch["input_ids"].shape
    assert B == 3 and batch["labels"].shape == (B, T)
    assert batch["pixel_values"].shape == (3, 3, 224, 224)
    # chỗ pad: labels phải -100
    lens = [len(ds[i]["input_ids"]) for i in range(3)]
    for i, L in enumerate(lens):
        assert (batch["labels"][i, L:] == -100).all()
