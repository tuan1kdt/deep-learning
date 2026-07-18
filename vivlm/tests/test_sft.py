import json
from dataclasses import replace

import torch
from PIL import Image

from vivlm.config import GPTConfig, PretrainConfig, SFTConfig
from vivlm.models.gpt import GPT
from vivlm.pretrain import save_ckpt
from vivlm.sft import build_vlm, load_vlm, save_vlm, train_sft
from vivlm.tests.test_fusion import TINY, StubEncoder


def _pretrain_ckpt(tmp_path, gpt_cfg=TINY):
    cfg = PretrainConfig(gpt=gpt_cfg)
    m = GPT(gpt_cfg)
    opt = m.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
    p = str(tmp_path / "pre.pt")
    save_ckpt(p, m, opt, 1, cfg, torch.Generator())
    return p


def _sft_files(tmp_path):
    (tmp_path / "images").mkdir()
    recs = []
    for i in range(4):
        name = f"images/x_{i}.jpg"
        Image.new("RGB", (40, 40), (i * 30, 10, 10)).save(tmp_path / name)
        recs.append({"image": name, "prompt": "Ảnh có gì?",
                     "response": f"vật thể {i}", "source": "viic"})
    for split in ("train", "val"):
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs),
            encoding="utf-8")


def _cfg(tmp_path):
    return SFTConfig(
        train_jsonl=str(tmp_path / "train.jsonl"),
        val_jsonl=str(tmp_path / "val.jsonl"),
        img_root=str(tmp_path), micro_batch=2, grad_accum=1,
        steps_projector=20, epochs_full=1, warmup_steps=2, val_every=100,
        max_text_len=32, out_dir=str(tmp_path / "out"),
        log_csv=str(tmp_path / "log.csv"),
        # StubEncoder trả (B,196,768) HẰNG SỐ (toàn 0) -> fc1.weight nhận
        # gradient=0 mọi bước (input=0 => dL/dW = dL/dout * x^T = 0); chỉ
        # fc1.bias/fc2.weight/fc2.bias học được. Với 20 bước x micro_batch=2
        # (2/4 mẫu mỗi bước, shuffle), lr mặc định 1e-3 quá nhỏ so với nhiễu
        # lấy mẫu -> loss[-1] vs loss[0] không ổn định (xác nhận thực nghiệm:
        # fail ở seed 0/1, pass ở seed 2). lr=1e-2 cho biên độ hội tụ rõ và ổn
        # định qua nhiều seed — không đổi default SFTConfig.lr_projector vì
        # SigLIP thật không cho đặc trưng ảnh hằng số như stub.
        lr_projector=1e-2)


def test_projector_phase_learns(tmp_path, tiny_tokenizer):
    _sft_files(tmp_path)
    torch.manual_seed(0)
    # TINY.vocab_size=64 (test_fusion.py) < tiny_tokenizer's actual vocab
    # (BPE trained to 300, see conftest.py) -> tok_emb IndexError. Build the
    # pretrain ckpt's GPT with a vocab large enough for the real tokenizer,
    # same pattern as test_sample.py's custom vocab=300 GPTConfig.
    gpt_cfg = replace(TINY, vocab_size=tiny_tokenizer.get_vocab_size())
    losses = train_sft(_cfg(tmp_path), "projector", "cpu",
                       pretrain_ckpt=_pretrain_ckpt(tmp_path, gpt_cfg),
                       encoder=StubEncoder(), tokenizer=tiny_tokenizer)
    assert losses[-1] < losses[0]        # overfit 4 mẫu -> loss phải giảm
    assert (tmp_path / "out" / "projector.pt").exists()


def test_save_load_roundtrip(tmp_path):
    vlm = build_vlm(_pretrain_ckpt(tmp_path), SFTConfig(), "cpu",
                    encoder=StubEncoder())
    p = str(tmp_path / "v.pt")
    save_vlm(p, vlm, 7)
    vlm2 = load_vlm(p, SFTConfig(), "cpu", encoder=StubEncoder())
    px = torch.randn(1, 3, 224, 224)
    ids = torch.randint(0, TINY.vocab_size, (1, 6))
    with torch.no_grad():
        a, _ = vlm(px, ids)
        b, _ = vlm2(px, ids)
    assert torch.allclose(a, b, atol=1e-6)
