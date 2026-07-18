import numpy as np
import torch

from vivlm.config import GPTConfig, PretrainConfig
from vivlm.pretrain import get_lr, train, load_ckpt

TINY_GPT = dict(vocab_size=64, n_layer=2, n_head=2, d_model=16,
                context=8, mlp_hidden=32)


def _tiny_cfg(tmp_path, max_steps=6):
    np.random.seed(0)   # hai run a/b phải cùng data thì resume mới so được
    np.random.randint(0, 64, 5000, dtype=np.uint16).tofile(tmp_path / "tr.bin")
    np.random.randint(0, 64, 1000, dtype=np.uint16).tofile(tmp_path / "va.bin")
    return PretrainConfig(
        gpt=GPTConfig(**TINY_GPT),
        train_bin=str(tmp_path / "tr.bin"), val_bin=str(tmp_path / "va.bin"),
        micro_batch=2, batch_tokens=32, max_steps=max_steps,
        warmup_steps=2, val_every=3, val_iters=2, ckpt_every=3,
        out_dir=str(tmp_path / "ckpt"), log_csv=str(tmp_path / "log.csv"),
        compile=False)


def test_lr_schedule():
    cfg = PretrainConfig(warmup_steps=100, max_steps=1000, lr=6e-4, min_lr=6e-5)
    assert get_lr(0, cfg) < get_lr(50, cfg) < get_lr(100, cfg)
    assert abs(get_lr(100, cfg) - 6e-4) < 1e-9          # đỉnh sau warmup
    assert abs(get_lr(1000, cfg) - 6e-5) < 1e-9         # min_lr ở cuối
    assert abs(get_lr(550, cfg) - (6e-4 + 6e-5) / 2) < 1e-5  # giữa cosine


def test_train_writes_ckpt_and_log(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    train(cfg, device="cpu")
    assert (tmp_path / "ckpt" / "latest.pt").exists()
    log = (tmp_path / "log.csv").read_text()
    assert "train" in log and "val" in log


def test_ckpt_key_set(tmp_path):
    # khóa contract checkpoint mà Task 7/8/10 phụ thuộc — đúng 7 key, không dư/thiếu
    cfg = _tiny_cfg(tmp_path)
    train(cfg, device="cpu")
    ck = torch.load(cfg.out_dir + "/latest.pt", weights_only=False)
    assert set(ck.keys()) == {
        "model", "optimizer", "step", "gpt_config",
        "gen_state", "torch_rng", "cuda_rng"}


def test_resume_equivalence(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    cfg_a = _tiny_cfg(tmp_path / "a", max_steps=6)
    train(cfg_a, device="cpu")
    w_full = load_ckpt(cfg_a.out_dir + "/latest.pt", "cpu")["model"]

    cfg_b = _tiny_cfg(tmp_path / "b", max_steps=3)
    train(cfg_b, device="cpu")                      # dừng ở step 3, có ckpt
    cfg_b.max_steps = 6
    train(cfg_b, device="cpu", resume=cfg_b.out_dir + "/latest.pt")
    w_resumed = load_ckpt(cfg_b.out_dir + "/latest.pt", "cpu")["model"]

    for k in w_full:
        assert torch.allclose(w_full[k], w_resumed[k], atol=1e-6), k
