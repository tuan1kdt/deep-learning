"""Cấu hình tập trung cho ViVLM-nano. Script KHÔNG hardcode hyperparameter."""
from dataclasses import dataclass, field

import torch

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|image|>"]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class GPTConfig:
    vocab_size: int = 20480
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    context: int = 1024
    rope_theta: float = 10000.0
    mlp_hidden: int = 2048        # SwiGLU hidden ≈ 8/3 * d_model


@dataclass
class PretrainConfig:
    gpt: GPTConfig = field(default_factory=GPTConfig)
    tokenizer_json: str = "vivlm/data/tokenizer.json"
    train_bin: str = "vivlm/data/bin/train.bin"
    val_bin: str = "vivlm/data/bin/val.bin"
    micro_batch: int = 32          # chỉnh sát 24GB trên trainbox bằng --micro-batch
    batch_tokens: int = 524288     # ~0.5M token/step -> grad_accum = 16 với micro 32
    max_steps: int = 6000          # ~3B token
    lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500        # spec ghi 2000 — sửa: 2000/6000 = 33% schedule là quá dài
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    val_every: int = 250
    val_iters: int = 50
    ckpt_every: int = 500
    out_dir: str = "vivlm/checkpoints/pretrain"
    log_csv: str = "vivlm/outputs/pretrain_log.csv"
    seed: int = 42
    compile: bool = True


@dataclass
class SFTConfig:
    siglip_name: str = "google/siglip-base-patch16-224"
    n_img_tokens: int = 49         # 196 patch pixel-shuffle scale 2 -> 49
    img_size: int = 224
    train_jsonl: str = "vivlm/data/sft/train.jsonl"
    val_jsonl: str = "vivlm/data/sft/val.jsonl"
    img_root: str = "vivlm/data/sft"
    max_text_len: int = 256        # caption/QA ngắn; seq thật = 49 + 255
    micro_batch: int = 24
    grad_accum: int = 4
    lr_projector: float = 1e-3
    steps_projector: int = 500
    lr_full: float = 1e-4
    min_lr_full: float = 1e-5
    warmup_steps: int = 100
    epochs_full: int = 2
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_every: int = 200
    out_dir: str = "vivlm/checkpoints/sft"
    log_csv: str = "vivlm/outputs/sft_log.csv"
    seed: int = 42


@dataclass
class SCSTConfig:
    refs_jsonl: str = "vivlm/data/sft/train_caption_refs.jsonl"
    caption_prompt: str = "Mô tả bức ảnh."
    batch_size: int = 32
    max_steps: int = 400
    max_new_tokens: int = 40
    lr: float = 1e-5
    grad_clip: float = 1.0
    ckpt_every: int = 100
    out_dir: str = "vivlm/checkpoints/scst"
    log_csv: str = "vivlm/outputs/scst_log.csv"
    seed: int = 42
