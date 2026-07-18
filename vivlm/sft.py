"""Giai đoạn B: SFT hai pha.

Pha "projector": chỉ train projector (GPT + encoder đóng băng) — cho projector
"bắt sóng" không gian embedding của GPT trước, tránh phá não ngôn ngữ.
Pha "full": mở GPT + projector (encoder vẫn đóng), lr nhỏ hơn 10x.
"""
import argparse
import csv
import math
import os
from dataclasses import asdict
from functools import partial

import torch
from torch.utils.data import DataLoader

from vivlm.config import GPTConfig, SFTConfig, pick_device
from vivlm.data.sft_dataset import SFTDataset, collate
from vivlm.models.gpt import GPT
from vivlm.models.projector import PixelShuffleProjector
from vivlm.models.vivlm import SiglipAdapter, ViVLM
from vivlm.pretrain import load_ckpt


def build_vlm(pretrain_ckpt, cfg, device, encoder=None):
    ck = load_ckpt(pretrain_ckpt, device)
    gpt = GPT(GPTConfig(**ck["gpt_config"]))
    gpt.load_state_dict(ck["model"])
    enc = encoder if encoder is not None else SiglipAdapter(cfg.siglip_name)
    proj = PixelShuffleProjector(in_dim=768, out_dim=gpt.cfg.d_model)
    return ViVLM(gpt, enc, proj).to(device)


def save_vlm(path, vlm, step):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"gpt": vlm.gpt.state_dict(),
                "projector": vlm.projector.state_dict(),
                "gpt_config": asdict(vlm.gpt.cfg), "step": step}, path)


def load_vlm(path, cfg, device, encoder=None):
    ck = torch.load(path, map_location=device, weights_only=False)
    gpt = GPT(GPTConfig(**ck["gpt_config"]))
    gpt.load_state_dict(ck["gpt"])
    enc = encoder if encoder is not None else SiglipAdapter(cfg.siglip_name)
    proj = PixelShuffleProjector(in_dim=768, out_dim=gpt.cfg.d_model)
    proj.load_state_dict(ck["projector"])
    return ViVLM(gpt, enc, proj).to(device)


def _cosine_lr(step, total, peak, floor, warmup):
    if step < warmup:
        return peak * (step + 1) / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def train_sft(cfg: SFTConfig, phase, device=None, pretrain_ckpt=None,
              init_ckpt=None, encoder=None, tokenizer=None, max_steps=None):
    from tokenizers import Tokenizer
    device = device or pick_device()
    device_type = device.split(":")[0]
    torch.manual_seed(cfg.seed)
    tok = tokenizer or Tokenizer.from_file("vivlm/data/tokenizer.json")
    pad_id = tok.token_to_id("<|endoftext|>")

    if phase == "projector":
        vlm = build_vlm(pretrain_ckpt, cfg, device, encoder)
        peak, floor = cfg.lr_projector, cfg.lr_projector   # lr hằng, pha ngắn
    elif phase == "full":
        vlm = load_vlm(init_ckpt, cfg, device, encoder)
        peak, floor = cfg.lr_full, cfg.min_lr_full
    else:
        raise ValueError(phase)

    for p in vlm.parameters():
        p.requires_grad_(False)
    params = vlm.trainable_parameters(phase)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=peak, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))

    train_ds = SFTDataset(cfg.train_jsonl, tok, cfg.img_root, cfg.max_text_len,
                          cfg.img_size)
    loader = DataLoader(train_ds, batch_size=cfg.micro_batch, shuffle=True,
                        num_workers=0 if device_type != "cuda" else 8,
                        collate_fn=partial(collate, pad_id=pad_id),
                        drop_last=True, pin_memory=(device_type == "cuda"))
    steps_per_epoch = max(1, len(loader) // cfg.grad_accum)
    total = (cfg.steps_projector if phase == "projector"
             else steps_per_epoch * cfg.epochs_full)
    if max_steps:
        total = min(total, max_steps)
    ctx = (torch.autocast(device_type, dtype=torch.bfloat16)
           if device_type == "cuda" else torch.autocast("cpu", enabled=False))

    vlm.train()
    losses, step, it = [], 0, iter(loader)
    while step < total:
        lr = _cosine_lr(step, total, peak, floor, cfg.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(cfg.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            px = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with ctx:
                _, loss = vlm(px, ids, labels)
            (loss / cfg.grad_accum).backward()
            acc += loss.item() / cfg.grad_accum
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        losses.append(acc)
        step += 1
        if step % 20 == 0 or step == 1:
            print(f"[{phase}] step {step}/{total} loss {acc:.4f} lr {lr:.2e}")
            _log(cfg.log_csv, [phase, step, f"{acc:.4f}", f"{lr:.2e}"])
    save_vlm(os.path.join(cfg.out_dir, f"{phase}.pt"), vlm, step)
    return losses


def _log(csv_path, row):
    new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["phase", "step", "loss", "lr"])
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["projector", "full"], required=True)
    ap.add_argument("--pretrain-ckpt",
                    default="vivlm/checkpoints/pretrain/latest.pt")
    ap.add_argument("--init-ckpt", default="vivlm/checkpoints/sft/projector.pt")
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()
    cfg = SFTConfig()
    if args.micro_batch:
        cfg.micro_batch = args.micro_batch
    train_sft(cfg, args.phase, pretrain_ckpt=args.pretrain_ckpt,
              init_ckpt=args.init_ckpt, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
