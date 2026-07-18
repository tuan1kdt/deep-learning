"""Giai đoạn A: pretrain GPT tiếng Việt trên token bin.

Resume-safe: checkpoint chứa optimizer + RNG (torch/cuda/generator sampling)
nên train 6 step liền = train 3 + resume 3 (test đảm bảo).
"""
import argparse
import csv
import math
import os
import time
from dataclasses import asdict

import torch

from vivlm.config import GPTConfig, PretrainConfig, pick_device
from vivlm.data.loader import TokenBin
from vivlm.models.gpt import GPT


def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))


def save_ckpt(path, model, optimizer, step, cfg, gen):
    raw = getattr(model, "_orig_mod", model)        # gỡ wrapper torch.compile
    torch.save({
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "gpt_config": asdict(cfg.gpt),
        "gen_state": gen.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
    }, path)


def load_ckpt(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def _log(csv_path, row):
    new = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "split", "loss", "lr", "tok_per_s", "vram_gb"])
        w.writerow(row)


@torch.no_grad()
def estimate_val_loss(model, val_bin, cfg, device, ctx):
    model.eval()
    g = torch.Generator().manual_seed(cfg.seed + 999)   # val cố định
    losses = []
    for _ in range(cfg.val_iters):
        x, y = val_bin.sample(cfg.micro_batch, cfg.gpt.context, device, g)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(cfg: PretrainConfig, device=None, resume=None, log_every=10):
    device = device or pick_device()
    device_type = device.split(":")[0]
    torch.manual_seed(cfg.seed)
    ctx = (torch.autocast(device_type, dtype=torch.bfloat16)
           if device_type == "cuda" else torch.autocast("cpu", enabled=False))

    model = GPT(cfg.gpt).to(device)
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.lr, (cfg.beta1, cfg.beta2), device_type)
    gen = torch.Generator().manual_seed(cfg.seed)
    start = 0
    if resume:
        ck = load_ckpt(resume, device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        gen.set_state(ck["gen_state"])
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        start = ck["step"]
        print(f"resume từ step {start}")
    if cfg.compile and device_type == "cuda":
        model = torch.compile(model)

    train_bin, val_bin = TokenBin(cfg.train_bin), TokenBin(cfg.val_bin)
    grad_accum = cfg.batch_tokens // (cfg.micro_batch * cfg.gpt.context)
    os.makedirs(cfg.out_dir, exist_ok=True)
    model.train()
    t0, tokens_done = time.time(), 0

    for step in range(start, cfg.max_steps):
        lr = get_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            x, y = train_bin.sample(cfg.micro_batch, cfg.gpt.context, device, gen)
            with ctx:
                _, loss = model(x, y)
            (loss / grad_accum).backward()
            loss_acc += loss.item() / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        tokens_done += cfg.batch_tokens

        if (step + 1) % log_every == 0 or step == start:
            dt = time.time() - t0
            tps = tokens_done / dt
            vram = (torch.cuda.max_memory_allocated() / 1e9
                    if device_type == "cuda" else 0.0)
            print(f"step {step+1}/{cfg.max_steps} loss {loss_acc:.4f} "
                  f"lr {lr:.2e} {tps/1e3:.1f}k tok/s vram {vram:.1f}GB")
            _log(cfg.log_csv, [step + 1, "train", f"{loss_acc:.4f}",
                               f"{lr:.2e}", f"{tps:.0f}", f"{vram:.2f}"])
        if (step + 1) % cfg.val_every == 0:
            vl = estimate_val_loss(model, val_bin, cfg, device, ctx)
            print(f"  val loss {vl:.4f}")
            _log(cfg.log_csv, [step + 1, "val", f"{vl:.4f}", "", "", ""])
        if (step + 1) % cfg.ckpt_every == 0 or step + 1 == cfg.max_steps:
            save_ckpt(os.path.join(cfg.out_dir, f"step{step+1:06d}.pt"),
                      model, optimizer, step + 1, cfg, gen)
            save_ckpt(os.path.join(cfg.out_dir, "latest.pt"),
                      model, optimizer, step + 1, cfg, gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro-batch", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--resume")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()
    cfg = PretrainConfig()
    if args.micro_batch:
        cfg.micro_batch = args.micro_batch
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.no_compile:
        cfg.compile = False
    train(cfg, device=args.device, resume=args.resume)


if __name__ == "__main__":
    main()
