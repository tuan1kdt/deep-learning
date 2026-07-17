"""Training loop SFT cho captioning — theo khuôn midterm/train.py.

Hai run chính của báo cáo (chung seed, chung feature, chỉ khác decoder):

    .venv/bin/python -m final.train --decoder lstm
    .venv/bin/python -m final.train --decoder transformer

Ablation attention (thí nghiệm #2):

    .venv/bin/python -m final.train --decoder lstm --no-attention

Smoke test local (subset 512 mẫu, 2 epoch):

    .venv/bin/python -m final.train --decoder lstm --smoke

Mỗi run lưu outputs/<run>/{config.json, history.json, curves.png} và
checkpoint tốt nhất theo VAL LOSS tại checkpoints/<run>.pt. history còn ghi
caption greedy của 3 ảnh val cố định sau MỖI epoch — nguyên liệu cho mục
"model học như thế nào" của báo cáo.
"""
import argparse
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from final.config import Config, pick_device
from final.data.dataset import load_caption_dataset, load_eval_data
from final.data.vocab import PAD_ID, Vocab
from final.models.caption_model import build_model
from final.models.decoding import greedy_decode


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_criterion(label_smoothing: float) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=label_smoothing)


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Per-step: warmup tuyến tính → cosine về 0. Transformer cần warmup vì
    pre-norm + AdamW lúc đầu có gradient lớn; LSTM đặt warmup_steps=0."""

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_validation(model, loader, criterion, device) -> float:
    model.eval()
    loss_sum, n_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            feats = batch["feat"].to(device)
            cap_in = batch["cap_in"].to(device)
            cap_out = batch["cap_out"].to(device)
            logits, _ = model(feats, cap_in)
            mask = cap_out != PAD_ID
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             cap_out.reshape(-1))
            loss_sum += loss.item() * mask.sum().item()
            n_tokens += mask.sum().item()
    return loss_sum / n_tokens


def plot_curves(history: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train")
    ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("epoch"), ax.set_ylabel("cross-entropy"), ax.legend()
    ax.set_title("Loss (per token)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(cfg: Config, smoke: bool = False) -> float:
    set_seed(cfg.seed)
    device = pick_device()
    print(f"Device: {device} | run: {cfg.run_name} | decoder: {cfg.decoder}"
          f" | attention: {cfg.use_attention}")

    vocab = Vocab.load(cfg.vocab_path)
    train_ds = load_caption_dataset(cfg, "train", vocab)
    val_ds = load_caption_dataset(cfg, "validation", vocab)
    # 3 ảnh val cố định để theo dõi caption qua các epoch
    probe_feats, probe_refs = load_eval_data(cfg, "validation")
    probe_feats = probe_feats[:3].to(device)
    if smoke:
        train_ds = Subset(train_ds, range(512))
        val_ds = Subset(val_ds, range(128))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    model = build_model(cfg, len(vocab)).to(device)
    total, trainable = model.count_parameters()
    print(f"Tham số: {total / 1e6:.1f}M tổng | {trainable / 1e6:.1f}M trainable")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    max_epochs = 2 if smoke else cfg.max_epochs
    scheduler = build_scheduler(optimizer, cfg.warmup_steps,
                                total_steps=max_epochs * len(train_loader))
    criterion = make_criterion(cfg.label_smoothing)

    out_dir = Path(cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.checkpoint_dir) / f"{cfg.run_name}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    history = {"train_loss": [], "val_loss": [], "samples": []}
    best_loss, epochs_no_improve = float("inf"), 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum, n_tokens = 0.0, 0
        for batch in train_loader:
            feats = batch["feat"].to(device)
            cap_in = batch["cap_in"].to(device)
            cap_out = batch["cap_out"].to(device)

            optimizer.zero_grad()
            logits, _ = model(feats, cap_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             cap_out.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            n = (cap_out != PAD_ID).sum().item()
            loss_sum += loss.item() * n
            n_tokens += n

        val_loss = run_validation(model, val_loader, criterion, device)
        seqs, _ = greedy_decode(model, probe_feats, cfg.max_len)
        samples = [vocab.decode(s) for s in seqs]
        history["train_loss"].append(loss_sum / n_tokens)
        history["val_loss"].append(val_loss)
        history["samples"].append(samples)
        print(f"Epoch {epoch:02d} | train {loss_sum / n_tokens:.4f}"
              f" | val {val_loss:.4f} | \"{samples[0]}\"")

        if val_loss < best_loss:
            best_loss, epochs_no_improve = val_loss, 0
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.to_dict(),
                "vocab_size": len(vocab),
                "epoch": epoch,
                "val_loss": val_loss,
            }, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"Early stopping tại epoch {epoch} (patience {cfg.patience})")
                break

    (out_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2))
    plot_curves(history, out_dir / "curves.png")
    print(f"Best val loss: {best_loss:.4f} | checkpoint: {ckpt_path}")
    return best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train captioning Flickr8k")
    parser.add_argument("--decoder", default="lstm",
                        choices=["lstm", "transformer"])
    parser.add_argument("--no-attention", action="store_true",
                        help="LSTM dùng mean-pool thay attention (ablation #2)")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--smoke", action="store_true",
                        help="512 mẫu + 2 epoch: kiểm tra pipeline end-to-end")
    args = parser.parse_args()

    use_attention = not args.no_attention
    default_name = f"{args.decoder}{'' if use_attention else '_noattn'}"
    run_name = args.run_name or (f"{default_name}_smoke" if args.smoke
                                 else default_name)
    cfg = Config(decoder=args.decoder, use_attention=use_attention,
                 run_name=run_name)
    if args.smoke:
        cfg.batch_size = 64
        cfg.num_workers = 0
    train(cfg, smoke=args.smoke)


if __name__ == "__main__":
    main()
